import logging
import torch
import numpy as np

from abc import ABC
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


# =============================================================================
# Client class
# =============================================================================

import sys

class Client(ABC):
    
    def __init__(self, data, model, t=None, config=None):

        if config is None:
            config = {}


        #self._config = self.get_default_config()
        self._config = config
        
        # Set data partition and likelihood
        self.data = data
        self.model = model
        
        # Set likelihood approximating term
        self.t = t
        
        self.log = defaultdict(list)
        self._can_update = True

        #self.optimiser = None

        self.freeze_var_updates = config['freeze_var_updates']
        self.update_counter = 0

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = {**self._config, **config}

    @classmethod
    def get_default_config(cls):
        return {}

    def can_update(self):
        """
        A check to see if this client can indeed update. Examples of reasons
        one may not be is that they haven't finished optimisation.
        """
        return self._can_update
    
    def fit(self, q, init_q=None, global_prior=None):
        """
        Computes the refined approximating posterior (q) and associated
        approximating likelihood term (t). This method differs from client to
        client, but in all cases it calls Client.q_update internally.
        """
        return self.update_q(q, init_q, global_prior)

    def update_q(self, q, init_q=None, global_prior=None):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """

        # Type(q) is self.model.conjugate_family.
        if str(type(q)) == str(self.model.conjugate_family) \
                and not self.config["train_model"]:
            # No need to make q trainable.
            q_new, self.t = self.model.conjugate_update(self.data, q, self.t)
        else:
            # Pass a trainable copy to optimise.
            q_new, self.t = self.gradient_based_update(p=q, init_q=init_q, global_prior=global_prior)

        return q_new, self.t

    def gradient_based_update(self, p, init_q=None, global_prior=None):
        # Cannot update during optimisation.
        self._can_update = False

        if self.config['batch_size'] is None:
            batch_size = int(np.floor(self.config['sampling_frac_q']*len(self.data["y"])))
        else:
            batch_size = self.config['batch_size']
        #print(f"batch size {batch_size}, noise std:{self.config['dp_sigma']}")
        
        # Copy the approximate posterior, make old posterior non-trainable.
        q_old = p.non_trainable_copy()

        if self.t is None:
            # Standard VI: prior = old posterior.
            q_cav = p.non_trainable_copy()
        else:
            # TODO: check if valid distribution.

            q_cav = p.non_trainable_copy()
            q_cav.nat_params = {k: v - self.t.nat_params[k]
                                for k, v in q_cav.nat_params.items()}

        if init_q is not None:
            q = init_q.trainable_copy()
        else:
            # Initialise to prior.
            q = p.trainable_copy()

        # Parameters are those of q(θ) and self.model.
        if self.config["train_model"]:
            if "model_optimiser_params" in self.config:
                parameters = [
                    {"params": q.parameters()},
                    {"params": self.model.parameters(),
                     **self.config["model_optimiser_params"]}
                ]
            else:
                parameters = [
                    {"params": q.parameters()},
                    {"params": self.model.parameters()}
                ]
        else:
            #print(q._unc_params['log_scale'].requires_grad)
            if self.freeze_var_updates > self.update_counter:
                logger.debug('Freezing log_scale params')
                q._unc_params['log_scale'].requires_grad = False
            parameters = q.parameters()

        #if self.optimiser is None:
        # restart optimiser after each global update
        optimiser = getattr(torch.optim, self.config["optimiser"])(
        parameters, **self.config["optimiser_params"])
        #self.optimiser = optimiser
        if self.config['use_lr_scheduler']:
            lr_scheduler = getattr(torch.optim.lr_scheduler,
                           self.config["lr_scheduler"])(
                            optimiser, **self.config["lr_scheduler_params"])
            try:
                lr_scheduler.load_state_dict(self.lr_scheduler_states)
            except:
                pass
        #else:
        #    optimiser = self.optimiser

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)


        loader = DataLoader(tensor_dataset,
                            batch_size=batch_size,
                            shuffle=True)

        # sample either given number of full epochs or given number of batches
        if self.config['dp_mode'] == 'nondp_batches':
            n_epochs = 1
            n_samples = self.config['epochs']
        elif self.config['dp_mode'] == 'nondp_epochs':
            n_epochs = self.config['epochs']
            n_samples = len(loader)
        else:
            raise ValueError(f"Unknown dp_mode: {self.config['dp_mode']}!")

        # Dict for logging optimisation progress
        training_curve = {
            "elbo" : [],
            "kl"   : [],
            "ll"   : [],
            "logt" : [],
        }
        
        # Gradient-based optimisation loop -- loop over epochs
        #epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch",
        #                  leave=True)
        epoch_iter = tqdm(range(n_epochs), desc="Epoch", leave=True, disable=self.config['pbar'])
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = {
                "elbo" : 0,
                "kl"   : 0,
                "ll"   : 0,
                "logt" : 0,
            }
            
            # Loop over batches in current epoch
            tmp = iter(loader)
            #for i_step, (x_batch, y_batch) in enumerate(iter(loader)):

            for i_step in range(n_samples):
                try:
                    #(x_batch, y_batch) = tmp.next()
                    (x_batch, y_batch) = next(tmp)
                except StopIteration as err:
                    tmp = iter(loader)
                    #(x_batch, y_batch) = tmp.next()
                    (x_batch, y_batch) = next(tmp)

                #logger.debug(f'optimiser starting step {i_step} with total batch_size {len(y_batch)}')

                optimiser.zero_grad()
                batch = {
                    "x" : x_batch,
                    "y" : y_batch,
                }

                # Compute KL divergence between q and q_cav.
                try:
                    kl = q.kl_divergence(q_cav).sum()/len(self.data["x"])
                except ValueError as err:
                    # NOTE: removed dirty fix: q_cav not guaranteed to give proper std, might give errors
                    logger.warning('\nException in KL: probably caused by invalid cavity distribution')
                    raise err

                # Sample θ from q and compute p(y | θ, x) for each θ
                ll = self.model.expected_log_likelihood(
                    batch, q, self.config["num_elbo_samples"]).sum()
                ll /=len(x_batch)

                if self.t is not None:
                    # how slow is this?
                    # Compute E_q[log t(θ)].this is only for bookkeeping, not used in loss
                    logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                    #logt /= len(x) # use full data len to be comparable; doesn't matter since not used in optimisation

                # Negative local free energy is KL minus log-probability.
                loss = kl - ll
                loss.backward()

                # try natural params
                if 0:
                    for i_weight, p_ in enumerate(filter(lambda p_: p_.requires_grad, q.parameters())):
                        #print(p_.grad)
                        if i_weight == 0:
                            p_.grad = (torch.exp(list(q.parameters())[1]*2) * p_.grad).detach().clone()
                        elif i_weight == 1:
                            p_.grad = (torch.exp(list(q.parameters())[1]*2)/2 * p_.grad).detach().clone()
                        else:
                            raise ValueError('Got more than 2 set of weights!')

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item() / n_samples
                epoch["kl"] += kl.item() / n_samples
                epoch["ll"] += ll.item() / n_samples
                if self.t is not None:
                    epoch["logt"] += logt.item() / n_samples

                optimiser.step()
                i_step += 1

                ### end loop over minibatches ###

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["ll"].append(epoch["ll"])

            if self.t is not None:
                training_curve["logt"].append(epoch["logt"])

            if i % self.config["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"log t: {epoch['logt']:.3f}, "
                             f"Epochs: {i}.")

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"], logt=epoch["logt"],
                                   lr=optimiser.param_groups[0]["lr"])

            ### end loop over local steps ###

        # Update learning rate.
        if self.config['use_lr_scheduler']:
            lr_scheduler.step()
            # optimiser zeroed after each global update, so change lr by hand
            self.config['optimiser_params']['lr'] = lr_scheduler.get_last_lr()[0]
            self.lr_scheduler_state = lr_scheduler.state_dict()

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)
        
        # Create non-trainable copy to send back to server
        q_new = q.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        self.update_counter += 1

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                q, q_old, damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
                update_log_coeff=self.config["update_log_coeff"])

            # note: only t is currently used by server:
            return q_new, t_new

        else:
            logger.debug('Note: client not returning t')
            return q_new, None


class ClientBayesianHypers(ABC):
    """
    PVI client with Bayesian treatment of model hyperparameters.
    """
    def __init__(self, data, model, t=None, teps=None, config=None):

        if config is None:
            config = {}

        #self._config = self.get_default_config()
        self._config = config

        # Set data partition and likelihood
        self.data = data
        self.model = model

        # Set likelihood approximating term
        self.t = t
        self.teps = teps

        self.log = defaultdict(list)
        self._can_update = True

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = {**self._config, **config}

    @classmethod
    def get_default_config(cls):
        return {}
        '''
            "damping_factor": 1.,
            "valid_factors": False,
            "epochs": 1,
            "batch_size": 100,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 0.05},
            "num_elbo_samples": 10,
            "num_elbo_hyper_samples": 1,
            "print_epochs": 1
        }'''

    def can_update(self):
        """
        A check to see if this client can indeed update. Examples of reasons
        one may not be is that they haven't finished optimisation.
        :return:
        """
        return self._can_update

    def fit(self, q, qeps):
        """
        Computes the refined approximating posterior (q) and associated
        approximating likelihood term (t). This method differs from client to
        client, but in all cases it calls Client.q_update internally.
        """
        return self.update_q(q, qeps)

    def update_q(self, q, qeps):
        """
        Computes a refined approximate posterior and the associated
        approximating likelihood term.
        """
        # Pass a trainable copy to optimise.
        q_new, qeps_new, self.t, self.teps = self.gradient_based_update(
            q.trainable_copy(), qeps.trainable_copy())

        return q_new, qeps_new, self.t, self.teps

    def gradient_based_update(self, q, qeps):
        # Cannot update during optimisation.
        self._can_update = False

        if self.t is None:
            # Old posterior = prior, make non-trainable.
            q_cav = q.non_trainable_copy()
            qeps_cav = qeps.non_trainable_copy()
        else:
            q_cav = q.non_trainable_copy()
            q_cav.nat_params = {k: v - self.t.nat_params[k]
                                for k, v in q_cav.nat_params.items()}

            qeps_cav = qeps.non_trainable_copy()
            for k1, v1 in qeps.distributions.items():
                qeps.distributions[k1].nat_params = {
                    k2: v2 - self.teps.factors[k1].nat_params[k2]
                    for k2, v2 in v1.nat_params.items()}

        parameters = list(q.parameters()) + qeps.parameters()

        # Reset optimiser. Parameters are those of q(θ) and q(ε).
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=True)

        # Dict for logging optimisation progress
        training_curve = {
            "elbo": [],
            "kl": [],
            "kleps": [],
            "ll": [],
            "logt": [],
            "logteps": [],
        }

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(self.config["epochs"]), desc="Epoch")
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = {
                "elbo": 0,
                "kl": 0,
                "kleps": 0,
                "ll": 0,
                "logt": 0,
                "logteps": 0,
            }

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute KL divergence between q and p.
                kl = q.kl_divergence(q_cav).sum() / len(x)
                kleps = sum(qeps.kl_divergence(qeps_cav).values()) / len(x)

                # Estimate E_q[log p(y | x, θ, ε)].
                ll = 0
                for _ in range(self.config["num_elbo_hyper_samples"]):
                    eps = qeps.rsample()
                    self.model.hyperparameters = eps
                    ll += self.model.expected_log_likelihood(
                        batch, q, self.config["num_elbo_samples"]).sum()

                ll /= (self.config["num_elbo_hyper_samples"] * len(x_batch))

                if self.t is not None:
                    logt = self.t.eqlogt(q, self.config["num_elbo_samples"])
                    logteps = sum(self.teps.eqlogt(
                        qeps, self.config["num_elbo_samples"]).values())
                    logt /= len(x)
                    logteps /= len(x)

                    # loss = kl + kleps - ll + logt - logteps

                loss = kl + kleps - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["kleps"] += kleps.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

                if self.t is not None:
                    epoch["logt"] += logt.item() / len(loader)
                    epoch["logteps"] += logteps.item() / len(loader)

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["kleps"].append(epoch["kleps"])
            training_curve["ll"].append(epoch["ll"])

            if self.t is not None:
                training_curve["logt"].append(epoch["logt"])
                training_curve["logteps"].append(epoch["logteps"])

            if i % self.config["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"KL eps: {epoch['kleps']:.3f}, "
                             f"log t: {epoch['logt']: .3f},"
                             f"log teps: {epoch['logteps']: .3f},"
                             f"Epochs: {i}.")

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"], kleps=epoch["kleps"],
                                   logt=epoch["logt"],
                                   logteps=epoch["logteps"])

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)

        # Create non_trainable_copy to send back to server.
        q_new = q.non_trainable_copy()
        qeps_new = qeps.non_trainable_copy()

        # Finished optimisation, can now update.
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                q, q_cav, damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"])
            teps_new = self.teps.compute_refined_factor(
                qeps, qeps_cav, damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"])

            return q_new, qeps_new, t_new, teps_new

        else:
            return q_new, qeps_new, None, None
