
from abc import ABC
from collections import defaultdict
import itertools
import logging
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

from .base import Client

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

# =============================================================================
# Local federated averaging client class
# =============================================================================


class VaryingBatchSampler():
    """
    Batch sampler class that supports different batch size for each batch
    """

    def __init__(self, batch_sizes):
        self.batch_sizes = batch_sizes

    def __iter__(self):
        return iter([list(range(cur-b,cur)) for cur,b in zip(itertools.accumulate(self.batch_sizes),iter(self.batch_sizes))])

    def __len__(self):
        return len(self.batch_sizes)


class LFA_Client(Client):
    
    def __init__(self, data, model, t, config=None):
        

        if config is None:
            config = {}

        super().__init__(data=data, model=model, t=t, config=config)

        # Initialise optimiser states
        if config['batch_size'] is None:
            self.n_local_models = int(np.ceil(1/(config['sampling_frac_q']))  )
        else:
            self.n_local_models = int(np.floor((self.data['y'].shape[-1])/config['batch_size']))
        self.optimiser_states = None
        self.lr_scheduler_states = None
        

    def gradient_based_update(self, p, init_q=None, global_prior=None):
        # Cannot update during optimisation.
        self._can_update = False
        
        # Copy the approximate posterior, make old posterior non-trainable.
        q_old = p.non_trainable_copy()

        # use different b for different models to use all data once: first models might have more data than last
        logger.debug('Using differing batch_size in local models to use all local data')
        tmp1 = int(np.floor((self.data['y'].shape[-1])/self.n_local_models))
        if tmp1 < 1:
            raise ValueError('Using batch_size < 1! Try increasing sampling frac!')
        tmp2 = len(self.data['y']) - tmp1*self.n_local_models
        batch_sizes = np.zeros(self.n_local_models, dtype=int) + tmp1
        batch_sizes[:tmp2] += 1

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
            raise NotImplementedError("LFA not implemented when train_model=True")
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
            if self.freeze_var_updates > self.update_counter:
                logger.debug('Freezing log_scale params')
                q._unc_params['log_scale'].requires_grad = False
            parameters = q.parameters()

        param_accumulator = {}
        if self.config['noisify_np']:
            for k in q.nat_params:
                param_accumulator[k] = torch.zeros_like(q._nat_params[k])
        else:
            for i_p,p in enumerate(q.parameters()):
                param_accumulator[str(i_p)] = torch.zeros_like(p)

        # create optimiser states on the first call
        if self.optimiser_states is None:

            optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])

            self.optimiser_states = []
            self.lr_scheduler_states = []
            for i_state in range(self.n_local_models):
                self.optimiser_states.append(optimiser.state_dict())
                if self.config['use_lr_scheduler']:
                    lr_scheduler = getattr(torch.optim.lr_scheduler, self.config["lr_scheduler"])(
                                    optimiser, **self.config["lr_scheduler_params"])
                    self.lr_scheduler_states.append(lr_scheduler.state_dict())

        # Set up data
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)

        # when only using single batch size can use standard DataLoader
        if len(np.unique(batch_sizes)) == 1:
            loader = DataLoader(tensor_dataset,
                        batch_size=int(batch_sizes[0]),
                        shuffle=False)
        else:
            loader = DataLoader(tensor_dataset,
                        batch_sampler=VaryingBatchSampler(batch_sizes),
                        shuffle=False
                        )

        # Dict for logging optimisation progress
        # note: for LFA log just means (mean over all local training step)
        training_curve = {
            "elbo" : [],
            "kl"   : [],
            "ll"   : [],
            "logt" : [],
        }
        
        # Gradient-based optimisation loop -- loop over epochs

        # NOTE: should fix these, maybe use means from all samples
        epoch = {
            "elbo" : 0,
            "kl"   : 0,
            "ll"   : 0,
            "logt" : 0,
        }
        
        model_checkpoint = q.trainable_copy()

        # Loop over samples
        batch_iter = tqdm(iter(loader), desc="Batch", leave=True, disable=self.config['pbar'])

        for i_batch, (x_batch, y_batch) in enumerate(batch_iter):
            #print(f'batch=model number {i_batch} ')

            # start optimiser for the current sample from existing state
            optimiser = getattr(torch.optim, self.config["optimiser"])(
            q.parameters(), **self.config["optimiser_params"])
            optimiser.load_state_dict(self.optimiser_states[i_batch])
            if self.config['use_lr_scheduler']:
                lr_scheduler = getattr(torch.optim.lr_scheduler,
                                       self.config["lr_scheduler"])(
                    optimiser, **self.config["lr_scheduler_params"])
                lr_scheduler.load_state_dict(self.lr_scheduler_states[i_batch])

            batch = {
                "x" : x_batch,
                "y" : y_batch,
            }

            # optimise separate model on each batch for some number of steps
            for i_step in range(self.config['epochs']):

                optimiser.zero_grad()

                # Compute KL divergence between q and q_cav.
                try:
                    kl = q.kl_divergence(q_cav).sum() / len(x)
                except ValueError as err:
                    # NOTE: removed dirty fix: q_cav not guaranteed to give proper std, might give errors
                    print('\nException in KL: probably caused by invalid cavity distribution')
                    raise err

                # Sample θ from q and compute p(y | θ, x) for each θ
                ll = self.model.expected_log_likelihood(
                    batch, q, self.config["num_elbo_samples"]).sum()
                ll /= batch_sizes[i_batch]

                if self.t is not None:
                    # Compute E_q[log t(θ)]. this is only for bookkeeping, not used in loss
                    logt = self.t.eqlogt(q, self.config["num_elbo_samples"])

                # Negative local free energy is KL minus log-probability.
                loss = kl - ll # NOTE: doesn't have LFA regularizer at the moment, should add?

                loss.backward()

                if self.config['track_client_norms']:
                    delta_param_norm = 0
                    for p0, p in zip(model_checkpoint.parameters(), q.parameters()):
                        delta_param_norm += torch.sum((p0-p)**2)
                    delta_param_norm = torch.sqrt(delta_param_norm)
                    if i_batch == 0 and i_step == 0:
                        self.pre_dp_norms.append(np.zeros(self.config['epochs'] ))
                    self.pre_dp_norms[-1][i_step] += delta_param_norm.item()/len(loader)

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item() / self.config['epochs']
                epoch["kl"] += kl.item() / self.config['epochs']
                epoch["ll"] += ll.item() / self.config['epochs']

                if self.t is not None:
                    epoch["logt"] += logt.item() / self.config['epochs'] 

                # Log progress for current epoch: use mean over all samples for each local epochs
                if i_batch == 0:
                    training_curve["elbo"].append(epoch["elbo"] / len(loader))
                    training_curve["kl"].append(epoch["kl"] / len(loader))
                    training_curve["ll"].append(epoch["ll"] / len(loader))

                    if self.t is not None:
                        training_curve["logt"].append(epoch["logt"] / len(loader))
                else:
                    training_curve["elbo"][i_step] += epoch["elbo"] / len(loader)
                    training_curve["kl"][i_step] += epoch["kl"] / len(loader)
                    training_curve["ll"][i_step] += epoch["ll"] / len(loader)

                    if self.t is not None:
                        training_curve["logt"][i_step] += epoch["logt"] / len(loader)

                optimiser.step()

            batch_iter.set_postfix(elbo=epoch["elbo"]/(i_batch+1), kl=epoch["kl"]/(i_batch+1),
                               ll=epoch["ll"]/(i_batch+1), logt=epoch["logt"]/(i_batch+1))
                               #lr=optimiser.param_groups[0]["lr"])

            # save optimiser state
            if self.config['use_lr_scheduler']:
                lr_scheduler.step()
                self.lr_scheduler_states[i_batch] = lr_scheduler.state_dict()
            # note: need to save optimiser state after scheduler step to include possible lr changes
            self.optimiser_states[i_batch] = optimiser.state_dict()
           
            # get norm of the change in params
            with torch.no_grad():
                delta_param_norm = 0
                # clip and noisify natural params as opposed to (unconstrained) loc-scale
                if self.config['noisify_np']:
                    for k in q.nat_params:
                        delta_param_norm += torch.sum((q.nat_params[k] - model_checkpoint.nat_params[k])**2)
                else:
                    for p0, p in zip(model_checkpoint.parameters(), q.parameters()):
                        delta_param_norm += torch.sum((p0-p)**2)
                delta_param_norm = torch.sqrt(delta_param_norm)
                logger.debug(f'delta norm: {delta_param_norm}')

                # clip, accumulate clipped change in params, and return to the model checkpoint
                if self.config['noisify_np']:

                    for k in q.nat_params:
                        param_accumulator[k] += ((q.nat_params[k]-model_checkpoint.nat_params[k])/torch.clamp(delta_param_norm/self.config['dp_C'], min=1)).detach().clone()

                    # return to model checkpoint
                    for i_param, (p0,p) in enumerate(zip(model_checkpoint.parameters(), q.parameters())):
                        p.data = p0.detach().clone()

                else:
                    for i_param, (p0,p) in enumerate(zip(model_checkpoint.parameters(), q.parameters())):
                        param_accumulator[str(i_param)] += ((p-p0)/torch.clamp(delta_param_norm/self.config['dp_C'], min=1)).detach().clone()

                        # return to model checkpoint for the next sample
                        p.data = p0.detach().clone()

            for p in q.parameters():
                p.requires_grad = True

        # add noise to sum of clipped change in parameters, take avg, add DP change in params to starting point to get new params
        if self.config['noisify_np']:
            tmp = {}
            # pre clip noise to mitigate bias from clipping
            if self.config['pre_clip_sigma'] > 0:
                for k in q.nat_params:
                    tmp[k] = self.config['pre_clip_sigma']*torch.randn_like(q.nat_params[k])

            for k in q.nat_params:
                tmp[k] = (model_checkpoint.nat_params[k] + (param_accumulator[k] + self.config['dp_C']*self.config['dp_sigma']*torch.randn_like(q.nat_params[k]) )/self.n_local_models).detach().clone()

            tmp = q._unc_from_std(q._std_from_nat(tmp))
            for p_new, k in zip(q.parameters(),tmp):
                p_new.data =  tmp[k]

        else:
            if self.config['pre_clip_sigma'] > 0:
                raise NotImplementedError('Pre clip noise not implemented!')
            for i_param, (p0,p) in enumerate(zip(model_checkpoint.parameters(), q.parameters())):
                p.data = (p0 +  (param_accumulator[str(i_param)] + self.config['dp_C']*self.config['dp_sigma']*torch.randn_like(p))/self.n_local_models).detach().clone()


        if self.config['track_client_norms']:
            # get norm of the change in params
            delta_param_norm = 0
            if self.config['noisify_np']:
                for k in q.nat_params:
                    delta_param_norm += torch.sum((q.nat_params[k] - model_checkpoint.nat_params[k])**2)
            else:
                for p0, p in zip(model_checkpoint.parameters(), q.parameters()):
                    delta_param_norm += torch.sum((p0-p)**2)
            delta_param_norm = torch.sqrt(delta_param_norm)
            #param_norm_trace[i_batch] = delta_param_norm.cpu().numpy()
            self.post_dp_norms.append(delta_param_norm.item())

        #if i % self.config["print_epochs"] == 0:
        logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                     f"LL: {epoch['ll']:.3f}, "
                     f"KL: {epoch['kl']:.3f}, "
                     f"log t: {epoch['logt']:.3f}, ")
                     #f"Epochs: {i}.")

        # Update learning rate.
        #lr_scheduler.step()

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)
        
        # Create non-trainable copy to send back to server
        q_new = q.non_trainable_copy()
        
        # Finished optimisation, can now update.
        self.update_counter += 1
        self._can_update = True

        if self.t is not None:
            # Compute new local contribution from old distributions
            t_new = self.t.compute_refined_factor(
                q, q_old, damping=self.config["damping_factor"],
                valid_dist=self.config["valid_factors"],
                update_log_coeff=self.config["update_log_coeff"])

            return q_new, t_new
        else:
            logger.debug('Note: client not returning t')
            return q_new, None
