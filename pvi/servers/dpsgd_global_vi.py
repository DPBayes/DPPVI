
from collections import defaultdict
import logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

from .base import Server
from pvi.utils.dataset import ListDataset

logger = logging.getLogger(__name__)

# =============================================================================
# Global VI class: use DPSGD for private learning on full data
# =============================================================================


class GlobalVIServer(Server):
    def __init__(self, model, p, clients, config=None, init_q=None):
        super().__init__(model, p, clients, config, init_q)


        # Global VI server has access to the entire dataset.
        self.data = {k: torch.cat([client.data[k] for client in self.clients],
                                  dim=0)
                     for k in self.clients[0].data.keys()}

        # Tracks number of epochs.
        self.epochs = 0

        if config['track_client_norms']:
            self.pre_dp_norms = []
            self.post_dp_norms = []

        # Initial q(θ) for optimisation.
        if self.init_q is not None:
            self.q = init_q.non_trainable_copy()

    def get_default_config(self):
        # NOTE: set all confs in calling script instead of in here
        return {}

    def tick(self):
        """
        Trains the model using the entire dataset.
        """
        if self.should_stop():
            return False

        p = self.p.non_trainable_copy()
        q = self.q.trainable_copy()


        if self.config['batch_size'] is None:
            batch_size = int(np.floor(self.config['sampling_frac_q']*len(self.data["y"])))
        else:
            batch_size = self.config['batch_size']
        print(f'batch size: {batch_size}')

        # Parameters are those of q(θ) and self.model.
        if self.config["train_model"]:
            parameters = [
                {"params": q.parameters()},
                {"params": self.model.parameters(),
                 **self.config["model_optimiser_params"]}
            ]
        else:
            parameters = q.parameters()

        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])
        lr_scheduler = getattr(torch.optim.lr_scheduler,
                               self.config["lr_scheduler"])(
            optimiser, **self.config["lr_scheduler_params"])

        # Set up data: global VI server can access the entire dataset.

        # TODO: current only supports homogenous minibatches.
        if self.config["homogenous_split"]:
            x = self.data["x"]
            y = self.data["y"]

            # Shuffle data.
            tensor_dataset = TensorDataset(x, y)

            # set up data loader with chosen sampling type
            # sequential data pass modes
            if self.config['dp_mode'] not in ['dpsgd','param_fixed']:
                loader = DataLoader(tensor_dataset,
                                batch_size=batch_size,#self.config["batch_size"],
                                shuffle=True)
                n_epochs = self.config['epochs']
                n_samples = len(loader)

            # swor data pass modes
            else:
                # regular SWOR sampler
                if self.config['dp_mode'] == 'dpsgd':
                    sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(tensor_dataset, replacement=False), batch_size=batch_size, drop_last=False)
                
                    loader = DataLoader(tensor_dataset, batch_sampler=sampler)

                # use only fixed single minibatch for local learning for each global update
                elif self.config['dp_mode'] == 'param_fixed':
                    inds = torch.randint(low=0,high=len(tensor_dataset),size=(batch_size,))
                    loader = DataLoader( torch.utils.data.Subset(tensor_dataset, indices=inds) )
                else:
                    raise ValueError(f"Unexpected dp_mode in base client: {self.config['dp_mode']}")

                n_epochs = 1
                n_samples = self.config['epochs']

            #loader = DataLoader(tensor_dataset,
            #                    batch_size=self.config["batch_size"],
            #                    shuffle=True)

        else:
            raise NotImplementedError('Inhomogeneous split not supported!')


        # Dict for logging optimisation progress
        training_curve = {
            "elbo": [],
            "kl": [],
            "ll": [],
        }

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(n_epochs), desc="Epoch", leave=True, disable=self.config['pbar'])
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = {
                "elbo" : 0,
                "kl"   : 0,
                "ll"   : 0,
            }
            
            # Loop over batches in current epoch
            tmp = iter(loader)
            for i_step in range(n_samples):
                try:
                    #(x_batch, y_batch) = tmp.next()
                    (x_batch, y_batch) = next(tmp)
                except StopIteration as err:
                    tmp = iter(loader)
                    #(x_batch, y_batch) = tmp.next()
                    (x_batch, y_batch) = next(tmp)

                self.communications += 1

                if self.config['dp_mode'] == 'dpsgd':
                    # simple DP-SGD implementation
                    trace_tmp = np.zeros(2) # for keeping track of ELBO and logl

                    # initialise grad accumulator
                    cum_grads = {}
                    for i_weight, p_ in enumerate(filter(lambda p_: p_.requires_grad, q.parameters())):
                        #if p_.grad is not None:
                        cum_grads[str(i_weight)] = torch.zeros_like(p_)

                    grad_norm_tracker = 0
                    # quick hack for DP-SGD: process each sample separately
                    for x_single, y_single in zip(x_batch,y_batch):
                        optimiser.zero_grad()

                        batch = {
                            "x" : torch.unsqueeze(x_single,0),
                            "y" : torch.unsqueeze(y_single,0),
                        }

                        # Compute KL divergence between q and p.
                        kl = q.kl_divergence(p).sum() / len(self.data["x"])

                        # note: avg over minibatch only after clipping per-example grads & noising the sum
                        # Sample θ from q and compute p(y | θ, x) for each θ
                        ll = self.model.expected_log_likelihood(
                            batch, q, self.config["num_elbo_samples"]).sum()
                        #ll /= self.config['batch_size'] # rescale when using minibatches

                        # Negative local free energy is KL minus log-probability.
                        loss = kl - ll

                        loss.backward(retain_graph=False) # keep graph when kl is computed outside loop

                        trace_tmp[0] += ll.item()
                        trace_tmp[1] += -loss.item()

                        # NOTE: assume that all parameters for dp are from q
                        g_norm = torch.zeros(1)
                        for p_ in filter(lambda p_: p_.requires_grad, q.parameters()):
                            #if p_.grad is not None:
                            g_norm += torch.sum(p_.grad**2)
                        g_norm = torch.sqrt(g_norm)
                        
                        
                        if self._config['track_client_norms']:
                        #    # track mean grad norm over samples in the minibatch
                            grad_norm_tracker += g_norm.item()/batch_size

                        # clip and accumulate grads
                        for i_weight, p_ in enumerate(filter(lambda p_: p_.requires_grad, q.parameters())):
                            #if p_.grad is not None:
                            cum_grads[str(i_weight)] += (p_.grad/torch.clamp(g_norm/self.config['dp_C'], min=1)).detach().clone()
                    
                    # add noise to clipped grads and avg
                    for key, p_ in zip( cum_grads, filter(lambda p_: p_.requires_grad, q.parameters()) ):
                        p_.grad = self.config['dp_C']*self.config['dp_sigma']*torch.randn_like(p_.grad) + cum_grads[key]
                        p_.grad /= batch_size

                    if self._config['track_client_norms']:
                        self.pre_dp_norms.append(grad_norm_tracker)
                        # calculate grad norm post DP treatment
                        g_norm = torch.zeros(1)
                        for p_ in filter(lambda p_: p_.requires_grad, q.parameters()):
                            g_norm += torch.sum(p_.grad**2)
                        g_norm = torch.sqrt(g_norm)
                        self.post_dp_norms.append(g_norm.item())

                    ### end loop over single samples in minibatch ###

                # no dpsgd
                else:
                    optimiser.zero_grad()
                    batch = {
                        "x" : x_batch,
                        "y" : y_batch,
                    }

                    # Compute KL divergence between q and p.
                    kl = q.kl_divergence(p).sum() / len(self.data["x"])

                    # Sample θ from q and compute p(y | θ, x) for each θ
                    ll = self.model.expected_log_likelihood(
                        batch, q, self.config["num_elbo_samples"]).sum()
                    ll /=len(x_batch)

                    # Negative local free energy is KL minus log-probability.
                    loss = kl - ll
                    loss.backward()

                optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"],
                                   lr=optimiser.param_groups[0]["lr"])

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["ll"].append(epoch["ll"])

            if i % self.config["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"Epochs: {i}.")

            # Log q so we can track performance each epoch.
            self.log["q"].append(q.non_trainable_copy())
            self.log["communications"].append(self.communications)
            self.epochs += 1

            # Update learning rate.
            lr_scheduler.step()

        # Update global posterior.
        self.q = q.non_trainable_copy()

        self.iterations += 1

        # Log training.
        self.log["training_curves"].append(training_curve)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1

