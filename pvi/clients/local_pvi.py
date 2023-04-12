
from abc import ABC
from collections import defaultdict
import copy
import itertools
import logging
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

from .base import Client
from .param_dp_client import Param_DP_Client
from pvi.distributions.exponential_family_distributions import MeanFieldGaussianDistribution
from pvi.distributions.exponential_family_factors import MeanFieldGaussianFactor
from pvi.servers.synchronous_server import SynchronousServer
from pvi.servers.sequential_server import SequentialServer

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

# =============================================================================
# Local PVI client class
# =============================================================================


class VaryingBatchSampler():
    """
    Batch sampler class that supports different batch size for each batch
    """

    def __init__(self, batch_sizes):
        self.batch_sizes = batch_sizes

    def __iter__(self):
        return iter([list(range(cur-b,cur)) for cur,b in zip(itertools.accumulate(self.batch_sizes),iter(self.batch_sizes))])

    def __len__(self) -> int:
        return len(self.batch_sizes)


class Local_PVI_Client(Client):
    
    def __init__(self, data, model, t, config=None):
        
        super().__init__(data=data, model=model, t=t, config=config)

        if config is None:
            config = {}

        self._config = config
        
        #print(self.config)
        
        # Set data partition and likelihood
        self.data = data
        self.model = model

        # Initialise optimiser states
        if config['batch_size'] is None:
            self.n_local_models = int(np.ceil(1/(config['sampling_frac_q']))  )
        else:
            self.n_local_models = int(np.floor((self.data['y'].shape[-1])/config['batch_size']))
            #sys.exit()
        
        #self.optimiser_states = None
        
        # Set likelihood approximating term
        self.t = t # note: use internal partitions instead of single t for optimising, keep track of joint t with this
        
        self.log = defaultdict(list)
        self._can_update = True

        self.pseudo_clients = None
        self.pseudo_server = None
        self.batch_sizes = None

        # note: most tracking currently not working
        # actually tracks norm of change in params for LFA
        if self._config['track_client_norms']:
            self.pre_dp_norms = []
            self.post_dp_norms = []
            self.noise_norms = []


    def gradient_based_update(self, p, init_q=None, global_prior=None):
        # Cannot update during optimisation.
        self._can_update = False
 
        # use different b for different models to use all data once: first models might have more data than last
        if self.batch_sizes is None:
            tmp1 = int(np.floor((self.data['y'].shape[-1])/self.n_local_models))
            if tmp1 == 0:
                raise ValueError('Using batch_size=0! Try increasing sampling frac!')
            tmp2 = len(self.data['y']) - tmp1*self.n_local_models
            batch_sizes = np.zeros(self.n_local_models, dtype=int) + tmp1
            batch_sizes[:tmp2] += 1
            self.batch_sizes = batch_sizes
        else:
            batch_sizes = self.batch_sizes

        # create internal pseudo clients if don't exist
        if self.pseudo_clients is None:
            self.pseudo_clients = []
            cur_ind = 0
            for i_client, b in enumerate(batch_sizes):
                # Data of ith client
                data = {'x' : self.data['x'][cur_ind:(cur_ind+b),:], 'y' : self.data['y'][cur_ind:(cur_ind+b)] }
                cur_ind += b

                client_config =  copy.deepcopy(self.config)
                client_config['dp_mode'] = 'param'
                client_config['dp_sigma'] = self.config['dp_sigma']/(np.sqrt(self.n_local_models)) # without secure aggregation
                client_config['clients'] = self.n_local_models
                client_config['batch_size'] = int(np.maximum(np.floor(self.config['pseudo_client_q']*b),1)) # this matches param client


                t = MeanFieldGaussianFactor(nat_params = copy.deepcopy(self.t.nat_params))

                # Create client and store
                client = Param_DP_Client(data=data, model=self.model, t=t, config=client_config)
                self.pseudo_clients.append(client)

        # init internal pseudo-server
        if self.pseudo_server is None:
            server_config = {
                        'max_iterations' : self.config['n_global_updates'],
                        'train_model' : False,
                        'model_update_freq': 1,
                        'dp_C' : self.config['dp_C'], # no clipping on server, applied by pseudo-clients
                        'dp_sigma' : 0., # note: noise added by pseudo-clients
                        'enforce_pos_var' : self.config['enforce_pos_var'],
                        'dp_mode' : 'param',
                        "pbar" : self.config['pbar'], 
                        }

            ChosenServer = SynchronousServer
            #ChosenServer = SequentialServer # use sequential only for debugging; can't use with DP
            self.pseudo_server = ChosenServer(model=self.model,
                                    p=p.non_trainable_copy(), # global prior=eff prior for given client?
                                    init_q=init_q.non_trainable_copy(), 
                                    clients=self.pseudo_clients,
                                    config=server_config)

            # fix init_q problem with sequential global server:
            # local server uses init_q on 0th iteration, even when not actually 0th globally
            if init_q is None:
                server_config['max_iterations'] += 1
                self.pseudo_server.iterations = 1
                self.pseudo_server.q = p.non_trainable_copy()
                self.pseudo_server.p = global_prior.non_trainable_copy()
        else:
            # update pseudo-server model to match the updated global model
            self.pseudo_server.q = p.non_trainable_copy()
            
        pseudo_server = self.pseudo_server
        
        # run training loop (just a single update)
        pseudo_server.tick()

        # Dict for logging optimisation progress
        # for local PVI log just means (mean over all local training step)
        training_curve = {
            "elbo" : [],
            "kl"   : [],
            "ll"   : [],
            "logt" : [],
        }
        

        if self.config['n_step_dict'] is None:
            for k in training_curve:
                training_curve[k] = np.zeros(self.config['epochs'])
                for i_client, client in enumerate(self.pseudo_clients):
                    training_curve[k] += np.array(client.log['training_curves'][-1][k])/len(self.pseudo_clients)
            self.log["training_curves"].append(training_curve)

        
        # Finished optimisation, can now update.
        self._can_update = True
        self.update_counter += 1

        if self.config['track_client_norms']:
            tmp = np.zeros(3)
            for client in self.pseudo_clients:
                tmp[0] += client.pre_dp_norms[-1]
                tmp[1] += client.post_dp_norms[-1]
                tmp[2] += client.noise_norms[-1]
            self.pre_dp_norms.append(tmp[0]/len(self.pseudo_clients))
            self.post_dp_norms.append(tmp[1]/len(self.pseudo_clients))
            self.noise_norms.append(tmp[2]/len(self.pseudo_clients))


        # Create non-trainable copy to send back to server
        #q_new = pseudo_server.q.non_trainable_copy()

        #print(self.t.__dict__)
        #sys.exit()
        if self.t is not None:
            # Compute new local contribution from old distributions
            # note: possible damping is done on the pseudo-client level
            tmp = {}
            for k,v in self.t.nat_params.items():
                tmp[k] = torch.zeros_like(v)
                for i_client, client in enumerate(self.pseudo_clients):
                    tmp[k] += client.t.nat_params[k].detach().clone()

            # Create and return refined t of the same type
            # note: doesn't actually track log_coeff
            t_new = type(self.t)(nat_params=tmp, log_coeff=self.t.log_coeff, enforce_pos_var = self.t.enforce_pos_var)
            self.t = copy.deepcopy(t_new)

            # note for DP: only t is currently used by server:
            # NOTE: q_new doesn't do any damping!
            #return q_new, t_new
            return None, t_new
        else:
            raise NotImplementedError('Local PVI only implemented with explicit t factors!')
            #logger.debug('Note: client not returning t')
            #return q_new, None




