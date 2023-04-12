from abc import ABC, abstractmethod
from torch import nn

import torch


# =============================================================================
# Base class for approximating likelihood factors of the exponential family
# =============================================================================


class ExponentialFamilyFactor(ABC):
    """
    Base class for exponential family (EF) approximating likelihood factors.
    The exponential family is made up of distributions which can be written
    in the form
    
        p(θ | v) = h(θ) exp(ν.T f(θ) + A(ν)).
    
    For a list of exponential family distributions see:
        https://en.wikipedia.org/wiki/Category:Exponential_family_distributions
        https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions
        
    There are many more members but these are especially useful for our
    applications.
    """
    
    def __init__(self, nat_params, log_coeff=0., enforce_pos_var=False):
        
        self.nat_params = nat_params
        self.log_coeff = log_coeff
        self.enforce_pos_var = enforce_pos_var

    def compute_refined_factor(self, q1, q2, damping=1., valid_dist=False,
                               update_log_coeff=True):
        """
        Computes the log-coefficient and natural parameters of the
        approximating likelihood term **t** given by
        
            t(θ) = q1(θ) / q2(θ) t_(θ)
            
        where **t_** is the approximating likelihood term corresponding
        to **self**. Note that the log-coefficient computed here includes
        the normalising constants of the q-distributions as well as the
        coefficient of t_.
        """
        #print(q1.__dict__)
        #print(q1._unc_params)
        #print(q1._unc_params['loc'], q1._unc_params['log_scale'])
        #print(torch.exp(q1._unc_params['log_scale']))

        #import sys
        #sys.exit()

        # Convert distributions to log-coefficients and natural parameters
        np1 = q1.nat_params
        np2 = q2.nat_params
        
        # Compute natural parameters of the new t-factor (detach gradients)
        # update without clipping or noise:
        delta_np = {k: (np1[k].detach().clone() - np2[k].detach().clone())
                    for k in self.nat_params.keys()}

        nat_params = {k: v.detach().clone() + delta_np[k] * damping
                      for k, v in self.nat_params.items()}


        if valid_dist:
            # Constraint natural parameters to form valid distribution.
            nat_params = self.valid_nat_from_nat(nat_params)

        if update_log_coeff and not valid_dist:
            # TODO: does not work unless valid_dist = False.
            log_coeff = self.log_coeff + (q2.log_a() - q1.log_a()) * damping
            log_coeff = log_coeff.detach().clone()
        else:
            log_coeff = 0.
            
        # Create and return refined t of the same type
        t = type(self)(nat_params=nat_params, log_coeff=log_coeff, enforce_pos_var = self.enforce_pos_var)
        
        return t

    def __call__(self, thetas):
        """
        Returns the value of log t(θ) (up to a const. independent of θ)
        
            log t(θ) = log h(θ) + ν.T f(θ) + const.
            
        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        
        # Compute inner product ν.T f(θ), log h(θ), log t(θ) (all shape (N,))
        npf = self.npf(thetas)
        log_h = self.log_h(thetas)
        log_t = log_h + npf
        
        return log_t

    @abstractmethod
    def log_h(self, thetas):
        """
        Returns the value of log h(θ). Input **thetas** is assumed to be
        a torch.tensor of shape (N, D) where N is the batch dimension and
        D is the dimension of the distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def npf(self, thetas):
        """
        Rearranges NPs to ν vector for this member of the exponential family.
        """
        raise NotImplementedError

    def eqlogt(self, q, num_samples=1):
        """
        Computes E_q[log t(θ)] = ν.T E_q[f(θ)] + E_q[log h(θ)], ignoring the
        latter term.
        :param q: q(θ).
        :param num_samples: Number of samples to form MC estimate with, if
        closed-form solution not specified.
        :return: ν.T E_q[f(θ)].
        """
        thetas = q.rsample((num_samples,))
        return self(thetas).mean(0)

    @abstractmethod
    def nat_from_dist(self, q):
        """
        Takes a torch.distribution **q**, assumed to be in the EF
        and extracts its leading coefficient and natural parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def dist_from_nat(self, np):
        """
        Takes a dictionary of natural parameters **np** and returns a
        torch.distribution defined by these natural parameters.
        """
        raise NotImplementedError

    def valid_nat_from_nat(self, nat_params):
        return nat_params


# =============================================================================
# Base exponential family distribution
# =============================================================================

class ExponentialFamilyDistribution(ABC, nn.Module):
    
    def __init__(self,
                 std_params=None,
                 nat_params=None,
                 is_trainable=False,
                 enforce_pos_var=False):
        
        super().__init__()
    
        # Specify whether the distribution is trainable wrt its NPs
        self.is_trainable = is_trainable

        # Set all to None.
        self._nat_params = None
        self._std_params = None
        self._unc_params = None

        self._enforce_pos_var = enforce_pos_var
        
        # Initialise standard and natural parameters
        if is_trainable:
            # Only initialise std_params (initialises unc_params).
            if std_params is not None:
                self.std_params = std_params
            elif nat_params is not None:
                self.std_params = self._std_from_nat(nat_params)
            else:
                # No intitial parameter values specified.
                raise ValueError("No initial parameterisation specified. "
                                 "Cannot create optimisable parameters.")
        else:
            self.std_params = std_params
            self.nat_params = nat_params

    @abstractmethod
    def log_a(self, nat_params=None):
        """
        :param nat_params: Natural parameters η.
        :return: Log partition function, A(η).
        """
        raise NotImplementedError
        
    @property
    def enforce_pos_var(self):
        return self._enforce_pos_var

    @property
    def std_params(self):
        
        if self.is_trainable:
            return self._std_from_unc(self._unc_params)
        
        elif self._std_params is None:
            return self._std_from_nat(self._nat_params)
        
        else:
            return self._std_params

    @std_params.setter
    def std_params(self, std_params):

        if self.is_trainable:
            self._unc_params = nn.ParameterDict(self._unc_from_std(std_params))

        else:
            self._std_params = std_params
        
    @property
    def nat_params(self):
        
        # If _nat_params None or distribution trainable compute nat params
        if self.is_trainable or self._nat_params is None:
            self._nat_params = self._nat_from_std(self.std_params)
        
        return self._nat_params

    @nat_params.setter
    def nat_params(self, nat_params):

        self._nat_params = nat_params

    @property
    def mean_params(self):
        return self._mean_from_std(self.std_params)

    @abstractmethod
    def _std_from_unc(self, unc_params):
        raise NotImplementedError

    @abstractmethod
    def _unc_from_std(self, std_params):
        raise NotImplementedError

    @abstractmethod
    def _nat_from_std(self, std_params):
        raise NotImplementedError
    
    @abstractmethod
    def _std_from_nat(self, nat_params):
        raise NotImplementedError

    @abstractmethod
    def _mean_from_std(self, std_params):
        raise NotImplementedError

    def non_trainable_copy(self):

        if self.is_trainable:
            nat_params = None
            std_params = {k: v.detach().clone()
                          for k, v in self.std_params.items()}

        else:
            if self._std_params is not None:
                std_params = {k: v.detach().clone()
                              for k, v in self.std_params.items()}
            else:
                std_params = None

            if self._nat_params is not None:
                nat_params = {k: v.detach().clone()
                              for k, v in self.nat_params.items()}
            else:
                nat_params = None
        
        return type(self)(std_params, nat_params, is_trainable=False)

    def trainable_copy(self):

        if self.is_trainable:
            nat_params = None
            std_params = {k: v.detach().clone()
                          for k, v in self.std_params.items()}

        else:
            if self._std_params is not None:
                std_params = {k: v.detach().clone()
                              for k, v in self.std_params.items()}
            else:
                std_params = None

            if self._nat_params is not None:
                nat_params = {k: v.detach().clone()
                              for k, v in self.nat_params.items()}
            else:
                nat_params = None

        return type(self)(std_params, nat_params, is_trainable=True, enforce_pos_var=self._enforce_pos_var)

    @property
    def distribution(self):
        return self.torch_dist_class(**self.std_params)

    def kl_divergence(self, other):
        return torch.distributions.kl_divergence(self.distribution,
                                                 other.distribution)
    
    def log_prob(self, *args, **kwargs):
        return self.distribution.log_prob(*args, **kwargs)
    
    def sample(self, *args, **kwargs):
        return self.distribution.sample(*args, **kwargs)

    def rsample(self, *args, **kwargs):
        return self.distribution.rsample(*args, **kwargs)

    @property
    @abstractmethod
    def torch_dist_class(self):
        raise NotImplementedError

    def create_new(self, **kwargs):
        return type(self)(**kwargs)
