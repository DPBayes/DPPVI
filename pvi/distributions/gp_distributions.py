"""
Identical to the multivariate Gaussian distributions in exponential_family_*.py
files, except they maintain their own inducing locations at which they are
defined.
"""
import torch

from pvi.distributions.exponential_family_distributions import \
    MultivariateGaussianDistribution
from pvi.distributions.exponential_family_factors import \
    MultivariateGaussianFactor

from torch import nn


class MultivariateGaussianDistributionWithZ(MultivariateGaussianDistribution):
    def __init__(self,
                 inducing_locations=None,
                 std_params=None,
                 nat_params=None,
                 is_trainable=False,
                 train_inducing=True):
        super().__init__(std_params, nat_params, is_trainable)

        self.train_inducing = train_inducing

        if is_trainable and inducing_locations is not None:
            self._inducing_locations = nn.Parameter(
                inducing_locations, requires_grad=self.train_inducing)
        else:
            self._inducing_locations = inducing_locations

    @property
    def inducing_locations(self):
        return self._inducing_locations

    @inducing_locations.setter
    def inducing_locations(self, value):
        if self.is_trainable:
            self._inducing_locations = nn.Parameter(
                value, requires_grad=self.train_inducing)
        else:
            self._inducing_locations = value

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

        if self._inducing_locations is not None:
            inducing_locations = self.inducing_locations.detach().clone()
        else:
            inducing_locations = None

        return type(self)(inducing_locations, std_params, nat_params,
                          is_trainable=False,
                          train_inducing=self.train_inducing)

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

        if self._inducing_locations is not None:
            inducing_locations = self.inducing_locations.detach().clone()
        else:
            inducing_locations = None

        return type(self)(inducing_locations, std_params, nat_params,
                          is_trainable=True,
                          train_inducing=self.train_inducing)

    def create_new(self, **kwargs):
        if "inducing_locations" not in kwargs:
            kwargs = {
                **kwargs,
                "inducing_locations": self.inducing_locations
            }
        if "train_inducing" not in kwargs:
            kwargs = {
                **kwargs,
                "train_inducing": self.train_inducing
            }

        return type(self)(**kwargs)


class MultivariateGaussianFactorWithZ(MultivariateGaussianFactor):
    def __init__(self, inducing_locations=None, nat_params=None,
                 train_inducing=True):
        super().__init__(nat_params)

        self.train_inducing = train_inducing
        self._inducing_locations = inducing_locations

    def compute_refined_factor(self, q1, q2, damping=1., valid_dist=False):
        """
        Computes the log-coefficient and natural parameters of the
        approximating likelihood term **t** given by

            t(θ) = q1(θ) / q2(θ) t_(θ)

        where **t_** is the approximating likelihood term corresponding
        to **self**. Note that the log-coefficient computed here includes
        the normalising constants of the q-distributions as well as the
        coefficient of t_.
        """

        assert torch.allclose(q1.inducing_locations.detach(),
                              q2.inducing_locations.detach()), \
            "Inducing locations must be the same."

        # Convert distributions to log-coefficients and natural parameters
        np1 = q1.nat_params
        np2 = q2.nat_params
        inducing_locations = q1.inducing_locations.detach().clone()

        # Compute natural parameters of the new t-factor (detach gradients)
        delta_np = {k: (np1[k].detach().clone() - np2[k].detach().clone())
                    for k in self.nat_params.keys()}
        nat_params = {k: v.detach().clone() + delta_np[k] * damping
                      for k, v in self.nat_params.items()}

        if valid_dist:
            # Constraint natural parameters to form valid distribution.
            nat_params = self.valid_nat_from_nat(nat_params)

        # Create and return refined t of the same type.
        t = type(self)(inducing_locations=inducing_locations,
                       nat_params=nat_params,
                       train_inducing=self.train_inducing)

        return t

    def forward(self, x):
        raise NotImplementedError

    @property
    def inducing_locations(self):
        return self._inducing_locations

    @inducing_locations.setter
    def inducing_locations(self, value):
        self._inducing_locations = value
