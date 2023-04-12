from .base import ExponentialFamilyFactor
from .exponential_family_distributions import *

import torch

MIN_PRECISION = 1e-6

# =============================================================================
# Mean field Gaussian factor
# =============================================================================
    

class MeanFieldGaussianFactor(ExponentialFamilyFactor):
    
    distribution_class = MeanFieldGaussianDistribution
    
    def log_h(self, thetas):
        """
        Returns the value of log h(θ) for the MeanFieldFactor class. For
        a mean-field multivariate Gaussian, log h(θ) = 0.
            
        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        return torch.zeros(size=thetas.shape[:1])

    def npf(self, thetas):
        
        np1 = self.nat_params["np1"]
        np2 = self.nat_params["np2"]
        
        npf = torch.mv(thetas, np1)
        npf = npf + torch.mv(thetas ** 2, np2)
        
        return npf

    def eqlogt(self, q, num_samples=1):

        np1 = self.nat_params["np1"]
        np2 = self.nat_params["np2"]

        loc = q.std_params["loc"]
        scale = q.std_params["scale"]

        eqlogt = np1.dot(loc) + np2.dot(scale ** 2 + loc ** 2)
        eqlogt += self.log_coeff

        return eqlogt

    def nat_from_dist(self, q):
        
        loc = q.loc.detach()
        scale = q.scale.detach()
        
        std = {
            "loc" : loc,
            "scale" : scale
        }
        
        return self.distribution_class._nat_from_std(std)
    
    def dist_from_nat(self, nat):
        
        std = self.distribution_class._std_from_nat(nat)
        dist = torch.distributions.Normal(**std)
        
        return dist

    def valid_nat_from_nat(self, nat_params):
        prec = -2 * nat_params["np2"]

        prec[prec <= 0] = MIN_PRECISION
        # NOTE: fix this: enabled by bool flags?
        # not currently used (valid=False in all tests)
        # NOTE: bug here, although not sure if mean should be fixed
        #nat_params["np1"][prec <= 0] = 0
        loc = nat_params["np1"] / prec

        nat_params["np2"] = -0.5 * prec
        nat_params["np1"] = loc * prec

        return nat_params

    
# =============================================================================
# Multivariate Gaussian factor
# =============================================================================


class MultivariateGaussianFactor(ExponentialFamilyFactor):

    distribution_class = MultivariateGaussianDistribution

    def log_h(self, thetas):
        """
        Returns the value of log h(θ) for the MultivariateGaussian class. For
        a multivariate Gaussian, log h(θ) = 0.
            
        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        return torch.zeros(size=thetas.shape[:1])

    def npf(self, thetas):
        
        np1 = self.nat_params["np1"]
        np2 = self.nat_params["np2"]

        npf = torch.mv(thetas, np1)
        npf = npf + torch.sum(thetas * torch.mm(thetas, np2), dim=1)
        
        return npf

    def eqlogt(self, q, num_samples=1):
        """
        E_q[log t(θ)] = nu.T E_q[f(θ)] + const (assumed independent of θ).
        """
        np1 = self.nat_params["np1"]
        np2 = self.nat_params["np2"].flatten()

        loc = q.std_params["loc"]
        cov = q.std_params["covariance_matrix"]

        m1 = loc
        m2 = (cov + loc.matmul(loc.T)).flatten()

        eqlogt = np1.dot(m1) + np2.dot(m2)
        eqlogt += self.log_coeff

        return eqlogt

    def nat_from_dist(self, q):
        
        loc = q.loc.detach()
        cov = q.covariance_matrix.detach()
        
        std = {
            "loc": loc,
            "covariance_matrix": cov
        }
        
        return self.distribution_class._nat_from_std(std)
    
    def dist_from_nat(self, nat):
        
        std = self.distribution_class._std_from_nat(nat)
        dist = torch.distributions.MultivariateNormal(**std)
        
        return dist


# =============================================================================
# Gamma factor
# =============================================================================


class GammaFactor(ExponentialFamilyFactor):

    distribution_class = GammaDistribution

    def log_h(self, thetas):
        """
        Returns the value of log h(θ) for the Gamma class. For a Gamma
        distribution, log h(θ) = 0.

        Input **thetas** is assumed to be a torch.tensor of shape (N, D)
        where N is the batch dimension and D is the dimension of the
        distribution.
        """
        return torch.zeros(size=thetas.shape[:1])

    def npf(self, thetas):

        np1 = self.nat_params["np1"]
        np2 = self.nat_params["np2"]

        npf = thetas.log() * np1
        npf = npf + thetas * np2

        return npf

    def nat_from_dist(self, q):
        concentration = q.concentration.detach()
        rate = q.rate.detach()

        std = {
            "concentration": concentration,
            "rate": rate
        }

        return self.distribution_class._nat_from_std(std)

    def dist_from_nat(self, nat):
        std = self.distribution_class._std_from_nat(nat)
        dist = torch.distributions.Gamma(**std)

        return dist

# =============================================================================
# Log-Normal factor
# =============================================================================


class LogNormalFactor(MeanFieldGaussianFactor):

    distribution_class = LogNormalDistribution

    def dist_from_nat(self, nat):
        std = self.distribution_class._std_from_nat(nat)
        dist = torch.distributions.LogNormal(**std)

        return dist
