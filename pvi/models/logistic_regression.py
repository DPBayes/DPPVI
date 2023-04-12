import torch
import numpy as np

from torch import distributions, nn, optim
from .base import Model
from pvi.distributions.exponential_family_distributions import \
    MultivariateGaussianDistribution


class LogisticRegressionModel(Model, nn.Module):
    """
    Logistic regression model with a multivariate Gaussian approximate
    posterior.
    """
    
    conjugate_family = None

    def __init__(self, include_bias=True, **kwargs):
        self.include_bias = include_bias

        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

    def get_default_nat_params(self):
        if self.include_bias:
            return {
                "np1": torch.tensor([0.]*(self.config["D"] + 1)),
                "np2": torch.tensor(
                    [1.]*(self.config["D"] + 1)).diag_embed(),
            }
        else:
            return {
                "np1": torch.tensor([0.] * self.config["D"]),
                "np2": torch.tensor([1.] * self.config["D"]).diag_embed(),
            }

    def get_default_config(self):
        return {}

    def get_default_hyperparameters(self):
        """
        :return: A default set of ε for the model.
        """
        return {}

    def forward(self, x, q, **kwargs):
        """
        Returns the (approximate) predictive posterior distribution of a
        Bayesian logistic regression model.
        :param x: The input locations to make predictions at.
        :param q: The approximate posterior distribution q(θ).
        :return: ∫ p(y | θ, x) q(θ) dθ ≅ (1/M) Σ_m p(y | θ_m, x) θ_m ~ q(θ).
        """
        if self.config["use_probit_approximation"]:
            # Use Probit approximation.
            q_loc = q.std_params["loc"]

            if self.include_bias:
                x_ = torch.cat((x, torch.ones(len(x)).unsqueeze(-1)), dim=1)
            else:
                x_ = x

            x_ = x_.unsqueeze(-1)

            if str(type(q)) == str(MultivariateGaussianDistribution):
                q_cov = q.std_params["covariance_matrix"]
            else:
                q_scale = q.std_params["scale"]
                q_cov = q_scale.diag_embed() ** 2

            denom = x_.transpose(-1, -2).matmul(q_cov).matmul(x_).reshape(-1)
            denom = (1 + np.pi * denom / 8) ** 0.5
            logits = q_loc.unsqueeze(-2).matmul(x_).reshape(-1) / denom

            return distributions.Bernoulli(logits=logits)

        else:
            thetas = q.distribution.sample(
                (self.config["num_predictive_samples"],))

            comp_ = self.likelihood_forward(x, thetas)
            comp = distributions.Bernoulli(logits=comp_.logits.T)
            mix = distributions.Categorical(torch.ones(len(thetas),))

            return distributions.MixtureSameFamily(mix, comp)

    def likelihood_forward(self, x, theta, **kwargs):
        """
        Returns the model's likelihood p(y | θ, x).
        :param x: Input of shape (*, D).
        :param theta: Parameters of shape (*, D + 1).
        :return: Bernoulli distribution.
        """
        assert len(x.shape) in [1, 2], "x must be (*, D)."
        assert len(theta.shape) in [1, 2], "theta must be (*, D)."

        if self.include_bias:
            x_ = torch.cat((x, torch.ones(len(x)).unsqueeze(-1)), dim=1)
        else:
            x_ = x

        if len(theta.shape) == 1:
            logits = x_.unsqueeze(-2).matmul(theta.unsqueeze(-1))
        else:
            if len(x.shape) == 1:
                x_r = x_.unsqueeze(0).repeat(len(theta), 1)
                logits = x_r.unsqueeze(-2).matmul(
                    theta.unsqueeze(-1)).reshape(-1)
            else:
                x_r = x_.unsqueeze(0).repeat(len(theta), 1, 1)
                theta_r = theta.unsqueeze(1).repeat(1, len(x_), 1)
                logits = x_r.unsqueeze(-2).matmul(
                    theta_r.unsqueeze(-1)).reshape(len(theta), len(x_))

        return distributions.Bernoulli(logits=logits)

    def conjugate_update(self, data, q, t=None):
        """
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :param t: The the local factor t(θ).
        :return: q_new, t_new, the new global posterior and the new local
        contribution.
        """
        raise NotImplementedError
