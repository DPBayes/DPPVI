import logging
import torch

from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pvi.clients.base import Client, ClientBayesianHypers
from pvi.utils.psd_utils import psd_inverse, add_diagonal
from pvi.distributions.gp_distributions import \
    MultivariateGaussianDistributionWithZ

logger = logging.getLogger(__name__)

JITTER = 1e-4


class ContinualLearningSGPClient(Client):
    def __init__(self, data, model, inducing_locations, config=None):
        super().__init__(data, model, config=config)

        # Private inducing locations Z_b.
        self.inducing_locations = inducing_locations

    def update_q(self, q):
        """
        Computes a refined approximate posterior.
        """
        return self.gradient_based_update(q)

    def gradient_based_update(self, q):
        """
        The gradient based update in the streaming SGP setting involves
        completely overhalling the current approximate posterior q with new
        inducing points, hence we override the default gradient_based_update
        function.
        :param q: The current approximate posterior, q(a | Z_a).
        :return q_new: The new approximate posterior, q(a, b | Z_a, Z_b).
        """
        # Cannot update during optimisation.
        self._can_update = False

        # Set up data etc.
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=True)

        # Copy current approximate posterior, ensuring non-trainable.
        qa = q.non_trainable_copy()
        za = qa.inducing_locations

        # Parameterise as q(b | a) N(b; mb + Aa, Sb), initialised as
        # q(b | a) = p(b | a).
        zb = self.inducing_locations
        mb = torch.zeros(len(zb))

        if za is not None:
            # Fixed during optimisation.
            qa_cov = qa.std_params["covariance_matrix"]
            qa_loc = qa.std_params["loc"]

            kaa = add_diagonal(self.model.kernel(za, za).detach(), JITTER)
            ikaa = psd_inverse(kaa)
            kbb = add_diagonal(self.model.kernel(zb, zb).detach(), JITTER)
            kba = self.model.kernel(zb, za).detach()
            # Initialise Sb = Kbb - Kba Kaa^{-1} Kab.
            sb = kbb - kba.matmul(ikaa).matmul(kba.T)
            sb_chol = torch.cholesky(sb)
        else:
            # Initialise Sb = Kbb.
            kbb = add_diagonal(self.model.kernel(zb, zb).detach(), JITTER)
            sb = kbb
            sb_chol = torch.cholesky(sb)

        # Variational parameters.
        zb = nn.Parameter(zb, requires_grad=True)
        mb = nn.Parameter(mb, requires_grad=True)
        sb_chol = nn.Parameter(sb_chol, requires_grad=True)
        variational_parameters = [zb, mb, sb_chol]

        if self.config["train_model"]:
            if "model_optimiser_params" in self.config:
                parameters = [
                    {"params": variational_parameters},
                    {"params": self.model.parameters(),
                     **self.config["model_optimiser_params"]}
                ]
            else:
                parameters = [
                    {"params": variational_parameters},
                    {"params": self.model.parameters()}
                ]
        else:
            parameters = variational_parameters

        # Reset optimiser
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])

        # Dict for logging optimisation progress
        training_curve = {
            "elbo": [],
            "kl": [],
            "ll": [],
        }

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(self.config["epochs"]), "Epochs")
        for i in epoch_iter:
            epoch = {
                "elbo": 0,
                "kl": 0,
                "ll": 0,
            }

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch
                }

                if za is None:
                    """
                    No inducing points yet. Perform standard VI approach, with 

                    F(q(u)) = -KL(q(u) || p(u)) + E_q(u)[log p(y | f)].

                    Just set q(u) = q(b) and Z = Z_b.
                    """
                    q = MultivariateGaussianDistributionWithZ(
                        inducing_locations=zb,
                        std_params={
                            "loc": mb,
                            "covariance_matrix": sb_chol.matmul(sb_chol.T)
                        }
                    )
                    z = zb

                else:
                    """
                    Perform modified VI approach, with

                    q(a, b) ⍺ q(a) p(b | a) t(b)

                    F(q(f)) = -KL(q(a, b) || p(a, b)) + E_q(a, b)[log p(y | 
                    f)]    
                                + terms not depending on t(b) or ε.

                    Just set q(a, b) ⍺ q(a) q(b | a) and Z = {Z_a, Z_b}
                    and ignore the terms not depending on t(b) or ε.
                    """
                    z = torch.cat([za, zb], axis=0)

                    kba = self.model.kernel(zb, za).detach()
                    a = kba.matmul(ikaa)

                    q_loc = torch.empty(len(z))
                    q_cov = torch.empty(len(z), len(z))

                    q_loc[:len(za)] = qa_loc
                    q_loc[len(za):] = mb + a.matmul(qa_loc)

                    q_cov[:len(za), :len(za)] = qa_cov
                    q_cov[len(za):, len(za):] = (
                            sb_chol.matmul(sb_chol.T)
                            + a.matmul(qa_cov).matmul(a.T))
                    q_cov[len(za):, :len(za)] = a.matmul(qa_cov)
                    q_cov[:len(za), len(za):] = qa_cov.matmul(a.T)

                    q = type(q)(
                        inducing_locations=z,
                        std_params={
                            "loc": q_loc,
                            "covariance_matrix": q_cov
                        },
                    )

                # Everything is the same from here on in.
                kzz = add_diagonal(self.model.kernel(z, z), JITTER)
                p = type(q)(
                    inducing_locations=z,
                    std_params={
                        "loc": torch.zeros(len(z)),
                        "covariance_matrix": kzz,
                    }
                )

                # Compute KL(q || p).
                kl = q.kl_divergence(p).sum()

                # Compute E_q[log p(y | f)].
                ll = self.model.expected_log_likelihood(
                    batch, q, self.config["num_elbo_samples"]).sum()

                # Normalise values w.r.t. batch size.
                kl /= len(x)
                ll /= len(x_batch)

                loss = kl - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["ll"].append(epoch["ll"])

            if i % self.config["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"Epochs: {i}.")

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   ll=epoch["ll"],
                                   outputscale=self.model.kernel.outputscale.item())

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)

        # Update clients inducing points.
        self.inducing_locations = zb.detach()

        # Create non_trainable_copy to send back to server.
        q_new = q.non_trainable_copy()

        self._can_update = True

        return q_new, None


class ContinualLearningSGPClientBayesianHypers(ClientBayesianHypers):
    """
    Continual learning SGP client with Bayesian treatment of model
    hyperparameters.
    """
    def __init__(self, data, model, inducing_locations, config=None):
        super().__init__(data, model, config=config)

        # Private inducing locations Z_b.
        self.inducing_locations = inducing_locations

    def update_q(self, q, qeps):
        """
        Computes a refined approximate posterior.
        """
        return self.gradient_based_update(q, qeps.trainable_copy())

    def gradient_based_update(self, q, qeps):
        """
        The gradient based update in the streaming SGP setting involves
        completely overhalling the current approximate posterior q with new
        inducing points, hence we override the default gradient_based_update
        function.
        :param q: The current approximate posterior, q(a | Z_a).
        :param qeps: The current approximate posterior, q(ε).
        :return q_new, qeps_new: The new approximate posteriors,
        q(a, b | Z_a, Z_b) and q(ε).
        """
        # Cannot update during optimisation.
        # self._can_update = False

        # Set up data etc.
        x = self.data["x"]
        y = self.data["y"]

        tensor_dataset = TensorDataset(x, y)
        loader = DataLoader(tensor_dataset,
                            batch_size=self.config["batch_size"],
                            shuffle=True)

        # Copy current approximate posterior, ensuring non-trainable.
        qa = q.non_trainable_copy()
        peps = qeps.non_trainable_copy()
        za = qa.inducing_locations

        # Parameterise as q(b | a) N(b; mb + Aa, Sb), initialised as
        # q(b | a) = p(b | a).
        zb = self.inducing_locations
        mb = torch.zeros(len(zb))

        if za is not None:
            # Fixed during optimisation.
            qa_cov = qa.std_params["covariance_matrix"]
            qa_loc = qa.std_params["loc"]

            kaa = add_diagonal(self.model.kernel(za, za).detach(), JITTER)
            ikaa = psd_inverse(kaa)
            kbb = add_diagonal(self.model.kernel(zb, zb).detach(), JITTER)
            kba = self.model.kernel(zb, za).detach()

            # Initialise Sb = Kbb - Kba Kaa^{-1} Kab.
            sb = kbb - kba.matmul(ikaa).matmul(kba.T)
            sb_chol = torch.cholesky(sb)

            # Initialise Ab = KbaKaa^{-1}.
            ab = kba.matmul(ikaa)
            ab = nn.Parameter(ab, requires_grad=True)
        else:
            # Initialise Sb = Kbb.
            kbb = add_diagonal(self.model.kernel(zb, zb).detach(), JITTER)
            sb = kbb
            sb_chol = torch.cholesky(sb)

            # Ab is None.
            ab = nn.Parameter(torch.tensor(0.), requires_grad=False)

        # Variational parameters of q(b | a) = N(b; A_b a + m_b, S_b).
        zb = nn.Parameter(zb, requires_grad=True)
        mb = nn.Parameter(mb, requires_grad=True)
        sb_chol = nn.Parameter(sb_chol, requires_grad=True)

        # + variational parameters of q(ε).
        parameters = [zb, mb, sb_chol] + qeps.parameters()

        # Reset optimiser
        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"])

        # Dict for logging optimisation progress
        training_curve = {
            "elbo": [],
            "kl": [],
            "kleps": [],
            "ll": [],
        }

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(self.config["epochs"]), "Epochs")
        for i in epoch_iter:
            epoch = {
                "elbo": 0,
                "kl": 0,
                "kleps": 0,
                "ll": 0,
            }

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch
                }

                if za is None:
                    """
                    No inducing points yet. Perform standard VI approach, with 

                    F(q(u)) = KL(q(ε) || p(ε)) + ∫q(ε) KL(q(u) || p(u | ε)) dε 
                              - ∫q(ε)q(f | ε) log p(y | f, ε) dεdu.

                    Just set q(u) = q(b) and Z = Z_b.
                    """
                    q = MultivariateGaussianDistributionWithZ(
                        inducing_locations=zb,
                        std_params={
                            "loc": mb,
                            "covariance_matrix": sb_chol.matmul(sb_chol.T)
                        }
                    )
                    z = zb

                else:
                    """
                    Perform modified VI approach, with

                    q(a, b) ⍺ q(a) q(b | a)

                    F(q(f)) = KL(q(ε) || p(ε)) 
                              + ∫q(ε) KL(q(a, b) || p(a, b | ε)) dε 
                              - ∫q(ε) KL(q(a) || p(a | ε)) dε
                              - ∫q(ε)q(f | ε) log p(y | f, ε) dεdu.

                    Just set q(a, b) ⍺ q(a) q(b | a) and Z = {Z_a, Z_b}.
                    """
                    z = torch.cat([za, zb], axis=0)

                    q_loc = torch.empty(len(z))
                    q_cov = torch.empty(len(z), len(z))

                    q_loc[:len(za)] = qa_loc
                    q_loc[len(za):] = mb + ab.matmul(qa_loc)

                    q_cov[:len(za), :len(za)] = qa_cov
                    q_cov[len(za):, len(za):] = (
                            sb_chol.matmul(sb_chol.T)
                            + ab.matmul(qa_cov).matmul(ab.T))
                    q_cov[len(za):, :len(za)] = ab.matmul(qa_cov)
                    q_cov[:len(za), len(za):] = qa_cov.matmul(ab.T)

                    q = type(q)(
                        inducing_locations=z,
                        std_params={
                            "loc": q_loc,
                            "covariance_matrix": q_cov
                        },
                    )

                kleps = sum(sum(qeps.kl_divergence(peps).values())) / len(x)

                ll = 0
                kl = 0
                for _ in range(self.config["num_elbo_hyper_samples"]):
                    eps = qeps.rsample()
                    # Set model hyperparameters.
                    self.model.hyperparameters = eps

                    kzz = add_diagonal(self.model.kernel(z, z), JITTER)

                    p = type(q)(
                        inducing_locations=z,
                        std_params={
                            "loc": torch.zeros(len(z)),
                            "covariance_matrix": kzz,
                        }
                    )
                    # Compute KL(q(a, b) || p(a, b | ε)).
                    kl += q.kl_divergence(p).sum() / len(x)

                    if za is not None:
                        kaa = add_diagonal(
                            self.model.kernel(za, za), JITTER)

                        pa = type(q)(
                            inducing_locations=za,
                            std_params={
                                "loc": torch.zeros(len(za)),
                                "covariance_matrix": kaa,
                            }
                        )
                        # Compute KL(q(a) || p(a | ε)).
                        kl -= qa.kl_divergence(pa).sum() / len(x)

                    # Compute E_q(f | ε)[log p(y | f, ε)].
                    ll += self.model.expected_log_likelihood(
                        batch, q, self.config["num_elbo_samples"]).sum()

                # Normalise values.
                kl /= self.config["num_elbo_hyper_samples"]
                ll /= (self.config["num_elbo_hyper_samples"] * len(x_batch))

                loss = kl + kleps - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["kleps"] += kleps.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

            # Log progress for current epoch
            training_curve["elbo"].append(epoch["elbo"])
            training_curve["kl"].append(epoch["kl"])
            training_curve["kleps"].append(epoch["kleps"])
            training_curve["ll"].append(epoch["ll"])

            if i % self.config["print_epochs"] == 0:
                logger.debug(f"ELBO: {epoch['elbo']:.3f}, "
                             f"LL: {epoch['ll']:.3f}, "
                             f"KL: {epoch['kl']:.3f}, "
                             f"KL eps: {epoch['kleps']:.3f}, "
                             f"Epochs: {i}.")

            epoch_iter.set_postfix(elbo=epoch["elbo"], kl=epoch["kl"],
                                   kleps=epoch["kleps"], ll=epoch["ll"])

        # Log the training curves for this update
        self.log["training_curves"].append(training_curve)

        # Update clients inducing points.
        self.inducing_locations = zb.detach()

        # Create non_trainable_copy to send back to server.
        q_new = q.non_trainable_copy()
        qeps_new = qeps.non_trainable_copy()

        self._can_update = True

        return q_new, qeps_new, None, None
