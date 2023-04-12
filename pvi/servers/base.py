import logging
import torch

from abc import ABC, abstractmethod
from collections import defaultdict

logger = logging.getLogger(__name__)


class Server(ABC):
    """
    An abstract class for the server.
    """
    def __init__(self, model, p, clients, config=None, init_q=None):

        if config is None:
            config = {}

        self._config = config

        # Shared probabilistic model.
        self.model = model

        # Global prior p(θ).
        self.p = p

        # Global posterior q(θ).
        self.q = p.non_trainable_copy()

        # Initial q(θ) for first client update.
        self.init_q = init_q

        # Clients.
        self.clients = clients

        # Internal iteration counter.
        self.iterations = 0

        # Internal communication counter.
        self.communications = 0

        self.param_update_norms = []

        self.log = defaultdict(list)

        self.log["q"].append(self.q.non_trainable_copy())
        self.log["communications"].append(self.communications)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = {**self._config, **config}

    def get_default_config(self):
        return {}
        '''pass all configs in call instead of here
            "train_model": False,
            "model_update_freq": 1,
            "hyper_optimiser": "SGD",
            "hyper_optimiser_params": {"lr": 1},
            "hyper_updates": 1,
        }
        '''

    @abstractmethod
    def tick(self):
        """
        Defines what the server should do on each update round. Could be a
        synchronous update, asynchronous update etc.
        """
        pass

    @abstractmethod
    def should_stop(self):
        """
        Defines when the server should stop running.
        """
        pass

    def update_hyperparameters(self):
        """
        Updates the model hyperparameters according to
        dF / dε = (μ_q - μ_0)^T dη_0 / dε + Σ_m dF_m / dε.
        """
        # TODO: currently performs single optimisation step. Perform more?

        # Zero gradients as accumulated during client optimisation.
        for param in self.model.parameters():
            if param.grad is not None:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)

                param.grad.zero_()

        # Ensure clients have same model as server and get E[log p(y | x, θ)].
        vfe = 0
        for client in self.clients:
            vfe += self.model.expected_log_likelihood(
                client.data, self.q, client.config["num_elbo_samples"]).sum()

        if hasattr(self.model, "prior"):
            # Compute prior using model's current hyperparameters.
            p_old = self.model.prior(q=self.q)
            mq = {k: v.detach() for k, v in self.q.mean_params.items()}
            mp = {k: v.detach() for k, v in p_old.mean_params.items()}

            vfe += sum([(mq[k] - mp[k]).flatten().dot(v.flatten())
                        for k, v in zip(mq.keys(), p_old.nat_params.values())])
        else:
            p_old = None

        # Make VFE per data-points. Improves stability.
        vfe /= sum([len(client.data["x"]) for client in self.clients])

        # Compute gradients.
        vfe.backward()

        # Update model parameters, and pass to clients.
        for param in self.model.parameters():
            param.data += (
                self.config["hyper_optimiser_params"]["lr"] * param.grad)

        for client in self.clients:
            client.model = self.model

        # Update prior term in maintained q.
        if p_old is not None:
            # Compute prior using model's current hyperparameters.
            p_new = self.model.prior(q=self.q)
            q_new_nps = {k: (v - p_old.nat_params[k].detach()
                             + p_new.nat_params[k].detach())
                         for k, v in self.q.nat_params.items()}

            self.q = self.q.create_new(nat_params=q_new_nps,
                                       is_trainable=False)

        parameters = {k: v.data for k, v in self.model.named_parameters()}
        logger.debug(f"Updated model hyperparameters."
                     f"\nNew model hyperparameters:\n{parameters}\n.")
        print(f"Updated model hyperparameters."
              f"\nNew model hyperparameters:\n{parameters}\n.")

        return

    def model_predict(self, x, **kwargs):
        """
        Returns the current models predictive posterior distribution.
        :return: ∫ p(y | θ, x) q(θ) dθ.
        """
        return self.model(x, self.q, **kwargs)

    def add_client(self, client):
        self.clients.append(client)

    def get_compiled_log(self):
        """
        Get full log, including logs from each client.
        :return: full log.
        """
        final_log = {
            "server": self.log
        }

        client_logs = [client.log for client in self.clients]
        for i, log in enumerate(client_logs):
            final_log["client_" + str(i)] = log

        return final_log


class ServerBayesianHypers(Server):
    def __init__(self, model, p, peps, clients, config=None, init_q=None,
                 init_qeps=None):
        super().__init__(model, p, clients, config, init_q)

        # Global prior p(ε).
        self.peps = peps

        # Global posterior q(ε).
        self.qeps = peps.non_trainable_copy()

        # Initial q(ε) for first client.
        self.init_qeps = init_qeps

        self.log["qeps"].append(self.qeps.non_trainable_copy())

    @classmethod
    def get_default_config(cls):
        return {}
        '''
            "num_eps_samples": 1,
        }'''

    @abstractmethod
    def tick(self):
        """
        Defines what the server should do on each update round. Could be a
        synchronous update, asynchronous update etc.
        """
        pass

    @abstractmethod
    def should_stop(self):
        """
        Defines when the server should stop running.
        """
        pass

    def model_predict(self, x, **kwargs):
        """
        Returns the current models predictive posterior distribution.
        :return: ∫ p(y | x, θ, ε) q(θ)q(ε) dθ dε.
        """
        neps = self.config["num_eps_samples"]
        dists = []
        for _ in range(neps):
            eps = self.qeps.sample()
            self.model.hyperparameters = eps
            dists.append(self.model(x, self.q, **kwargs))

        return dists
