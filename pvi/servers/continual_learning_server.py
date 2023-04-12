import logging
import copy

from .base import *

logger = logging.getLogger(__name__)


class ContinualLearningServer(Server):
    def __init__(self, model, p, clients, config=None, init_q=None):
        super().__init__(model, p, clients, config, init_q)

        # Loop through each client just once.
        self.config = {"max_iterations": len(self.clients)}

        self.client_idx = 0

        if self.config["train_model"]:
            self.log["model_state_dict"].append(
                copy.deepcopy(self.model.state_dict()))

            for client in self.clients:
                # Ensure clients know to train the model.
                client.config["train_model"] = True
                client.config["model_optimiser_params"] = \
                    self.config["model_optimiser_params"]

                # Tie model hyperparameters together.
                client.model = self.model

    def get_default_config(self):
        return {
            **super().get_default_config(),
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        client = self.clients[self.client_idx]

        if client.can_update():

            if self.iterations == 0:
                # First iteration. Pass q_init(θ) to client.
                q_new, _ = client.fit(self.q, self.init_q)
            else:
                q_new, _ = client.fit(self.q)

            self.q = q_new.non_trainable_copy()
            self.communications += 1

            self.log["q"].append(self.q.non_trainable_copy())
            self.log["communications"].append(self.communications)

            if self.config["train_model"]:
                self.log["model_state_dict"].append(
                    copy.deepcopy(self.model.state_dict()))

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1
        self.client_idx = (self.client_idx + 1) % len(self.clients)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class ContinualLearningServerBayesianHypers(ServerBayesianHypers):
    def __init__(self, model, p, peps, clients, config=None, init_q=None,
                 init_qeps=None):
        super().__init__(model, p, peps, clients, config, init_q, init_qeps)

        # Loop through each client just once.
        self.config = {"max_iterations": len(self.clients)}

        self.client_idx = 0

    def get_default_config(self):
        return {
            **super().get_default_config(),
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        client = self.clients[self.client_idx]

        if client.can_update():
            if self.iterations == 0:
                q_new, qeps_new, _, _ = client.fit(self.q, self.qeps,
                                                   self.init_q, self.init_qeps)
            else:
                q_new, qeps_new, _, _ = client.fit(self.q, self.qeps)

            self.q = q_new.non_trainable_copy()
            self.qeps = qeps_new.non_trainable_copy()
            self.communications += 1

            self.log["q"].append(self.q.non_trainable_copy())
            self.log["qeps"].append(self.qeps.non_trainable_copy())
            self.log["communications"].append(self.communications)

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1
        self.client_idx = (self.client_idx + 1) % len(self.clients)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
