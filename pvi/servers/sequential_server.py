import logging

from tqdm.auto import tqdm
from .base import *

logger = logging.getLogger(__name__)


class SequentialServer(Server):

    def get_default_config(self):
        return {}
        #    **super().get_default_config(),
        #    "max_iterations": 25,
        #    "damping_factor": 1.,
        #}

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        clients_updated = 0

        for i, client in enumerate(self.clients):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_i_old = client.t

                if self.iterations == 0:
                    # First iteration. Pass q_init(θ) to client.
                    _, t_i_new = client.fit(self.q, self.init_q, global_prior=self.p)
                else:
                    _, t_i_new = client.fit(self.q, global_prior=self.p)

                # Compute change in natural parameters.
                delta_np = {k: (t_i_new.nat_params[k] - t_i_old.nat_params[k])
                            for k in self.q.nat_params.keys()}

                logger.debug(
                    "Received client update. Updating global posterior.")
                # Update global posterior.
                q_new_nps = {k: v + delta_np[k]
                             for k, v in self.q.nat_params.items()}

                self.q = self.q.create_new(nat_params=q_new_nps,
                                           is_trainable=False)
                clients_updated += 1
                self.communications += 1

                # Log q after each update.
                self.log["q"].append(self.q.non_trainable_copy())
                self.log["communications"].append(self.communications)

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        # Update hyperparameters.
        if self.config["train_model"] and \
                self.iterations % self.config["model_update_freq"] == 0:
            self.update_hyperparameters()

        self.log["clients_updated"].append(clients_updated)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class SequentialServerBayesianHypers(ServerBayesianHypers):

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 5,
            "damping_factor": 1.,
        }

    def tick(self):
        if self.should_stop():
            return False

        logger.debug("Getting client updates.")

        clients_updated = 0

        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_old = client.t
                teps_old = client.teps

                if self.iterations == 0:
                    _, _, t_new, teps_new = client.fit(
                        self.q, self.qeps, self.init_q, self.init_qeps)
                else:
                    _, _, t_new, teps_new = client.fit(self.q, self.qeps)

                # Compute change in natural parameters.
                q_delta_np = {k: t_new.nat_params[k] - t_old.nat_params[k]
                              for k in self.q.nat_params.keys()}
                qeps_delta_np = {
                    k1: {k2: (teps_new.nat_params[k1][k2]
                              - teps_old.nat_params[k1][k2])
                         for k2 in self.qeps.nat_params[k1].keys()}
                    for k1 in self.qeps.nat_params.keys()}

                logger.debug(
                    "Received client update. Updating global posterior.")
                # Update global posterior.
                q_new_nps = {k: v + q_delta_np[k]
                             for k, v in self.q.nat_params.items()}
                qeps_new_nps = {
                    k1: {k2: v2 + qeps_delta_np[k1][k2]
                         for k2, v2 in self.qeps.nat_params[k1].items()}
                    for k1 in self.qeps.nat_params.keys()}
                qeps_new_distributions = {
                    k: self.qeps.distributions[k].create_new(
                        nat_params=v, is_trainable=False)
                    for k, v in qeps_new_nps.items()}

                self.q = self.q.create_new(nat_params=q_new_nps,
                                           is_trainable=False)
                self.qeps = type(self.qeps)(
                    distributions=qeps_new_distributions)
                clients_updated += 1
                self.communications += 1

                # Log q after each update.
                self.log["q"].append(self.q.non_trainable_copy())
                self.log["qeps"].append(self.qeps.non_trainable_copy())
                self.log["communications"].append(self.communications)

        logger.debug(f"Iteration {self.iterations} complete."
                     f"\nNew natural parameters:\n{self.q.nat_params}\n.")

        self.iterations += 1

        self.log["clients_updated"].append(clients_updated)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
