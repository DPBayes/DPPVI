from .base import Client


# =============================================================================
# Asynchronous client class
# =============================================================================


class AsynchronousClient(Client):
    
    def __init__(self, server, client_id, data, model, t, config=None):
        
        super().__init__(data=data, model=model, t=t, config=config)
        
        # Set server and the client id
        self.server = server
        self.client_id = client_id
    
    def fit(self):
        """
        Perpetually queries the server for the latest q and updates it.
        
        TODO:
        
            - Add more sophisticated scheduling, e.g. maximum number of updates
        or maximum number of updates per unit time etc.
            - Add a step which checks if the q has been updated since the last
            time the server was queried, to avoid excess computations.
        """
        while True: self._fit()

    def _fit(self):
        """
        Computes a refined posterior and its associated approximating
        likelihood term. This method is called within AsynchronousClient.fit.
        """
        
        # Retrieve newest q from server
        q = self.server.q
        
        # Compute new posterior (ignored) and approximating likelihood term
        self.q, self.t = self.update_q(q, self.t)
        
        # Send new appeoximating likelihood term to server
        self.server.update_posterior(self.t, client_id)
        
        return self.t
