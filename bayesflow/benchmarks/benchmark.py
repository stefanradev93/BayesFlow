import numpy as np
from bayesflow.types import Tensor


class Benchmark:
    def sample(self, batch_size: int) -> dict[str, Tensor]:
        """Runs simulated benchmark and returns `batch_size` parameter
        and observation batches

        Parameters
        ----------
        batch_size: int
            Number of parameter-observation batches to simulate

        Returns
        -------
        dict[str, Tensor]: simulated parameters and observables
            with shapes (`batch_size`, ...)
        """
        return ...  # TODO
        # return batched_call(self, (batch_size,))

    def prior(self):
        raise NotImplementedError

    def observation_model(self, params: np.ndarray):
        raise NotImplementedError

    def __call__(self):
        prior_draws = self.prior()
        observables = self.observation_model(prior_draws)
        return dict(parameters=prior_draws.astype(np.float32), observables=observables.astype(np.float32))
