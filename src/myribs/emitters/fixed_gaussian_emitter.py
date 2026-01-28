import numpy as np
from ribs.emitters import GaussianEmitter


class FixedGaussianEmitter(GaussianEmitter):
    """Modified from :cls:`ribs.emitters.GaussianEmitter`. The only change is 
    that :meth:`ask` only uses :attr:`x0` as parents when applying gaussian 
    noise. It no longer samples elites from the archive as parents because 
    it is intended for domain randomization, which only samples near some 
    starting configuration.
    """
    def ask(self):
        if self._initial_solutions is not None:
            return np.clip(self._initial_solutions, self.lower_bounds,
                            self.upper_bounds)
        parents = np.repeat(self.x0[None], repeats=self._batch_size, axis=0)

        return self._operator.ask(parents=parents)
