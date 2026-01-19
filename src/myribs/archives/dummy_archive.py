import numpy as np
from ribs.archives import ArchiveBase
from ribs.archives._transforms import batch_entries_with_threshold


class DummyArchive(ArchiveBase):
    def __init__(self, seed_solutions, **kwargs):
        batch_size, solution_dim = seed_solutions.shape
        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            cells=batch_size,
            measure_dim=1,
            learning_rate=None,
            threshold_min=-np.inf,
            qd_score_offset=0.0,
            **kwargs,
        )
        self._store.add(
            indices=np.arange(batch_size),
            new_data={
                "solution": seed_solutions,
                "objective": np.zeros(batch_size),
                "measures": np.zeros((batch_size, 1)),
            },
            extra_args={
                "dtype": self._dtype,
                "learning_rate": self._learning_rate,
                "threshold_min": self._threshold_min,
                "objective_sum": self._objective_sum,
            },
            transforms=[
                batch_entries_with_threshold,
            ],
        )

    def index_of(self, measures):
        raise NotImplementedError

    def index_of_single(self, measures):
        raise NotImplementedError

    def add(self, solution, objective, measures, **fields):
        batch_size = solution.shape[0]
        # Dummy add_info.
        return {
            "status": np.zeros(batch_size, dtype=np.int32),
            "value": np.zeros(batch_size, dtype=np.float64),
        }

    def add_single(self, solution, objective, measures, **fields):
        # Dummy add_info.
        return {
            "status": np.zeros(1, dtype=np.int32),
            "value": np.zeros(1, dtype=np.float64),
        }
