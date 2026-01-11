import hashlib
import logging

import numpy as np
from ribs._utils import check_finite
from ribs._utils import check_shape
from ribs._utils import validate_batch
from ribs._utils import validate_single
from ribs.archives import ArchiveBase
from ribs.archives import ArrayStore
from ribs.archives._transforms import batch_entries_with_threshold
from ribs.archives._transforms import single_entry_with_threshold

logger = logging.getLogger(__name__)


class LoggingArchive(ArchiveBase):
    """Dummy archive that does not implement QD space elitism and simply saves
    everything that it got. Used in domain randomization experiments.
    """

    hash_retry_limit = 10000

    def __init__(self, solution_dim, starting_cells, measure_dim, **kwargs):
        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            cells=starting_cells,
            measure_dim=measure_dim,
            **kwargs,
        )

    def _upsize(self, new_cells):
        """Increases the number of cells in the archive to ``new_cells``. Since
        each cell within the QD framework should only contain a single solution,
        this effectively increases the maximum number of solutions the current
        archive may hold.

        Args:
            new_cells (int): New number of cells the archive can contain. Must
                be larger than the current :attr:`_cells`.
        """
        assert new_cells > self._cells, f"new_cells {new_cells} must be larger than current cells {self._cells}"

        self._cells = new_cells

        # Re-hash the solutions since range has increased
        cur_data = self.data()
        del cur_data["index"]
        self._store = ArrayStore(self._store.field_desc, capacity=self._cells)
        self.add(**cur_data)

    def index_of(self, measures):
        return np.asarray([self.index_of_single(mea) for mea in measures], dtype=np.int32)

    def index_of_single(self, measures):
        """Uses a hash to return a unique index. The index returned by this
        function is NOT intended for :meth:`retrieve` since it is not
        guaranteed the same measures will get mapped to the same index.
        """
        measures = np.asarray(measures)
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        hash = hashlib.sha256(measures.tobytes()).digest()
        hash_value = int.from_bytes(hash, byteorder="big")
        index = hash_value % (self._cells + 1)

        occupied, _ = self._store.retrieve(index)

        num_hash_retry = 0
        while occupied:
            index += 1
            occupied, _ = self._store.retrieve(index)
            num_hash_retry += 1
            if num_hash_retry > self.hash_retry_limit:
                raise ValueError(f"exceeded hash_retry_limit {self.hash_retry_limit}")

        return index

    def add(self, solution, objective, measures, **fields):
        data = validate_batch(
            self,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
        )

        batch_size = data["solution"].shape[0]
        if len(self) + batch_size >= self._cells:
            new_cells = 2 * self._cells
            logger.warning(f"Upsizing from {self._cells} to {new_cells}")
            self._upsize(new_cells)

        self._store.add(
            self.index_of(data["measures"]),
            data,
            {
                "dtype": self._dtype,
                "learning_rate": self._learning_rate,
                "threshold_min": self._threshold_min,
                "objective_sum": self._objective_sum,
            },
            [
                batch_entries_with_threshold,
            ],
        )

        # Dummy add_info.
        return {
            "status": np.zeros(batch_size, dtype=np.int32),
            "value": np.zeros(batch_size, dtype=np.float64),
        }

    def add_single(self, solution, objective, measures, **fields):
        data = validate_single(
            self,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
        )

        for name, arr in data.items():
            data[name] = np.expand_dims(arr, axis=0)

        if len(self) + 1 >= self._cells:
            new_cells = 2 * self._cells
            logger.warning(f"Upsizing from {self._cells} to {new_cells}")
            self._upsize(new_cells)

        self._store.add(
            np.expand_dims(self.index_of_single(measures), axis=0),
            data,
            {
                "dtype": self._dtype,
                "learning_rate": self._learning_rate,
                "threshold_min": self._threshold_min,
                "objective_sum": self._objective_sum,
            },
            [
                single_entry_with_threshold,
            ],
        )

        # Dummy add_info.
        return {
            "status": np.zeros(1, dtype=np.int32),
            "value": np.zeros(1, dtype=np.float64),
        }

    def retrieve(self, measures):
        raise NotImplementedError(
            "Retrieve not supported since index cannot be recovered; use sample_elites to get solutions from the archive"
        )

    def retrieve_single(self, measures):
        raise NotImplementedError(
            "Retrieve not supported since index cannot be recovered; use sample_elites to get solutions from the archive"
        )
