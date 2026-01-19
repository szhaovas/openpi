from ribs.archives import GridArchive


class GridArchiveWrapper(GridArchive):
    """Simple wrapper for :cls:`ribs.archives.GridArchive`. It only replaces
    the ``solution_dim`` parameter from __init__ with ``seed_solutions`` to
    make the API compatible with LoggingArchive.
    """

    def __init__(self, seed_solutions, **kwargs):
        _, solution_dim = seed_solutions.shape
        super().__init__(
            solution_dim=solution_dim,
            **kwargs,
        )
