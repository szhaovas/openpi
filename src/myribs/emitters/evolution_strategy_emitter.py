import ribs
from ribs._utils import validate_batch


class EvolutionStrategyEmitter(ribs.emitters.EvolutionStrategyEmitter):
    """Modified from :cls:`ribs.emitters.EvolutionStrategyEmitter`. 
    """
    def __init__(self, *args, restart_mode, **kwargs):
        super().__init__(*args, **kwargs)
        assert restart_mode in ["stepping_stone", "fixed"]
        self._restart_mode = restart_mode

    @property
    def restart_mode(self):
        return self._restart_mode

    def tell(self, solution, objective, measures, add_info, **fields):
        data, add_info = validate_batch(
            self.archive,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
            add_info,
        )

        # Increase iteration counter.
        self._itrs += 1

        # Count number of new solutions.
        new_sols = add_info["status"].astype(bool).sum()

        # Sort the solutions using ranker.
        indices, ranking_values = self._ranker.rank(self, self.archive, data,
                                                    add_info)

        # Select the number of parents.
        num_parents = (new_sols if self._selection_rule == "filter" else
                       self._batch_size // 2)

        # Update Evolution Strategy.
        self._opt.tell(indices, ranking_values, num_parents)

        # Check for reset.
        if (self._opt.check_stop(ranking_values[indices]) or
                self._check_restart(new_sols)):
            if self.archive.empty or self.restart_mode == "fixed":
                self._opt.reset(self.x0)
            else:
                self._opt.reset(self.archive.sample_elites(1)["solution"][0])
            
            self._ranker.reset(self, self.archive)
            self._restarts += 1
