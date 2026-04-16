import numba as nb
import numpy as np
from ribs._utils import check_shape
from ribs._utils import validate_batch
from ribs.emitters import EmitterBase
from ribs.emitters import EvolutionStrategyEmitter
from ribs.emitters.opt._cma_es import CMAEvolutionStrategy
from ribs.emitters.rankers import _get_ranker
from threadpoolctl import threadpool_limits


class CMAEvolutionStrategyExternal(CMAEvolutionStrategy):
    """Modified from :cls:`ribs.emitters.opt.CMAEvolutionStrategy`.
    Supports updating its internal distribution with external solutions (i.e.
    solutions that were not generated with :meth:`ask`). In our case, external
    solutions come from MILP solutions repaired from :meth:`ask` solutions to
    ensure validity.

    This implementation follows <https://arxiv.org/abs/1110.4181>. The main
    idea is to downscale updates caused by external solutions according to
    their Mahalanobis distance to the current internal distribution. Far away
    solutions are downscaled more.
    """

    def __init__(self, delta_sm=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_sm = delta_sm

    def _calc_weights(self, num_parents):
        weights = np.log(num_parents + 0.5) - np.log(np.arange(1, num_parents + 1))
        total_weights = np.sum(weights)
        return weights / total_weights

    def _calc_strat_params(self, num_parents):
        """Modified from :meth:`CMAEvolutionStrategy._calc_strat_params`.
        Computes additional parameters cy and cmy for external update scaling.
        See Table 1 from <https://arxiv.org/abs/1110.4181>.
        """
        # Create fresh weights for the number of parents found.
        weights = self._calc_weights(num_parents)
        # Note: Since `weights` changes on the line above, np.sum(weights)
        # is NOT the same as total_weights.
        mueff = np.sum(weights) ** 2 / np.sum(weights**2)

        # Dynamically update these strategy-specific parameters.
        cc = (4 + mueff / self.solution_dim) / (self.solution_dim + 4 + 2 * mueff / self.solution_dim)
        cs = (mueff + 2) / (self.solution_dim + mueff + 5)
        c1 = 2 / ((self.solution_dim + 1.3) ** 2 + mueff)
        cmu = min(
            1 - c1,
            2 * (mueff - 2 + 1 / mueff) / ((self.solution_dim + 2) ** 2 + mueff),
        )

        cy = np.sqrt(self.solution_dim) + 2 * self.solution_dim / (self.solution_dim + 2)
        cmy = np.sqrt(2 * self.solution_dim) + 2 * self.solution_dim / (self.solution_dim + 2)

        return weights, mueff, cc, cs, c1, cmu, cy, cmy

    @staticmethod
    @nb.jit(nopython=True)
    def _calc_cov_update(cov, c1a, cmu, c1, pc, rank_mu_update, weights):
        """No need to divide rank_mu_update by sigma**2 since rank_mu_update
        is now computed with standardized y.
        """
        rank_one_update = c1 * np.outer(pc, pc)
        return cov * (1 - c1a - cmu * np.sum(weights)) + rank_one_update * c1 + rank_mu_update * cmu

    @threadpool_limits.wrap(limits=1, user_api="blas")
    def tell(self, ranking_indices, ranking_values, num_parents, solutions=None, injected=None):
        """Modified from :meth:`CMAEvolutionStrategy.tell`. Supports updating
        its internal distribution with external solutions.

        Args:
            ranking_indices (np.ndarray): An array of shape (batch_size,)
                containing a descending argsort of all solutions.
            ranking_values (np.ndarray): Not actually used. Kept for
                compatibility.
            num_parents (int): How many solutions to actually use for the
                update. This number may be smaller than batch_size depending on
                the selection rule.
            solutions (np.ndarray): An array of shape (batch_size, solution_dim)
                containing solutions to be used in this update iteration. It
                can be a mix of :meth:`ask` solutions and external solutions. If
                None, :meth:`ask` solutions will be used for the update (same
                as vanilla CMA-ES).
            injected_indices (np.ndarray): An array of shape (batch_size,) of
                boolean indicating whether the solution at each index from
                ``solutions`` was injected.
        """
        if solutions is not None:
            self._solutions = solutions
            assert isinstance(injected, np.ndarray)
            assert injected.dtype == bool
            assert injected.shape[0] == solutions.shape[0]
        else:
            injected = np.full(self._solutions.shape[0], False)

        self.current_eval += len(self._solutions[ranking_indices])

        if num_parents == 0:
            return

        parents = self._solutions[ranking_indices][:num_parents]

        weights, mueff, cc, cs, c1, cmu, cy, cmy = self._calc_strat_params(num_parents)

        damps = (
            1
            + 2
            * max(
                0,
                np.sqrt((mueff - 1) / (self.solution_dim + 1)) - 1,
            )
            + cs
        )

        y = (parents - np.expand_dims(self.mean, axis=0)) / self.sigma

        # Eq.3
        should_discount = injected[ranking_indices][:num_parents]
        y[should_discount] *= np.clip(cy / np.linalg.norm(y @ self.cov.invsqrt, axis=1), a_min=None, a_max=1.0)[
            should_discount
        ]

        # Eq.4
        delta_m_original = np.sum(y[~should_discount] * np.expand_dims(weights[~should_discount], axis=1), axis=0)
        # NOTE: The original paper didn't have weights for injected solutions,
        # but since our repaired solutions usually aren't too far from the
        # original, we weigh injected solutions to accelerate learning.
        delta_m_injected = np.sum(y[should_discount] * np.expand_dims(weights[should_discount], axis=1), axis=0)

        # Eq.6
        delta_m_injected *= min(1.0, cmy / (np.sqrt(mueff) * np.linalg.norm(self.cov.invsqrt @ delta_m_injected)))
        delta_m = delta_m_original + delta_m_injected

        self.mean += delta_m

        # Update the evolution path.
        z = np.matmul(self.cov.invsqrt, delta_m)
        self.ps = (1 - cs) * self.ps + (np.sqrt(cs * (2 - cs) * mueff) / self.sigma) * z
        left = (
            np.sum(np.square(self.ps)) / self.solution_dim / (1 - (1 - cs) ** (2 * self.current_eval / self.batch_size))
        )
        right = 2 + 4.0 / (self.solution_dim + 1)
        hsig = 1 if left < right else 0

        self.pc = (1 - cc) * self.pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * delta_m

        # Adapt the covariance matrix.
        weighted_ys = y * np.expand_dims(weights, axis=1)
        # Equivalent to calculating the outer product of each ys[i] with itself
        # and taking a weighted sum of the outer products.
        rank_mu_update = np.einsum("ki,kj", weighted_ys, y)
        c1a = c1 * (1 - (1 - hsig**2) * cc * (2 - cc))
        self.cov.cov = self._calc_cov_update(self.cov.cov, c1a, cmu, c1, self.pc, rank_mu_update, weights)

        # Update sigma.
        cn, sum_square_ps = cs / damps, np.sum(np.square(self.ps))
        self.sigma *= np.exp(min(self.delta_sm, cn * (sum_square_ps / self.solution_dim - 1) / 2))


class EvolutionStrategyEmitterExternal(EvolutionStrategyEmitter):
    """Modified from :cls:`ribs.emitters.EvolutionStrategyEmitter`. Has
    CMAEvolutionStrategyExternal as its optimizer.
    """

    def __init__(
        self,
        archive,
        *,
        x0,
        sigma0,
        ranker="2imp",
        es_kwargs=None,
        selection_rule="filter",
        restart_rule="no_improvement",
        bounds=None,
        batch_size=None,
        seed=None,
    ):
        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )

        seed_sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        opt_seed, ranker_seed = seed_sequence.spawn(2)

        self._x0 = np.array(x0, dtype=archive.dtype)
        check_shape(self._x0, "x0", archive.solution_dim, "archive.solution_dim")
        self._sigma0 = sigma0

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule

        self._restart_rule = restart_rule
        self._restarts = 0
        self._itrs = 0

        # Check if the restart_rule is valid, discard check_restart result.
        _ = self._check_restart(0)

        self._opt = CMAEvolutionStrategyExternal(
            sigma0=sigma0,
            batch_size=batch_size,
            solution_dim=self._solution_dim,
            seed=opt_seed,
            dtype=self.archive.dtype,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            **(es_kwargs if es_kwargs is not None else {}),
        )
        self._opt.reset(self._x0)

        self._ranker = _get_ranker(ranker, ranker_seed)
        self._ranker.reset(self, archive)

        self._batch_size = self._opt.batch_size

    def tell(self, solution, objective, measures, add_info, **fields):
        """Modified from :meth:`EvolutionStrategyEmitter.tell`. Passes
        additional parameters `solution` and `injected` to accommodate external
        solution injection.
        """
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

        assert "injected" in fields
        injected = fields["injected"]

        # Increase iteration counter.
        self._itrs += 1

        # Count number of new solutions.
        new_sols = add_info["status"].astype(bool).sum()

        # Sort the solutions using ranker.
        indices, ranking_values = self._ranker.rank(self, self.archive, data, add_info)

        # Select the number of parents.
        num_parents = new_sols if self._selection_rule == "filter" else self._batch_size // 2

        # Update Evolution Strategy.
        self._opt.tell(indices, ranking_values, num_parents, solutions=data["solution"], injected=injected)

        # Check for reset.
        if self._opt.check_stop(ranking_values[indices]) or self._check_restart(new_sols):
            new_x0 = self.archive.sample_elites(1)["solution"][0]
            self._opt.reset(new_x0)
            self._ranker.reset(self, self.archive)
            self._restarts += 1
