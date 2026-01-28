import numpy as np
from ribs.schedulers import Scheduler


class SchedulerExternal(Scheduler):
    """
    Modified from :cls:`ribs.schedulers.Scheduler`. Supports injecting external
    solutions that don't come from the emitter.
    """

    def __init__(
        self,
        archive,
        emitters,
        num_active,
        *,
        result_archive=None,
        add_mode="batch"
    ):
        super().__init__(
            archive, emitters, result_archive=result_archive, add_mode=add_mode
        )
        self._num_active = num_active
        self._emitters = np.array(self._emitters)
        self._active_arr = np.zeros_like(self._emitters, dtype=bool)

    @property
    def emitters(self):
        """Returns the"""
        return self._emitters[self._active_arr]

    def _validate_tell_data(self, data):
        """No longer sets data['solution'] to :attr:`_cur_solutions` because
        this scheduler needs to receive repaired solutions."""
        for name, arr in data.items():
            data[name] = np.asarray(arr)
            self._check_length(name, arr)

        return data

    def ask_dqd(self):
        if self._last_called in ["ask", "ask_dqd"]:
            raise RuntimeError(
                "ask_dqd cannot be called immediately after "
                + self._last_called
            )
        self._last_called = "ask_dqd"

        if not self._active_arr.any():
            self._active_arr[: self._num_active] = True
        else:
            self._active_arr = np.roll(self._active_arr, self._num_active)

        self._cur_solutions = []

        for i, emitter in enumerate(self.emitters):
            emitter_sols = emitter.ask_dqd()
            self._cur_solutions.append(emitter_sols)
            self._num_emitted[i] = len(emitter_sols)

        # In case the emitters didn't return any solutions.
        self._cur_solutions = (
            np.concatenate(self._cur_solutions, axis=0)
            if self._cur_solutions
            else np.empty((0, self._solution_dim))
        )
        return self._cur_solutions

    def ask(self):
        if self._last_called in ["ask", "ask_dqd"]:
            raise RuntimeError(
                "ask cannot be called immediately after " + self._last_called
            )
        self._last_called = "ask"

        if not self._active_arr.any():
            self._active_arr[: self._num_active] = True
        else:
            self._active_arr = np.roll(self._active_arr, self._num_active)

        self._cur_solutions = []

        for i, emitter in enumerate(self.emitters):
            emitter_sols = emitter.ask()
            self._cur_solutions.append(emitter_sols)
            self._num_emitted[i] = len(emitter_sols)

        # In case the emitters didn't return any solutions.
        self._cur_solutions = (
            np.concatenate(self._cur_solutions, axis=0)
            if self._cur_solutions
            else np.empty((0, self._solution_dim))
        )
        return self._cur_solutions

    def tell_dqd(self, objective, measures, jacobian, **fields):
        if self._last_called != "ask_dqd":
            raise RuntimeError(
                "tell_dqd() was called without calling ask_dqd()."
            )
        self._last_called = "tell_dqd"

        assert "solution" in fields
        data = self._validate_tell_data(
            {
                "solution": fields["solution"],
                "objective": objective,
                "measures": measures,
                **{k: v for k, v in fields.items() if k != "injected"},
            }
        )

        jacobian = np.asarray(jacobian)
        self._check_length("jacobian", jacobian)

        add_info = self._add_to_archives(data)

        assert "injected" in fields
        data["injected"] = fields["injected"]

        # Keep track of pos because emitters may have different batch sizes.
        pos = 0
        for emitter, n in zip(self.emitters, self._num_emitted):
            end = pos + n
            emitter.tell_dqd(
                **{name: arr[pos:end] for name, arr in data.items()},
                jacobian=jacobian[pos:end],
                add_info={name: arr[pos:end] for name, arr in add_info.items()},
            )
            pos = end

    def tell(self, objective, measures, **fields):
        if self._last_called != "ask":
            raise RuntimeError("tell() was called without calling ask().")
        self._last_called = "tell"

        assert "solution" in fields
        data = self._validate_tell_data(
            {
                "solution": fields["solution"],
                "objective": objective,
                "measures": measures,
                **{k: v for k, v in fields.items() if k != "injected"},
            }
        )

        add_info = self._add_to_archives(data)

        assert "injected" in fields
        data["injected"] = fields["injected"]

        # Keep track of pos because emitters may have different batch sizes.
        pos = 0
        for emitter, n in zip(self.emitters, self._num_emitted):
            end = pos + n
            emitter.tell(
                **{name: arr[pos:end] for name, arr in data.items()},
                add_info={name: arr[pos:end] for name, arr in add_info.items()},
            )
            pos = end
