import numpy as np
from ribs.schedulers import Scheduler


class SchedulerExternal(Scheduler):
    """
    Modified from :cls:`ribs.schedulers.Scheduler`. Supports injecting external
    solutions that don't come from the emitter.
    """

    def _validate_tell_data(self, data):
        """No longer sets data['solution'] to :attr:`_cur_solutions` because
        this scheduler needs to receive repaired solutions."""
        for name, arr in data.items():
            data[name] = np.asarray(arr)
            self._check_length(name, arr)

        return data

    def tell_dqd(self, objective, measures, jacobian, **fields):
        if self._last_called != "ask_dqd":
            raise RuntimeError("tell_dqd() was called without calling ask_dqd().")
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
        for emitter, n in zip(self._emitters, self._num_emitted):
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
        for emitter, n in zip(self._emitters, self._num_emitted):
            end = pos + n
            emitter.tell(
                **{name: arr[pos:end] for name, arr in data.items()},
                add_info={name: arr[pos:end] for name, arr in add_info.items()},
            )
            pos = end
