import warnings
from collections import defaultdict

import numpy as np
from ribs.schedulers import BanditScheduler, Scheduler


class BanditSchedulerExternal(BanditScheduler):
    """
    Modified from :cls:`ribs.schedulers.BanditScheduler`. Supports injecting
    external solutions that don't come from the emitter.
    """

    def __init__(self, archive, emitters, num_active, **kwargs):
        super().__init__(
            archive=archive,
            emitter_pool=emitters,
            num_active=num_active,
            **kwargs,
        )
        self._add_mode = "batch"

    @property
    def emitters(self):
        return self._emitter_pool[self._active_arr]

    def _validate_tell_data(self, data):
        """No longer sets data['solution'] to :attr:`_cur_solutions` because
        this scheduler needs to receive repaired solutions."""
        for name, arr in data.items():
            data[name] = np.asarray(arr)
            self._check_length(name, arr)

        return data

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

        archive_empty_before = self.archive.empty
        if self._result_archive is not None:
            # Check self._result_archive here since self.result_archive is a
            # property that always provides a proper archive.
            result_archive_empty_before = self.result_archive.empty

        # Add solutions to the archive.
        add_info = self.archive.add(**data)

        # Add solutions to result_archive.
        if self._result_archive is not None:
            self._result_archive.add(**data)

        # Warn the user if nothing was inserted into the archives.
        if archive_empty_before and self.archive.empty:
            warnings.warn(Scheduler.EMPTY_WARNING.format(name="archive"))
        if self._result_archive is not None:
            if result_archive_empty_before and self.result_archive.empty:
                warnings.warn(
                    Scheduler.EMPTY_WARNING.format(name="result_archive")
                )

        assert "injected" in fields
        data["injected"] = fields["injected"]

        # Keep track of pos because emitters may have different batch sizes.
        pos = 0
        for i in np.where(self._active_arr)[0]:
            emitter = self._emitter_pool[i]
            n = self._num_emitted[i]

            end = pos + n
            self._selection[i] += n
            self._success[i] += np.count_nonzero(add_info["status"][pos:end])
            emitter.tell(
                **{name: arr[pos:end] for name, arr in data.items()},
                add_info={name: arr[pos:end] for name, arr in add_info.items()},
            )
            pos = end
