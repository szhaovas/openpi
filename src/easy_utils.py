"""Contains util functions with easy dependencies."""

import csv
import glob
import importlib
import pickle as pkl
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


@contextmanager
def patch_pkl_load():
    """Run pickle.load within this context if you see an error with
    __generator_ctor expecting 1 argument when loading from a pkl checkpoint.
    """
    mod = importlib.import_module("numpy.random._pickle")
    orig = getattr(mod, "__generator_ctor", None)

    def compat_generator_ctor(*args, **kwargs):
        # Try to be permissive: if orig accepts the args, call it
        try:
            return orig(*args, **kwargs)
        except TypeError:
            # If orig expects a single arg (the more common case), call with first arg
            if len(args) >= 1:
                return orig(args[0])
            # fallback: raise original error
            raise

    if orig is not None:
        mod.__generator_ctor = compat_generator_ctor

    try:
        yield
    finally:
        mod.__generator_ctor = orig


def safe_pkl_dump(obj: Any, save_path: Path):
    """Checks free disk space before picking an object. This prevents using up
    all remaining disk space with a partially-written and unusable object.
    """
    obj_size = len(pkl.dumps(obj, protocol=pkl.HIGHEST_PROTOCOL))
    free_space = shutil.disk_usage(save_path.parent).free

    if obj_size > free_space:
        raise OSError(
            f"Not enough disk space: need {obj_size} bytes, have {free_space}"
        )

    with open(save_path, "wb") as f:
        pkl.dump(obj, f)


def extract_scheduler_nevals(experiment_logdir: str) -> Dict[str, int]:
    """Extracts the numbers of evaluations at all scheduler checkpoints within
    an experiment log. Returns a dictionary matching checkpoint filenames with
    extracted numbers of evaluations.

    A scheduler checkpoint is expected to be named after the
    ``scheduler_[0-9]{8}.pkl`` format, in which the digits record its number
    of evaluations.
    """
    all_scheduler_ckpt = glob.glob(
        f"{experiment_logdir}/scheduler_{'[0-9]'*8}.pkl"
    )

    result = {}
    pattern = r"scheduler_(\d{8})\.pkl"
    for filename in all_scheduler_ckpt:
        match = re.search(pattern, filename)
        if match:
            result[filename] = int(match.group(1))

    return result


@contextmanager
def suppress_tqdm():
    original = tqdm.__init__

    def disabled_init(self, *args, **kwargs):
        kwargs["disable"] = True
        original(self, *args, **kwargs)

    tqdm.__init__ = disabled_init
    try:
        yield
    finally:
        tqdm.__init__ = original


def update_csv_column(
    input_csv: Path, column_name: str, values: List, save_to: Path
) -> None:
    """Replace an existing column or add a new one in a CSV file."""
    with open(input_csv) as fin, open(save_to, "w") as fout:
        input_csv_rows = list(csv.reader(fin))
        writer = csv.writer(fout)

        header = input_csv_rows[0]
        data = input_csv_rows[1:]
        assert len(values) == len(data)

        if column_name in header:
            col_index = header.index(column_name)
            is_new_column = False
        else:
            col_index = len(header)
            header += [column_name]
            is_new_column = True

        writer.writerow(header)

        for row, val in zip(data, values):
            if is_new_column:
                row.append(val)
            else:
                row[col_index] = val

            writer.writerow(row)
