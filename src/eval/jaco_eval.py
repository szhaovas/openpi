"""Evaluator for custom Jaco environments"""

from typing import List, Tuple

from .libero_eval import LiberoEval


class JacoEval(LiberoEval):
    task_suite = "custom"

    @staticmethod
    def get_default_env_params(task_id: int = 0) -> Tuple[List[float], str]:
        return ([0.01, 0.31, -0.18, 0.32, 0.06, 0.20], "pick up the black bowl next to the plate and place it on the plate")
