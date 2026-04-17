"""Evaluator for custom Jaco environments"""

from typing import List, Tuple

from .libero_eval import LiberoEval


class JacoEval(LiberoEval):
    task_suite = "custom"

    @staticmethod
    def get_default_env_params(task_id: int = 0) -> Tuple[List[float], str]:
        raise NotImplementedError
