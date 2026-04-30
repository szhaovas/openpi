"""Evaluator for custom Jaco environments"""

from typing import List, Tuple

from .libero_eval import LiberoEval


class JacoEval(LiberoEval):
    task_suite = "custom"

    @staticmethod
    def get_default_env_params(task_id: int = 0) -> Tuple[List[float], str]:
        # For now ``env_params`` consists of:
        #   [
        #       akita_black_bowl_1_x, akita_black_bowl_1_y,
        #       akita_black_bowl_2_x, akita_black_bowl_2_y,
        #       plate_1_x, plate_1_y,
        #       light_x, light_y, light_z
        #       camera_x, camera_y, camera_z,
        #       table_r, table_g, table_b,
        #       camera_r1, camera_r2, camera_r3,
        #       light_spec_r, light_spec_g, light_spec_b,
        #   ]

        bowl_starting_xy = [
            [0.1, -0.05, -0.31, -0.11],
            [-0.31, -0.11, 0.1, -0.05],
        ]

        jaco_custom_prompts = [
            "pick up the black bowl from table center and place it on the plate",
            "pick up the black bowl next to the plate and place it on the plate",
        ]
        
        return (
            bowl_starting_xy[task_id]
            + [
                -0.25,
                0.0,
                1.0,
                1.0,
                4.0,
                0.76,
                0,
                1.83,
                0.5,
                0.5,
                0.5,
                0.0,
                0.0,
                0.0,
                0.3,
                0.3,
                0.3,
            ], 
            jaco_custom_prompts[task_id],
        )
