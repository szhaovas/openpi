"""Evaluator for LIBERO-Spatial task suite."""

from typing import List, Tuple

from .libero_eval import LiberoEval


class LiberoSpatialEval(LiberoEval):
    task_suite = "milp_libero_spatial"

    @staticmethod
    def get_default_env_params(task_id: int = 0) -> Tuple[List[float], str]:
        # For now ``env_params`` consists of:
        #   [
        #       akita_black_bowl_1_x, akita_black_bowl_1_y,
        #       akita_black_bowl_2_x, akita_black_bowl_2_y,
        #       cookies_1_x, cookies_1_y,
        #       glazed_rim_porcelain_ramekin_1_x,
        #       glazed_rim_porcelain_ramekin_1_y,
        #       plate_1_x, plate_1_y,
        #       light_x, light_y, light_z
        #       camera_x, camera_y, camera_z,
        #       table_r, table_g, table_b,
        #       camera_r1, camera_r2, camera_r3,
        #       light_spec_r, light_spec_g, light_spec_b,
        #   ]
        # These starting params are derived from default libero-spatial; make sure
        # task_order_index=0 when loading benchmark_dict
        bowl_starting_xy = [
            [-0.05, 0.20, -0.18, 0.32],
            [-0.18, 0.32, 0.13, -0.07],
            [-0.075, 0.0, 0.01, 0.31],
            [0.07, 0.03, 0.03, -0.27],
            [0.08, -0.15, 0.03, -0.27],
            [-0.20, 0.20, 0.07, 0.03],
            [0.13, -0.07, -0.41, -0.14],
            [-0.41, -0.14, 0.03, -0.27],
            [0.01, 0.31, -0.18, 0.32],
            [0.03, -0.27, -0.41, -0.14],
        ]

        libero_spatial_prompts = [
            "pick up the black bowl between the plate and the ramekin and place it on the plate",
            "pick up the black bowl next to the ramekin and place it on the plate",
            "pick up the black bowl from table center and place it on the plate",
            "pick up the black bowl on the cookie box and place it on the plate",
            "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
            "pick up the black bowl on the ramekin and place it on the plate",
            "pick up the black bowl next to the cookie box and place it on the plate",
            "pick up the black bowl on the stove and place it on the plate",
            "pick up the black bowl next to the plate and place it on the plate",
            "pick up the black bowl on the wooden cabinet and place it on the plate",
        ]

        return (
            bowl_starting_xy[task_id]
            + [
                0.07,
                0.03,
                -0.20,
                0.20,
                0.06,
                0.20,
                1.0,
                1.0,
                4.0,
                0.66,
                0,
                1.61,
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
            libero_spatial_prompts[task_id],
        )
