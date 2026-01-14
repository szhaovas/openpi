import torch

from pathlib import Path
from functools import partial

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

class Task_5():
    def __init__(self, name, config):
        self.name = name

        # Initializing task 5 libero environment
        self.bddl = (
            Path(get_libero_path("bddl_files"))
                / "custom"
                / config.bddl_fn
        )

        self.TASK_ENV = partial(
            OffScreenRenderEnv,
            bddl_file_name=self.bddl,
            camera_heights=config.camera_heights,
            camera_widths=config.camera_widths,
        )

        print(f"Loaded {self.name} libero module")

    def get_bddl_path(self):
        return self.bddl
    
    def get_task_env(self):
        return self.TASK_ENV
    
# python -m src.generate_env task=task_5 qd=cma_me env_search=latent_2d