import torch

from pathlib import Path
from functools import partial

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

class CMA_ME():
    def __init__(self, name, config):
        self.name = name
        # Initializing qd settings
        self.grid_archive = GridArchive(solution_dim=config.solution_dim,
                                        dims=config.archive_dims,
                                        ranges=config.archive_ranges,
                                        seed=config.seed)
        
        self.emitters = [
            EvolutionStrategyEmitter(
                archive=self.grid_archive,
                x0=config.x0,
                sigma0=config.sigma0,
                batch_size=config.batch_size,
                seed=config.seed+i
            )
            for i in range(config.num_emitters)
        ]

        self.scheduler = Scheduler(self.grid_archive, self.emitters)

        print(f"Loaded {self.name} QD module")
    
    def get_scheduler(self):
        return self.scheduler