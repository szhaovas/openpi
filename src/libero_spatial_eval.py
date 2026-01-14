import collections
import datetime
import logging
import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
from dask.distributed import Client
from libero.libero.envs import OffScreenRenderEnv

from libero.libero import benchmark
from src.vla_client.websocket_client_policy import WebsocketClientPolicy

logger = logging.getLogger(__name__)

benchmark_dict = benchmark.get_benchmark_dict()
custom_task_suite = benchmark_dict["custom"]()


def get_default_env_params(task_id=0):
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
    # These starting params are derived from default libero-spatial
    bowl_starting_xy = [
        [-0.075, 0.0, 0.01, 0.31],
        [0.01, 0.31, -0.18, 0.32],
        [-0.20, 0.20, 0.07, 0.03],
        [-0.18, 0.32, 0.13, -0.07],
        [0.07, 0.03, 0.03, -0.27],
        [-0.41, -0.14, 0.03, -0.27],
        [0.13, -0.07, -0.41, -0.14],
        [-0.05, 0.20, -0.18, 0.32],
        [0.08, -0.15, 0.03, -0.27],
        [0.03, -0.27, -0.41, -0.14],
    ]

    return bowl_starting_xy[task_id] + [
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
    ]


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


@dataclass
class Trajectory:
    prompt: Optional[str] = (
        None  # TODO: Find some way to merge this with env params
    )
    success: bool = False
    image: List[np.ndarray] = field(default_factory=list)
    wrist_image: List[np.ndarray] = field(default_factory=list)
    state: List[np.ndarray] = field(default_factory=list)
    action: List[np.ndarray] = field(default_factory=list)
    # TODO: store policy embeddings(?)


class LiberoSpatialEval:
    def __init__(
        self,
        task_id: int,
        num_trials_per_sol: int = 1,
        max_steps: int = 220,
        num_steps_wait: int = 10,
        replan_steps: int = 5,
        seed: int = 42,
        dask_client: Optional[Client] = None,
        repair_config: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        self.task_id = task_id
        self.num_trials_per_sol = num_trials_per_sol
        self.max_steps = max_steps
        self.num_steps_wait = num_steps_wait
        self.replan_steps = replan_steps
        self._seed = seed

        self._dask_client = dask_client
        if self._dask_client is not None:
            nworkers = len(self._dask_client.scheduler_info()["workers"])
            if self.num_trials_per_sol > nworkers:
                logger.warning(
                    f"num_trials_per_sol={self.num_trials_per_sol} exceeds the number of workers {nworkers}"
                )

        self.repair_config = repair_config
        self._eval_stub = partial(
            OffScreenRenderEnv,
            *args,
            bddl_file_name=custom_task_suite.tasks[self.task_id].bddl_file,
            repair_config=self.repair_config,
            **kwargs,
        )

    def evaluate_single(
        self, solution: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[float, float], List[Trajectory]]:
        """Evaluates a single solution by creating a LIBERO env from it and
        doing VLA rollout for :attr:`num_trials_per_sol` times. If
        :attr:`dask_client` is set, the rollouts will be parallelized. If
        :attr:`repair_config` is set, solution is repaired to ensure validity
        before being used to create the environment.

        Args:
            solution (np.ndarray): Array of shape (solution_dim,) containing a
                single solution to be evaluated.

        Returns:
            repaired_solution (np.ndarray): Array of shape (solution_dim,)
                containing a single solution that has been repaired. If no
                repair was done, returns a copy of the original solution.
            objective (float): Entropy of VLA's success rate on the
                environment created from ``solution`` across all trials.
            measures (Tuple[float, float]): The spread and similarity
                corresponding to the generated environment. The former
                represents how well do objects cover the table. The latter
                represents how tightly are objects clustered.
            trajectories (List[Trajectory]): Array of shape (ntrials,)
                containing all rollout trajectories.
        """
        try:
            env = self._eval_stub(env_params=solution)
            env.reset()
            repaired_solution = env.env.env_params.copy()
        except Exception as e:
            logger.warning(e)
            # TODO: How to represent a solution that cannot be repaired
            return (
                solution.copy(),
                1e-6,
                (0.0, 0.0),
                [
                    Trajectory(success=False)
                    for _ in range(self.num_trials_per_sol)
                ],
            )

        spread, similarity = env.env.compute_spread_similarity()

        success_rate = 0
        trajectories = []
        if self._dask_client is not None:
            futures = [
                self._dask_client.submit(
                    rollout,
                    env_params=repaired_solution,
                    vla_client_port=8000 + trial_id,
                    eval_stub=self._eval_stub,
                    max_steps=self.max_steps,
                    num_steps_wait=self.num_steps_wait,
                    replan_steps=self.replan_steps,
                    seed=self._seed + trial_id,
                    pure=False,
                )
                for trial_id in range(self.num_trials_per_sol)
            ]
            results = self._dask_client.gather(futures)

            for succ, traj in results:
                success_rate += succ / self.num_trials_per_sol
                trajectories.append(traj)
        else:
            for trial_id in range(self.num_trials_per_sol):
                succ, traj = rollout(
                    env_params=repaired_solution,
                    vla_client_port=8000,
                    eval_stub=self._eval_stub,
                    max_steps=self.max_steps,
                    num_steps_wait=self.num_steps_wait,
                    replan_steps=self.replan_steps,
                    seed=self._seed + trial_id,
                )

                success_rate += succ / self.num_trials_per_sol
                trajectories.append(traj)

        # Maximizes entropy as objective, i.e. we want more uncertain success rates
        success_rate = np.clip(success_rate, 1e-4, 1 - 1e-4)
        entropy = -success_rate * math.log2(success_rate) - (
            1 - success_rate
        ) * math.log2(1 - success_rate)

        return repaired_solution, entropy, (spread, similarity), trajectories

    def evaluate(
        self, solutions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[Trajectory]]]:
        """Wrapper for :meth:`evaluate_single`.

        Args:
            solutions (np.ndarray): Array of shape (batch_size, solution_dim)
                containing a batch of solutions to be evaluated.

        Returns:
            repaired_solution (np.ndarray): Repaired solution array of shape
                (batch_size, solution_dim).
            objective (np.ndarray): Entropy array of shape (batch_size,).
            measures (np.ndarray): Measures array of shape (batch_size, 2).
            trajectories (List[List[Trajectory]]): Trajectories array of shape
                (batch_size, ntrials).
        """
        batch_size = solutions.shape[0]

        repaired, objective, measures, trajectories = [], [], [], []
        for sol_id, sol in enumerate(solutions):
            logger.info(
                f"------------------- Evaluating solution {sol_id + 1}/{batch_size} (task_id: {self.task_id})"
            )

            rep, obj, meas, traj = self.evaluate_single(sol)

            trial_success = [
                f"trial{tid}: {'success' if tr.success else 'fail'}"
                for tid, tr in enumerate(traj)
            ]
            logger.info(f"\n{trial_success}")
            logger.info(f"\n\t objective: {obj}\n\t measures: {meas}")

            repaired.append(rep)
            objective.append(obj)
            measures.append(meas)
            trajectories.append(traj)

        return (
            np.array(repaired),
            np.array(objective),
            np.array(measures),
            trajectories,
        )


def rollout(
    env_params: np.ndarray,
    vla_client_port: int,
    eval_stub: partial,
    max_steps: int,
    num_steps_wait: int,
    replan_steps: int,
    seed: int,
    video_logdir: Optional[str] = None,
) -> Tuple[bool, Trajectory]:
    """Rolls out one set of ``env_params`` for one trial and returns success
    indicator and trajectory.

    Args:
        env_params (np.ndarray): Array of shape (solution_dim,) containing a single
            solution to be evaluated.
        eval_stub (partial): ``eval_stub(env_params)`` returns a Libero env.
        vla_client_port (int): On which localhost port should this rollout
            contact the VLA server.
        max_steps (int): The maximum number of rollout steps before the task is
            failed.
        num_steps_wait (int): How many steps to wait at the start before VLA
            starts doing stuff. This allows the env to settle.
        replan_steps (int): How often to query VLA for replan.
        seed (int): Seed.
        video_logdir (str): Folder for saving rollout videos. If None no video
            is saved.

    Returns:
        success (bool): Whether the VLA successfully completed the task during
            rollout on the environment generated from ``env_params``.
        trajectory (Trajectory): Each rollout trajectory is a dictionary of
            the following format:
            {
                "success": bool,
                "prompt": str,
                "image": List,
                "wrist_image": List,
                "state": List,
                "action": List
            }
    """
    np.random.seed(seed)

    env = eval_stub(env_params=env_params)
    env.seed(seed)
    obs = env.reset()
    if obs is None:
        # TODO: How to handle solutions that fail to evaluate
        return False, Trajectory(success=False)

    vla_client = WebsocketClientPolicy("0.0.0.0", vla_client_port)

    if video_logdir is not None:
        # ID each sol with datetime to prevent overwriting
        sol_logdir = (
            Path(video_logdir)
            / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        sol_logdir.mkdir(parents=True)

    action_plan = collections.deque()
    trajectory = Trajectory(prompt=env.language_instruction)
    for t in range(max_steps + num_steps_wait):
        if t < num_steps_wait:
            # Do nothing at the start to wait for env to settle
            obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
            continue

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1, ::-1]
        )
        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )

        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": env.language_instruction,
            }

            action_chunk = vla_client.infer(element)["actions"]
            if len(action_chunk) < replan_steps:
                logger.warning(
                    f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                )
            action_chunk = action_chunk[:replan_steps]
            action_plan.extend(action_chunk)

        action = action_plan.popleft()

        # save step to trajectory
        trajectory.image.append(img)
        trajectory.wrist_image.append(wrist_img)
        trajectory.state.append(state)
        trajectory.action.append(action)

        obs, reward, done, info = env.step(action.tolist())
        if done:
            trajectory.success = True
            break

    if video_logdir is not None:
        imageio.mimwrite(
            sol_logdir
            / f"trial{trial_id}_{'success' if trajectory.success else 'fail'}.mp4",
            [np.asarray(x) for x in trajectory.image],
            fps=10,
        )

    vla_client._ws.close()

    return trajectory.success, trajectory
