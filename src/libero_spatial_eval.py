import collections
import datetime
import logging
import math
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
from dask.distributed import Client
from libero.libero.envs import OffScreenRenderEnv
from numpy.typing import NDArray

from libero.libero import benchmark
from src.dataset_utils import Trajectory
from src.encoder import EncoderManager
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


class LiberoSpatialEval:
    def __init__(
        self,
        task_id: int,
        objective_func: Optional[str] = "entropy",
        measure_func: Optional[str] = None,
        num_trials_per_sol: int = 1,
        max_steps: int = 220,
        num_steps_wait: int = 10,
        replan_steps: int = 5,
        server_port_start: int = 8000,
        seed: int = 42,
        dask_client: Optional[Client] = None,
        repair_config: Optional[Dict] = None,
        measure_encoder: Optional[EncoderManager] = None,
        **kwargs,
    ):
        self.task_id = task_id

        if objective_func not in ["entropy", "success_rate"]:
            raise ValueError(
                f"Unknown objective_func {objective_func} (must be one of {['entropy', 'success_rate']})"
            )
        self.objective_func = objective_func

        if measure_func not in [None, "spread_similarity", "policy_embedding"]:
            raise ValueError(
                f"Unknown measure_func {measure_func} (must be one of {[None, 'spread_similarity', 'policy_embedding']})"
            )
        if measure_func == "policy_embedding":
            if measure_encoder is None:
                raise ValueError(
                    "If measure_func='policy_embedding', measure_encoder must also be set"
                )
            self._measure_enc = measure_encoder
        self.measure_func = measure_func

        self.num_trials_per_sol = num_trials_per_sol
        self.max_steps = max_steps
        self.num_steps_wait = num_steps_wait
        self.replan_steps = replan_steps
        self.server_port_start = server_port_start
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
            bddl_file_name=custom_task_suite.tasks[self.task_id].bddl_file,
            repair_config=self.repair_config,
            **kwargs,
        )

    def evaluate_single(
        self, solution: np.ndarray
    ) -> Tuple[np.ndarray, float, NDArray[np.floating], List[Trajectory]]:
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
            objective (float):
                - If :attr:`objective_func` is ``success_rate``, this is VLA's
                success rate on the environment created from ``solution``
                across all trials.
                - If :attr:`objective_func` is ``entropy``, this is entropy of
                the aforementioned success rate.
            measures (np.ndarray): Measure values corresponding to this
                solution and the measure function defined in
                :attr:`measure_func`. Returns a random number if
                :attr:`measure_func` is not set.
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
                np.random.rand(1),
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
                    vla_client_port=self.server_port_start + trial_id,
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
                    vla_client_port=self.server_port_start,
                    eval_stub=self._eval_stub,
                    max_steps=self.max_steps,
                    num_steps_wait=self.num_steps_wait,
                    replan_steps=self.replan_steps,
                    seed=self._seed + trial_id,
                )

                success_rate += succ / self.num_trials_per_sol
                trajectories.append(traj)

        if self.objective_func == "success_rate":
            objective = success_rate
        elif self.objective_func == "entropy":
            # Maximizes entropy as objective, i.e. we want more uncertain success rates
            success_rate = np.clip(success_rate, 1e-6, 1 - 1e-6)
            entropy = -success_rate * math.log2(success_rate) - (
                1 - success_rate
            ) * math.log2(1 - success_rate)
            objective = entropy
        else:
            raise RuntimeError

        if self.measure_func is None:
            measures = np.random.rand(1)
        elif self.measure_func == "spread_similarity":
            measures = np.array([spread, similarity])
        elif self.measure_func == "policy_embedding":
            measures = self._measure_enc.encode(trajectories)
            # Average embeddings across all rollouts on this solution
            measures = np.mean(measures, axis=0)
        else:
            raise RuntimeError

        return repaired_solution, objective, measures, trajectories

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

    # TODO: clean this up
    def get_single_trajectories(
        self, solutions: np.ndarray
    ) -> List[Trajectory]:
        """Useful for collecting embeddings for autoencoder training. We define
        this seperately from :meth:`evaluate` because each solution only gets
        evaluated once when collecting embeddings and it no longer makes sense
        to parallelize by rollouts. Instead, this function parallelizes by
        solutions.
        """
        assert self.num_trials_per_sol == 1

        if self._dask_client is not None:
            futures = [
                self._dask_client.submit(
                    rollout,
                    env_params=sol,
                    vla_client_port=self.server_port_start + sol_id,
                    eval_stub=self._eval_stub,
                    max_steps=self.max_steps,
                    num_steps_wait=self.num_steps_wait,
                    replan_steps=self.replan_steps,
                    seed=self._seed,
                    pure=False,
                )
                for sol_id, sol in enumerate(solutions)
            ]
            results = self._dask_client.gather(futures)
            trajectories = [traj for _, traj in results]
        else:
            trajectories = []
            for _, sol in enumerate(solutions):
                _, traj = rollout(
                    env_params=sol,
                    vla_client_port=self.server_port_start,
                    eval_stub=self._eval_stub,
                    max_steps=self.max_steps,
                    num_steps_wait=self.num_steps_wait,
                    replan_steps=self.replan_steps,
                    seed=self._seed,
                )
                trajectories.append(traj)

        return trajectories


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
                "action": List,
                "embedding": List
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

            inference_obj = vla_client.infer(element)
            trajectory.embedding.append(
                np.mean(inference_obj["pre_logits"], axis=0)
            )

            action_chunk = inference_obj["actions"]
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
