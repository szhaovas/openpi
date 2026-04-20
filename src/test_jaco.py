from functools import partial
from typing import Dict, List, Optional, Tuple

import os
import pathlib
import imageio
import collections
import logging
import math

from PIL import Image

from pathlib import Path

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from tqdm import trange, tqdm

from src.libero_spatial_eval import get_default_env_params

from src.vla_client.websocket_client_policy import WebsocketClientPolicy

logger = logging.getLogger(__name__)
benchmark_dict = benchmark.get_benchmark_dict()
custom_task_suite = benchmark_dict["custom"]()

def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])

def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image

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

def rollout(
    vla_server_uri: str,
    prompt: str,
    max_steps: int,
    num_steps_wait: int,
    replan_steps: int,
    seed: int,
):
    np.random.seed(seed)

    task_id = 0
    task = custom_task_suite.tasks[task_id]

    env = partial(
        OffScreenRenderEnv,
        env_params=get_default_env_params(task_id=0),
        bddl_file_name=os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file
        ),
        robots=["Jaco6DOF"],
    )

    env = env()

    env.seed(seed)
    obs = env.reset()

    ip, port = vla_server_uri.split(":")
    vla_policy = WebsocketClientPolicy(ip, int(port))

    action_plan = collections.deque()

    replay_imgs = []
    replay_wrist_imgs = []

    success = False
    for t in trange(max_steps + num_steps_wait):
        if t < num_steps_wait:
            # Do nothing at the start to wait for env to settle
            obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
            continue
        
        print(f"timestep {t}")

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1, ::-1]
        )

        img = convert_to_uint8(
            resize_with_pad(img, 224, 224)
        )
        wrist_img = convert_to_uint8(
            resize_with_pad(wrist_img, 224, 224)
        )

        replay_imgs.append(img)
        replay_wrist_imgs.append(wrist_img)

        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"] if len(obs["robot0_gripper_qpos"]) == 2 else [obs["robot0_gripper_qpos"][0], obs["robot0_gripper_qpos"][2]],
            )
        )
            
        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "observation/proprio": obs["robot0_joint_pos"],
                "prompt": prompt,
            }

            inference_obj = vla_policy.infer(element)

            action_chunk = np.atleast_2d(inference_obj["actions"])
            if len(action_chunk) < replan_steps:
                logger.warning(
                    f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                )
            action_chunk = action_chunk[:replan_steps]
            action_plan.extend(action_chunk)

        action = action_plan.popleft()

        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
    
    save_dir = Path("test_jaco_vids")
    save_dir.mkdir(parents=True, exist_ok=True)    

    imageio.mimwrite(
        save_dir / f"rollout_succ_{success}.mp4",
        [np.asarray(x) for x in replay_imgs],
        fps=10,
    )

    imageio.mimwrite(
        save_dir / f"rollout_wrist_succ_{success}.mp4",
        [np.asarray(x) for x in replay_wrist_imgs],
        fps=10,
    )

    vla_policy._ws.close()

rollout(vla_server_uri="0.0.0.0:8000",
        prompt="pick up the black bowl between the plate and the ramekin and place it on the plate",
        max_steps=220,
        num_steps_wait=10,
        replan_steps=5,
        seed=42)