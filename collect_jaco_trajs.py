import os
import time
from enum import Enum
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygame
from libero.libero.envs import OffScreenRenderEnv

from libero.libero import benchmark, get_libero_path
from src.dataset_utils import TempDataset, Trajectory
from src.eval.libero_eval import _quat2axisangle

# matplotlib defaults (prevent key clashes)
plt.rcParams["keymap.quit"].remove("q")
plt.rcParams["keymap.save"].remove("s")

# ── Controller initialization ──────────────────────────────────────────
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("No Nintendo Switch controller found on the Mac!")

js = pygame.joystick.Joystick(0)
js.init()

print(f"Controller connected: {js.get_name()}")
print(f"Axes: {js.get_numaxes()}, Buttons: {js.get_numbuttons()}")

# ── Controller mapping (Nintendo Switch Pro Controller) ────────────────
# Axes
AX_LX, AX_LY, AX_RX, AX_RY = 0, 1, 2, 3  # Left stick XY, Right stick XY

# Buttons
BTN_A = 2
BTN_B = 1
BTN_X = 3
BTN_Y = 0
BTN_MINUS = 8
BTN_HOME = 12
BTN_PLUS = 9
BTN_L = 4
BTN_R = 5
BTN_ZL = 6
BTN_ZR = 7

# ── Control parameters ─────────────────────────────────────────────────
VEL_TRANSLATE = 0.3
VEL_ROTATE = 0.15
DEADZONE = 0.1

display_width = 224
display_height = 224


class ControlMode(Enum):
    TRA = 0
    ROT = 1

    def toggle(self) -> "ControlMode":
        return ControlMode.TRA if self is ControlMode.ROT else ControlMode.ROT


class GripperState(float, Enum):
    OPEN = -1
    CLOSE = 1

    def toggle(self) -> "GripperState":
        return (
            GripperState.OPEN
            if self is GripperState.CLOSE
            else GripperState.CLOSE
        )


control_mode: ControlMode = ControlMode.TRA
gripper_state: GripperState = GripperState.OPEN


def axis(i) -> float:
    """Get axis value with deadzone filtering."""
    v = js.get_axis(i)
    return v if abs(v) > DEADZONE else 0.0


def get_controller_inputs() -> Tuple[np.ndarray, bool, Tuple[bool, bool]]:
    """Generate 10-float action vector from controller input."""
    global control_mode, gripper_state

    pygame.event.pump()

    dx = dy = dz = 0.0
    droll = dpitch = dyaw = 0.0

    # Translation control mode: left stick for dx dy, right stick for dz
    if control_mode == ControlMode.TRA:
        dx = VEL_TRANSLATE * (axis(AX_LY))
        dy = VEL_TRANSLATE * (-axis(AX_LX))
        dz = VEL_TRANSLATE * (-axis(AX_RY))

    # Rotation control mode: left stick for droll dpitch; right stick for dyaw
    elif control_mode == ControlMode.ROT:
        droll = VEL_ROTATE * axis(AX_LX)
        dpitch = VEL_ROTATE * (-axis(AX_LY))
        dyaw = VEL_ROTATE * (axis(AX_RX))

    debounce = False

    # R shoulder for toggling gripper state
    if js.get_button(BTN_ZR):
        gripper_state = gripper_state.toggle()
        debounce = True

    # L shoulder for toggling control mode
    if js.get_button(BTN_ZL):
        control_mode = control_mode.toggle()
        debounce = True

    if debounce:
        time.sleep(0.3)

    return (
        np.array(
            [dx, dy, dz, droll, dpitch, dyaw, gripper_state], dtype=np.float32
        ),
        # R for No-op (e.g. when waiting for gripper to toggle)
        js.get_button(BTN_R),
        # + for starting episode; - for discarding episode
        (js.get_button(BTN_PLUS), js.get_button(BTN_MINUS)),
    )


def print_controls():
    """Display control mapping for reference."""
    print("\n" + "=" * 60)
    print("NINTENDO SWITCH PRO CONTROLLER - ROBOT TELEOPERATION")
    print("=" * 60)
    print("Left Shoulder  → Toggle Control Mode")
    print("Right Shoulder → Toggle Gripper State")
    print()
    print("Translation Control Mode:")
    print("  Left Stick ↑↓  → Forward/Backward")
    print("  Left Stick ←→  → Left/Right")
    print("  Right Stick ↑↓ → Up/Down")
    print()
    print("Rotation Control Mode:")
    print("  Left Stick ←→  → Roll Left/Right")
    print("  Left Stick ↑↓  → Pitch Up/Down")
    print("  Right Stick ←→ → Yaw Left/Right")
    print()
    print("+ Button      → Start Episode")
    print("- Button      → Abort Episode")
    print()
    print("=" * 60)


def collect_on_env(env_params: np.ndarray, num_collect: int, logdir: str):
    global control_mode, gripper_state

    print_controls()

    # create a jaco custom environment
    benchmark_dict = benchmark.get_benchmark_dict()
    custom_task_suite = benchmark_dict["custom"]()
    task = custom_task_suite.tasks[0]
    env = OffScreenRenderEnv(
        env_params=env_params,
        bddl_file_name=os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        ),
        camera_widths=display_width,
        camera_heights=display_height,
    )

    # lpane for overhead camera; rpane for wrist camera
    plt.ion()
    _, ax = plt.subplots(1, 2)
    overhead_display = ax[0].imshow(
        np.zeros((display_height, display_width, 3), dtype=np.uint8)
    )
    ax[0].axis("off")
    wrist_display = ax[1].imshow(
        np.zeros((display_height, display_width, 3), dtype=np.uint8)
    )
    ax[1].axis("off")

    trajectory_dataset = TempDataset(Path(logdir))
    while len(trajectory_dataset) < num_collect:
        obs = env.reset()
        if obs is None:
            raise RuntimeError("Env repair failed")

        _, _, (wants_start, wants_discard) = get_controller_inputs()
        while not wants_start:
            overhead_display.set_data(
                np.zeros((display_height, display_width, 3), dtype=np.uint8)
            )
            wrist_display.set_data(
                np.zeros((display_height, display_width, 3), dtype=np.uint8)
            )
            plt.pause(0.001)
            time.sleep(0.1)
            _, _, (wants_start, wants_discard) = get_controller_inputs()
            if wants_discard:
                return

        # set default control mode and gripper state
        control_mode = ControlMode.TRA
        gripper_state = GripperState.OPEN

        step_count = 0
        trajectory = Trajectory(prompt=task.language, success=True)
        while True:
            # noop steps to allow libero to settle
            if step_count < 10:
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                step_count += 1
                continue

            # update display
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(
                obs["robot0_eye_in_hand_image"][::-1, ::-1]
            )
            overhead_display.set_data(img)
            wrist_display.set_data(wrist_img)
            plt.pause(0.001)

            action, noop, (_, wants_discard) = get_controller_inputs()
            if wants_discard:
                print(f"\nEpisode aborted by - button (step {step_count})")
                break

            print(
                f"\rStep {step_count:3d} | Action: "
                + " | ".join(f"{a:+.02f}" for a in action)
                + f" | {control_mode}",
                end="",
                flush=True,
            )
            if np.all(action[:6] == 0) and not noop:
                time.sleep(0.02)
                continue

            trajectory.image.append(img)
            trajectory.wrist_image.append(wrist_img)
            trajectory.state.append(
                np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
            )
            trajectory.proprio.append(obs["robot0_joint_pos"])
            trajectory.action.append(action)

            obs, _, done, _ = env.step(action.tolist())
            step_count += 1

            if step_count >= 1000:
                print("\nEpisode aborted by exceeding max steps 1000")
                break

            if done:
                trajectory_dataset.write_episode(trajectory)
                print(f"\nEpisode saved to {logdir} (step {step_count})")
                break


if __name__ == "__main__":
    collect_on_env(
        env_params=np.array([0.01, 0.31, -0.18, 0.32, 0.06, 0.20]),
        num_collect=3,
        logdir="testtest",
    )
