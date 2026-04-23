#!/usr/bin/env python3
"""
mac_controller.py - Nintendo Switch Pro Controller for Robot Teleoperation
Stream 8-float action packets via SSH tunnel at optimized rate.

SSH Tunnel Command:
ssh -N -L 60000:localhost:5555 -o Compression=no -o TCPKeepAlive=yes -o ServerAliveInterval=10 fhliang@10.136.109.136 -p 42022
Controls (Intuitive Left/Right Hand Separation)
────────────────────────────────────────────────────────────
LEFT HAND - TRANSLATION:
  Left Stick X/Y        : X/Y translate (forward/back, left/right)
  L Button              : Z translate UP
  ZL Trigger            : Z translate DOWN

RIGHT HAND - ROTATION:
  Right Stick X         : Roll rotate (left/right)
  Right Stick Y         : Pitch rotate (nose up/down)
  R Button              : Yaw rotate RIGHT
  ZR Trigger            : Yaw rotate LEFT

GRIPPER & CONTROL:
  A Button              : Close gripper
  B Button              : Open gripper
  + Button              : Start episode / Abort episode / Save failure
  - Button              : Abort episode / Discard failure  
  Home Button           : (unused)

SPEED CONTROL:
  Y Button (hold)       : Precision mode (50% speed)
  X Button (hold)       : Fast mode (200% speed)
  Normal                : 100% speed
"""

import time, socket, struct, pygame, numpy as np, sys
import math
import os
from datetime import datetime
from pathlib import Path
import time
import socket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from PIL import Image
import fire
import cv2

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from src.libero_spatial_eval import get_default_env_params
from functools import partial

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
BTN_A = 2          # Close gripper
BTN_B = 1          # Open gripper
BTN_X = 3          # Fast mode modifier
BTN_Y = 0          # Precision mode modifier
BTN_MINUS = 8      # Discard episode
BTN_HOME = 12       # Home button
BTN_PLUS = 9       # Quit episode
BTN_L = 4          # Z up (Left Bumper)
BTN_R = 5         # Roll left (Right Bumper)
BTN_ZL = 6
BTN_ZR = 7

# ── Control parameters ─────────────────────────────────────────────────
VEL_TRANSLATE = 0.03    # Base translation speed
VEL_ROTATE = 0.15      # Base rotation speed (halved)
VEL_GRIPPER = 0.1     # Gripper sensitivity
DEADZONE = 0.05        # Stick deadzone

# Speed modifiers
PRECISION_MULTIPLIER = 0.5   # Y button: 50% speed
FAST_MULTIPLIER = 2.0        # X button: 200% speed

def axis(i):
    """Get axis value with deadzone filtering."""
    v = js.get_axis(i)
    return v if abs(v) > DEADZONE else 0.0

def get_speed_multiplier():
    """Calculate speed multiplier based on modifier buttons."""
    if js.get_button(BTN_Y):     # Y button: precision mode
        return PRECISION_MULTIPLIER
    elif js.get_button(BTN_X):   # X button: fast mode  
        return FAST_MULTIPLIER
    else:
        return 1.0               # Normal speed

def get_action():
    """Generate 10-float action vector from controller input."""
    pygame.event.pump()
    
    # Get speed modifier
    speed_mult = get_speed_multiplier()
    
    # ── LEFT HAND: TRANSLATION ─────────────────────────────────────────
    dx = VEL_TRANSLATE * (-axis(AX_LY)) * speed_mult    # Left stick up/down → forward/back
    dy = VEL_TRANSLATE * (-axis(AX_LX)) * speed_mult     # Left stick left/right → left/right
    
    # Z movement on left shoulder buttons
    dz = 0.0
    if js.get_button(BTN_L):                            # L button → up
        dz = VEL_TRANSLATE * speed_mult
    elif js.get_button(BTN_ZL):                    # ZL trigger (analog) → down
        dz = -VEL_TRANSLATE * speed_mult
    
    # ── RIGHT HAND: ROTATION ────────────────────────────────────────────
    # Roll on right stick X-axis (swapped from yaw)
    droll = 0.0
    if axis(AX_RX) < -0.1:                              # Right stick left → roll left
        droll = -VEL_ROTATE * (-axis(AX_RX)) * speed_mult
    elif axis(AX_RX) > 0.1:                             # Right stick right → roll right
        droll = VEL_ROTATE * axis(AX_RX) * speed_mult

    dpitch = VEL_ROTATE * (-axis(AX_RY)) * speed_mult   # Right stick up/down → pitch

    # Yaw on right shoulder buttons (swapped from roll)
    dyaw = 0.0
    if js.get_button(BTN_R):                            # R button → yaw right (inverted)
        dyaw = VEL_ROTATE * speed_mult
    elif js.get_button(BTN_ZR):                    # ZR trigger (analog) → yaw left (inverted)
        dyaw = -VEL_ROTATE * speed_mult
    
    # ── GRIPPER & CONTROL ───────────────────────────────────────────────
    grip = 0.0
    if js.get_button(BTN_A):                            # A button → close
        grip = -VEL_GRIPPER
    elif js.get_button(BTN_B):                          # B button → open
        grip = VEL_GRIPPER
    
    # Control flags
    plusf = 1.0 if js.get_button(BTN_PLUS) else 0.0    # + button → quit/save
    startf = 1.0 if js.get_button(BTN_HOME) else 0.0   # Home button → start episode
    minusf = 1.0 if js.get_button(BTN_MINUS) else 0.0  # - button → discard
    
    return np.array([dx, dy, dz, droll, dpitch, dyaw, grip, plusf, startf, minusf], dtype=np.float32)

_last = np.zeros(10, dtype=np.float32)

def get_switch_action():
    global _last
    _last = get_action()
    return _last[:7]

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

def wants_quit() -> bool:
    """PLUS button flag sent as the 8th float."""
    return bool(_last[7])

def wants_start() -> bool:
    """PLUS button flag sent as the 8th float (same as quit)."""
    return bool(_last[7])

def wants_discard() -> bool:
    """MINUS button flag sent as the 10th float."""
    return bool(_last[9])

def clear_start_flag():
    """Clear the start flag after episode begins."""
    global _last
    _last[7] = 0.0

def clear_discard_flag():
    """Clear the discard flag after handling."""
    global _last
    _last[9] = 0.0

def clear_all_flags():
    """Clear all button flags."""
    global _last
    _last[7] = 0.0  # plus/quit/start
    _last[8] = 0.0  # home (unused)
    _last[9] = 0.0  # minus/discard

# matplotlib defaults (prevent key clashes)
plt.rcParams["keymap.quit"].remove("q")
plt.rcParams["keymap.save"].remove("s")

def print_controls():
    """Display control mapping for reference."""
    print("\n" + "="*60)
    print("NINTENDO SWITCH PRO CONTROLLER - ROBOT TELEOPERATION")
    print("="*60)
    print("LEFT HAND (Translation):")
    print("  Left Stick ↑↓  → Forward/Backward")
    print("  Left Stick ←→  → Left/Right") 
    print("  L Button       → Move UP")
    print("  ZL Trigger     → Move DOWN")
    print()
    print("RIGHT HAND (Rotation):")
    print("  Right Stick ←→ → Roll Left/Right")
    print("  Right Stick ↑↓ → Pitch Up/Down")
    print("  R Button       → Yaw Right")
    print("  ZR Trigger     → Yaw Left")
    print()
    print("GRIPPER & CONTROL:")
    print("  A Button       → Close Gripper")
    print("  B Button       → Open Gripper")
    print("  + Button       → Start Episode / Abort Episode / Save Failure")
    print("  - Button       → Abort Episode / Discard Failure")
    print("  Home Button    → (unused)")
    print()
    print("SPEED CONTROL:")
    print("  Y Button (hold) → Precision Mode (50% speed)")
    print("  X Button (hold) → Fast Mode (200% speed)")
    print("  Normal          → 100% speed")
    print("="*60)

def main():
    print_controls()

    benchmark_dict = benchmark.get_benchmark_dict()
    custom_task_suite = benchmark_dict["custom"]()

    task = custom_task_suite.tasks[0]
    env_params = get_default_env_params(task_id=0)

    prompt = task.language

    env = partial(
        OffScreenRenderEnv,
        env_params=env_params,
        bddl_file_name=os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file
        ),
        robots=["Jaco6DOF"],
    )

    env = env()
    obs = env.reset()

    # main loop
    plt.ion()
    fig, ax = plt.subplots()
    ax.axis("off")
    im_plot = ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
    plt.show(block=False)
    plt.pause(1.0)

    while wants_start():
        time.sleep(0.1)
        get_switch_action()
    
    while not wants_start():
        time.sleep(0.1)
        get_switch_action()

    clear_start_flag()
    ignore_quit_for = 10 # for debouncing

    step_count = 0
    while True:
        img = Image.fromarray(obs["agentview_image"][::-1, ::-1])
        frame = np.array(img)

        ax.clear()
        ax.imshow(frame)
        ax.axis("off")
        plt.pause(0.001)
        
        if ignore_quit_for > 0:
            ignore_quit_for -= 1
        elif wants_quit() or wants_discard():
            if wants_quit():
                print(f"\nEpisode aborted by + button (step {step_count})")
            else:
                print(f"\nEpisode aborted by - button (step {step_count})")
            
            print("QUIT")
            return

        action = get_switch_action()
        print(
            f"\rStep {step_count:3d} | Action: "
            + " | ".join(f"{a:+.02f}" for a in action),
            end="",
            flush=True,
        )

        if np.any(np.abs(action) > 1e-3):
            obs, reward, success, truncated = env.step(action.tolist())
            step_count += 1
        else:
            time.sleep(0.02)

if __name__ == "__main__":
    main()
