"""Interactive policy rehearsal and keyboard teleoperation in Genesis World."""
from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import onnxruntime as ort

from microduck_genesis.constants import (
    DEFAULT_ROBOT,
    DEFAULT_SCENE,
    HOME_POSE,
    MODEL_JOINT_NAMES,
    MOUTH_INDEX,
)
from microduck_genesis.sim.body import gravity_in_trunk
from microduck_genesis.sim.world import World

ACTION_SCALE = 0.25


class PolicyRunner:
    """Evaluates an exported ONNX policy in Genesis simulation with interactive control."""

    def __init__(self, onnx_path: str | Path, headless: bool = False):
        self.session = ort.InferenceSession(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name

        self.world = World(count=1, headless=headless)
        self.body = self.world.bodies[0]

        # Place robot standing
        self.body.place(None, 0.125, 0.0)
        self.body.set_torque(True)

        # 14-DOF default angles (skipping mouth dummy)
        self.default_qpos = np.array(
            [HOME_POSE[i] for i in range(len(HOME_POSE)) if i != MOUTH_INDEX],
            dtype=np.float32,
        )

        # Command states
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.head_cmd = np.zeros(4, dtype=np.float32)
        self.body_cmd = np.zeros(6, dtype=np.float32)
        self.last_action = np.zeros(14, dtype=np.float32)

    def assemble_obs(self) -> np.ndarray:
        """Assemble 61-dim observation contract:

        - ang_vel (3)
        - projected_gravity (3)
        - commands (13: twist(3), head(4), body(6))
        - dof_pos_residual (14)
        - dof_vel (14)
        - last_action (14)
        """
        sensors = self.body.sensors()
        pos15 = sensors["positions"]
        vel15 = sensors["velocities"]
        imu = sensors["imu"]

        # 14 model joints
        pos14 = np.array([pos15[i] for i in range(len(pos15)) if i != MOUTH_INDEX], dtype=np.float32)
        vel14 = np.array([vel15[i] for i in range(len(vel15)) if i != MOUTH_INDEX], dtype=np.float32)

        ang_vel = np.array(imu["gyro"], dtype=np.float32)
        gravity = np.array(imu["gravity"], dtype=np.float32)

        twist = np.array([self.vx, self.vy, self.vyaw], dtype=np.float32)
        commands = np.concatenate([twist, self.head_cmd, self.body_cmd])

        pos_residual = pos14 - self.default_qpos

        obs = np.concatenate([
            ang_vel,
            gravity,
            commands,
            pos_residual,
            vel14,
            self.last_action,
        ]).astype(np.float32)

        return obs.reshape(1, -1)

    def step_policy(self) -> None:
        obs = self.assemble_obs()
        action = self.session.run(None, {self.input_name: obs})[0][0]
        self.last_action = action.copy()

        # Compute position target and control DOFs
        target14 = self.default_qpos + action * ACTION_SCALE

        # Expand to 15 joints with mouth
        target15 = np.zeros(15, dtype=np.float32)
        slot = 0
        for i in range(15):
            if i == MOUTH_INDEX:
                target15[i] = 0.0
            else:
                target15[i] = target14[slot]
                slot += 1

        self.body.set_targets(target15.tolist())
        self.world.step(4)  # 20 ms


def run_interactive(onnx_path: str | Path, headless: bool = False) -> None:
    runner = PolicyRunner(onnx_path, headless=headless)
    print("\n== Interactive Policy Running in Genesis")
    print("Controls:")
    print("  W/S: Forward/Backward velocity")
    print("  A/D: Turn Left/Right")
    print("  Space: Stop motion")
    print("  Q: Quit\n")

    # Non-blocking terminal input
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        while True:
            t_start = time.perf_counter()

            # Poll keyboard
            r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if r:
                ch = sys.stdin.read(1)
                if ch in ("q", "Q", "\x1b"):
                    break
                elif ch in ("w", "W"):
                    runner.vx = min(0.4, runner.vx + 0.05)
                    print(f"vx: {runner.vx:.2f}")
                elif ch in ("s", "S"):
                    runner.vx = max(-0.2, runner.vx - 0.05)
                    print(f"vx: {runner.vx:.2f}")
                elif ch in ("a", "A"):
                    runner.vyaw = min(1.0, runner.vyaw + 0.2)
                    print(f"vyaw: {runner.vyaw:.2f}")
                elif ch in ("d", "D"):
                    runner.vyaw = max(-1.0, runner.vyaw - 0.2)
                    print(f"vyaw: {runner.vyaw:.2f}")
                elif ch == " ":
                    runner.vx = 0.0
                    runner.vy = 0.0
                    runner.vyaw = 0.0
                    print("Stopped velocity commands.")

            runner.step_policy()

            elapsed = time.perf_counter() - t_start
            slack = 0.020 - elapsed
            if slack > 0:
                time.sleep(slack)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
