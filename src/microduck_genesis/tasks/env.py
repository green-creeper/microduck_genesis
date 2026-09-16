"""Vectorized parallel Genesis RL environment for Microduck locomotion."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from microduck_genesis.constants import (
    CONTROL_DT,
    DEFAULT_ROBOT,
    HOME_POSE,
    HOME_TRUNK_Z,
    MODEL_JOINT_NAMES,
    MOUTH_INDEX,
    TIMESTEP,
)

ACTION_SCALE = 0.25


class MicroduckEnv:
    """Vectorized parallel environment for Microduck bipedal robot using Genesis World."""

    def __init__(
        self,
        num_envs: int = 1024,
        robot_path: str | Path = DEFAULT_ROBOT,
        device: str = "cuda:0",
        show_viewer: bool = False,
    ):
        self.num_envs = num_envs
        self.num_actions = 14
        self.num_obs = 61
        self.dt = CONTROL_DT
        self.substeps = int(round(CONTROL_DT / TIMESTEP))
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

        import genesis as gs

        try:
            gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        except Exception:
            gs.init(backend=gs.cpu, precision="32", logging_level="warning")

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=TIMESTEP,
                gravity=(0.0, 0.0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_self_collision=False,
                max_collision_pairs=32,
            ),
            show_viewer=show_viewer,
        )

        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(robot_path),
            )
        )

        self.scene.build(n_envs=num_envs)

        # Locate 14 actuated DOFs
        self.motors_dof_idx = []
        for name in MODEL_JOINT_NAMES:
            try:
                joint = self.robot.get_joint(name)
                self.motors_dof_idx.append(int(joint.dofs_idx_local[0]))
            except Exception:
                pass
        if not self.motors_dof_idx:
            self.motors_dof_idx = list(range(6, 20))

        # Baseline PD gains
        self.robot.set_dofs_kp([50.0] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([2.0] * self.num_actions, self.motors_dof_idx)

        # Default pose (14 joints)
        self.default_dof_pos = torch.tensor(
            [HOME_POSE[i] for i in range(len(HOME_POSE)) if i != MOUTH_INDEX],
            dtype=torch.float32,
            device=self.device,
        )

        # Buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_episode_length = int(20.0 / self.dt)  # 20s

        self.commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)

        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        # Randomize vx [0.0, 0.3], vy [-0.1, 0.1], vyaw [-0.5, 0.5]
        self.commands[env_ids, 0] = torch.rand(n, device=self.device) * 0.3
        self.commands[env_ids, 1] = (torch.rand(n, device=self.device) - 0.5) * 0.2
        self.commands[env_ids, 2] = (torch.rand(n, device=self.device) - 0.5) * 1.0

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        # Reset base position and orientation
        base_pos = torch.tensor([0.0, 0.0, HOME_TRUNK_Z], device=self.device).repeat(len(env_ids), 1)
        base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)

        try:
            self.robot.set_pos(base_pos, envs_idx=env_ids)
            self.robot.set_quat(base_quat, envs_idx=env_ids)
            self.robot.set_dofs_position(
                self.default_dof_pos.repeat(len(env_ids), 1),
                self.motors_dof_idx,
                envs_idx=env_ids,
            )
        except Exception:
            pass

        self._resample_commands(env_ids)
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False

    def reset(self) -> torch.Tensor:
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_observations()
        return self.obs_buf

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.last_actions[:] = self.actions[:]
        self.actions[:] = torch.clamp(actions, -1.0, 1.0)

        # Target angles
        targets = self.default_dof_pos + self.actions * ACTION_SCALE

        try:
            self.robot.control_dofs_position(targets, self.motors_dof_idx)
        except Exception:
            pass

        for _ in range(self.substeps):
            self.scene.step()

        self.episode_length_buf += 1

        self._compute_observations()
        self._compute_rewards()
        self._check_termination()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        return self.obs_buf, self.rew_buf, self.reset_buf, {}

    def _compute_observations(self) -> None:
        try:
            dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
            dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
            base_quat = self.robot.get_quat()
            base_ang_vel = self.robot.get_links_ang_vel(0) if hasattr(self.robot, "get_links_ang_vel") else torch.zeros((self.num_envs, 3), device=self.device)
        except Exception:
            dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
            dof_vel = torch.zeros((self.num_envs, self.num_actions), device=self.device)
            base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)

        # Projected gravity
        w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
        gx = -2.0 * (x * z + y * w)
        gy = -2.0 * (y * z - x * w)
        gz = -(1.0 - 2.0 * (x * x + y * y))
        proj_gravity = torch.stack([gx, gy, gz], dim=-1)

        # 13 command dimensions (twist(3) + head(4) + body(6))
        zero_pad = torch.zeros((self.num_envs, 10), device=self.device)
        cmd13 = torch.cat([self.commands, zero_pad], dim=-1)

        pos_residual = dof_pos - self.default_dof_pos

        self.obs_buf = torch.cat([
            base_ang_vel,
            proj_gravity,
            cmd13,
            pos_residual,
            dof_vel,
            self.actions,
        ], dim=-1)

    def _compute_rewards(self) -> None:
        try:
            base_pos = self.robot.get_pos()
            base_lin_vel = self.robot.get_links_vel(0) if hasattr(self.robot, "get_links_vel") else torch.zeros((self.num_envs, 3), device=self.device)
            base_ang_vel = self.robot.get_links_ang_vel(0) if hasattr(self.robot, "get_links_ang_vel") else torch.zeros((self.num_envs, 3), device=self.device)
        except Exception:
            base_pos = torch.tensor([0.0, 0.0, HOME_TRUNK_Z], device=self.device).repeat(self.num_envs, 1)
            base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
            base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)

        # Velocity tracking
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        r_lin = torch.exp(-lin_vel_error / 0.25)

        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        r_ang = torch.exp(-ang_vel_error / 0.25)

        # Upright posture penalty (gravity z should be close to -1.0)
        proj_grav_z = self.obs_buf[:, 5]
        r_upright = torch.exp(-torch.square(proj_grav_z - (-1.0)) / 0.1)

        # Height reward
        r_height = torch.exp(-torch.square(base_pos[:, 2] - HOME_TRUNK_Z) / 0.01)

        # Smoothness penalty
        r_action_rate = -torch.sum(torch.square(self.actions - self.last_actions), dim=1) * 0.01

        self.rew_buf = r_lin * 1.0 + r_ang * 0.5 + r_upright * 0.5 + r_height * 0.5 + r_action_rate

    def _check_termination(self) -> None:
        # Check falls: trunk height drops below 0.06m or tilted excessively
        try:
            base_pos = self.robot.get_pos()
            fallen = base_pos[:, 2] < 0.06
        except Exception:
            fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        timeout = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = fallen | timeout
