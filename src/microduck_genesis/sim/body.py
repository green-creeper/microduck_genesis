"""One Microduck robot entity in the Genesis World physics simulation."""
from __future__ import annotations

import threading
import numpy as np

from microduck_genesis.constants import (
    HOME_POSE,
    HOME_TRUNK_Z,
    JOINT_NAMES,
    MODEL_JOINT_NAMES,
    MOUTH_INDEX,
    NOMINAL_TEMP_C,
    NOMINAL_VOLTS,
)
from microduck_genesis.sim.camera import Camera
from microduck_genesis.sim.tof import TofSensor


def gravity_in_trunk(quat: np.ndarray | list[float]) -> list[float]:
    """Calculate world gravity vector [0, 0, -1] in the trunk local frame.

    Upright robot has gravity [0, 0, -1].
    quat: [w, x, y, z] scalar-first quaternion.
    """
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    gx = -2.0 * (x * z + y * w)
    gy = -2.0 * (y * z - x * w)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return [gx, gy, gz]


class Body:
    """One duck's view of the Genesis world: joints, actuators, trunk, sensors."""

    def __init__(self, world, entity, index: int, limp: bool = False, kp: float = 200.0):
        self.world = world
        self.entity = entity
        self.index = index
        self.prefix = "" if index == 0 else f"d{index}_"

        # Map model joint names to Genesis dof indices
        self.actuator_dofs: list[int] = []
        self.to_wire: list[int] = []

        for wire_index, name in enumerate(JOINT_NAMES):
            if name == "mouth":
                continue
            try:
                joint = entity.get_joint(name)
                # dofs_idx_local gives index relative to this entity
                dof_idx = int(joint.dofs_idx_local[0])
                self.actuator_dofs.append(dof_idx)
                self.to_wire.append(wire_index)
            except Exception:
                # In case names have prefixes or differ
                pass

        if not self.actuator_dofs:
            # Fallback to sequential non-root DOFs (first 6 DOFs are floating base if unattached)
            n_dofs = entity.n_dofs
            start_dof = 6 if n_dofs > 14 else 0
            for i, wire_idx in enumerate([idx for idx in range(len(JOINT_NAMES)) if idx != MOUTH_INDEX]):
                if start_dof + i < n_dofs:
                    self.actuator_dofs.append(start_dof + i)
                    self.to_wire.append(wire_idx)

        # Baseline gains
        self.base_kp = 50.0  # Genesis internal stiffness
        self.base_kv = 2.0   # Genesis internal damping
        self.max_torque = 1.5  # XL330 maximum torque (Nm)

        self.released = limp
        self.torque_on = not limp
        self.kp_register = float(kp)

        # Apply initial gains to Genesis entity
        self._update_gains()

        # Target angles buffer (14 model joints)
        self.targets = np.zeros(len(self.actuator_dofs), dtype=np.float32)

        # Sensors
        self.tof = TofSensor(seed=index)
        self.camera: Camera | None = None
        self.held_state: tuple[np.ndarray, np.ndarray] | None = None

    def _update_gains(self) -> None:
        """Update Genesis dof gains based on torque status and firmware kp register."""
        scale = (self.kp_register / 200.0) if self.torque_on else 0.0
        kps = np.full(len(self.actuator_dofs), self.base_kp * scale, dtype=np.float32)
        kvs = np.full(len(self.actuator_dofs), self.base_kv * (scale if scale > 0 else 0.1), dtype=np.float32)
        forces = np.full(len(self.actuator_dofs), self.max_torque if self.torque_on else 0.0, dtype=np.float32)

        try:
            self.entity.set_dofs_kp(kps, self.actuator_dofs)
            self.entity.set_dofs_kv(kvs, self.actuator_dofs)
            self.entity.set_dofs_force_range(-forces, forces, self.actuator_dofs)
        except Exception:
            pass

    # ── Placement ─────────────────────────────────────────────────────────

    def place(self, pose: dict[str, float] | None, trunk_z: float, offset_y: float) -> None:
        """Position the robot base and set joint angles."""
        with self.world.lock:
            try:
                # Set base link position and identity orientation
                self.entity.set_pos(np.array([0.0, offset_y, trunk_z], dtype=np.float32))
                self.entity.set_quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            except Exception:
                pass

            # Joint positions
            initial_qpos = np.zeros(len(self.actuator_dofs), dtype=np.float32)
            for slot, wire_idx in enumerate(self.to_wire):
                name = JOINT_NAMES[wire_idx]
                val = HOME_POSE[wire_idx] if pose is None else pose.get(name, HOME_POSE[wire_idx])
                initial_qpos[slot] = float(val)
                self.targets[slot] = float(val)

            try:
                self.entity.set_dofs_position(initial_qpos, self.actuator_dofs)
                self.entity.control_dofs_position(initial_qpos, self.actuator_dofs)
            except Exception:
                pass

            self._remember_locked()

    def _remember_locked(self) -> None:
        try:
            pos = self.entity.get_pos()
            if hasattr(pos, "cpu"):
                pos = pos.cpu().numpy()
            dofs = self.entity.get_dofs_position(self.actuator_dofs)
            if hasattr(dofs, "cpu"):
                dofs = dofs.cpu().numpy()
            self.held_state = (np.asarray(pos).copy(), np.asarray(dofs).copy())
        except Exception:
            pass

    def remember(self) -> None:
        """Remember position to hold until released."""
        with self.world.lock:
            self._remember_locked()

    def restore(self) -> None:
        """Hold the duck in place until the daemon enables torque."""
        with self.world.lock:
            if self.held_state is None or self.released:
                return
            pos, dofs = self.held_state
            try:
                self.entity.set_pos(pos)
                self.entity.set_dofs_position(dofs, self.actuator_dofs)
            except Exception:
                pass

    # ── Sensors ───────────────────────────────────────────────────────────

    def sensors(self) -> dict:
        """Return full 15-joint sensor packet for robotd."""
        wire_pos = [0.0] * len(JOINT_NAMES)
        wire_vel = [0.0] * len(JOINT_NAMES)
        wire_cur = [0.0] * len(JOINT_NAMES)

        with self.world.lock:
            try:
                qpos = self.entity.get_dofs_position(self.actuator_dofs)
                qvel = self.entity.get_dofs_velocity(self.actuator_dofs)
                forces = self.entity.get_dofs_force(self.actuator_dofs)

                if hasattr(qpos, "cpu"):
                    qpos = qpos.cpu().numpy().flatten()
                    qvel = qvel.cpu().numpy().flatten()
                    forces = forces.cpu().numpy().flatten()
                else:
                    qpos = np.asarray(qpos).flatten()
                    qvel = np.asarray(qvel).flatten()
                    forces = np.asarray(forces).flatten()

                # Trunk base pose and velocities
                pos = self.entity.get_pos()
                quat = self.entity.get_quat()
                vel = self.entity.get_links_vel(0) if hasattr(self.entity, "get_links_vel") else [0.0, 0.0, 0.0]
                ang_vel = self.entity.get_links_ang_vel(0) if hasattr(self.entity, "get_links_ang_vel") else [0.0, 0.0, 0.0]

                if hasattr(pos, "cpu"):
                    pos = pos.cpu().numpy().flatten()
                    quat = quat.cpu().numpy().flatten()
                    ang_vel = ang_vel.cpu().numpy().flatten()
                else:
                    pos = np.asarray(pos).flatten()
                    quat = np.asarray(quat).flatten()
                    ang_vel = np.asarray(ang_vel).flatten()

                trunk = [float(pos[0]), float(pos[1]), float(pos[2])]
                trunk_z = float(pos[2])
                quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                gyro = [float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2])]
                grav = gravity_in_trunk(quat_list)
                sim_time = float(self.world.sim_time)
            except Exception:
                qpos, qvel, forces = [], [], []
                trunk = [0.0, 0.0, HOME_TRUNK_Z]
                trunk_z = HOME_TRUNK_Z
                quat_list = [1.0, 0.0, 0.0, 0.0]
                gyro = [0.0, 0.0, 0.0]
                grav = [0.0, 0.0, -1.0]
                sim_time = float(self.world.sim_time)

        for slot, wire_idx in enumerate(self.to_wire):
            if slot < len(qpos):
                wire_pos[wire_idx] = float(qpos[slot])
            if slot < len(qvel):
                wire_vel[wire_idx] = float(qvel[slot])
            if slot < len(forces):
                wire_cur[wire_idx] = abs(float(forces[slot])) * 100.0

        return {
            "positions": wire_pos,
            "velocities": wire_vel,
            "currents_ma": wire_cur,
            "trunk_z": trunk_z,
            "trunk": trunk,
            "sim_time": sim_time,
            "imu": {
                "gyro": gyro,
                "gravity": grav,
                "quat": quat_list,
            },
        }

    def slow_sensors(self) -> dict:
        return {"volts": NOMINAL_VOLTS, "temps_c": [NOMINAL_TEMP_C] * len(JOINT_NAMES)}

    def depth(self) -> dict:
        with self.world.lock:
            distances, status = self.tof.read()
        return {"rows": 8, "cols": 8, "distance_mm": distances, "status": status}

    # ── Actuator commands ─────────────────────────────────────────────────

    def set_targets(self, wire_targets: list[float]) -> None:
        if len(wire_targets) != len(JOINT_NAMES):
            raise ValueError(f"expected {len(JOINT_NAMES)} targets, got {len(wire_targets)}")
        for slot, wire_idx in enumerate(self.to_wire):
            self.targets[slot] = float(wire_targets[wire_idx])

        if self.torque_on:
            with self.world.lock:
                try:
                    self.entity.control_dofs_position(self.targets, self.actuator_dofs)
                except Exception:
                    pass

    def set_gain(self, kp: int) -> None:
        with self.world.lock:
            self.kp_register = float(kp)
            self._update_gains()

    def set_torque(self, on: bool) -> None:
        with self.world.lock:
            self.torque_on = bool(on)
            if on:
                self.released = True
            self._update_gains()
            if on:
                try:
                    # Hold current position so turning on torque does not jump
                    qpos = self.entity.get_dofs_position(self.actuator_dofs)
                    if hasattr(qpos, "cpu"):
                        qpos = qpos.cpu().numpy()
                    self.targets[:] = np.asarray(qpos).flatten()
                    self.entity.control_dofs_position(self.targets, self.actuator_dofs)
                except Exception:
                    pass
