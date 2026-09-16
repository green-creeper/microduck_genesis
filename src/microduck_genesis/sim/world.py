"""Genesis World scene wrapper for single and multi-duck simulation."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

from microduck_genesis.constants import (
    DEFAULT_ROBOT,
    DEFAULT_SCENE,
    KEYFRAMES,
    SPACING,
    SUBSTEPS,
    TIMESTEP,
)
from microduck_genesis.sim.body import Body
from microduck_genesis.sim.camera import FPS as CAMERA_FPS


class World:
    """The Genesis physics simulation, shared across all duck bodies in it."""

    def __init__(self, scene_path: Path = DEFAULT_SCENE, robot_path: Path = DEFAULT_ROBOT, count: int = 1, headless: bool = False, keyframe: str = "SIT"):
        self.scene_path = Path(scene_path)
        self.robot_path = Path(robot_path)
        self.count = count
        self.headless = headless
        self.keyframe = keyframe
        self.timestep = TIMESTEP
        self.sim_time = 0.0
        self.lock = threading.RLock()
        self.bodies: list[Body] = []

        import genesis as gs

        # Initialize Genesis (CPU backend is 15x faster on Apple Silicon for single robot)
        if not getattr(gs, "_initialized", False):
            try:
                gs.init(backend=gs.cpu, precision="32", logging_level="warning")
            except Exception:
                gs.init(precision="32", logging_level="warning")

        # Build Scene
        viewer_opt = None if headless else gs.options.ViewerOptions(
            res=(1280, 720),
            camera_pos=(1.5, -2.0, 1.0),
            camera_lookat=(0.0, 0.5, 0.2),
            camera_fov=40,
            refresh_rate=30,
            realtime_factor=None,  # run_loop controls the 50 Hz wall-clock pacing
        )

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.timestep,
                gravity=(0.0, 0.0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_self_collision=True,
            ),
            viewer_options=viewer_opt,
            show_viewer=not headless,
        )

        # Add ground plane
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # Add robot entities (MJCF trunk_base already carries z=0.12)
        self.entities = []
        for i in range(count):
            entity = self.scene.add_entity(
                gs.morphs.MJCF(
                    file=str(self.robot_path),
                    pos=(0.0, i * SPACING, 0.0),
                )
            )
            self.entities.append(entity)

        # Build Genesis scene
        self.scene.build()

        # Wrap into Body instances
        pose_dict, trunk_z = KEYFRAMES.get(keyframe, (None, 0.125))
        for i, entity in enumerate(self.entities):
            body = Body(self, entity, index=i)
            self.bodies.append(body)
            body.place(pose_dict, trunk_z, offset_y=i * SPACING)

        # Warm up 1 step so JIT compiles before server opens and timing starts
        self.step(1)
        self.sim_time = 0.0

    def step(self, times: int = SUBSTEPS) -> None:
        """Advance the world by `times` physics steps, protected by world lock."""
        with self.lock:
            for i in range(times):
                is_last = (i == times - 1)
                self.scene.step(update_visualizer=is_last)
                self.sim_time += self.timestep
                for body in self.bodies:
                    if not body.released:
                        body.restore()


def run_loop(world: World, headless: bool = False) -> None:
    """Run simulation loop synchronized with wall-clock real time."""
    dt = world.timestep
    batch = SUBSTEPS  # 4 * 0.005 = 0.020s (50 Hz)
    period = batch * dt

    eyes = [b for b in world.bodies if b.camera is not None]
    passes_per_eye = max(1, round((1.0 / CAMERA_FPS) / period))

    step_count = 0
    next_step = time.perf_counter()
    behind = 0

    try:
        while True:
            world.step(batch)

            next_step += period
            slack = next_step - time.perf_counter()

            if slack > 0.001:
                time.sleep(slack)
            elif slack < -0.25:
                behind += 1
                if behind % 10 == 0:
                    print(
                        f"== behind real time by {-slack:.2f}s (x{behind}) — consider --headless",
                        flush=True,
                    )
                next_step = time.perf_counter()
                time.sleep(0.001)
            else:
                time.sleep(0.001)

            step_count += 1
            if eyes and step_count % passes_per_eye == 0:
                for b in eyes:
                    if b.camera is not None:
                        b.camera.render()
    except KeyboardInterrupt:
        pass
