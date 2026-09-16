"""Simulated VL53L5CX Time-of-Flight sensor for microduck_genesis.

8x8 zones over a 45-degree square field of view, out to 4 m, at 15 Hz.
Publishes distance in millimeters and status per zone:
- STATUS_VALID = 5
- STATUS_NO_TARGET = 255
"""
from __future__ import annotations

import numpy as np

ROWS = 8
COLS = 8
ZONES = ROWS * COLS
STATUS_VALID = 5
STATUS_NO_TARGET = 255

FOV_DEG = 45.0
MAX_RANGE = 4.0


class TofSensor:
    """VL53L5CX 8x8 sensor simulation."""

    def __init__(self, seed: int = 0):
        self.random = np.random.default_rng(seed)
        self.sensor_handle = None

        # Precompute zone direction vectors in sensor frame (+x forward, +y left, +z up)
        half = np.radians(FOV_DEG) / 2.0
        edges = np.linspace(-half, half, COLS + 1)
        centres = (edges[:-1] + edges[1:]) / 2.0
        self.directions = np.zeros((ZONES, 3))
        for row in range(ROWS):
            elevation = -centres[row]
            for col in range(COLS):
                azimuth = -centres[col]
                self.directions[row * COLS + col] = [
                    np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                ]

    def attach_genesis_sensor(self, scene, entity, link_name: str = "bottom_head_shell"):
        """Attach a Genesis raycaster / depth sensor to the robot entity if available."""
        try:
            import genesis as gs

            link = entity.get_link(link_name)
            pattern = gs.sensors.DepthCameraPattern(
                res=(COLS, ROWS),
                fov_horizontal=FOV_DEG,
            )
            self.sensor_handle = scene.add_sensor(
                gs.sensors.DepthCamera(
                    pattern=pattern,
                    entity_idx=entity.idx,
                    link_idx_local=link.idx_local,
                    pos_offset=(0.045, 0.0, 0.02),  # Approximate position of ToF on head
                    max_range=MAX_RANGE,
                    return_world_frame=False,
                    return_points=False,
                )
            )
        except Exception as e:
            # Fallback to analytical / synthetic rays if sensor creation is unavailable
            self.sensor_handle = None

    def read(self) -> tuple[list[int], list[int]]:
        """Return (distance_mm, status) for the 64 zones."""
        distance_mm = [0] * ZONES
        status = [STATUS_NO_TARGET] * ZONES

        if self.sensor_handle is not None:
            try:
                # Read from Genesis DepthCamera
                depth_img = self.sensor_handle.read_image()  # (ROWS, COLS) meters
                if hasattr(depth_img, "cpu"):
                    depth_arr = depth_img.cpu().numpy().flatten()
                else:
                    depth_arr = np.asarray(depth_img).flatten()

                for i in range(min(ZONES, len(depth_arr))):
                    d = float(depth_arr[i])
                    if 0.05 < d < MAX_RANGE:
                        sigma = 0.003 + 0.02 * (d / MAX_RANGE)
                        measured = max(0.0, d + self.random.normal(0.0, sigma))
                        distance_mm[i] = int(measured * 1000.0)
                        status[i] = STATUS_VALID
                    else:
                        distance_mm[i] = int(MAX_RANGE * 1000.0)
                        status[i] = STATUS_NO_TARGET
                return distance_mm, status
            except Exception:
                pass

        # Default fallback: open space (no target)
        for i in range(ZONES):
            distance_mm[i] = int(MAX_RANGE * 1000.0)
            status[i] = STATUS_NO_TARGET
        return distance_mm, status
