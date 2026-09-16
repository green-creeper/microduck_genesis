"""Canonical constants, joint mappings, poses, and asset paths for microduck_genesis."""
from __future__ import annotations

from pathlib import Path

PROTOCOL = 1

# 200 Hz physics (0.005s), 50 Hz control loop (0.020s decimation = 4)
TIMESTEP = 0.005
CONTROL_DT = 0.020
SUBSTEPS = 4

# Placement defaults
HOME_TRUNK_Z = 0.125
SPACING = 0.5

# Wire protocol joint names (15 elements).
# Index 9 is 'mouth' dummy joint: absent in the 14-servo model, padded in wire protocol.
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
MOUTH_INDEX = JOINT_NAMES.index("mouth")

# The 14 actuated servo joints in the robot model
MODEL_JOINT_NAMES = tuple(name for name in JOINT_NAMES if name != "mouth")

# Default stand-ready posture (15 wire joints, matching STAND keyframe in scene.xml)
# Note: right leg pitch/roll/ankle is mirrored, not symmetric.
HOME_POSE = (
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
)

# Keyframes mapping name -> (joint_dict, trunk_z)
KEYFRAMES: dict[str, tuple[dict[str, float] | None, float]] = {
    "HOME": (None, HOME_TRUNK_Z),
    "STAND": (
        {
            "left_hip_yaw": 0.0,
            "left_hip_roll": -0.087266,
            "left_hip_pitch": -0.457924,
            "left_knee": -0.004940,
            "left_ankle": 0.452984,
            "neck_pitch": 0.349066,
            "head_pitch": 0.349066,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.087266,
            "right_hip_pitch": 0.457924,
            "right_knee": 0.004940,
            "right_ankle": -0.452984,
        },
        0.12,
    ),
    "SIT": (
        {
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": -0.5236,
            "left_knee": 1.0472,
            "left_ankle": 0.0,
            "neck_pitch": 0.5,
            "head_pitch": 1.6,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": 0.5236,
            "right_knee": -1.0472,
            "right_ankle": 0.0,
        },
        0.07,
    ),
    "FOLD": (
        {
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": 1.57,
            "left_knee": 1.57,
            "left_ankle": 0.0,
            "neck_pitch": 1.0,
            "head_pitch": 1.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": -1.57,
            "right_knee": -1.57,
            "right_ankle": 0.0,
        },
        0.07,
    ),
}

# Nominal telemetry values
NOMINAL_VOLTS = 7.4
NOMINAL_TEMP_C = 32.0

# Assets directory
ASSETS_DIR = Path(__file__).resolve().parent / "robot" / "assets"
DEFAULT_SCENE = ASSETS_DIR / "scene.xml"
DEFAULT_ROBOT = ASSETS_DIR / "robot_groundcontact.xml"
ROBOT_ALLCOLLISIONS = ASSETS_DIR / "robot_allcollisions.xml"
ROBOT_WALK = ASSETS_DIR / "robot_walk.xml"
SCENE_APARTMENT = ASSETS_DIR / "scene_apartment.xml"
SCENE_ROLLERS = ASSETS_DIR / "scene_rollers.xml"
