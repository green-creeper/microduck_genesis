import pytest
from microduck_genesis.constants import (
    ASSETS_DIR,
    DEFAULT_ROBOT,
    DEFAULT_SCENE,
    HOME_POSE,
    HOME_TRUNK_Z,
    JOINT_NAMES,
    KEYFRAMES,
    MODEL_JOINT_NAMES,
    MOUTH_INDEX,
    PROTOCOL,
    TIMESTEP,
)


def test_joint_definitions():
    assert len(JOINT_NAMES) == 15
    assert JOINT_NAMES[MOUTH_INDEX] == "mouth"
    assert len(MODEL_JOINT_NAMES) == 14
    assert "mouth" not in MODEL_JOINT_NAMES
    assert len(HOME_POSE) == 15


def test_protocol_constants():
    assert PROTOCOL == 1
    assert TIMESTEP == 0.005
    assert HOME_TRUNK_Z > 0.0


def test_keyframes():
    for name in ("HOME", "STAND", "SIT", "FOLD"):
        assert name in KEYFRAMES
        pose_dict, trunk_z = KEYFRAMES[name]
        assert trunk_z > 0.0
        if pose_dict is not None:
            assert len(pose_dict) == 14


def test_asset_files_exist():
    assert ASSETS_DIR.exists()
    assert DEFAULT_SCENE.exists()
    assert DEFAULT_ROBOT.exists()
