import pytest
from microduck_genesis.constants import DEFAULT_ROBOT


def test_genesis_scene_step():
    """Verify that Genesis can parse the Microduck MJCF model and step physics."""
    try:
        import genesis as gs
    except ImportError:
        pytest.skip("Genesis World not installed in test environment")

    # Initialize CPU backend for headless testing
    try:
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    except Exception:
        pass

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005),
        rigid_options=gs.options.RigidOptions(enable_collision=True),
        show_viewer=False,
    )

    plane = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=str(DEFAULT_ROBOT),
            pos=(0.0, 0.0, 0.12),
        )
    )

    scene.build()
    scene.step()

    pos = robot.get_pos()
    assert pos is not None
