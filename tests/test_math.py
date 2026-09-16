import math
import numpy as np
import pytest

from microduck_genesis.sim.body import gravity_in_trunk
from microduck_genesis.sim.camera import to_uyvy


def test_gravity_in_trunk():
    # 1. Upright robot
    g_upright = gravity_in_trunk([1.0, 0.0, 0.0, 0.0])
    assert pytest.approx(g_upright[0], abs=1e-5) == 0.0
    assert pytest.approx(g_upright[1], abs=1e-5) == 0.0
    assert pytest.approx(g_upright[2], abs=1e-5) == -1.0

    # 2. Roll 90 deg about X
    s = math.sqrt(0.5)
    g_roll90 = gravity_in_trunk([s, s, 0.0, 0.0])
    assert pytest.approx(g_roll90[0], abs=1e-5) == 0.0
    assert pytest.approx(g_roll90[1], abs=1e-5) == 1.0
    assert pytest.approx(g_roll90[2], abs=1e-5) == 0.0

    # 3. Pitch 90 deg about Y
    g_pitch90 = gravity_in_trunk([s, 0.0, s, 0.0])
    assert pytest.approx(g_pitch90[0], abs=1e-5) == -1.0
    assert pytest.approx(g_pitch90[1], abs=1e-5) == 0.0
    assert pytest.approx(g_pitch90[2], abs=1e-5) == 0.0


def test_uyvy_conversion():
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    # Fill with white
    rgb[:] = 255
    uyvy = to_uyvy(rgb)
    assert isinstance(uyvy, bytes)
    # 640 * 360 * 2 bytes per pixel in UYVY format = 460800 bytes
    assert len(uyvy) == 640 * 360 * 2
