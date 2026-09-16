"""Standalone test runner using standard library unittest to verify microduck_genesis."""
import json
import math
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

# Add src to sys.path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

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
from microduck_genesis.sim.body import gravity_in_trunk
from microduck_genesis.sim.camera import to_uyvy
from microduck_genesis.sim.body_server import Handler, Server


class TestConstants(unittest.TestCase):
    def test_joint_names(self):
        self.assertEqual(len(JOINT_NAMES), 15)
        self.assertEqual(JOINT_NAMES[MOUTH_INDEX], "mouth")
        self.assertEqual(len(MODEL_JOINT_NAMES), 14)
        self.assertNotIn("mouth", MODEL_JOINT_NAMES)
        self.assertEqual(len(HOME_POSE), 15)

    def test_protocol_and_keyframes(self):
        self.assertEqual(PROTOCOL, 1)
        self.assertEqual(TIMESTEP, 0.005)
        self.assertGreater(HOME_TRUNK_Z, 0.0)
        for k in ("HOME", "STAND", "SIT", "FOLD"):
            self.assertIn(k, KEYFRAMES)
            pose_dict, trunk_z = KEYFRAMES[k]
            self.assertGreater(trunk_z, 0.0)

    def test_assets(self):
        self.assertTrue(ASSETS_DIR.exists())
        self.assertTrue(DEFAULT_SCENE.exists())
        self.assertTrue(DEFAULT_ROBOT.exists())


class TestMath(unittest.TestCase):
    def test_gravity_projection(self):
        # Upright
        g = gravity_in_trunk([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(g[0], 0.0, places=5)
        self.assertAlmostEqual(g[1], 0.0, places=5)
        self.assertAlmostEqual(g[2], -1.0, places=5)

        # Roll 90 deg about X
        s = math.sqrt(0.5)
        g_roll = gravity_in_trunk([s, s, 0.0, 0.0])
        self.assertAlmostEqual(g_roll[0], 0.0, places=5)
        self.assertAlmostEqual(g_roll[1], -1.0, places=5)
        self.assertAlmostEqual(g_roll[2], 0.0, places=5)

        # Pitch 90 deg about Y
        g_pitch = gravity_in_trunk([s, 0.0, s, 0.0])
        self.assertAlmostEqual(g_pitch[0], 1.0, places=5)
        self.assertAlmostEqual(g_pitch[1], 0.0, places=5)
        self.assertAlmostEqual(g_pitch[2], 0.0, places=5)

    def test_uyvy_conversion(self):
        import numpy as np
        rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        rgb[:] = 200
        uyvy = to_uyvy(rgb)
        self.assertEqual(len(uyvy), 640 * 360 * 2)


class MockBody:
    def __init__(self):
        self.index = 0
        self.target_called = False
        self.gain_called = False
        self.torque_called = False

    def sensors(self):
        return {
            "positions": [0.1] * 15,
            "velocities": [0.0] * 15,
            "currents_ma": [12.0] * 15,
            "trunk_z": 0.125,
            "trunk": [0.0, 0.0, 0.125],
            "sim_time": 1.0,
            "imu": {
                "gyro": [0.0, 0.0, 0.0],
                "gravity": [0.0, 0.0, -1.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        }

    def slow_sensors(self):
        return {"volts": 7.4, "temps_c": [32.0] * 15}

    def depth(self):
        return {"rows": 8, "cols": 8, "distance_mm": [2000] * 64, "status": [5] * 64}

    def set_targets(self, targets):
        self.target_called = True

    def set_gain(self, kp):
        self.gain_called = True

    def set_torque(self, on):
        self.torque_called = True


class TestProtocol(unittest.TestCase):
    def test_tcp_protocol_flow(self):
        body = MockBody()
        server = Server(("127.0.0.1", 0), Handler)
        server.body = body
        port = server.server_address[1]

        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.05)

        s = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        f = s.makefile("rw", buffering=1)

        def send(req: dict) -> dict:
            f.write(json.dumps(req) + "\n")
            f.flush()
            line = f.readline()
            return json.loads(line)

        # Hello
        r = send({"op": "hello", "protocol": 1, "joints": 15})
        self.assertEqual(r, {"protocol": 1})

        # Read
        r = send({"op": "read"})
        self.assertEqual(len(r["positions"]), 15)
        self.assertEqual(r["trunk_z"], 0.125)
        self.assertEqual(r["imu"]["gravity"], [0.0, 0.0, -1.0])

        # Write
        r = send({"op": "write", "targets": [0.0] * 15})
        self.assertEqual(r, {})
        self.assertTrue(body.target_called)

        # Gain
        r = send({"op": "gain", "kp": 200})
        self.assertEqual(r, {})
        self.assertTrue(body.gain_called)

        # Torque
        r = send({"op": "torque", "on": True})
        self.assertEqual(r, {})
        self.assertTrue(body.torque_called)

        # Slow
        r = send({"op": "slow"})
        self.assertEqual(r["volts"], 7.4)

        # Tof
        r = send({"op": "tof"})
        self.assertEqual(r["rows"], 8)
        self.assertEqual(len(r["distance_mm"]), 64)

        s.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    unittest.main()
