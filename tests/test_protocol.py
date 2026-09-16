import json
import socket
import threading
import time
import pytest

from microduck_genesis.constants import JOINT_NAMES, PROTOCOL
from microduck_genesis.sim.body_server import Handler, Server


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


def test_tcp_protocol():
    body = MockBody()
    # Find free port
    server = Server(("127.0.0.1", 0), Handler)
    server.body = body
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    time.sleep(0.05)

    s = socket.create_connection(("127.0.0.1", port), timeout=2.0)
    f = s.makefile("rw", buffering=1)

    def send(req: dict) -> dict:
        f.write(json.dumps(req) + "\n")
        f.flush()
        line = f.readline()
        return json.loads(line)

    # 1. Hello
    resp = send({"op": "hello", "protocol": 1, "joints": 15})
    assert resp == {"protocol": 1}

    # 2. Read
    resp = send({"op": "read"})
    assert len(resp["positions"]) == 15
    assert resp["trunk_z"] == 0.125
    assert resp["imu"]["gravity"] == [0.0, 0.0, -1.0]

    # 3. Write
    resp = send({"op": "write", "targets": [0.0] * 15})
    assert resp == {}
    assert body.target_called

    # 4. Gain
    resp = send({"op": "gain", "kp": 200})
    assert resp == {}
    assert body.gain_called

    # 5. Torque
    resp = send({"op": "torque", "on": True})
    assert resp == {}
    assert body.torque_called

    # 6. Slow
    resp = send({"op": "slow"})
    assert resp["volts"] == 7.4

    # 7. ToF
    resp = send({"op": "tof"})
    assert resp["rows"] == 8 and resp["cols"] == 8
    assert len(resp["distance_mm"]) == 64

    s.close()
    server.shutdown()
    server.server_close()
