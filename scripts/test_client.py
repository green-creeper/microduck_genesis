#!/usr/bin/env python3
"""Test client simulating robotd and tofd connecting to microduck_genesis body server."""
import json
import socket
import time

HOST = "127.0.0.1"
PORT = 7801


def test_connection():
    print(f"Connecting to Genesis body server at {HOST}:{PORT}...")
    s = socket.create_connection((HOST, PORT), timeout=5.0)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    f = s.makefile("rw", buffering=1)

    def send(req: dict) -> dict:
        f.write(json.dumps(req) + "\n")
        f.flush()
        line = f.readline()
        if not line:
            raise RuntimeError("Server closed connection")
        return json.loads(line)

    # 1. Handshake
    hello_resp = send({"op": "hello", "protocol": 1, "joints": 15})
    print("1. Handshake response:", hello_resp)
    assert hello_resp.get("protocol") == 1, f"Expected protocol 1, got {hello_resp}"

    # 2. Read sensors
    read_resp = send({"op": "read"})
    pos = read_resp["positions"]
    vel = read_resp["velocities"]
    cur = read_resp["currents_ma"]
    trunk_z = read_resp["trunk_z"]
    imu = read_resp["imu"]
    print(f"2. Read sensors: 15 joints, trunk_z={trunk_z:.3f}m, gravity={imu['gravity']}")
    assert len(pos) == 15, f"Expected 15 positions, got {len(pos)}"
    assert len(vel) == 15, f"Expected 15 velocities, got {len(vel)}"
    assert len(cur) == 15, f"Expected 15 currents, got {len(cur)}"
    assert len(imu["gravity"]) == 3, "Expected 3-dim gravity"

    # 3. Write target angles
    write_resp = send({"op": "write", "targets": pos})
    print("3. Write response:", write_resp)
    assert write_resp == {} or "error" not in write_resp

    # 4. Gain & Torque
    gain_resp = send({"op": "gain", "kp": 200})
    print("4. Gain response:", gain_resp)
    torque_resp = send({"op": "torque", "on": True})
    print("5. Torque response:", torque_resp)

    # 5. Slow telemetry
    slow_resp = send({"op": "slow"})
    print("6. Slow sensors:", slow_resp)
    assert "volts" in slow_resp and len(slow_resp["temps_c"]) == 15

    # 6. ToF Depth
    tof_resp = send({"op": "tof"})
    print(f"7. ToF depth: {tof_resp['rows']}x{tof_resp['cols']} zones, sample distance: {tof_resp['distance_mm'][:4]}mm")
    assert tof_resp["rows"] == 8 and tof_resp["cols"] == 8
    assert len(tof_resp["distance_mm"]) == 64
    assert len(tof_resp["status"]) == 64

    print("\n✅ All Protocol 1 tests passed successfully!")
    s.close()


if __name__ == "__main__":
    test_connection()
