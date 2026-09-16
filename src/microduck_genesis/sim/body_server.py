"""A Microduck body in Genesis World, served to real `robotd` and `tofd` over TCP Protocol 1."""
from __future__ import annotations

import json
import socket
import socketserver
import threading

from microduck_genesis.constants import PROTOCOL
from microduck_genesis.sim.body import Body


class Handler(socketserver.StreamRequestHandler):
    """Handles one daemon connection to a specific duck body over TCP."""

    def handle(self) -> None:
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        body: Body = self.server.body
        print(f"== duck {body.index}: daemon connected from {self.client_address}", flush=True)

        try:
            for raw in self.rfile:
                line = raw.decode().strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    answer = self.dispatch(body, req)
                except Exception as error:
                    answer = {"error": str(error)}

                self.wfile.write((json.dumps(answer) + "\n").encode())
                self.wfile.flush()
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            try:
                print(f"== duck {body.index}: daemon disconnected", flush=True)
            except OSError:
                pass

    def dispatch(self, body: Body, request: dict) -> dict:
        op = request.get("op")
        if op == "hello":
            asked = request.get("protocol")
            if asked != PROTOCOL:
                raise ValueError(
                    f"the daemon speaks protocol {asked} and this simulator speaks {PROTOCOL}"
                )
            return {"protocol": PROTOCOL}
        if op == "read":
            return body.sensors()
        if op == "write":
            body.set_targets(request["targets"])
            return {}
        if op == "gain":
            body.set_gain(int(request["kp"]))
            return {}
        if op == "torque":
            body.set_torque(bool(request["on"]))
            return {}
        if op == "slow":
            return body.slow_sensors()
        if op == "tof":
            return body.depth()
        raise ValueError(f"unknown op {op!r}")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    body: Body
