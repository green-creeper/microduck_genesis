"""Interactive keyboard teleoperation for Microduck daemons via UNIX domain socket."""
from __future__ import annotations

import argparse
import json
import os
import select
import socket
import sys
import termios
import threading
import time
import tty
from pathlib import Path


DEFAULT_STATE_DIR = Path(os.environ.get("DUCK_SIM_STATE", Path.home() / ".cache" / "duck-sim"))


def get_key(timeout: float = 0.05) -> str | None:
    """Read a keypress or escape sequence from stdin non-blockingly."""
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        r2, _, _ = select.select([sys.stdin], [], [], 0.05)
        if r2:
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r3:
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":
                        return "up"
                    elif ch3 == "B":
                        return "down"
                    elif ch3 == "C":
                        return "right"
                    elif ch3 == "D":
                        return "left"
            return "esc"
        return "esc"
    return ch


class Teleop:
    def __init__(self, sock_path: Path):
        self.sock_path = Path(sock_path)
        self.sock: socket.socket | None = None
        self.writer = None
        self.vx = 0.0
        self.vyaw = 0.0
        self.running = True
        self.lock = threading.Lock()

    def connect(self) -> None:
        if not self.sock_path.exists():
            raise FileNotFoundError(
                f"Socket not found at {self.sock_path}.\n"
                "Is the simulation running? Start it first with: ./scripts/duck-sim"
            )
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.connect(str(self.sock_path))
        self.writer = self.sock.makefile("w")

    def send_skill(self, skill: str) -> None:
        if not self.writer:
            return
        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": "robot.do",
            "params": {"skill": skill},
        }) + "\n"
        try:
            self.writer.write(payload)
            self.writer.flush()
        except Exception as err:
            print(f"\n[!] Failed to send skill: {err}", flush=True)

    def _sender_loop(self) -> None:
        while self.running:
            with self.lock:
                vx = self.vx
                vyaw = self.vyaw

            if self.writer and (vx != 0.0 or vyaw != 0.0):
                try:
                    payload = json.dumps({
                        "jsonrpc": "2.0",
                        "method": "robot.move",
                        "params": {"vx": vx, "vy": 0.0, "vyaw": vyaw},
                    }) + "\n"
                    self.writer.write(payload)
                    self.writer.flush()
                except Exception:
                    pass
            time.sleep(0.1)

    def run(self) -> None:
        self.connect()

        # Start 10 Hz background sender
        sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        sender_thread.start()

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        def print_banner():
            print("\033[2J\033[H", end="")  # Clear screen
            print("======================================================")
            print("         Microduck Keyboard Teleoperation             ")
            print("======================================================")
            print(f"Connected to: {self.sock_path}")
            print("\nControls:")
            print("  [W] or [▲ Up]     : Walk Forward (+0.05 m/s, min 0.30)")
            print("  [S] or [▼ Down]   : Walk Backward / Decel (-0.05 m/s)")
            print("  [A] or [◀ Left]   : Turn Left (+0.20 rad/s)")
            print("  [D] or [▶ Right]  : Turn Right (-0.20 rad/s)")
            print("  [Space]           : Stop Motion (E-stop)")
            print("  [R]               : Trick: Roulade")
            print("  [Q] or [Ctrl-C]   : Quit\n")
            print("------------------------------------------------------")

        print_banner()

        try:
            tty.setcbreak(fd)
            last_status = ""

            while True:
                key = get_key(timeout=0.05)
                if key:
                    with self.lock:
                        if key in ("w", "W", "up"):
                            if self.vx < 0.25:
                                self.vx = 0.30  # Jump to forward gait threshold
                            else:
                                self.vx = min(0.50, round(self.vx + 0.05, 2))
                        elif key in ("s", "S", "down"):
                            if self.vx > 0.0:
                                self.vx = max(0.0, round(self.vx - 0.05, 2))
                            else:
                                self.vx = max(-0.20, round(self.vx - 0.05, 2))
                        elif key in ("a", "A", "left"):
                            self.vyaw = min(1.2, round(self.vyaw + 0.2, 2))
                        elif key in ("d", "D", "right"):
                            self.vyaw = max(-1.2, round(self.vyaw - 0.2, 2))
                        elif key == " ":
                            self.vx = 0.0
                            self.vyaw = 0.0
                        elif key in ("r", "R"):
                            self.send_skill("roulade")
                        elif key in ("q", "Q", "esc", "\x03"):
                            break

                with self.lock:
                    vx = self.vx
                    vyaw = self.vyaw

                # Status indicator
                if vx > 0:
                    motion = "▲ FORWARD"
                elif vx < 0:
                    motion = "▼ REVERSE"
                elif vyaw > 0:
                    motion = "◀ TURNING LEFT"
                elif vyaw < 0:
                    motion = "▶ TURNING RIGHT"
                else:
                    motion = "■ STOPPED"

                status = f"\rStatus: {motion:<16} | vx: {vx:+.2f} m/s | vyaw: {vyaw:+.2f} rad/s   "
                if status != last_status:
                    sys.stdout.write(status)
                    sys.stdout.flush()
                    last_status = status

        finally:
            self.running = False
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print("\n\nStopped teleoperation cleanly.")
            if self.sock:
                self.sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for Microduck robotd daemon")
    parser.add_argument(
        "--socket",
        type=Path,
        default=DEFAULT_STATE_DIR / "duck.sock",
        help="Path to robotd UNIX domain socket",
    )
    parser.add_argument(
        "--duck",
        default="",
        help="Specific duck name in multi-duck runs (e.g. duck-a, duck-b, duck-c)",
    )
    args = parser.parse_args()

    sock_path = args.socket
    if args.duck:
        sock_path = DEFAULT_STATE_DIR / f"{args.duck}.sock"

    teleop = Teleop(sock_path)
    try:
        teleop.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
