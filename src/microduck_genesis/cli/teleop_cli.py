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
        self.last_action_msg = ""
        self.current_duck_name = self.sock_path.stem.replace(".sock", "")
        if self.current_duck_name == "duck":
            self.current_duck_name = "duck-a"

    def connect(self) -> None:
        if not self.sock_path.exists():
            raise FileNotFoundError(
                f"Socket not found at {self.sock_path}.\n"
                "Is the simulation running? Start it first with: ./scripts/duck-sim"
            )
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.connect(str(self.sock_path))
        self.writer = self.sock.makefile("w")

    def switch_duck(self, duck_name: str) -> None:
        candidate = DEFAULT_STATE_DIR / f"{duck_name}.sock"
        if not candidate.exists():
            self.last_action_msg = f"[!] {duck_name}.sock not found"
            return
        with self.lock:
            self.vx = 0.0
            self.vyaw = 0.0
            self.sock_path = candidate
            self.current_duck_name = duck_name
            try:
                self.connect()
                self.last_action_msg = f"Switched to {duck_name}"
            except Exception as err:
                self.last_action_msg = f"[!] Failed to switch: {err}"

    def send_skill(self, skill: str, label: str) -> None:
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
            self.last_action_msg = f"Skill: {label}"
        except Exception as err:
            self.last_action_msg = f"[!] Failed skill {label}: {err}"

    def send_sound(self, tag: str, label: str) -> None:
        if not self.writer:
            return
        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": "robot.sound",
            "params": {"tag": tag},
        }) + "\n"
        try:
            self.writer.write(payload)
            self.writer.flush()
            self.last_action_msg = f"Sound: {label}"
        except Exception as err:
            self.last_action_msg = f"[!] Failed sound {label}: {err}"

    def send_look(self, x: float, y: float, z: float, label: str) -> None:
        if not self.writer:
            return
        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": "robot.look",
            "params": {"x": x, "y": y, "z": z},
        }) + "\n"
        try:
            self.writer.write(payload)
            self.writer.flush()
            self.last_action_msg = f"Gaze: {label}"
        except Exception as err:
            self.last_action_msg = f"[!] Failed gaze: {err}"

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
            print("\033[2J\033[H", end="")  # Clear terminal
            print("========================================================================")
            print("                     Microduck Keyboard Cockpit                         ")
            print("========================================================================")
            print(f"Target: [{self.current_duck_name}]  (Socket: {self.sock_path})")
            print("\n  [Locomotion]")
            print("    [W] / [▲ Up]    : Walk Forward (+0.05 m/s, jump to 0.30 m/s)")
            print("    [S] / [▼ Down]  : Walk Backward / Decelerate (-0.05 m/s)")
            print("    [A] / [◀ Left]  : Turn Left (+0.20 rad/s)")
            print("    [D] / [▶ Right] : Turn Right (-0.20 rad/s)")
            print("    [Space]         : E-Stop / Stop Motion (0.0 m/s)")
            print("\n  [Postures & Skills]")
            print("    [X] : Sit ⇄ Stand Toggle (sit_toggle)")
            print("    [P] : Ground Pick / Bow (ground_pick)")
            print("    [J] : Left Kick (kick_left)")
            print("    [K] : Right Kick (kick_right)")
            print("    [R] : Roulade Somersault (roulade)")
            print("\n  [Head Camera Gaze]")
            print("    [I] : Look Up       [M] : Look Down")
            print("    [U] : Look Left     [O] : Look Right    [C] : Center Head")
            print("\n  [Sounds & Quacks]")
            print("    [Q] / [F] : 🦆 Quack! (chirp)   [G] : Greet (wak-wak)")
            print("    [H]       : Honk / Alarm         [Z] : Coo (sleepy)")
            print("\n  [Duck Selector in Multi-Duck]")
            print("    [1] duck-a   [2] duck-b   [3] duck-c   [4] duck-d")
            print("\n  [Quit]")
            print("    [Esc] / [Ctrl-C] : Exit cleanly")
            print("========================================================================")

        print_banner()

        try:
            tty.setcbreak(fd)
            last_line = ""

            while True:
                key = get_key(timeout=0.05)
                if key:
                    with self.lock:
                        # Movement
                        if key in ("w", "W", "up"):
                            if self.vx < 0.25:
                                self.vx = 0.30  # Jump above RL walking deadband
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
                            self.last_action_msg = "E-Stop (motion zeroed)"

                        # Skills & Postures
                        elif key in ("x", "X"):
                            self.send_skill("sit_toggle", "Sit ⇄ Stand (sit_toggle)")
                        elif key in ("p", "P"):
                            self.send_skill("ground_pick", "Ground Pick (bow down)")
                        elif key in ("j", "J"):
                            self.send_skill("kick_left", "Left Kick (kick_left)")
                        elif key in ("k", "K"):
                            self.send_skill("kick_right", "Right Kick (kick_right)")
                        elif key in ("r", "R"):
                            self.send_skill("roulade", "Roulade Somersault (roulade)")

                        # Head Gaze
                        elif key in ("i", "I"):
                            self.send_look(1.0, 0.0, 0.25, "Up")
                        elif key in ("m", "M"):
                            self.send_look(0.5, 0.0, -0.25, "Down")
                        elif key in ("u", "U"):
                            self.send_look(1.0, 0.4, 0.0, "Left")
                        elif key in ("o", "O"):
                            self.send_look(1.0, -0.4, 0.0, "Right")
                        elif key in ("c", "C"):
                            self.send_look(1.0, 0.0, 0.0, "Center")

                        # Sounds
                        elif key in ("q", "Q", "f", "F"):
                            self.send_sound("chirp", "🦆 Quack! (chirp)")
                        elif key in ("g", "G"):
                            self.send_sound("greet", "Greet (wak-wak)")
                        elif key in ("h", "H"):
                            self.send_sound("alarm", "Honk / Alarm")
                        elif key in ("z", "Z"):
                            self.send_sound("coo", "Coo (purr)")

                        # Duck Selector
                        elif key in ("1", "2", "3", "4"):
                            target = f"duck-{'abcd'[int(key) - 1]}"
                            self.switch_duck(target)
                            print_banner()

                        # Exit
                        elif key in ("esc", "\x03", "\x1b"):
                            break

                with self.lock:
                    vx = self.vx
                    vyaw = self.vyaw

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

                act_msg = f" | {self.last_action_msg}" if self.last_action_msg else ""
                line = f"\r[{self.current_duck_name}] {motion:<16} | vx: {vx:+.2f} m/s | vyaw: {vyaw:+.2f} rad/s{act_msg:<35}   "
                if line != last_line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    last_line = line

        finally:
            self.running = False
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print("\n\nStopped teleoperation cleanly.")
            if self.sock:
                self.sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard cockpit for Microduck robotd daemon")
    parser.add_argument(
        "--socket",
        type=Path,
        default=DEFAULT_STATE_DIR / "duck.sock",
        help="Path to robotd UNIX domain socket",
    )
    parser.add_argument(
        "--duck",
        default="",
        help="Specific duck name in multi-duck runs (e.g. duck-a, duck-b, duck-c, duck-d)",
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
