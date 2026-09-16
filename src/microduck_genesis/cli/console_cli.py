"""Web console and live camera stream for Microduck on macOS and Linux."""
from __future__ import annotations

import argparse
import io
import json
import os
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_STATE_DIR = Path(os.environ.get("DUCK_SIM_STATE", Path.home() / ".cache" / "duck-sim"))


class CameraReceiver:
    """Connects to Genesis camera TCP stream and continuously converts frames to JPEG."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7901, fps: int = 20):
        self.host = host
        self.port = port
        self.period = 1.0 / max(1, fps)
        self.latest_jpeg: bytes | None = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False

    def start(self) -> None:
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self) -> None:
        while self.running:
            try:
                s = socket.socket()
                s.settimeout(3.0)
                s.connect((self.host, self.port))
                self.connected = True
                print(f"== Connected to camera stream on {self.host}:{self.port}", flush=True)

                while self.running:
                    t_start = time.perf_counter()
                    header = s.recv(4)
                    if not header or len(header) < 4:
                        break
                    length = struct.unpack("<I", header)[0]
                    payload = bytearray()
                    while len(payload) < length:
                        chunk = s.recv(min(65536, length - len(payload)))
                        if not chunk:
                            break
                        payload.extend(chunk)

                    if len(payload) == length and length == 640 * 360 * 2:
                        jpeg = self._uyvy_to_jpeg(bytes(payload))
                        with self.lock:
                            self.latest_jpeg = jpeg

                    elapsed = time.perf_counter() - t_start
                    slack = self.period - elapsed
                    if slack > 0:
                        time.sleep(slack)
            except Exception:
                self.connected = False
                time.sleep(1.0)
            finally:
                try:
                    s.close()
                except Exception:
                    pass

    @staticmethod
    def _uyvy_to_jpeg(payload: bytes) -> bytes:
        uyvy = np.frombuffer(payload, dtype=np.uint8).reshape((360, 320, 4))
        u = uyvy[:, :, 0].astype(np.float32) - 128.0
        y0 = uyvy[:, :, 1].astype(np.float32)
        v = uyvy[:, :, 2].astype(np.float32) - 128.0
        y1 = uyvy[:, :, 3].astype(np.float32)

        r0 = np.clip(y0 + 1.402 * v, 0, 255).astype(np.uint8)
        g0 = np.clip(y0 - 0.344136 * u - 0.714136 * v, 0, 255).astype(np.uint8)
        b0 = np.clip(y0 + 1.772 * u, 0, 255).astype(np.uint8)

        r1 = np.clip(y1 + 1.402 * v, 0, 255).astype(np.uint8)
        g1 = np.clip(y1 - 0.344136 * u - 0.714136 * v, 0, 255).astype(np.uint8)
        b1 = np.clip(y1 + 1.772 * u, 0, 255).astype(np.uint8)

        rgb = np.empty((360, 640, 3), dtype=np.uint8)
        rgb[:, 0::2, 0] = r0
        rgb[:, 0::2, 1] = g0
        rgb[:, 0::2, 2] = b0
        rgb[:, 1::2, 0] = r1
        rgb[:, 1::2, 1] = g1
        rgb[:, 1::2, 2] = b1

        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def get_frame(self) -> bytes | None:
        with self.lock:
            return self.latest_jpeg


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Microduck Web Console</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f111a; color: #e2e8f0; display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 20px; }
    header { width: 100%; max-width: 960px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
    h1 { font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
    .badge { background: #22c55e20; color: #4ade80; border: 1px solid #22c55e40; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
    .main-grid { display: grid; grid-template-columns: 1fr; gap: 20px; width: 100%; max-width: 960px; }
    @media (min-width: 768px) { .main-grid { grid-template-columns: 640px 1fr; } }
    .video-card { background: #1a1d2e; border: 1px solid #2d3748; border-radius: 12px; overflow: hidden; position: relative; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    .video-feed { width: 100%; height: auto; display: block; background: #000; aspect-ratio: 16/9; object-fit: contain; }
    .video-overlay { position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.6); backdrop-filter: blur(4px); padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; color: #38bdf8; }
    .panel { background: #1a1d2e; border: 1px solid #2d3748; border-radius: 12px; padding: 18px; display: flex; flex-direction: column; gap: 16px; }
    .section-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }
    .btn-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    button { background: #262b42; color: #f1f5f9; border: 1px solid #3b4261; border-radius: 8px; padding: 10px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s; display: flex; align-items: center; justify-content: center; gap: 6px; }
    button:hover { background: #333a59; border-color: #60a5fa; }
    button:active { transform: scale(0.97); }
    .btn-accent { background: #2563eb; border-color: #3b82f6; }
    .btn-accent:hover { background: #1d4ed8; }
    .btn-quack { background: #eab308; color: #000; border-color: #facc15; font-size: 15px; }
    .btn-quack:hover { background: #ca8a04; }
    .btn-danger { background: #dc2626; border-color: #ef4444; }
    .btn-danger:hover { background: #b91c1c; }
    .d-pad { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; max-width: 180px; margin: 0 auto; }
    .kbd-guide { font-size: 12px; color: #64748b; line-height: 1.6; background: #121524; padding: 12px; border-radius: 8px; border: 1px solid #1e2538; }
    kbd { background: #2d3748; color: #e2e8f0; padding: 2px 6px; border-radius: 4px; font-size: 11px; }
    #log { font-family: monospace; font-size: 11px; color: #38bdf8; height: 38px; overflow: hidden; background: #0b0d17; padding: 8px; border-radius: 6px; border: 1px solid #1f293d; }
  </style>
</head>
<body>
  <header>
    <h1>🦆 Microduck Head Camera</h1>
    <span class="badge" id="status-badge">● LIVE 15 FPS</span>
  </header>

  <div class="main-grid">
    <div class="video-card">
      <div class="video-overlay">640×360 UYVY · GENESIS</div>
      <img class="video-feed" src="/stream.mjpg" alt="Live Camera Feed">
    </div>

    <div class="panel">
      <div class="section-title">Locomotion</div>
      <div class="d-pad">
        <div></div>
        <button onclick="walk(0.35, 0)">▲</button>
        <div></div>
        <button onclick="walk(0, 0.4)">◀</button>
        <button class="btn-danger" onclick="walk(0, 0)">■</button>
        <button onclick="walk(0, -0.4)">▶</button>
        <div></div>
        <button onclick="walk(-0.15, 0)">▼</button>
        <div></div>
      </div>

      <div class="section-title">Postures & Skills</div>
      <div class="btn-grid">
        <button class="btn-accent" onclick="skill('sit_toggle')">🪑 Sit ⇄ Stand</button>
        <button onclick="skill('ground_pick')">🌾 Ground Pick</button>
        <button onclick="skill('kick_left')">⚽ Kick Left</button>
        <button onclick="skill('kick_right')">⚽ Kick Right</button>
        <button style="grid-column: span 2;" onclick="skill('roulade')">🤸 Roulade Roll</button>
      </div>

      <div class="section-title">Vocalizations</div>
      <button class="btn-quack" onclick="sound('chirp')">🦆 Quack!</button>

      <div id="log">Ready. Use on-screen buttons or keyboard.</div>

      <div class="kbd-guide">
        <strong>Keyboard Shortcuts:</strong><br>
        <kbd>W</kbd>/<kbd>S</kbd>: Walk · <kbd>A</kbd>/<kbd>D</kbd>: Turn · <kbd>Space</kbd>: Stop<br>
        <kbd>X</kbd>: Sit/Stand · <kbd>P</kbd>: Pick · <kbd>R</kbd>: Roll · <kbd>Q</kbd>: Quack
      </div>
    </div>
  </div>

  <script>
    const logEl = document.getElementById("log");
    function log(msg) { logEl.textContent = msg; }

    async function rpc(method, params = {}) {
      try {
        const res = await fetch("/api/rpc", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ jsonrpc: "2.0", method, params, id: Date.now() })
        });
        const data = await res.json();
        log(`${method}: ${JSON.stringify(params)}`);
        return data;
      } catch (err) {
        log(`Error: ${err}`);
      }
    }

    let walkInterval = null;
    function walk(vx, vyaw) {
      if (walkInterval) clearInterval(walkInterval);
      if (vx === 0 && vyaw === 0) {
        rpc("robot.move", { vx: 0, vy: 0, vyaw: 0 });
        return;
      }
      rpc("robot.move", { vx, vy: 0, vyaw });
      walkInterval = setInterval(() => {
        rpc("robot.move", { vx, vy: 0, vyaw });
      }, 150);
      setTimeout(() => { clearInterval(walkInterval); }, 3000);
    }

    function skill(name) { rpc("robot.do", { skill: name }); }
    function sound(tag) { rpc("robot.sound", { tag }); }
    function look(x, y, z) { rpc("robot.look", { x, y, z }); }

    window.addEventListener("keydown", (e) => {
      if (["INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
      const k = e.key.toLowerCase();
      if (k === "w" || k === "arrowup") walk(0.35, 0);
      else if (k === "s" || k === "arrowdown") walk(-0.15, 0);
      else if (k === "a" || k === "arrowleft") walk(0, 0.4);
      else if (k === "d" || k === "arrowright") walk(0, -0.4);
      else if (k === " ") walk(0, 0);
      else if (k === "x") skill("sit_toggle");
      else if (k === "p") skill("ground_pick");
      else if (k === "r") skill("roulade");
      else if (k === "q") sound("chirp");
    });
  </script>
</body>
</html>
"""


class ConsoleHandler(BaseHTTPRequestHandler):
    camera: CameraReceiver
    sock_path: Path

    def log_message(self, format, *args):
        pass  # Quiet logger

    def do_HEAD(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            content = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            try:
                while True:
                    frame = self.camera.get_frame()
                    if frame is not None:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)
            except (ConnectionResetError, BrokenPipeError):
                pass
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/rpc":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                s = socket.socket(socket.AF_UNIX)
                s.connect(str(self.sock_path))
                s.sendall(body + b"\n")
                res = s.recv(4096)
                s.close()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res)))
                self.end_headers()
                self.wfile.write(res)
            except Exception as err:
                err_resp = json.dumps({"error": str(err)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_resp)))
                self.end_headers()
                self.wfile.write(err_resp)
            return

        self.send_response(404)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Web console and camera viewer for Microduck")
    parser.add_argument("--port", type=int, default=8080, help="Web console HTTP port (default: 8080)")
    parser.add_argument("--frame-port", type=int, default=7901, help="Genesis camera frame port (default: 7901)")
    parser.add_argument("--socket", type=Path, default=DEFAULT_STATE_DIR / "duck.sock", help="Path to robotd socket")
    args = parser.parse_args()

    camera = CameraReceiver(port=args.frame_port)
    camera.start()

    ConsoleHandler.camera = camera
    ConsoleHandler.sock_path = args.socket

    server = ThreadingHTTPServer(("0.0.0.0", args.port), ConsoleHandler)
    print(f"== Microduck Web Console listening at http://127.0.0.1:{args.port}/", flush=True)
    print(f"==   Camera source: 127.0.0.1:{args.frame_port}", flush=True)
    print(f"==   Robot socket: {args.socket}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
