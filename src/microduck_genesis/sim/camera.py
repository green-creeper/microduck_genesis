"""What a duck sees, rendered via Genesis and streamed as UYVY to `mediad`."""
from __future__ import annotations

import socket
import socketserver
import struct
import threading
import time
import numpy as np

WIDTH = 640
HEIGHT = 360
FPS = 15

# BT.601 coefficients matching mediad / duck_detect
_Y = np.array([0.299, 0.587, 0.114], dtype=np.float32)
_U = np.array([-0.168736, -0.331264, 0.5], dtype=np.float32)
_V = np.array([0.5, -0.418688, -0.081312], dtype=np.float32)


def to_uyvy(rgb: np.ndarray) -> bytes:
    """Convert RGB frame (H, W, 3) to packed UYVY bytes: U Y0 V Y1 per pixel pair."""
    frame = rgb.astype(np.float32)
    luma = frame @ _Y
    chroma_u = frame @ _U + 128.0
    chroma_v = frame @ _V + 128.0

    pairs = frame.shape[1] // 2
    packed = np.empty((frame.shape[0], pairs, 4), dtype=np.uint8)
    packed[:, :, 0] = np.clip((chroma_u[:, 0::2] + chroma_u[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 1] = np.clip(luma[:, 0::2], 0, 255)
    packed[:, :, 2] = np.clip((chroma_v[:, 0::2] + chroma_v[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 3] = np.clip(luma[:, 1::2], 0, 255)
    return packed.tobytes()


class Camera:
    """Head camera rendered on demand via Genesis Camera object."""

    def __init__(self, cam_handle, width: int = WIDTH, height: int = HEIGHT):
        self.cam_handle = cam_handle
        self.width = width
        self.height = height
        self.latest: bytes | None = None
        self.lock = threading.Lock()

    def render(self) -> None:
        """Render one frame and update the latest UYVY buffer."""
        if self.cam_handle is not None:
            try:
                rgb_arr, _, _, _ = self.cam_handle.render(rgb=True)
                if hasattr(rgb_arr, "cpu"):
                    rgb_arr = rgb_arr.cpu().numpy()
                else:
                    rgb_arr = np.asarray(rgb_arr)
                if rgb_arr.ndim == 4:
                    rgb_arr = rgb_arr[0]
                packed = to_uyvy(rgb_arr)
                with self.lock:
                    self.latest = packed
                return
            except Exception:
                pass

        # Fallback test pattern: neutral gray frame
        blank = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        packed = to_uyvy(blank)
        with self.lock:
            self.latest = packed

    def frame(self) -> bytes | None:
        with self.lock:
            return self.latest


class FrameHandler(socketserver.BaseRequestHandler):
    """Streams length-prefixed raw UYVY frames over TCP."""

    def handle(self) -> None:
        camera: Camera = self.server.camera
        fps: int = getattr(self.server, "fps", FPS)
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        period = 1.0 / max(1, fps)
        next_frame = time.perf_counter()
        try:
            while True:
                frame = camera.frame()
                if frame is not None:
                    # 4 bytes little-endian length, followed by payload
                    self.request.sendall(struct.pack("<I", len(frame)) + frame)
                next_frame += period
                slack = next_frame - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_frame = time.perf_counter()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


class FrameServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    camera: Camera
    fps: int = FPS
