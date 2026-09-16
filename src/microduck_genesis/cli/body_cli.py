"""CLI entry point to launch the Genesis World Microduck body server."""
from __future__ import annotations

import argparse
import threading
from pathlib import Path

from microduck_genesis.constants import (
    ASSETS_DIR,
    DEFAULT_ROBOT,
    DEFAULT_SCENE,
    KEYFRAMES,
    SCENE_APARTMENT,
    SPACING,
)
from microduck_genesis.sim.body_server import Handler, Server
from microduck_genesis.sim.camera import FPS as CAMERA_FPS
from microduck_genesis.sim.camera import Camera, FrameHandler, FrameServer
from microduck_genesis.sim.world import World, run_loop


def parse_cameras(cameras_str: str, ducks: int) -> set[int]:
    cameras_str = cameras_str.strip()
    if not cameras_str:
        return set()
    if cameras_str == "all":
        return set(range(ducks))
    result = set()
    for c in cameras_str.split(","):
        c = c.strip()
        if c:
            idx = ord(c) - ord("a")
            if 0 <= idx < ducks:
                result.add(idx)
    return result


def resolve_scene_path(scene_arg: str | Path) -> Path:
    s = str(scene_arg)
    if s == "apartment":
        return SCENE_APARTMENT
    p = Path(s)
    if p.exists():
        return p
    # Check inside assets directory
    in_assets = ASSETS_DIR / f"{s}.xml"
    if in_assets.exists():
        return in_assets
    return DEFAULT_SCENE


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis World simulation server for Microduck")
    parser.add_argument("--scene", default=str(DEFAULT_SCENE), help="Scene XML path or preset (e.g. apartment)")
    parser.add_argument("--robot", default=str(DEFAULT_ROBOT), help="Robot XML model path")
    parser.add_argument("--ducks", type=int, default=1, help="Number of ducks sharing the world")
    parser.add_argument("--host", default="127.0.0.1", help="TCP bind address")
    parser.add_argument("--port", type=int, default=7801, help="Base TCP port for duck-body protocol")
    parser.add_argument("--headless", action="store_true", help="Run without graphical viewer")
    parser.add_argument("--cameras", default="", help="Which ducks render cameras ('a', 'a,b', 'all')")
    parser.add_argument("--frame-port", type=int, default=7901, help="Base TCP port for camera frames")
    parser.add_argument("--camera-fps", type=int, default=CAMERA_FPS, help="Camera frame rate")
    parser.add_argument("--limp", action="store_true", help="Start with no torque (collapses on floor)")
    parser.add_argument("--keyframe", default="SIT", choices=list(KEYFRAMES.keys()), help="Initial keyframe pose")

    args = parser.parse_args()

    scene_path = resolve_scene_path(args.scene)
    robot_path = Path(args.robot)
    if not robot_path.exists():
        robot_path = DEFAULT_ROBOT

    wanted_cameras = parse_cameras(args.cameras, args.ducks)

    # Bind TCP sockets early so wait_for_port succeeds immediately while Genesis compiles scene
    servers = []
    ready_event = threading.Event()
    for idx in range(args.ducks):
        body_server = Server((args.host, args.port + idx), Handler)
        body_server.ready_event = ready_event
        body_server.body = None
        threading.Thread(target=body_server.serve_forever, daemon=True).start()
        servers.append(body_server)

    print(f"== Initializing Genesis World simulation ({args.ducks} duck(s), keyframe: {args.keyframe})...", flush=True)
    world = World(
        scene_path=scene_path,
        robot_path=robot_path,
        count=args.ducks,
        headless=args.headless,
        keyframe=args.keyframe,
    )

    pose_dict, trunk_z = KEYFRAMES.get(args.keyframe, (None, 0.125))

    for idx, body in enumerate(world.bodies):
        if args.limp:
            body.set_torque(False)

        servers[idx].body = body

        # Optional camera frame server
        if idx in wanted_cameras:
            body.camera = Camera(world.scene, body.entity)
            frame_server = FrameServer((args.host, args.frame_port + idx), FrameHandler)
            frame_server.camera = body.camera
            frame_server.fps = args.camera_fps
            threading.Thread(target=frame_server.serve_forever, daemon=True).start()
            servers.append(frame_server)

        cam_info = f" · camera on {args.host}:{args.frame_port + idx}" if idx in wanted_cameras else ""
        print(f"==   duck {idx}: robotd --sim {args.host}:{args.port + idx}{cam_info}", flush=True)

    ready_event.set()

    print(f"== Genesis simulation running. Press Ctrl+C to stop.", flush=True)
    run_loop(world, headless=args.headless)


if __name__ == "__main__":
    main()
