# Microduck Genesis

Physics simulation environment for the [Microduck bipedal robot](https://github.com/pollen-robotics/microduck), built on [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world).

Provides drop-in simulation capabilities with full parity to [microduck_rl](https://github.com/pollen-robotics/microduck_rl):
1. **TCP `duck-body` Server (`Protocol 1`)**: Serves simulated robot bodies to the real software daemons in [`microduck`](https://github.com/pollen-robotics/microduck) (`robotd --sim`, `tofd --sim`, `mediad --sim-camera`).
2. **Interactive Policy Rehearsal (`genesis-infer`)**: Test trained ONNX policies with keyboard teleoperation in an interactive 3D Genesis viewer.
3. **Vectorized Reinforcement Learning**: High-throughput parallel environment (`MicroduckEnv`) leveraging GPU/Metal compilation.

---

## Quickstart

Requires Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
cd microduck_genesis
uv sync
```

---

## 1. Running the Body Server (`genesis-duck-body`)

The body server runs the physics simulation at 200 Hz (`dt=0.005`) with 50 Hz control loop decimation and listens on TCP port **7801** (Protocol 1):

### Single Duck (Default: Folded Boot Pose with 3D Viewer)
Starts one duck folded on the floor (`SIT`), ready for `robotd` to initialize and stand it up:
```bash
uv run genesis-duck-body
```

### Multiple Ducks in One World
Simulates 4 ducks sharing a floor with mutual collisions (ports 7801, 7802, 7803, 7804):
```bash
uv run genesis-duck-body --ducks 4
```

### Apartment Scene with Head Camera
Simulates the multi-room indoor scene, streaming length-prefixed raw UYVY head-camera frames to port 7901:
```bash
uv run genesis-duck-body --scene apartment --cameras a --frame-port 7901
```

### Headless Mode
Runs the physics simulation without opening a GUI window (for background runs or CI):
```bash
uv run genesis-duck-body --headless
```

---

## 2. Drop-in Replacement for `scripts/duck-sim` (`DUCK_SIM_RL`)

You can run `microduck_genesis` directly through the official `microduck/scripts/duck-sim` launcher simply by pointing the `DUCK_SIM_RL` environment variable to this repository:

```bash
cd path/to/microduck

# Point the simulator path to microduck_genesis checkout
export DUCK_SIM_RL=$(pwd)/../microduck_genesis

# Launch everything in one command (simulator + daemons + standing gait)
./scripts/duck-sim
```

From there, all standard commands work identically:
```bash
./scripts/duck-sim drive 0.30   # walk forward 8 seconds (0.30 m/s)
./scripts/duck-sim ctl health   # check daemon health (50 Hz control loop)
./scripts/duck-sim realtime     # report real-time physics factor (1.00x)
./scripts/duck-sim down         # cleanly shut down simulator and daemons
```

> [!NOTE]
> **Walking Policy Gait Initiation Threshold (`alpha_walking.onnx`):**
> In both MuJoCo and Genesis, the reinforcement-learning trained walking policy has a velocity threshold ($\sim 0.24\text{ m/s}$). Commands below this speed (such as the default `0.15` in bare `./scripts/duck-sim drive`) cause the policy to balance upright and sway in-place without lifting feet. Specifying `0.25` or `0.30` (e.g. `./scripts/duck-sim drive 0.30`) triggers full alternating forward strides.

> [!TIP]
> You can also pass standard flags like `DUCK_SIM_VIEWER=0 ./scripts/duck-sim` for headless execution or `DUCK_SIM_SCENE=apartment ./scripts/duck-sim`.

### Multi-Robot Simulation (`DUCK_SIM_DUCKS`)

To run multiple ducks simultaneously in the same Genesis world:

```bash
cd path/to/microduck
export DUCK_SIM_RL=$(pwd)/../microduck_genesis

# Launch 4 ducks with host daemons and inter-duck radio
DUCK_SIM_DUCKS=4 ./scripts/duck-sim
```

> [!NOTE]
> In `scripts/duck-sim`, positional arguments for duck count (e.g. `./duck-sim boot 4`) apply only to the containerized `boot` command. For host daemon runs (`./duck-sim` or `./duck-sim up`), duck count is controlled via the `DUCK_SIM_DUCKS` environment variable (`DUCK_SIM_DUCKS=4 ./scripts/duck-sim`).

**What happens under the hood:**
- **Scene Placement:** Genesis spawns 4 robot instances (`duck-a`, `duck-b`, `duck-c`, `duck-d`) spaced along the Y-axis (0.5 m spacing) with full mutual physics collision.
- **Port Allocation:** Sockets are allocated starting from base port 7801 (`7801 + i`):
  - `duck-a`: TCP port 7801, socket `~/.cache/duck-sim/duck-a.sock`
  - `duck-b`: TCP port 7802, socket `~/.cache/duck-sim/duck-b.sock`
  - `duck-c`: TCP port 7803, socket `~/.cache/duck-sim/duck-c.sock`
  - `duck-d`: TCP port 7804, socket `~/.cache/duck-sim/duck-d.sock`
- **Radio Mesh (`duck-ether`):** An inter-duck virtual radio network connects all active `robotd` daemons so they can exchange chorale and beacon packets.
- **Early Socket Binding:** `microduck_genesis` binds all TCP server sockets immediately on launch while the Genesis scene compiles in parallel, ensuring `duck-sim`'s health probe (`wait_for_port`) passes reliably without timing out.

### Keyboard Teleoperation (`genesis-teleop`)

Once your duck simulation is running via `./scripts/duck-sim`, open a second terminal to drive the robot with your keyboard in real time:

```bash
cd path/to/microduck_genesis
uv run genesis-teleop
```

For multi-duck setups, specify which duck to command:
```bash
uv run genesis-teleop --duck duck-b
```

**Keybindings:**

| Category | Key | Action |
|---|---|---|
| **Locomotion** | `W` / `▲ Up` | Walk Forward (accelerates $+0.05\text{ m/s}$; starts at $0.30\text{ m/s}$) |
| | `S` / `▼ Down` | Walk Backward / Decelerate ($-0.05\text{ m/s}$) |
| | `A` / `◀ Left` | Turn Left ($+0.20\text{ rad/s}$) |
| | `D` / `▶ Right` | Turn Right ($-0.20\text{ rad/s}$) |
| | `Space` | E-Stop / Stop Motion ($0.0\text{ m/s}$) |
| **Postures & Skills** | `X` | **Sit ⇄ Stand Toggle** (`sit_toggle` via `alpha_sitstand.onnx`) |
| | `P` | **Ground Pick / Bow** (`ground_pick` via `alpha_ground_pick.onnx`) |
| | `J` | **Left Kick** (`kick_left` via `ball_kick_left.onnx`) |
| | `K` | **Right Kick** (`kick_right` via `ball_kick_right.onnx`) |
| | `R` | **Roulade Somersault** (`roulade` 360° roll) |
| **Head Camera Gaze** | `I` / `M` | Look Up / Look Down |
| | `U` / `O` | Look Left / Look Right |
| | `C` | Center Head Straight Ahead |
| **Vocalizations** | `Q` / `F` | 🦆 **Quack!** (`chirp`) |
| | `G` | Greet (`greet` wak-wak) |
| | `H` | Honk / Alarm (`alarm`) |
| | `Z` | Coo / Purr (`coo`) |
| **Multi-Duck Selection** | `1`–`4` | Switch active duck (`duck-a`, `duck-b`, `duck-c`, `duck-d`) |
| **Exit** | `Esc` / `Ctrl-C` | Exit Teleoperation cleanly |

---

## 3. Connecting Daemons Manually

Once the simulator is running in one terminal:
```bash
uv run genesis-duck-body --port 7801 --keyframe SIT
```

Open another terminal and connect the real binaries from [`microduck`](../microduck):
```bash
cd ../microduck

# 1. Connect the robot daemon (50 Hz control loop, policies, kinematics, IPC)
cargo run --bin robotd -- --sim 127.0.0.1:7801

# 2. (Optional) Connect the VL53L5CX Time-of-Flight depth daemon
cargo run --bin tofd -- --sim 127.0.0.1:7801 --socket /tmp/duck-tof.sock

# 3. (Optional) Connect the head camera daemon (if simulator started with --cameras a)
cargo run --bin mediad -- --sim-camera 127.0.0.1:7901 --robot-socket /tmp/duck.sock
```

---

## 4. Verifying with the Test Client

You can verify the entire Protocol 1 lifecycle (handshake, joint reading, actuator commands, IMU gravity projection, and ToF depth) without running the real daemons:

```bash
# Terminal 1: Start simulator
uv run genesis-duck-body --headless

# Terminal 2: Run verification client
uv run python scripts/test_client.py
```

---

## 5. Interactive Standalone Policy Rehearsal (`genesis-infer`)

Run any exported walking or trick ONNX policy directly in Genesis with interactive keyboard controls:
```bash
uv run genesis-infer ../microduck_rl/walk.onnx
```

**Keyboard Controls:**
| Key | Action |
|---|---|
| `W` / `S` | Increase / decrease forward walking velocity ($v_x$) |
| `A` / `D` | Turn left / right ($\omega_{\text{yaw}}$) |
| `Space` | Stop moving (zero all velocity commands) |
| `Q` / `Esc` | Quit runner |

---

## 6. Parallel Vectorized RL Environment

To train locomotion policies with `rsl-rl` or interact with 1,024 parallel environments on your GPU/Metal compiler:

```python
import torch
from microduck_genesis.tasks import MicroduckEnv

# Initialize 1024 parallel environments on GPU
env = MicroduckEnv(num_envs=1024, show_viewer=False)

# Reset environments
obs = env.reset()
print("Observation batch shape:", obs.shape)  # [1024, 61]

# Step simulation with actions [1024, 14]
actions = torch.zeros((1024, 14))
next_obs, rewards, dones, infos = env.step(actions)
```

---

## CLI Options Reference (`genesis-duck-body`)

| Option | Default | Description |
|---|---|---|
| `--scene` | `scene.xml` | Scene XML path or preset (`apartment`, `scene.xml`, etc.) |
| `--robot` | `robot_groundcontact.xml` | Robot XML model path |
| `--ducks` | `1` | Number of ducks sharing the world |
| `--host` | `127.0.0.1` | TCP bind address |
| `--port` | `7801` | Base TCP port for duck-body protocol (+1 per duck) |
| `--headless` | `false` | Run without graphical 3D viewer window |
| `--cameras` | `""` | Ducks with camera enabled (`a`, `a,b`, `all`) |
| `--frame-port` | `7901` | Base TCP port for UYVY camera frames (+1 per camera) |
| `--camera-fps` | `15` | Head camera frame rate |
| `--limp` | `false` | Start with no torque (duck collapses on floor) |
| `--keyframe` | `SIT` | Initial keyframe pose (`SIT`, `STAND`, `HOME`, `FOLD`) |

---

## Protocol 1 Reference

The simulator communicates over newline-delimited JSON on TCP (port 7801 + duck index):

| Request (`op`) | Arguments | Response |
|---|---|---|
| `hello` | `{"protocol": 1, "joints": 15}` | `{"protocol": 1}` |
| `read` | — | `{"positions": [15], "velocities": [15], "currents_ma": [15], "trunk_z": f64, "trunk": [3], "sim_time": f64, "imu": {"gyro": [3], "gravity": [3], "quat": [4]}}` |
| `write` | `{"targets": [15]}` | `{}` |
| `gain` | `{"kp": u16}` | `{}` (scales stiffness relative to nominal 200) |
| `torque` | `{"on": bool}` | `{}` (torque off puts joints limp) |
| `slow` | — | `{"volts": 7.4, "temps_c": [15]}` |
| `tof` | — | `{"rows": 8, "cols": 8, "distance_mm": [64], "status": [64]}` |

---

## Project Structure

```
microduck_genesis/
├── pyproject.toml                     # uv package configuration
├── README.md                          # Comprehensive documentation & quickstart
├── src/microduck_genesis/
│   ├── constants.py                   # 15 wire joint names, default poses, timesteps
│   ├── sim/
│   │   ├── world.py                   # Genesis Scene wrapper & real-time sync loop
│   │   ├── body.py                    # Body state, actuation & IMU gravity projection
│   │   ├── body_server.py             # TCP Protocol 1 server for robotd / tofd
│   │   ├── camera.py                  # Head camera & UYVY frame streamer for mediad
│   │   └── tof.py                     # VL53L5CX 8x8 Time-of-Flight depth sensor
│   ├── robot/
│   │   └── assets/                    # Bundled MJCF definitions & 3D STL meshes
│   ├── tasks/
│   │   └── env.py                     # Vectorized parallel RL locomotion environment
│   ├── cli/
│   │   ├── body_cli.py                # `genesis-duck-body` CLI entry point
│   │   ├── infer_cli.py               # `genesis-infer` CLI entry point
│   │   └── teleop_cli.py              # `genesis-teleop` keyboard teleoperation CLI
│   └── infer.py                       # Interactive ONNX policy runner
├── scripts/
│   ├── run_body.py                    # Runner script
│   ├── teleop.py                      # Interactive keyboard teleoperation script
│   └── test_client.py                 # Protocol 1 verification client
└── tests/
    ├── test_constants.py              # Constants and mapping tests
    ├── test_genesis_scene.py          # Genesis scene build and step integration test
    ├── test_math.py                   # Analytical gravity & UYVY math tests
    ├── test_protocol.py               # TCP Protocol 1 dispatch tests
    └── run_tests.py                   # Standalone fast test runner
```

---

## Testing

Run the full pytest suite in the Genesis environment:
```bash
uv run --python 3.12 --with pytest pytest tests/
```

Or run the fast standalone test suite:
```bash
uv run python tests/run_tests.py
```

---

## Related Projects

- [**microduck**](https://github.com/pollen-robotics/microduck): Official daemon, firmware, and control stack (`robotd`, `tofd`, `mediad`, `robotctl`) for the Microduck robot.
- [**microduck_rl**](https://github.com/pollen-robotics/microduck_rl): Reinforcement learning training environments, MuJoCo simulation, and official ONNX policies.
- [**genesis-world**](https://github.com/Genesis-Embodied-AI/genesis-world): Generative world and physics simulation platform for embodied AI and robotics.

