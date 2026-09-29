# arm_sim

Teach a simulated Franka Panda to follow natural-language commands
("stack the blue cube on the red cube") using a vision-language-action
policy, with ROS 2 + C++ for the control plumbing and MuJoCo for physics.

Learning goals: C++, ROS 2, PyTorch, robotics simulation.

## Stack

| Piece | Choice | Why |
|---|---|---|
| Physics | MuJoCo | Fast, runs natively on Apple Silicon, has a C API for C++ work |
| Robot | Franka Panda | In `mujoco_menagerie` and Isaac Lab; standard in VLA papers |
| ROS | ROS 2 Jazzy via RoboStack + pixi | Only sane way to get ROS 2 on macOS; Jazzy is the mature LTS |
| Policy | LeRobot: ACT first, then SmolVLA; OpenVLA LoRA as a stretch | SmolVLA fine-tunes in hours on one rented GPU |
| Heavy compute | Rented Linux GPU box | SmolVLA/OpenVLA training; optional Isaac Sim port later |

## Setup (macOS, Apple Silicon)

```bash
brew install pixi
pixi install          # creates .pixi/envs/default with ROS 2 Jazzy, clang, cmake, ninja, MuJoCo, PyTorch
```

There are two pixi environments in one manifest:

| Env | Contents | Use |
|---|---|---|
| `default` | ROS 2 Jazzy, C++ toolchain, MuJoCo, conda PyTorch | `pixi run <task>` / `pixi shell` |
| `train` | LeRobot (+ `dataset`, `smolvla` extras), PyPI PyTorch, MuJoCo, no ROS | `pixi run -e train <task>` / `pixi shell -e train` |

They are separate because LeRobot pins torch/numpy versions that conflict with the
conda-forge builds ROS is linked against. `pixi install -e train` sets up the second one.

## Tasks

```bash
pixi run build        # colcon build of ros2_ws (Ninja, symlink install)
pixi run fake-arm     # run the C++ joint-state publisher node
pixi run echo-joints  # in a second terminal: watch /joint_states
pixi run hz-joints    # measure publish rate
pixi run clean        # wipe build/ install/ log/

pixi run env-check    # load the MuJoCo scene, print cameras and cube positions
pixi run record-demo  # wave the arm over random cubes, write outputs/demo.mp4 (3 cameras tiled)

pixi run expert -- --seeds 20 --video outputs/expert.mp4   # scripted stacking expert, success rate
pixi run collect -- --episodes 200 --out data/raw           # record expert demos (front + wrist cams)
pixi run replay -- data/raw/episode_000000                  # re-run an episode's actions, check it reproduces

pixi run -e train to-lerobot -- --raw data/raw --out data/lerobot/panda_stack   # LeRobot v3 dataset
```

## Layout

```
pixi.toml               environment + tasks (channels: robostack-jazzy, conda-forge)
ros2_ws/src/
  arm_sim_cpp/          C++ ROS 2 nodes (sim bridge, controllers)
    src/fake_arm_node.cpp   first node: sine-wave JointState publisher for the Panda
sim/
  scene.xml             MuJoCo workspace: Panda + table + 3 cubes + front/overhead cameras
  arm_env.py            PandaCubeEnv: reset() / step(action) / render(camera) / get_obs()
  record_demo.py        Phase 2 smoke test, records all cameras to MP4
  inverse_kinematics.py DiffIK: damped least-squares inverse kinematics on the TCP site
  expert.py             StackingExpert: hover/descend/grasp/lift/place state machine over DiffIK
  tasks.py              task sampling + language instruction templates
  run_expert.py         run the expert on N seeds, report success, optional video
  collect_episodes.py   record demos to data/raw/episode_XXXXXX (meta.json, steps.npz, <cam>.mp4)
  replay_episode.py     open-loop replay of a recorded episode; asserts reproducibility
  to_lerobot.py         convert data/raw -> LeRobot v3 dataset (train env)
data/raw/               recorded demonstrations (gitignored)
data/lerobot/           converted LeRobot datasets (gitignored)
  assets/franka_emika_panda/   vendored from mujoco_menagerie (Apache-2.0), see SOURCE.txt
outputs/                videos and frames (gitignored)
```

## Sim conventions

- Panda base at the origin, +x into the workspace, +z up. Cubes spawn in
  x ∈ [0.35, 0.65], y ∈ [-0.25, 0.25], at least 9 cm apart, random yaw.
- Physics at 500 Hz (timestep 0.002 s), control at 20 Hz (25 substeps per action).
- **Action** = 8 floats: 7 joint position targets in radians, then gripper 0..255 (255 = open).
- **Observation** = dict with `qpos`, `qvel` (7 each), `gripper` opening in metres,
  `ee_pos`/`ee_quat` of the hand body, `cubes` {name: xyz}, `time`.
- Cameras: `front`, `overhead`, `wrist` (attached to the hand body via MjSpec in Python
  so the vendored model stays untouched). Default render 240×320.
- A `tcp` site sits between the fingertip pads (0.1034 m below the hand flange). IK targets it.
- The gripper servo is stiffened via MjSpec (kp 100 → 1000 N/m, ~20 N pinch); the menagerie
  default let cubes creep out of the grasp during transport.

## Expert and dataset

The scripted expert reads true cube poses (privileged) and walks a fixed sequence:
hover over source → descend → grasp → lift → hover over target → descend → release → retreat.
The commanded TCP follows a trapezoidal speed profile (0.3 m/s transit, 0.12 m/s when
lifting or lowering a cube) and DiffIK turns each commanded pose into joint targets.
Target position is latched before descending so bumping the target cube can't cause a
chase, and the in-hand offset of the held cube is measured so it lands centred.
Success = source cube centred within 1.5 cm on top of the target and the gripper clear.

Episode layout (framework-neutral; `to_lerobot.py` converts it):

```
data/raw/index.jsonl                 one JSON line per episode
data/raw/episode_000000/meta.json    instruction, src/dst cube, seed, success, fps, cameras
data/raw/episode_000000/steps.npz    action (T,8), qpos/qvel (T,7), gripper, tcp_pos/quat, cube_* (T,3), phase, time
data/raw/episode_000000/front.mp4    one video per camera, T frames at 20 fps
data/raw/episode_000000/wrist.mp4
```

The LeRobot export keeps only what a policy may see: `observation.images.front`,
`observation.images.wrist`, `observation.state` (7 joint angles + gripper opening),
`action` (7 joint targets + gripper command), and `task` (the instruction).
Seeds 0–199 are reserved for evaluation; datasets are recorded from seed 10000 up.
When loading on the Mac, pass `video_backend="pyav"` to `LeRobotDataset` to skip the
noisy torchcodec probe (torchcodec needs system ffmpeg libs; conda ffmpeg conflicts with
PyAV's bundled copy). On the Linux GPU box torchcodec can be set up properly.

## Roadmap

1. **ROS 2 + C++ toolchain** — done. First node builds and publishes.
2. **MuJoCo workspace** — done. Panda + table + red/blue/green cubes, three cameras, offscreen MP4.
3. **Scripted expert + dataset** — done. DiffIK pick-and-place solving "stack X on Y" with
   randomised colours/positions/yaw; episodes recorded with instruction, actions, state, and video.
4. **Train** — ACT (no language) to validate the pipeline, then SmolVLA on a rented GPU.
   Evaluate closed-loop success rate in sim, record rollout videos.
5. **ROS 2 bridge in C++** — MuJoCo sim as rclcpp nodes: camera + joint-state publishers,
   joint-command subscriber; policy runs as a Python node.
6. **Optional: Isaac Lab port** on a rented GPU for a sim-to-sim transfer demo.

## VS Code

`.vscode/settings.json` points CMake Tools at the pixi environment's cmake, clang, and
ROS install, and gives it its own build dir (`ros2_ws/build-ide/`) so it never fights
with colcon, which owns `ros2_ws/build/`. IntelliSense reads the compile database from
there. The Python interpreter is set to the pixi one so `mujoco`/`torch` resolve.
If CMake Tools ever says it cannot find `ament_cmake`, it is running outside that
environment: reload the window, or pick the kit/cmake path it lists above.

## Gotchas

### macOS: C++ node aborts with `symbol not found in flat namespace '_PyExc_RuntimeError'`

RoboStack's Jazzy interface packages export their Python-bindings library
(`lib<pkg>__rosidl_generator_py`) in `<pkg>_TARGETS` and in rclcpp's transitive
deps. Linking it into a pure C++ executable makes dyld abort at load because no
Python interpreter is present. RoboStack's own binaries avoid it because
conda-build links with `-dead_strip_dylibs`. Two things fix it, both in
`ros2_ws/src/arm_sim_cpp/CMakeLists.txt`:

- `add_link_options("LINKER:-dead_strip_dylibs")` on Apple, and
- link specific targets (`rclcpp::rclcpp`, `sensor_msgs::sensor_msgs__rosidl_typesupport_cpp`)
  instead of `ament_target_dependencies()` or `${sensor_msgs_TARGETS}`.

Upstream fix so far only in Rolling: https://github.com/RoboStack/ros-rolling/pull/51

### MuJoCo: keyframes and MjSpec

- `panda.xml` ships a `home` keyframe sized for the bare arm. Once cubes with free joints
  are added, MuJoCo silently zero-pads it, which would drop every cube at the origin.
  `PandaCubeEnv.reset()` sets the home pose from Python instead.
- `<compiler meshdir>` in an included file resolves relative to the *including* file, so
  `scene.xml` overrides it after the include.
- To edit a vendored model without touching its files, load it as `mujoco.MjSpec`, modify
  (e.g. `spec.body("hand").add_camera(...)`), then `spec.compile()`.
