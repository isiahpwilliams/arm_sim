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
pixi install          # creates .pixi/envs/default with ROS 2 Jazzy + clang + cmake + ninja
```

Everything runs inside the pixi environment, either `pixi run <task>` or
`pixi shell` for an interactive shell.

## Tasks

```bash
pixi run build        # colcon build of ros2_ws (Ninja, symlink install)
pixi run fake-arm     # run the C++ joint-state publisher node
pixi run echo-joints  # in a second terminal: watch /joint_states
pixi run hz-joints    # measure publish rate
pixi run clean        # wipe build/ install/ log/
```

## Layout

```
pixi.toml               environment + tasks (channels: robostack-jazzy, conda-forge)
ros2_ws/src/
  arm_sim_cpp/          C++ ROS 2 nodes (sim bridge, controllers)
    src/fake_arm_node.cpp   first node: sine-wave JointState publisher for the Panda
```

## Roadmap

1. **ROS 2 + C++ toolchain** — done. First node builds and publishes.
2. **MuJoCo workspace** — Panda + table + red/blue cubes, wrist + overhead cameras, offscreen video.
3. **Scripted expert + dataset** — IK pick-and-place solving "stack X on Y" with randomised
   colours/positions; record thousands of episodes to a LeRobot dataset with the instruction attached.
4. **Train** — ACT (no language) to validate the pipeline, then SmolVLA on a rented GPU.
   Evaluate closed-loop success rate in sim, record rollout videos.
5. **ROS 2 bridge in C++** — MuJoCo sim as rclcpp nodes: camera + joint-state publishers,
   joint-command subscriber; policy runs as a Python node.
6. **Optional: Isaac Lab port** on a rented GPU for a sim-to-sim transfer demo.

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
