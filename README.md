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
pixi run -e train to-lerobot -- --raw data/raw --out data/lerobot/panda_stack_cond --task-cond   # + (src,dst) one-hot in the state
pixi run -e train eval -- outputs/train/act/checkpoints/last/pretrained_model --seeds 20 --video outputs/eval.mp4
pixi run -e train eval -- <ckpt> --pair cube_green,cube_blue --temporal-ensemble 0.01   # single-task ACT
pixi run -e train check-predictions -- <ckpt> --episodes 2 9 19    # offline action error on training frames
```

Training uses LeRobot's own CLI, `lerobot-train`, inside the train env. The exact
invocations that work are in `cloud/train_act.sh` and `cloud/train_smolvla.sh`; on the
Mac swap `--policy.device=cuda` for `mps` and use a small batch.

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
  eval_policy.py        closed-loop evaluation of a LeRobot checkpoint in the sim (train env)
  check_predictions.py  offline predicted-vs-recorded action error for a checkpoint (train env)
cloud/                  scripts for a rented Linux GPU box: setup, train ACT / SmolVLA, eval
data/raw/               recorded demonstrations (gitignored)
data/lerobot/           converted LeRobot datasets (gitignored)
outputs/train/          checkpoints (gitignored)
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
4. **Train** — done on a rented A10. Single-task ACT stacks cubes closed-loop
   (24%). Multi-task does not: SmolVLA reads neither its cameras nor its
   instruction (0-8%), and task-conditioned ACT reads its cameras but ignores the
   task label (2%). Diagnosis in "Training status" below.
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

## Training status

Throughput on one A10 (24 GB), batch 32: ACT 6.4 step/s, so 40k steps is 1h45m.
SmolVLA 1.0-1.5 step/s, so 20k steps is 3h40m. ACT is 52M parameters, all
trainable; SmolVLA fine-tunes 100M of 450M on the frozen recipe. On the Mac
(Metal) both run but ~20x slower: ACT ~3 step/s at batch 8, SmolVLA ~2.6 s/step
at batch 2.

### What works, and what does not

| | SmolVLA multi-task | SmolVLA single-pair | ACT single-pair |
|---|---|---|---|
| dataset | 198 eps / 34k frames, 6 pairs | 153 eps / 30k frames, 1 pair | 153 eps / 30k frames, 1 pair |
| steps / epochs | 20k / 18.7 | 20k / 21.0 | 40k / 42.1 |
| final loss | 0.012 | 0.021 | 0.044 (L1) |
| scene-swap shift | 0.78° | 0.95° | **8.05°** |
| instruction-swap shift | 0.29° | 0.54° | n/a (no text input) |
| grounding slope x / y | 0.13 / 0.53 | 0.69 / 0.85 | **1.03 / 1.11** |
| mean aim error | 13.3 cm | 5.6 cm | **2.4 cm** |
| correct cube at close | 8/16 | 15/16 | **16/16** |
| grasped and lifted | 1/50 | 11/50 | **29/50** |
| **success** | **0/50** | **4/50** | **12/50 (24%)** |

Success counts at 50 seeds carry a ~6 point standard error, a 95% interval about
25 points wide, so read that last column as **30% (60/200) at its best checkpoint,
30k steps**, falling to 22% (44/200) by 40k while the training loss keeps falling.
`sim/diagnose/README.md` has the arithmetic; use n=200 for any comparison you mean
to act on, and n=50 only as a smoke test.

Single-task ACT stacks cubes. That is the positive control the whole
investigation needed: the sim, the expert data, absolute joint-target actions at
20 Hz, 50-step chunks and the two cameras all support closed-loop stacking, so
anything that fails here fails on its own merits.

SmolVLA reads neither of the inputs it is supposed to read. Swapping in a
different scene's images moves its predicted action by 0.78°, against ACT's
8.05° and ~12° of action standard deviation. Changing the instruction to name a
different source cube moves it 0.29° — *less* than a harmless paraphrase at
0.42° — and only a blank string moves anything at 1.15°, which is a reaction to
token count, not to meaning. Its aim collapses to the centroid of the spawn
region: 13.7 cm from the TCP to the spawn centre against 13.3 cm to the cube it
was told to pick.

Both failures are real and they compound. Remove the ambiguity by training on
one pair and grounding partly recovers (13.3 cm → 5.6 cm aim, 8/16 → 15/16
correct cube, 0/50 → 4/50), so the broken language binding was doing real damage
on the multi-task set. But a vision pathway still ~8x less sensitive than ACT's
caps the single-task case at 4/50 against ACT's 12/50 on the same 153 episodes.

**Caveat on the multi-task column.** That run used 198 episodes across 6 pairs,
roughly 33 per pair, where the single-pair runs had 153 for one pair. So the
multi-task → single-pair improvement is confounded with ~4.6x more data per
behaviour and the two effects cannot be separated from these two runs alone. The
evidence that the language binding is broken does not rest on that comparison:
it comes from the instruction ablation on each checkpoint directly, where naming
a different cube barely moves the action.

### Ruled out

Undertraining (20k steps, 18-21 epochs, loss flat), the frozen vision encoder
(unfreezing bought a 31% sensitivity gain that training then spent), DART action
noise, the camera setup and image pipeline (dataset frames match freshly
rendered ones), and chunk length — re-planning more often made ACT monotonically
worse, 24% → 16% → 8% as `--temporal-ensemble` shortened the executed chunk, so
long open-loop chunks are load-bearing here rather than a liability. The
proprioceptive-shortcut theory is weakened too: ACT uses the same absolute joint
targets and the same qpos state and still reaches a grounding slope of 1.03.

`sim/diagnose/cube_probe.py` settles whether the task is possible at all. A 6M
from-scratch conv net reading only the front camera regresses cube xy to 3.5 cm
validation error against a 15.4 cm predict-the-mean baseline, depth and lateral
equally accurate, after 4 minutes of GPU. The pixels carry the cube positions;
SmolVLA's 13-16 cm is a learning failure, not an information ceiling.

### Do not trust the loss on this task

It ordered the models backwards. Multi-task SmolVLA reached loss 0.012 and
scored 0/50; the single-pair run reached 0.021 and scored 4/50. Same
architecture, same objective, directly comparable, and the better loss was the
useless policy. Watch closed-loop success and the input-sensitivity measures in
`sim/diagnose/` instead.

### Conditioning ACT on the task: tried, does not work

ACT takes no text input whatsoever — there is no tokenizer anywhere in
`modeling_act.py` — so on the full 823-episode set its six pairs are six
behaviours for identical pixels. `to_lerobot.py --task-cond` appends a 6-dim
one-hot of the (source, destination) pair to `observation.state`, which should
disambiguate them without a text encoder. 823 episodes is ~137 per pair, close
to the 153 the working single-task run had, so 24% was the number to beat.

It reaches 1/50. The policy ignores the one-hot almost exactly as SmolVLA
ignores language:

| at 40k steps / 7.9 epochs | |
|---|---|
| scene-swap shift | **8.71°** (single-pair ACT: 8.05°) |
| changing the source cube in the label | **0.21°** mid-episode, 0.32° at episode start |
| label fed **swapped** | correct cube 7/16 vs 6/16 normal, gain 0.39 vs 0.42 |
| correct cube at grasp | 6/16 — chance is 5.33/16 |
| success | 1/50 |

Vision is not the problem: scene-swap sensitivity matches the working policy and
*rises* with training (8.10° at 20k → 8.71° at 40k), and with the true images it
predicts the expert's action to 0.86°. Nor is the mapping learned backwards —
feeding `(dst, src)` instead of `(src, dst)` changes nothing closed-loop, which
is what `grounding_test.py --swap-cond` exists to check. The label is simply not
read, and more steps do not fix it: 20k → 40k moved correct-cube 5/16 → 6/16
while the loss fell 0.098 → 0.069.

The reason is the same shortcut that defeats SmolVLA, acting on the conditioning
channel instead of on vision. The label only earns its keep on the minority of
frames where proprioception does not already determine the action: by
mid-trajectory `qpos` encodes which cube the arm is heading to, so the one-hot is
redundant there and load-bearing only in each episode's opening steps. Its
gradient contribution is swamped, and ignoring it is the lower-loss solution.
`cond_ablation.py --first 12` measures exactly this, restricting to episode
openings — the label's influence roughly doubles there and is still ~11% of the
images'.

So six dims concatenated into a 14-dim state vector is too weak a channel.
Making conditioning structurally unavoidable (FiLM, cross-attention, or a
separate conditioning token) is the next thing to try; the cheap alternative is
to sidestep conditioning entirely, since single-pair ACT already stacks at 24%.

## Training on a rented GPU

The `train` environment is locked for `linux-64` as well as `osx-arm64`, and on Linux the
PyPI torch wheel is the CUDA build. On a fresh Ubuntu box with an NVIDIA driver:

```bash
git clone <this repo> && cd arm_sim
cloud/setup_gpu_box.sh                 # installs pixi + the train env, prints the GPU
# copy data/lerobot/panda_stack over (rsync/scp), then:
cloud/train_act.sh                     # ~52M params, batch 32, 40k steps (1h45m on an A10)
cloud/train_smolvla.sh                 # fine-tune lerobot/smolvla_base, batch 32, 20k steps (3h40m)
cloud/eval.sh outputs/train/act_gb/checkpoints/040000/pretrained_model 50 --pair cube_green,cube_blue
```

ACT has no language input, so give it one pair (`--dataset.episodes`) or a
task-conditioned dataset (`--task-cond`); multi-task ACT on raw data averages
the six behaviours. Evaluate single-task checkpoints with `--pair`, or the score
counts scenes the policy was never taught.

Sizing: ACT fits any 16 GB card. SmolVLA at batch 32 wants 24 GB (L4, A10, RTX 4090);
drop to 16 on smaller cards. Evaluation renders MuJoCo headless via EGL.

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
