"""PandaCubeEnv: MuJoCo workspace with a Franka Panda and three coloured cubes.

Action (8 floats): 7 joint position targets in radians, then a gripper command
in [0, 255] (255 = open). Observation keys: qpos, qvel, gripper (opening in m),
ee_pos/ee_quat (hand body), tcp_pos/tcp_quat (between the fingertips),
cubes {name: xyz}, time.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

SIM_DIR = Path(__file__).resolve().parent
SCENE_XML = SIM_DIR / "scene.xml"

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
CUBE_NAMES = ["cube_red", "cube_blue", "cube_green"]
CUBE_HALF = 0.02

HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0

SPAWN_X = (0.35, 0.65)
SPAWN_Y = (-0.25, 0.25)
MIN_CUBE_SPACING = 0.09


class PandaCubeEnv:
    def __init__(
        self,
        control_hz: float = 20.0,
        render_size: tuple[int, int] = (240, 320),
        seed: int | None = None,
    ):
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.control_hz = control_hz
        self.n_substeps = max(1, int(round(1.0 / (control_hz * self.model.opt.timestep))))
        self.rng = np.random.default_rng(seed)

        h, w = render_size
        self._renderer = mujoco.Renderer(self.model, height=h, width=w)

        self._arm_qpos_idx = np.array(
            [self.model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self._arm_qvel_idx = np.array(
            [self.model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self._finger_qpos_idx = self.model.joint("finger_joint1").qposadr[0]
        self._hand_id = self.model.body("hand").id
        self.tcp_site_id = self.model.site("tcp").id
        self.arm_qpos_idx = self._arm_qpos_idx
        self.arm_dof_idx = self._arm_qvel_idx
        self.arm_joint_range = self.model.jnt_range[
            [self.model.joint(j).id for j in ARM_JOINTS]].copy()
        self._cube_qpos_idx = {
            c: self.model.joint(c).qposadr[0] for c in CUBE_NAMES}
        self._cube_qvel_idx = {
            c: self.model.joint(c).dofadr[0] for c in CUBE_NAMES}
        self._cube_body_id = {c: self.model.body(c).id for c in CUBE_NAMES}

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.home_action = np.concatenate([HOME_QPOS, [GRIPPER_OPEN]])

    @staticmethod
    def _build_model() -> mujoco.MjModel:
        """Load scene.xml and add the wrist camera, TCP site, and a stiffer gripper via MjSpec."""
        spec = mujoco.MjSpec.from_file(str(SCENE_XML))

        hand = spec.body("hand")
        hand.add_camera(
            name="wrist",
            pos=[0.0, -0.08, 0.0],
            xyaxes=[1, 0, 0, 0, -0.95, 0.3],
            fovy=75,
        )
        hand.add_site(name="tcp", pos=[0.0, 0.0, 0.1034], size=[0.005, 0.005, 0.005])

        # The menagerie gripper servo (kp = 100 N/m) pinches with ~2 N and lets
        # cubes creep out during transport. kp = 1000 gives ~20 N, like a real Franka Hand.
        grip = spec.actuator("actuator8")
        kp, kv = 1000.0, 20.0
        grip.gainprm[0] = 0.04 * kp / 255.0
        grip.biasprm[1] = -kp
        grip.biasprm[2] = -kv
        return spec.compile()

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Not using the vendored 'home' keyframe: it predates the cubes' free
        # joints and MuJoCo would zero-pad it, dropping every cube at the origin.
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._arm_qpos_idx] = HOME_QPOS
        self.data.qpos[self._finger_qpos_idx] = 0.04
        self.data.qpos[self._finger_qpos_idx + 1] = 0.04
        self.data.ctrl[:] = self.home_action

        for name, (x, y) in zip(CUBE_NAMES, self._sample_cube_xy(len(CUBE_NAMES))):
            yaw = self.rng.uniform(-np.pi, np.pi)
            quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            i = self._cube_qpos_idx[name]
            self.data.qpos[i:i + 3] = [x, y, CUBE_HALF]
            self.data.qpos[i + 3:i + 7] = quat
            v = self._cube_qvel_idx[name]
            self.data.qvel[v:v + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step(self, action: np.ndarray) -> dict:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (8,):
            raise ValueError(f"action must have shape (8,), got {action.shape}")
        self.data.ctrl[:] = np.clip(action, self.ctrl_low, self.ctrl_high)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    def get_obs(self) -> dict:
        d = self.data
        return {
            "qpos": d.qpos[self._arm_qpos_idx].copy(),
            "qvel": d.qvel[self._arm_qvel_idx].copy(),
            "gripper": float(d.qpos[self._finger_qpos_idx]) * 2.0,
            "ee_pos": d.xpos[self._hand_id].copy(),
            "ee_quat": d.xquat[self._hand_id].copy(),
            "tcp_pos": d.site_xpos[self.tcp_site_id].copy(),
            "tcp_quat": self._site_quat(self.tcp_site_id),
            "cubes": {c: d.xpos[self._cube_body_id[c]].copy() for c in CUBE_NAMES},
            "time": float(d.time),
        }

    def render(self, camera: str = "front") -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    @property
    def cameras(self) -> list[str]:
        return [self.model.camera(i).name for i in range(self.model.ncam)]

    def _site_quat(self, site_id: int) -> np.ndarray:
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[site_id])
        return q

    def _sample_cube_xy(self, n: int) -> list[tuple[float, float]]:
        placed: list[tuple[float, float]] = []
        for _ in range(1000):
            if len(placed) == n:
                break
            x = self.rng.uniform(*SPAWN_X)
            y = self.rng.uniform(*SPAWN_Y)
            if all(np.hypot(x - px, y - py) >= MIN_CUBE_SPACING for px, py in placed):
                placed.append((x, y))
        if len(placed) < n:
            raise RuntimeError("could not place cubes without overlap; loosen SPAWN_* or spacing")
        return placed


if __name__ == "__main__":
    env = PandaCubeEnv()
    obs = env.reset(seed=0)
    print("cameras:", env.cameras)
    print("substeps per control step:", env.n_substeps)
    print("home hand position:", obs["ee_pos"].round(3), "tcp:", obs["tcp_pos"].round(3))
    for c, p in obs["cubes"].items():
        print(f"  {c}: {p.round(3)}")
