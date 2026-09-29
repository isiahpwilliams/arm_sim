"""Scripted pick-and-place expert that stacks one cube on another.

Phases: hover_src -> descend_src -> grasp -> lift -> hover_dst -> descend_dst -> release -> retreat.
Reads true cube poses (privileged); emits the same 8-float action the env accepts.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from arm_env import CUBE_HALF, GRIPPER_CLOSED, GRIPPER_OPEN, HOME_QPOS, PandaCubeEnv
from inverse_kinematics import DiffIK

# TCP orientation at home: fingers straight down. Targets are this rotated about z.
DOWN_QUAT = np.array([0.0, 0.7071068, 0.7071068, 0.0])

HOVER_HEIGHT = 0.12
GRASP_Z = CUBE_HALF + 0.005
STACK_Z = 3 * CUBE_HALF + 0.008
MAX_TCP_STEP = 0.015     # m per control step (0.3 m/s at 20 Hz)
SLOW_TCP_STEP = 0.006    # while lifting or lowering a cube
ACCEL_STEP = 0.002
POS_TOL = 0.02
SETTLE_STEPS = 4
GRIPPER_SETTLE_STEPS = 8


@dataclass
class Phase:
    name: str
    target_fn: object          # (obs) -> (xyz, yaw)
    gripper: float
    tol: float = POS_TOL
    hold_steps: int = 0
    speed: float = MAX_TCP_STEP
    timeout: int = 60


def yaw_quat(yaw: float) -> np.ndarray:
    rz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    out = np.empty(4)
    mujoco.mju_mulQuat(out, rz, DOWN_QUAT)
    return out


def cube_yaw(quat: np.ndarray) -> float:
    """Yaw folded into (-45°, 45°], since a cube has 4-fold symmetry."""
    w, x, y, z = quat
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4


class StackingExpert:
    def __init__(self, env: PandaCubeEnv):
        self.env = env
        self.ik = DiffIK(
            env.model, env.tcp_site_id, env.arm_qpos_idx, env.arm_dof_idx,
            env.arm_joint_range, HOME_QPOS)
        self.src = self.dst = None
        self.phases: list[Phase] = []
        self.phase_i = 0
        self.steps_in_phase = 0
        self.tcp_cmd = None
        self.speed = 0.0
        self.arrived_steps = 0
        self.latched: dict = {}

    def reset(self, obs: dict, src: str, dst: str) -> None:
        self.src, self.dst = src, dst
        self.phase_i = 0
        self.steps_in_phase = 0
        self.tcp_cmd = obs["tcp_pos"].copy()
        self.speed = 0.0
        self.arrived_steps = 0
        self.latched = {}

        src_yaw = cube_yaw(self._cube_quat(src))
        dst_yaw = cube_yaw(self._cube_quat(dst))
        src_xy = obs["cubes"][src][:2].copy()

        def dst_xy(o):
            # Latched so a bumped target cube is not chased.
            if "dst_xy" not in self.latched:
                self.latched["dst_xy"] = o["cubes"][dst][:2].copy()
            return self.latched["dst_xy"]

        def place_xy(o):
            # TCP position that centres the *held* cube over the target, using the
            # in-hand offset measured after transit.
            if "place_xy" not in self.latched:
                in_hand = o["tcp_pos"][:2] - o["cubes"][src][:2]
                self.latched["place_xy"] = dst_xy(o) + in_hand
            return self.latched["place_xy"]

        def above_src(o):  return (np.r_[src_xy, CUBE_HALF + HOVER_HEIGHT], src_yaw)
        def at_src(o):     return (np.r_[src_xy, GRASP_Z], src_yaw)
        def lifted(o):     return (np.r_[src_xy, CUBE_HALF + HOVER_HEIGHT + 0.03], src_yaw)
        def above_dst(o):  return (np.r_[dst_xy(o) + (o["tcp_pos"][:2] - o["cubes"][src][:2]), CUBE_HALF + HOVER_HEIGHT + 0.03], dst_yaw)
        def at_dst(o):     return (np.r_[place_xy(o), STACK_Z], dst_yaw)
        def retreat(o):    return (np.r_[place_xy(o), CUBE_HALF + HOVER_HEIGHT], dst_yaw)

        self.phases = [
            Phase("hover_src",   above_src, GRIPPER_OPEN),
            Phase("descend_src", at_src,    GRIPPER_OPEN, speed=SLOW_TCP_STEP),
            Phase("grasp",       at_src,    GRIPPER_CLOSED, hold_steps=GRIPPER_SETTLE_STEPS),
            Phase("lift",        lifted,    GRIPPER_CLOSED, speed=SLOW_TCP_STEP),
            Phase("hover_dst",   above_dst, GRIPPER_CLOSED),
            Phase("descend_dst", at_dst,    GRIPPER_CLOSED, speed=SLOW_TCP_STEP),
            Phase("release",     at_dst,    GRIPPER_OPEN, hold_steps=GRIPPER_SETTLE_STEPS),
            Phase("retreat",     retreat,   GRIPPER_OPEN),
        ]

    @property
    def phase(self) -> str:
        return self.phases[self.phase_i].name if self.phase_i < len(self.phases) else "done"

    @property
    def done(self) -> bool:
        return self.phase_i >= len(self.phases)

    def act(self, obs: dict) -> np.ndarray:
        if self.done:
            return np.concatenate([obs["qpos"], [GRIPPER_OPEN]])

        ph = self.phases[self.phase_i]
        target_pos, yaw = ph.target_fn(obs)

        # Trapezoidal speed profile on the commanded TCP; jerk shakes cubes loose.
        delta = target_pos - self.tcp_cmd
        dist = np.linalg.norm(delta)
        brake = np.sqrt(2.0 * ACCEL_STEP * dist)
        self.speed = min(self.speed + ACCEL_STEP, ph.speed, brake, dist)
        if dist > 1e-9:
            self.tcp_cmd = self.tcp_cmd + delta * (self.speed / dist)
        cmd_arrived = dist <= 1e-4

        q_target = self.ik.solve(obs["qpos"], self.tcp_cmd, yaw_quat(yaw))
        action = np.concatenate([q_target, [ph.gripper]])

        self.steps_in_phase += 1
        self.arrived_steps = self.arrived_steps + 1 if cmd_arrived else 0
        if ph.hold_steps:
            complete = self.steps_in_phase >= ph.hold_steps
        else:
            complete = (self.arrived_steps >= SETTLE_STEPS
                        and np.linalg.norm(obs["tcp_pos"] - target_pos) < ph.tol)
        if complete or self.steps_in_phase >= ph.timeout:
            self.phase_i += 1
            self.steps_in_phase = 0
            self.arrived_steps = 0
            self.speed = 0.0
        return action

    def success(self, obs: dict) -> bool:
        """Source cube centred on top of the target and the gripper clear of it."""
        s, t = obs["cubes"][self.src], obs["cubes"][self.dst]
        xy_ok = np.linalg.norm(s[:2] - t[:2]) < 0.015
        z_ok = abs(s[2] - (t[2] + 2 * CUBE_HALF)) < 0.008
        clear = obs["tcp_pos"][2] > s[2] + 0.05
        return bool(xy_ok and z_ok and clear)

    def _cube_quat(self, name: str) -> np.ndarray:
        i = self.env._cube_qpos_idx[name]
        return self.env.data.qpos[i + 3:i + 7].copy()
