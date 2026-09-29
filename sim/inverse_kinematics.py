"""Damped least-squares differential IK for the Panda's tool centre point."""
from __future__ import annotations

import mujoco
import numpy as np


class DiffIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        site_id: int,
        arm_qpos_idx: np.ndarray,
        arm_dof_idx: np.ndarray,
        joint_range: np.ndarray,
        home_qpos: np.ndarray,
        damping: float = 0.05,
        rot_weight: float = 0.5,
        nullspace_gain: float = 0.05,
        max_dq: float = 0.2,
        iters: int = 8,
    ):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.site_id = site_id
        self.qidx = arm_qpos_idx
        self.vidx = arm_dof_idx
        self.lo, self.hi = joint_range[:, 0], joint_range[:, 1]
        self.home = home_qpos
        self.damping = damping
        self.rot_weight = rot_weight
        self.nullspace_gain = nullspace_gain
        self.max_dq = max_dq
        self.iters = iters
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def solve(self, q_init: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        """Return 7 joint targets that put the TCP at (target_pos, target_quat)."""
        q = np.clip(q_init.copy(), self.lo, self.hi)
        d = self.scratch
        for _ in range(self.iters):
            d.qpos[self.qidx] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)

            pos_err = target_pos - d.site_xpos[self.site_id]
            rot_err = self._rot_error(target_quat)
            err = np.concatenate([pos_err, self.rot_weight * rot_err])

            mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.site_id)
            J = np.vstack([self._jacp[:, self.vidx], self.rot_weight * self._jacr[:, self.vidx]])

            JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)

            # Nullspace pull toward home: 7 joints for a 6-DOF target leaves one free.
            J_pinv = J.T @ np.linalg.inv(JJt)
            N = np.eye(7) - J_pinv @ J
            dq += N @ (self.nullspace_gain * (self.home - q))

            step = np.abs(dq).max()
            if step > self.max_dq:
                dq *= self.max_dq / step
            q = np.clip(q + dq, self.lo, self.hi)
            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-3:
                break
        return q

    def _rot_error(self, target_quat: np.ndarray) -> np.ndarray:
        """World-frame rotation vector from the current TCP orientation to the target."""
        cur = np.empty(4)
        mujoco.mju_mat2Quat(cur, self.scratch.site_xmat[self.site_id])
        local = np.empty(3)
        mujoco.mju_subQuat(local, target_quat, cur)
        world = np.empty(3)
        mujoco.mju_rotVecQuat(world, local, cur)
        return world

    def tcp_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.scratch
        d.qpos[self.qidx] = q
        mujoco.mj_kinematics(self.model, d)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, d.site_xmat[self.site_id])
        return d.site_xpos[self.site_id].copy(), quat
