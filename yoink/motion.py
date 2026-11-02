"""Motion layer: min-jerk trajectories, IK every control tick, position motors + gravity compensation."""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np
from armageddon import Chain, forward, solve
from armageddon.kinematics import rotation, rotvec_from_matrix
from armageddon.mjcf import FINGER_TRAVEL

CONTROL_HZ = 60
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0  # commanding past the cube width is what produces the squeeze


def min_jerk(s: float) -> float:
    """Smooth 0->1 profile with zero velocity and acceleration at both ends."""
    s = min(max(s, 0.0), 1.0)
    return s**3 * (10 - 15 * s + 6 * s**2)


def gripper_down(yaw: float) -> np.ndarray:
    """Tool pointing straight down, fingers closing along the horizontal direction yaw + 90 deg."""
    return rotation(np.array([0.0, 0.0, 1.0]), yaw) @ rotation(np.array([0.0, 1.0, 0.0]), np.pi)


class Controller:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, chain: Chain,
                 on_tick: Callable[[], None] | None = None):
        self.model, self.data, self.chain = model, data, chain
        self.n = chain.dof
        self.q_des = data.qpos[: self.n].copy()
        self.grip = GRIPPER_OPEN
        self.substeps = max(1, round(1 / (CONTROL_HZ * model.opt.timestep)))
        self.on_tick = on_tick
        self.ik_failures = 0

    # ----------------------------------------------------------- low level --

    def tick(self):
        self.data.ctrl[: self.n] = self.q_des
        self.data.ctrl[self.n:self.n + 2] = min(max(self.grip, 0.0), FINGER_TRAVEL)
        for _ in range(self.substeps):
            # Feed forward gravity/coriolis for the arm joints only (not the fingers or the cubes).
            self.data.qfrc_applied[: self.n] = self.data.qfrc_bias[: self.n]
            mujoco.mj_step(self.model, self.data)
        if self.on_tick:
            self.on_tick()

    def ticks(self, seconds: float) -> int:
        return max(1, round(seconds * CONTROL_HZ))

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pose = forward(self.chain, self.q_des)
        return pose.pos, pose.rot

    # ---------------------------------------------------------- primitives --

    def wait(self, seconds: float):
        for _ in range(self.ticks(seconds)):
            self.tick()

    def set_gripper(self, width: float, seconds: float = 0.4):
        self.grip = width
        self.wait(seconds)

    def move_joints(self, q_goal: np.ndarray, seconds: float):
        q0 = self.q_des.copy()
        n = self.ticks(seconds)
        for k in range(1, n + 1):
            self.q_des = q0 + (q_goal - q0) * min_jerk(k / n)
            self.tick()

    def move_linear(self, pos: np.ndarray, rot: np.ndarray, seconds: float):
        """Straight-line Cartesian move (slerped orientation), solved with warm-started IK each tick."""
        p0, R0 = self.ee_pose()
        w = rotvec_from_matrix(rot @ R0.T)
        angle = np.linalg.norm(w)
        axis = w / angle if angle > 1e-9 else np.array([0.0, 0.0, 1.0])
        n = self.ticks(seconds)
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            res = solve(self.chain, p0 + (pos - p0) * s, rotation(axis, angle * s) @ R0,
                        q0=self.q_des, max_iters=20)
            if not res.success:
                self.ik_failures += 1
            self.q_des = res.q
            self.tick()

    def plan_to(self, pos: np.ndarray, rot: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
        """IK for a far-away goal (used with move_joints). Prefers solutions near `seed`."""
        res = solve(self.chain, pos, rot, q0=seed, restarts=10, q_rest=seed, nullspace_gain=0.05,
                    rng=np.random.default_rng(0))
        return res.q if res.success else None
