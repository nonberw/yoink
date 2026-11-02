"""The sorting loop: look, pick the closest cube, carry it to the bin of its color, repeat."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from armageddon import HOME, default_arm

from .camera import RGBDCamera
from .motion import GRIPPER_CLOSED, GRIPPER_OPEN, Controller, gripper_down
from .perception import Detection, detect_cubes
from .scene import BINS, CUBE_HALF, Scene, bin_of, make_scene

# Arm leaned back and up, out of the camera's way, while it looks at the table.
OBSERVE = np.array([0.0, -0.35, 0.6, 0.0, 1.2, 0.0])
APPROACH_HEIGHT = 0.12
CARRY_HEIGHT = 0.17
GRASP_DEPTH = 0.004  # go a hair below the cube center so the pads cover more of the faces
DROP_SLOTS = 0.033 * np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])  # bin corners, in fill order


@dataclass
class EpisodeResult:
    colors: list[str]
    placed: list[str | None]          # bin each cube ended up in
    attempts: int
    sim_seconds: float
    ik_failures: int
    detections: list[list[Detection]] = field(default_factory=list)

    @property
    def correct(self) -> int:
        return sum(p == c for p, c in zip(self.placed, self.colors))


def grasp_yaw(det: Detection) -> float:
    """Cube yaw is only known mod 90 deg; pick the equivalent closest to the direction from the base,
    which keeps the wrist roll small and the forearm out of the neighbours' way."""
    natural = np.arctan2(det.pos[1], det.pos[0])
    options = det.yaw + np.pi / 2 * np.arange(-3, 4)
    return float(options[np.argmin(np.abs(options - natural))])


def seed_for(xy: np.ndarray) -> np.ndarray:
    q = HOME.copy()
    q[0] = np.arctan2(xy[1], xy[0])
    return q


class Sorter:
    def __init__(self, scene: Scene, camera: RGBDCamera, on_tick: Callable[[], None] | None = None,
                 on_detect: Callable[[np.ndarray, list[Detection]], None] | None = None):
        self.scene, self.camera = scene, camera
        self.ctrl = Controller(scene.model, scene.data, default_arm(), on_tick)
        self.on_detect = on_detect
        self.dropped = {color: 0 for color in BINS}

    def look(self) -> list[Detection]:
        self.ctrl.move_joints(OBSERVE, 1.0)
        self.ctrl.wait(0.2)  # let things settle before the shutter
        rgb, depth = self.camera.capture(self.scene.data)
        dets = detect_cubes(self.camera, self.scene.data, rgb, depth)
        if self.on_detect:
            self.on_detect(rgb, dets)
        return dets

    def pick_and_place(self, det: Detection) -> bool:
        c = self.ctrl
        yaw = grasp_yaw(det)
        R = gripper_down(yaw)
        above = det.pos + [0, 0, APPROACH_HEIGHT]
        q = c.plan_to(above, R, seed_for(det.pos))
        if q is None:
            return False
        c.grip = GRIPPER_OPEN
        c.move_joints(q, 1.2)
        c.move_linear(det.pos - [0, 0, GRASP_DEPTH], R, 0.8)
        c.set_gripper(GRIPPER_CLOSED, 0.5)
        c.move_linear(det.pos + [0, 0, CARRY_HEIGHT], R, 0.8)

        # Fill the bin corner by corner: dropping everything on one spot builds a pile that the
        # next drop shoves over the wall.
        k = self.dropped[det.color]
        self.dropped[det.color] += 1
        bx, by = np.array(BINS[det.color]) + DROP_SLOTS[k % len(DROP_SLOTS)]
        drop = np.array([bx, by, CARRY_HEIGHT])
        R_drop = gripper_down(np.arctan2(by, bx))
        q = c.plan_to(drop, R_drop, seed_for(drop))
        if q is None:
            c.set_gripper(GRIPPER_OPEN)
            return False
        c.move_joints(q, 1.4)
        c.move_linear(drop - [0, 0, 0.06], R_drop, 0.5)
        c.set_gripper(GRIPPER_OPEN, 0.4)
        c.move_linear(drop, R_drop, 0.4)
        return True

    def run(self, max_attempts: int | None = None) -> EpisodeResult:
        n = len(self.scene.colors)
        max_attempts = max_attempts or 2 * n
        history, attempts = [], 0
        while attempts < max_attempts:
            dets = self.look()
            history.append(dets)
            if not dets:
                break
            # Closest first: clears the near side so far cubes never sit behind the forearm.
            det = min(dets, key=lambda d: np.hypot(d.pos[0], d.pos[1]))
            attempts += 1
            self.pick_and_place(det)
        self.ctrl.move_joints(OBSERVE, 1.0)
        placed = [bin_of(self.scene.cube_pose(i)[0]) for i in range(n)]
        return EpisodeResult(self.scene.colors, placed, attempts, float(self.scene.data.time),
                             self.ctrl.ik_failures, history)


def run_episode(n_cubes: int = 4, seed: int | None = None, **kw) -> EpisodeResult:
    scene = make_scene(n_cubes, seed)
    scene.data.qpos[:6] = OBSERVE
    scene.data.ctrl[:6] = OBSERVE
    return Sorter(scene, RGBDCamera.shared(scene.model), **kw).run()


__all__ = ["EpisodeResult", "Sorter", "run_episode", "grasp_yaw", "OBSERVE", "CUBE_HALF"]
