"""Pinhole RGB-D camera on top of MuJoCo's renderer, with back-projection to world points."""

from __future__ import annotations

import mujoco
import numpy as np


class RGBDCamera:
    def __init__(self, model: mujoco.MjModel, name: str = "rgbd", width: int = 640, height: int = 480):
        self.model, self.name = model, name
        self.id = model.camera(name).id
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(model, height, width)
        # MuJoCo's fovy is the full vertical field of view, so focal length in pixels is:
        self.f = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[self.id]) / 2)
        self.cx, self.cy = width / 2, height / 2

    def close(self):
        """Free the GL context. Leaking a few of these makes later renders come back wrong."""
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """RGB (H, W, 3) uint8 and metric depth (H, W) along the optical axis."""
        self.renderer.update_scene(data, camera=self.name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.name)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def backproject(self, data: mujoco.MjData, u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Pixel coords (u right, v down) + depth -> (N, 3) world points."""
        x = (u + 0.5 - self.cx) / self.f * depth
        y = -(v + 0.5 - self.cy) / self.f * depth  # image v grows down, camera y points up
        z = -depth                                  # MuJoCo cameras look down their -z axis
        pts_cam = np.stack([x, y, z], axis=1)
        R = data.cam_xmat[self.id].reshape(3, 3)
        return data.cam_xpos[self.id] + pts_cam @ R.T

    _shared: dict[int, "RGBDCamera"] = {}

    @classmethod
    def shared(cls, model: mujoco.MjModel, name: str = "rgbd") -> "RGBDCamera":
        """One long-lived camera per model, for loops that run hundreds of episodes."""
        key = id(model)
        if key not in cls._shared:
            cls._shared[key] = cls(model, name)
        return cls._shared[key]

    def project(self, data: mujoco.MjData, points: np.ndarray) -> np.ndarray:
        """(N, 3) world points -> (N, 2) pixel coords. Handy for drawing overlays."""
        R = data.cam_xmat[self.id].reshape(3, 3)
        pc = (np.atleast_2d(points) - data.cam_xpos[self.id]) @ R
        d = -pc[:, 2]
        u = pc[:, 0] / d * self.f + self.cx - 0.5
        v = -pc[:, 1] / d * self.f + self.cy - 0.5
        return np.stack([u, v], axis=1)
