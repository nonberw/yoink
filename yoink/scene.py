"""The sorting scene: ARMageddon arm with a parallel gripper, colored cubes, bins, an RGB-D camera."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from armageddon import default_arm
from armageddon.mjcf import ARM_MATERIALS, arm_actuators, arm_xml

CUBE_HALF = 0.02
COLORS = {  # name -> rgb
    "red": (0.88, 0.12, 0.12),
    "yellow": (0.95, 0.78, 0.08),
    "green": (0.12, 0.72, 0.22),
}
BINS = {  # name -> bin center (x, y); inner size BIN_HALF*2 square
    "red": (0.04, 0.40),
    "yellow": (-0.26, 0.30),
    "green": (0.04, -0.40),
}
BIN_HALF = 0.075
SPAWN_LOW = np.array([0.30, -0.17])
SPAWN_HIGH = np.array([0.50, 0.17])
CAMERA_POS = np.array([0.95, 0.0, 0.72])
CAMERA_TARGET = np.array([0.40, 0.0, 0.0])


def _v(a) -> str:
    return " ".join(f"{x:.6g}" for x in np.asarray(a, dtype=float).ravel())


def _camera_xyaxes(pos, target) -> np.ndarray:
    fwd = (target - pos) / np.linalg.norm(target - pos)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.concatenate([right, up])


def _bin_xml(name, center, rgb) -> str:
    x, y = center
    h, t, w = 0.035, 0.006, BIN_HALF + 0.006
    walls = [((w - t / 2, 0), (t / 2, w, h)), ((-w + t / 2, 0), (t / 2, w, h)),
             ((0, w - t / 2), (w, t / 2, h)), ((0, -w + t / 2), (w, t / 2, h))]
    dark = tuple(c * 0.55 for c in rgb)
    geoms = "\n".join(
        f'      <geom type="box" pos="{dx:.4g} {dy:.4g} {h:.4g}" size="{_v(s)}" rgba="{_v(dark)} 1" contype="1" conaffinity="1"/>'
        for (dx, dy), s in walls
    )
    return f"""    <body name="bin_{name}" pos="{x} {y} 0">
      <geom type="box" pos="0 0 0.002" size="{w:.4g} {w:.4g} 0.002" rgba="{_v(rgb)} 0.35" contype="0" conaffinity="0"/>
{geoms}
    </body>"""


def build_scene_xml(colors: list[str]) -> str:
    chain = default_arm()
    cubes = "\n".join(
        f"""    <body name="cube{i}" pos="{0.3 + 0.06 * i} 0 {CUBE_HALF}">
      <freejoint/>
      <geom name="cube{i}" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" rgba="{_v(COLORS[c])} 1" mass="0.05"
            contype="1" conaffinity="1" condim="4" friction="1.2 0.02 0.002" solref="0.005 1"/>
    </body>"""
        for i, c in enumerate(colors)
    )
    bins = "\n".join(_bin_xml(n, c, COLORS[n]) for n, c in BINS.items())
    return f"""<mujoco model="yoink">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.35 0.35 0.4" diffuse="0.5 0.5 0.55" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.16 0.09 0.32" rgb2="0.01 0.01 0.04" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.13 0.12 0.2" rgb2="0.08 0.08 0.13" width="512" height="512" mark="edge" markrgb="0.45 0.3 0.9"/>
    <material name="floor" texture="grid" texrepeat="10 10" reflectance="0.1"/>
{ARM_MATERIALS}
  </asset>

  <default>
    <joint damping="3" armature="0.01"/>
    <geom contype="1" conaffinity="0"/>
  </default>

  <worldbody>
    <light pos="0.6 -0.6 2.2" dir="-0.3 0.3 -1" diffuse="0.9 0.85 1" castshadow="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor" contype="0" conaffinity="1" condim="3" friction="1 0.02 0.002"/>
    <geom name="mat" type="box" pos="0.40 0 0.0005" size="0.15 0.22 0.0005" rgba="0.32 0.31 0.36 1" contype="0" conaffinity="0"/>
    <camera name="rgbd" pos="{_v(CAMERA_POS)}" xyaxes="{_v(_camera_xyaxes(CAMERA_POS, CAMERA_TARGET))}" fovy="50"/>
{arm_xml(chain, gripper="parallel")}
{bins}
{cubes}
  </worldbody>

  <actuator>
{arm_actuators(chain, gripper="parallel")}
  </actuator>
</mujoco>
"""


@dataclass
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    colors: list[str]

    def cube_pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        body = self.data.body(f"cube{i}")
        return body.xpos.copy(), body.xmat.reshape(3, 3).copy()


def random_layout(n: int, rng: np.random.Generator, min_gap: float = 0.075) -> list[tuple[float, float, float]]:
    """Non-overlapping (x, y, yaw) placements inside the spawn area."""
    out: list[tuple[float, float, float]] = []
    while len(out) < n:
        xy = rng.uniform(SPAWN_LOW, SPAWN_HIGH)
        if all(np.hypot(xy[0] - x, xy[1] - y) > min_gap for x, y, _ in out):
            out.append((float(xy[0]), float(xy[1]), float(rng.uniform(0, np.pi / 2))))
    return out


_MODELS: dict[int, mujoco.MjModel] = {}


def scene_model(n_cubes: int) -> mujoco.MjModel:
    """One compiled model per cube count, reused across episodes; colors are set per episode.
    Reuse matters: every new model needs a new GL context to render, and Windows runs out of
    those after ~150 episodes even when each renderer is closed."""
    if n_cubes not in _MODELS:
        _MODELS[n_cubes] = mujoco.MjModel.from_xml_string(build_scene_xml(["red"] * n_cubes))
    return _MODELS[n_cubes]


def make_scene(n_cubes: int = 4, seed: int | None = None, colors: list[str] | None = None) -> Scene:
    """A fresh random table. Scenes with the same cube count share one model, so only the
    most recent one has the right cube colors."""
    rng = np.random.default_rng(seed)
    if colors is None:
        colors = [str(c) for c in rng.choice(list(COLORS), size=n_cubes)]
    model = scene_model(len(colors))
    for i, c in enumerate(colors):
        model.geom(f"cube{i}").rgba[:3] = COLORS[c]
    data = mujoco.MjData(model)
    for i, (x, y, yaw) in enumerate(random_layout(len(colors), rng)):
        adr = model.jnt_qposadr[model.body(f"cube{i}").jntadr[0]]
        data.qpos[adr:adr + 3] = (x, y, CUBE_HALF)
        data.qpos[adr + 3:adr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    mujoco.mj_forward(model, data)
    return Scene(model, data, colors)


def bin_of(pos: np.ndarray) -> str | None:
    """Which bin a point is inside of (in xy), if any."""
    for name, (x, y) in BINS.items():
        if abs(pos[0] - x) < BIN_HALF and abs(pos[1] - y) < BIN_HALF:
            return name
    return None
