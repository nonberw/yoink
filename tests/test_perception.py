import numpy as np
import pytest

from yoink.camera import RGBDCamera
from yoink.perception import (
    color_mask, connected_components, detect_cubes, min_area_rect, rgb_to_hsv, split_touching,
)
from yoink.scene import COLORS, make_scene


def square(center, yaw, half=0.02, n=30):
    g = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel()], 1)
    c, s = np.cos(yaw), np.sin(yaw)
    return pts @ np.array([[c, s], [-s, c]]) + center


@pytest.mark.parametrize("yaw", [0.0, 0.3, 0.7, 1.2, 1.5])
def test_min_area_rect_recovers_square(yaw):
    center, est, sides = min_area_rect(square(np.array([0.4, -0.1]), yaw))
    np.testing.assert_allclose(center, [0.4, -0.1], atol=1e-6)
    np.testing.assert_allclose(sides, [0.04, 0.04], atol=1e-4)
    err = abs((est - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    assert np.rad2deg(err) < 0.1


def test_touching_cubes_get_split():
    a, b = square(np.array([0.40, 0.0]), 0.2), square(np.array([0.44, 0.012]), 0.5)
    parts = split_touching(np.vstack([a, b]))
    centers = sorted(min_area_rect(p)[0][0] for p in parts)
    assert len(parts) == 2
    np.testing.assert_allclose(centers, [0.40, 0.44], atol=4e-3)


def test_single_cube_is_not_split():
    assert len(split_touching(square(np.zeros(2), 0.785))) == 1  # diagonal is 1.41x the side


def test_connected_components():
    m = np.zeros((10, 10), bool)
    m[1:3, 1:3] = True
    m[5:8, 5:8] = True
    m[8, 8] = True  # diagonal neighbor joins the second blob (8-connectivity)
    sizes = sorted(len(b) for b in connected_components(m))
    assert sizes == [4, 10]


@pytest.mark.parametrize("color", list(COLORS))
def test_pure_color_classified(color):
    rgb = (np.array(COLORS[color]) * 255).astype(np.uint8)[None, None]
    h, s, v = rgb_to_hsv(rgb)
    assert all(color_mask(h, s, v, c)[0, 0] == (c == color) for c in COLORS)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_detects_every_cube_accurately(seed):
    scene = make_scene(5, seed)
    with RGBDCamera(scene.model) as cam:
        rgb, depth = cam.capture(scene.data)
        dets = detect_cubes(cam, scene.data, rgb, depth)
    assert len(dets) == 5
    for i, color in enumerate(scene.colors):
        pos, _ = scene.cube_pose(i)
        err = min(np.linalg.norm(d.pos - pos) for d in dets if d.color == color)
        assert err < 2e-3
