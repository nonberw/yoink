import numpy as np
import pytest

from yoink.camera import RGBDCamera
from yoink.motion import gripper_down, min_jerk
from yoink.perception import Detection
from yoink.scene import BIN_HALF, BINS, bin_of, make_scene, random_layout
from yoink.task import grasp_yaw, run_episode


def test_min_jerk_profile():
    assert min_jerk(0) == 0 and min_jerk(1) == 1 and min_jerk(0.5) == pytest.approx(0.5)
    h = 1e-4
    for s in (0.0, 1.0):  # zero velocity at both ends
        assert abs(min_jerk(min(s + h, 1)) - min_jerk(max(s - h, 0))) / h < 1e-3


def test_gripper_points_down():
    R = gripper_down(0.7)
    np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-12)


@pytest.mark.parametrize("yaw", [0.0, 0.4, 1.3])
def test_grasp_yaw_is_equivalent_and_near_natural(yaw):
    det = Detection("red", np.array([0.4, 0.15, 0.02]), yaw, 100, (0, 0))
    g = grasp_yaw(det)
    natural = np.arctan2(0.15, 0.4)
    assert abs((g - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4) < 1e-9  # same cube face set
    assert abs(g - natural) <= np.pi / 4 + 1e-9


def test_layout_has_no_overlaps():
    pts = random_layout(6, np.random.default_rng(0))
    for i in range(6):
        for j in range(i):
            assert np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) > 0.075


def test_bin_of():
    for name, (x, y) in BINS.items():
        assert bin_of(np.array([x, y, 0.02])) == name
        assert bin_of(np.array([x + BIN_HALF + 0.01, y, 0.02])) != name
    assert bin_of(np.array([0.4, 0.0, 0.02])) is None


def test_camera_projection_roundtrip():
    scene = make_scene(1, seed=0)
    with RGBDCamera(scene.model) as cam:
        p = np.array([[0.4, 0.05, 0.03], [0.33, -0.12, 0.0]])
        uv = cam.project(scene.data, p)
        R = scene.data.cam_xmat[cam.id].reshape(3, 3)
        d = -((p - scene.data.cam_xpos[cam.id]) @ R)[:, 2]
        back = cam.backproject(scene.data, uv[:, 0], uv[:, 1], d)
    np.testing.assert_allclose(back, p, atol=1e-9)


def test_full_episode_sorts_everything():
    r = run_episode(3, seed=5)
    assert r.correct == 3
    assert r.ik_failures == 0
