"""Cube detection from RGB-D, from scratch: HSV color masks, connected components,
back-projection to 3D, and a minimum-area rectangle fit on the top face for position + yaw."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .camera import RGBDCamera
from .scene import CUBE_HALF, SPAWN_HIGH, SPAWN_LOW

# hue center (deg), hue tolerance (deg)
# Yellow is centered above its true hue on purpose: the brightly lit top face clips at
# full value and its hue drifts to ~60 deg, and the top face is what we measure.
HUES = {"red": (0.0, 14.0), "yellow": (53.0, 13.0), "green": (130.0, 25.0)}
MIN_SAT, MIN_VAL = 0.55, 0.25
MIN_PIXELS = 40
WORKSPACE_MARGIN = 0.06  # meters around the spawn area that still counts as "on the table"


@dataclass
class Detection:
    color: str
    pos: np.ndarray   # cube center in world, meters
    yaw: float        # radians in [0, pi/2): a cube looks the same every 90 degrees
    pixels: int
    centroid_px: tuple[float, float]


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint8 (H, W, 3) -> hue in degrees [0, 360), saturation and value in [0, 1]."""
    c = rgb.astype(np.float32) / 255.0
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx, mn = c.max(-1), c.min(-1)
    delta = mx - mn
    safe = np.where(delta > 1e-6, delta, 1.0)
    h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 60.0
    h = np.where(delta > 1e-6, h, 0.0)
    s = np.where(mx > 1e-6, delta / np.where(mx > 1e-6, mx, 1.0), 0.0)
    return h, s, mx


def color_mask(h, s, v, color: str) -> np.ndarray:
    center, tol = HUES[color]
    dist = np.abs((h - center + 180.0) % 360.0 - 180.0)  # circular hue distance
    return (dist < tol) & (s > MIN_SAT) & (v > MIN_VAL)


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected blobs of a boolean mask, each as an (N, 2) array of (row, col)."""
    seen = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    blobs = []
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        queue, pix = deque([(r0, c0)]), []
        while queue:
            r, c = queue.popleft()
            pix.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        queue.append((rr, cc))
        blobs.append(np.array(pix))
    return blobs


def min_area_rect(xy: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Smallest enclosing rectangle of 2D points: (center, yaw in [0, pi/2), side lengths).

    Coarse 1-degree sweep then a 0.05-degree refine. Rotating calipers would be O(n log n),
    but a square's top face is a few hundred points, so brute force is plenty.
    """
    def area_at(theta):
        c, s = np.cos(theta), np.sin(theta)
        p = xy @ np.array([[c, -s], [s, c]])  # rotate points by -theta
        lo, hi = p.min(0), p.max(0)
        return (hi - lo).prod(), lo, hi

    coarse = np.deg2rad(np.arange(0.0, 90.0, 1.0))
    best = coarse[np.argmin([area_at(t)[0] for t in coarse])]
    fine = best + np.deg2rad(np.arange(-1.0, 1.0, 0.05))
    theta = fine[np.argmin([area_at(t)[0] for t in fine])] % (np.pi / 2)
    _, lo, hi = area_at(theta)
    c, s = np.cos(theta), np.sin(theta)
    center = np.array([[c, -s], [s, c]]) @ ((lo + hi) / 2)
    return center, float(theta), hi - lo


def split_touching(top_xy: np.ndarray, max_side: float = 2.5 * CUBE_HALF) -> list[np.ndarray]:
    """Same-colored cubes that touch merge into one blob. If the top face is too wide for one
    cube, split it with 2-means seeded at the two extreme points along its long side, and recurse."""
    _, yaw, sides = min_area_rect(top_xy)
    if sides.max() <= max_side or len(top_xy) < 20:
        return [top_xy]
    long_axis = np.array([np.cos(yaw), np.sin(yaw)]) if sides[0] >= sides[1] else np.array([-np.sin(yaw), np.cos(yaw)])
    t = top_xy @ long_axis
    centers = np.stack([top_xy[t.argmin()], top_xy[t.argmax()]])
    for _ in range(10):
        label = np.linalg.norm(top_xy[:, None] - centers[None], axis=2).argmin(1)
        if label.min() == label.max():
            return [top_xy]
        centers = np.stack([top_xy[label == k].mean(0) for k in (0, 1)])
    return split_touching(top_xy[label == 0], max_side) + split_touching(top_xy[label == 1], max_side)


def detect_cubes(camera: RGBDCamera, data, rgb: np.ndarray, depth: np.ndarray) -> list[Detection]:
    h, s, v = rgb_to_hsv(rgb)
    lo, hi = SPAWN_LOW - WORKSPACE_MARGIN, SPAWN_HIGH + WORKSPACE_MARGIN
    out = []
    for color in HUES:
        for blob in connected_components(color_mask(h, s, v, color)):
            if len(blob) < MIN_PIXELS:
                continue
            rows, cols = blob[:, 0], blob[:, 1]
            pts = camera.backproject(data, cols.astype(float), rows.astype(float), depth[rows, cols])
            # Keep only points that sit on the table at cube height: kills the bins, the arm, reflections.
            keep = ((pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0]) & (pts[:, 1] > lo[1]) & (pts[:, 1] < hi[1])
                    & (pts[:, 2] > 0.003) & (pts[:, 2] < 3 * CUBE_HALF))
            pts = pts[keep]
            if len(pts) < MIN_PIXELS:
                continue
            z_top = np.percentile(pts[:, 2], 90)
            top = pts[pts[:, 2] > z_top - 0.004]
            if len(top) < 10:
                continue
            z = np.median(top[:, 2]) - CUBE_HALF
            parts = split_touching(top[:, :2])
            for part in parts:
                if len(parts) > 1 and len(part) < 10:
                    continue
                center, yaw, _ = min_area_rect(part)
                pixels = int(keep.sum()) // len(parts)
                out.append(Detection(color, np.array([center[0], center[1], z]), yaw, pixels,
                                     (float(cols.mean()), float(rows.mean()))))
    return out
