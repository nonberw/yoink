"""How accurate is the cube detector? Compares detections to ground truth on random tables.

    python scripts/perception_benchmark.py --scenes 40
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yoink.camera import RGBDCamera  # noqa: E402
from yoink.perception import detect_cubes  # noqa: E402
from yoink.scene import make_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--cubes", type=int, default=5)
    args = ap.parse_args()

    pos_err, yaw_err, times = [], [], []
    missed = extra = 0
    for seed in range(30_000, 30_000 + args.scenes):
        scene = make_scene(args.cubes, seed)
        cam = RGBDCamera.shared(scene.model)
        rgb, depth = cam.capture(scene.data)
        t0 = time.perf_counter()
        dets = detect_cubes(cam, scene.data, rgb, depth)
        times.append(time.perf_counter() - t0)
        used = set()
        for i, color in enumerate(scene.colors):
            pos, R = scene.cube_pose(i)
            cands = [(np.linalg.norm(d.pos - pos), k) for k, d in enumerate(dets) if d.color == color and k not in used]
            if not cands or min(cands)[0] > 0.02:
                missed += 1
                continue
            err, k = min(cands)
            used.add(k)
            pos_err.append(err)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            yaw_err.append(abs((dets[k].yaw - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4))
        extra += len(dets) - len(used)

    n = args.scenes * args.cubes
    pe, ye = np.array(pos_err) * 1e3, np.rad2deg(yaw_err)
    print("| Metric | Value |\n|---|---|")
    print(f"| Cubes | {n} on {args.scenes} random tables |")
    print(f"| Detected / missed / false positives | {len(pe)} / {missed} / {extra} |")
    print(f"| Position error, mean / p95 / max | {pe.mean():.2f} / {np.percentile(pe, 95):.2f} / {pe.max():.2f} mm |")
    print(f"| Yaw error, mean / max | {ye.mean():.2f} / {ye.max():.2f} deg |")
    print(f"| Detection time per frame (640x480, pure Python) | {1e3 * np.mean(times):.0f} ms |")


if __name__ == "__main__":
    main()
