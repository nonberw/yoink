"""Watch the robot sort a random table in real time.

    python scripts/play.py --cubes 5 --seed 3
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yoink.camera import RGBDCamera  # noqa: E402
from yoink.motion import CONTROL_HZ  # noqa: E402
from yoink.scene import make_scene  # noqa: E402
from yoink.task import OBSERVE, Sorter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    scene = make_scene(args.cubes, args.seed)
    scene.data.qpos[:6] = OBSERVE
    scene.data.ctrl[:6] = OBSERVE
    mujoco.mj_forward(scene.model, scene.data)

    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer, RGBDCamera(scene.model) as cam:
        viewer.cam.lookat[:] = [0.18, 0.0, 0.12]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.45, 205, -32
        clock = {"t": time.perf_counter()}

        def on_tick():
            if not viewer.is_running():
                raise SystemExit
            viewer.sync()
            clock["t"] += 1 / CONTROL_HZ
            time.sleep(max(0.0, clock["t"] - time.perf_counter()))

        result = Sorter(scene, cam, on_tick=on_tick).run()
        print(f"sorted {result.correct}/{len(result.colors)} cubes in {result.sim_seconds:.1f} s")
        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
