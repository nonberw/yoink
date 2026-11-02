"""Render assets/demo.gif: main view plus the robot's camera feed with detections drawn on top.

    python scripts/render_gif.py --seed 7 --cubes 4
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yoink.camera import RGBDCamera  # noqa: E402
from yoink.scene import COLORS, CUBE_HALF, make_scene  # noqa: E402
from yoink.task import OBSERVE, Sorter  # noqa: E402

INSET_CROP = (150, 130, 490, 330)  # camera pixels around the table
INSET_SIZE = (238, 140)
BOX_RGB = {"red": (255, 90, 90), "yellow": (255, 220, 60), "green": (90, 235, 120)}


def font(size):
    try:
        return ImageFont.truetype("consola.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_gif(frames, path, fps):
    """One shared palette and no dithering: with a static camera most pixels stay identical
    between frames, so the GIF encoder only stores the parts that move (~4x smaller)."""
    picked = frames[:: max(1, len(frames) // 12)]
    # The scene is mostly dark purple; without help median-cut spends the whole palette there
    # and the cubes come out grey. A swatch of the colors that matter reserves entries for them.
    key = [np.array(c) * 255 * k for c in COLORS.values() for k in (0.35, 0.55, 0.75, 0.9, 1.0)]
    key += [np.array(c) for c in BOX_RGB.values()] + [np.array((235, 230, 255)), np.array((180, 150, 255))]
    rows = len(picked) * frames[0].shape[0] // 4
    swatch = np.repeat(np.repeat(np.clip(key, 0, 255).astype(np.uint8)[None], rows, 0),
                       frames[0].shape[1] // len(key) + 1, 1)[:, : frames[0].shape[1]]
    sample = Image.fromarray(np.concatenate(picked + [swatch], axis=0))
    palette = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=0,
                 optimize=False, disposal=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cubes", type=int, default=4)
    ap.add_argument("--speed", type=int, default=3, help="playback speed-up")
    ap.add_argument("--out", default=str(ROOT / "assets" / "demo.gif"))
    args = ap.parse_args()

    W, H, FPS = 560, 360, 20
    scene = make_scene(args.cubes, args.seed)
    scene.data.qpos[:6] = OBSERVE
    scene.data.ctrl[:6] = OBSERVE
    mujoco.mj_forward(scene.model, scene.data)

    view = mujoco.Renderer(scene.model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.18, 0.0, 0.12]
    cam.distance, cam.azimuth, cam.elevation = 1.45, 205, -32

    inset = {"img": None}
    frames, tick = [], {"n": 0}
    every = round(60 * args.speed / FPS)
    big, small = font(15), font(12)

    def on_detect(rgb, dets):
        img = Image.fromarray(rgb)
        d = ImageDraw.Draw(img)
        for det in dets:
            corners = []
            c, s = np.cos(det.yaw), np.sin(det.yaw)
            for dx, dy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                # Draw the estimated footprint 1.8x oversized so it frames the cube instead of hiding on it.
                off = 1.8 * CUBE_HALF * np.array([c * dx - s * dy, s * dx + c * dy, 0.0])
                corners.append(sorter.camera.project(scene.data, det.pos + off)[0])
            pts = [tuple(p) for p in corners]
            d.polygon(pts, outline=BOX_RGB[det.color], width=3)
            u, v = sorter.camera.project(scene.data, det.pos + [0, 0, CUBE_HALF])[0]
            d.line((u - 5, v, u + 5, v), fill=(255, 255, 255), width=2)
            d.line((u, v - 5, u, v + 5), fill=(255, 255, 255), width=2)
        # Crop to the table so the boxes are actually visible at inset size.
        inset["img"] = img.crop(INSET_CROP).resize(INSET_SIZE, Image.LANCZOS)
        inset["n"] = len(dets)

    def on_tick():
        tick["n"] += 1
        if tick["n"] % every:
            return
        view.update_scene(scene.data, cam)
        img = Image.fromarray(view.render())
        d = ImageDraw.Draw(img)
        if inset["img"] is not None:
            iw, ih = INSET_SIZE
            img.paste(inset["img"], (W - iw - 8, 8))
            d.rectangle((W - iw - 9, 7, W - 7, ih + 9), outline=(180, 150, 255), width=2)
            d.text((W - iw - 4, ih + 12), f"robot cam: {inset['n']} cubes found", fill=(220, 210, 255), font=small)
        done = sum(1 for i in range(len(scene.colors)) if _in_right_bin(i))
        d.text((12, 10), f"sorted {done}/{len(scene.colors)}   t={scene.data.time:4.1f}s   x{args.speed}",
               fill=(235, 230, 255), font=big)
        frames.append(np.asarray(img))

    def _in_right_bin(i):
        from yoink.scene import bin_of
        return bin_of(scene.cube_pose(i)[0]) == scene.colors[i]

    with RGBDCamera(scene.model) as rgbd:
        sorter = Sorter(scene, rgbd, on_tick=on_tick, on_detect=on_detect)
        result = sorter.run()
    for _ in range(FPS):  # hold the final frame for a second
        frames.append(frames[-1])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, args.out, FPS)
    print(f"sorted {result.correct}/{len(result.colors)}; wrote {args.out} "
          f"({len(frames)} frames, {Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
