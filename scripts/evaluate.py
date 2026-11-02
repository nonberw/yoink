"""Run many randomized episodes and report how well the robot sorts.

    python scripts/evaluate.py --episodes 100
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yoink.task import run_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--min-cubes", type=int, default=3)
    ap.add_argument("--max-cubes", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=20_000,
                    help="episode seeds start here (10_000 was used while debugging, so it is not held out)")
    args = ap.parse_args()

    rng = np.random.default_rng(1234)
    cubes = correct = perfect = attempts = ik_fail = 0
    sim_time = 0.0
    failures = Counter()
    t0 = time.perf_counter()
    for ep in range(args.episodes):
        n = int(rng.integers(args.min_cubes, args.max_cubes + 1))
        r = run_episode(n, seed=args.seed_offset + ep)
        cubes += n
        correct += r.correct
        perfect += r.correct == n
        attempts += r.attempts
        ik_fail += r.ik_failures
        sim_time += r.sim_seconds
        for placed, color in zip(r.placed, r.colors):
            if placed != color:
                failures["left on the table" if placed is None else "wrong bin"] += 1
        print(f"\repisode {ep + 1}/{args.episodes}: {correct}/{cubes} cubes sorted", end="", flush=True)

    wall = time.perf_counter() - t0
    print(f"\n\n| Metric | Value |\n|---|---|")
    print(f"| Episodes | {args.episodes} ({args.min_cubes}-{args.max_cubes} cubes each, {cubes} cubes total) |")
    print(f"| Cubes sorted into the right bin | {correct}/{cubes} ({100 * correct / cubes:.1f}%) |")
    print(f"| Episodes with every cube sorted | {perfect}/{args.episodes} ({100 * perfect / args.episodes:.0f}%) |")
    print(f"| Pick attempts per cube | {attempts / cubes:.2f} |")
    print(f"| Sim time per cube | {sim_time / cubes:.1f} s |")
    print(f"| IK misses during straight-line moves | {ik_fail} |")
    print(f"| Wall time | {wall:.0f} s ({wall / cubes:.2f} s per cube) |")
    if failures:
        print("\nFailure modes:", dict(failures))


if __name__ == "__main__":
    main()
