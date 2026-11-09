"""Command-line entry point: `courtvision game.mp4` or `python -m courtvision game.mp4`."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

from .config import Config
from .pipeline import analyze, render
from .possession import detect_possession
from .stats import export, summarize
from .timing import StageTimer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="courtvision", description="Basketball video analytics")
    p.add_argument("video", help="input video")
    p.add_argument("-c", "--config", help="YAML config (see configs/default.yaml)")
    p.add_argument("-o", "--out-dir", default="outputs", help="where the video and stats go")
    p.add_argument("--cache", help="pickle of detections; reused if it exists, written if not")
    p.add_argument("--max-frames", type=int, help="only process the first N frames")
    p.add_argument("--no-render", action="store_true", help="skip writing the annotated video")
    p.add_argument("--benchmark", action="store_true", help="print per-stage timing and FPS")
    p.add_argument("--dump-possession", action="store_true",
                   help="also write per-frame possession for pose and box-only methods "
                        "(input for evaluation/evaluate_possession.py)")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    timer = StageTimer(enabled=args.benchmark)
    stem = os.path.splitext(os.path.basename(args.video))[0]

    print(f"[1/3] Analysing {args.video}")
    a = analyze(args.video, cfg, timer, args.cache, args.max_frames)

    print(f"[2/3] Writing stats")
    summary = summarize(a)
    for path in export(summary, args.out_dir, stem):
        print(f"  {path}")
    t0, t1 = summary["teams"]
    print(f"  ball control {t0['possession_pct']}% / {t1['possession_pct']}%, "
          f"passes {t0['passes']} / {t1['passes']}, interceptions {t0['interceptions']} / {t1['interceptions']}")

    if args.dump_possession:
        box_cfg = replace(cfg.possession, use_pose=False)
        dump = {
            "pose": a.possession,
            "bbox": detect_possession(a.frames, a.ball_centers, box_cfg),
        }
        path = os.path.join(args.out_dir, f"{stem}_possession.json")
        with open(path, "w") as f:
            json.dump(dump, f)
        print(f"  {path}")

    if not args.no_render:
        out = os.path.join(args.out_dir, f"{stem}_annotated.mp4")
        print(f"[3/3] Rendering {out}")
        render(a, out, cfg, timer)

    if args.benchmark:
        print()
        print(timer.report(len(a.frames)))


if __name__ == "__main__":
    main()
