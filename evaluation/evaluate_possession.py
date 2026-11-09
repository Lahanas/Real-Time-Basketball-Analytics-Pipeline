"""
Score possession predictions against hand labels.

1. Predictions:  courtvision clip.mp4 --dump-possession   -> outputs/clip_possession.json
2. Labels: watch outputs/clip_annotated.mp4 (frame numbers and track IDs are
   drawn) and write a CSV of who holds the ball:
       start_frame,end_frame,player_id
       0,37,4
       52,110,9
   Uncovered frames mean nobody has it (pass in flight, shot, loose ball).
3. Score:  python evaluation/evaluate_possession.py outputs/clip_possession.json labels/clip.csv

Metrics
  frame_acc   frames where the predicted holder (or "nobody") matches the label
  events      predicted possession spells
  false_pos   predicted spells where < 50% of frames match the labelled holder
  missed      labelled spells that < 50% of frames are predicted correctly
  switches    holder-to-holder changes (flicker shows up here)

Multiple prediction files can be passed; their methods are pooled across clips
when given matching label files in the same order.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from courtvision.types import NO_PLAYER  # noqa: E402


def load_labels(path: str, n: int) -> List[int]:
    gt = [NO_PLAYER] * n
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            s, e, pid = int(row["start_frame"]), int(row["end_frame"]), int(row["player_id"])
            for i in range(max(s, 0), min(e, n - 1) + 1):
                gt[i] = pid
    return gt


def spells(seq: List[int]) -> List[Tuple[int, int, int]]:
    out = []
    for i, p in enumerate(seq):
        if p == NO_PLAYER:
            continue
        if out and out[-1][2] == p and out[-1][1] == i - 1:
            out[-1] = (out[-1][0], i, p)
        else:
            out.append((i, i, p))
    return out


def _covered(seq: List[int], s: int, e: int, pid: int) -> bool:
    return sum(seq[i] == pid for i in range(s, e + 1)) >= 0.5 * (e - s + 1)


def evaluate(pred: List[int], gt: List[int]) -> Dict[str, float]:
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    ps, gs = spells(pred), spells(gt)
    return {
        "frames": n,
        "correct": sum(a == b for a, b in zip(pred, gt)),
        "events": len(ps),
        "false_pos": sum(not _covered(gt, s, e, p) for s, e, p in ps),
        "missed": sum(not _covered(pred, s, e, p) for s, e, p in gs),
        "labelled": len(gs),
        "switches": sum(a != b and NO_PLAYER not in (a, b) for a, b in zip(pred, pred[1:])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="predictions.json labels.csv [predictions.json labels.csv ...]")
    ap.add_argument("--baseline", default="bbox")
    args = ap.parse_args()
    if len(args.pairs) % 2:
        ap.error("pass predictions and labels in pairs")

    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pred_path, label_path in zip(args.pairs[::2], args.pairs[1::2]):
        with open(pred_path) as f:
            preds = json.load(f)
        n = len(next(iter(preds.values())))
        gt = load_labels(label_path, n)
        for method, seq in preds.items():
            for k, v in evaluate(seq, gt).items():
                totals[method][k] += v

    cols = ["frame_acc", "events", "false_pos", "missed", "switches"]
    print(f"{'method':<10}" + "".join(f"{c:>12}" for c in cols))
    for m, t in totals.items():
        acc = t["correct"] / t["frames"] if t["frames"] else 0.0
        print(f"{m:<10}{acc:>12.3f}" + "".join(f"{t[c]:>12}" for c in cols[1:]))
    any_t = next(iter(totals.values()))
    print(f"labelled possessions: {any_t['labelled']}, frames: {any_t['frames']}")

    base = totals.get(args.baseline)
    if base and base["false_pos"]:
        for m, t in totals.items():
            if m != args.baseline:
                pct = 100 * (t["false_pos"] - base["false_pos"]) / base["false_pos"]
                print(f"{m}: {pct:+.1f}% false-positive possession events vs {args.baseline}")


if __name__ == "__main__":
    main()
