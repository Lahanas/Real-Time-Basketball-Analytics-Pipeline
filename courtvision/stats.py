"""Game statistics from an Analysis, exported as JSON and CSV."""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np

from .types import NO_PLAYER, Analysis


def team_possession_counts(possession: List[int], teams: Dict[int, int]) -> np.ndarray:
    """Cumulative frames of possession per team: (N, 2)."""
    counts = np.zeros((len(possession), 2), dtype=np.int64)
    running = np.zeros(2, dtype=np.int64)
    for i, pid in enumerate(possession):
        t = teams.get(pid, -1) if pid != NO_PLAYER else -1
        if t in (0, 1):
            running[t] += 1
        counts[i] = running
    return counts


def summarize(a: Analysis, min_seconds_visible: float = 1.0) -> dict:
    fps = a.meta.fps
    visible = Counter(tid for f in a.frames for tid in f.players)
    held = Counter(pid for pid in a.possession if pid != NO_PLAYER)
    passes_made = Counter(e.from_player for e in a.events if e.kind == "pass")
    passes_recv = Counter(e.to_player for e in a.events if e.kind == "pass")
    steals = Counter(e.to_player for e in a.events if e.kind == "interception")

    players = []
    for tid, n in sorted(visible.items()):
        if n < min_seconds_visible * fps:
            continue  # flicker tracks: a referee's arm, a fan, a broken ID
        speed = a.speed_kmh.get(tid)
        dist = a.distance_m.get(tid)
        valid = speed[~np.isnan(speed)] if speed is not None else np.array([])
        players.append({
            "track_id": tid,
            "team": a.teams.get(tid, -1),
            "seconds_visible": round(n / fps, 2),
            "possession_s": round(held[tid] / fps, 2),
            "passes_made": passes_made[tid],
            "passes_received": passes_recv[tid],
            "interceptions": steals[tid],
            "distance_m": round(float(dist[-1]), 1) if dist is not None else None,
            "avg_speed_kmh": round(float(valid.mean()), 1) if valid.size else None,
            "top_speed_kmh": round(float(np.percentile(valid, 95)), 1) if valid.size else None,
        })

    team_frames = team_possession_counts(a.possession, a.teams)[-1] if a.possession else np.zeros(2)
    total = max(int(team_frames.sum()), 1)
    teams = []
    for t in (0, 1):
        teams.append({
            "team": t,
            "color_bgr": list(a.team_colors[t]),
            "possession_pct": round(100.0 * team_frames[t] / total, 1),
            "passes": sum(1 for e in a.events if e.kind == "pass" and e.team == t),
            "interceptions": sum(1 for e in a.events if e.kind == "interception" and e.team == t),
        })

    return {
        "video": os.path.basename(a.meta.path),
        "fps": fps,
        "frames": len(a.frames),
        "teams": teams,
        "players": players,
        "events": [
            {"type": e.kind, "frame": e.frame, "time_s": round(e.frame / fps, 2),
             "from": e.from_player, "to": e.to_player, "team": e.team}
            for e in a.events
        ],
    }


def export(summary: dict, out_dir: str, stem: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    p = os.path.join(out_dir, f"{stem}_stats.json")
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    paths.append(p)
    if summary["players"]:
        p = os.path.join(out_dir, f"{stem}_players.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary["players"][0]))
            w.writeheader()
            w.writerows(summary["players"])
        paths.append(p)
    return paths
