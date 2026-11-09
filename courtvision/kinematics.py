"""Court positions, speed and distance for every player."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .court import to_court


def court_tracks(frames, homographies: List[Optional[np.ndarray]], num_frames: int) -> Dict[int, np.ndarray]:
    """track_id -> (N, 2) court metres of the player's feet, NaN where not visible."""
    tracks: Dict[int, np.ndarray] = {}
    for f in frames:
        H = homographies[f.index]
        if H is None or not f.players:
            continue
        ids = list(f.players)
        feet = np.stack([f.players[t].foot for t in ids])
        xy = to_court(H, feet)
        for tid, p in zip(ids, xy):
            if tid not in tracks:
                tracks[tid] = np.full((num_frames, 2), np.nan)
            tracks[tid][f.index] = p
    return tracks


def _valid_runs(valid: np.ndarray):
    """(start, end_inclusive) of each run of True."""
    runs, start = [], None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(valid) - 1))
    return runs


def smooth_positions(xy: np.ndarray, window: int) -> np.ndarray:
    """
    Centred moving average within each visible run (never across a gap).

    The window stays symmetric and narrows near a run's ends; a one-sided
    window there would pull the first/last positions inward and undercount
    distance every time a player enters or leaves the frame.
    """
    if window <= 1:
        return xy.copy()
    half = window // 2
    out = np.full_like(xy, np.nan)
    for rs, re in _valid_runs(~np.isnan(xy[:, 0])):
        for i in range(rs, re + 1):
            k = min(half, i - rs, re - i)
            out[i] = xy[i - k:i + k + 1].mean(axis=0)
    return out


def speed_and_distance(xy: np.ndarray, fps: float, window: int, max_speed_kmh: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-frame speed (km/h) and cumulative distance (m).

    Steps faster than `max_speed_kmh` are homography glitches (feet hidden,
    a keypoint jumping) and add neither speed nor distance.
    """
    sm = smooth_positions(xy, window)
    n = len(sm)
    speed = np.full(n, np.nan)
    step = np.zeros(n)
    max_step = max_speed_kmh / 3.6 / fps
    for i in range(1, n):
        if np.isnan(sm[i, 0]) or np.isnan(sm[i - 1, 0]):
            continue
        d = float(np.linalg.norm(sm[i] - sm[i - 1]))
        if d <= max_step:
            step[i] = d
            speed[i] = d * fps * 3.6
    # Displayed speed: average over the smoothing window to avoid jitter
    if window > 1:
        speed = smooth_positions(np.stack([speed, speed], axis=1), window)[:, 0]
    return speed, np.cumsum(step)
