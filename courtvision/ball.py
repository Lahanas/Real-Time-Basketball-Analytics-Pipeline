"""Ball trajectory: choose the right candidate each frame, then fill short gaps.

Generic detectors find small, blurred broadcast balls only at a low confidence
threshold, which also lets in heads, shoes and spare balls. So the detector
keeps several candidates per frame and the tracker picks the one consistent
with motion: within reach of the last accepted position given the frames
elapsed, and closest to it. With no recent anchor, the most confident wins.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import BallConfig


def track_ball(candidates: List[Optional[np.ndarray]], max_speed_px: float,
               reanchor_after: int = 30) -> np.ndarray:
    """
    candidates[i]: (k, 3) array of (cx, cy, conf) or None.
    Returns (N, 2) centres, NaN where no plausible ball.

    If nothing has been accepted for `reanchor_after` frames, the last accepted
    position may itself have been wrong, so the next most-confident detection
    is trusted again.
    """
    out = np.full((len(candidates), 2), np.nan)
    last_i, last_xy = None, None
    for i, c in enumerate(candidates):
        if c is None or len(c) == 0:
            continue
        if last_i is None or i - last_i > reanchor_after:
            pick = c[int(np.argmax(c[:, 2])), :2]
        else:
            d = np.linalg.norm(c[:, :2] - last_xy, axis=1)
            ok = d <= max_speed_px * (i - last_i)
            if not ok.any():
                continue
            pick = c[np.where(ok)[0][np.argmin(d[ok])], :2]
        out[i] = pick
        last_i, last_xy = i, pick
    return out


def interpolate_gaps(centers: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation across gaps of at most `max_gap` frames."""
    out = centers.copy()
    known = np.where(~np.isnan(centers[:, 0]))[0]
    for a, b in zip(known[:-1], known[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            t = np.arange(1, gap + 1)[:, None] / (gap + 1)
            out[a + 1:b] = centers[a] + t * (centers[b] - centers[a])
    return out


def clean_ball_track(candidates: List[Optional[np.ndarray]], cfg: BallConfig) -> np.ndarray:
    return interpolate_gaps(track_ball(candidates, cfg.max_speed_px), cfg.max_interp_gap)
