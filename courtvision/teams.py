"""Unsupervised team assignment from jersey colour.

No labels, no per-game setup: during detection we sample the torso colour of
every track a few times, cluster all samples into two groups with k-means in
CIELAB space (perceptually uniform, so "distance" means "looks different"),
then give each track the team most of its samples fell into.

Voting across many samples per track makes the result robust to single
frames where a player is occluded or motion-blurred.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import TeamConfig

UNKNOWN_TEAM = -1


def torso_color(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    """Median CIELAB colour of the central torso, the part that is almost all jersey."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    tx1, tx2 = int(x1 + 0.30 * w), int(x1 + 0.70 * w)
    ty1, ty2 = int(y1 + 0.20 * h), int(y1 + 0.45 * h)
    H, W = frame.shape[:2]
    tx1, tx2 = max(tx1, 0), min(tx2, W)
    ty1, ty2 = max(ty1, 0), min(ty2, H)
    if tx2 - tx1 < 3 or ty2 - ty1 < 3:
        return None
    crop = cv2.cvtColor(frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(crop, axis=0).astype(np.float32)


class JerseySampler:
    """Collects colour samples while frames stream past."""

    def __init__(self, cfg: TeamConfig):
        self.cfg = cfg
        self.samples: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._last_frame: Dict[int, int] = {}

    def update(self, frame_idx: int, frame: np.ndarray, players: dict) -> None:
        for tid, p in players.items():
            if len(self.samples[tid]) >= self.cfg.samples_per_track:
                continue
            if frame_idx - self._last_frame.get(tid, -10 ** 9) < self.cfg.sample_every:
                continue
            c = torso_color(frame, p.bbox)
            if c is not None:
                self.samples[tid].append(c)
                self._last_frame[tid] = frame_idx


def assign_teams(samples: Dict[int, List[np.ndarray]], min_samples: int = 3,
                 seed: int = 0) -> Tuple[Dict[int, int], List[tuple]]:
    """
    Returns (track_id -> team 0/1, [BGR colour of team 0, team 1]).
    Team 0 is always the lighter jersey, so labels are stable across runs.
    """
    from sklearn.cluster import KMeans

    all_samples = [s for ss in samples.values() for s in ss]
    if len(all_samples) < 2:
        return {tid: UNKNOWN_TEAM for tid in samples}, [(255, 255, 255), (0, 0, 0)]

    X = np.stack(all_samples)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X)
    order = np.argsort(-km.cluster_centers_[:, 0])          # by lightness, light first
    remap = {int(old): new for new, old in enumerate(order)}
    centers = km.cluster_centers_[order]

    teams: Dict[int, int] = {}
    for tid, ss in samples.items():
        if not ss:
            teams[tid] = UNKNOWN_TEAM
            continue
        S = np.stack(ss)
        if len(ss) >= min_samples:
            votes = np.bincount([remap[int(l)] for l in km.predict(S)], minlength=2)
            teams[tid] = int(np.argmax(votes))
        else:
            d = np.linalg.norm(centers - S.mean(axis=0), axis=1)
            teams[tid] = int(np.argmin(d))

    lab = centers.reshape(1, 2, 3).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).reshape(2, 3)
    return teams, [tuple(int(v) for v in c) for c in bgr]
