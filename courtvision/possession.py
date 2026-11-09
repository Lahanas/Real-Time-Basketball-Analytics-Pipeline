"""Ball possession from hand keypoints, with hysteresis smoothing.

A player has the ball when it is at their hands, not merely inside their
bounding box. Boxes overlap constantly in basketball (screens, post-ups,
contested shots), so box-based rules hand the ball to defenders. Here each
player is scored by palm-to-ball distance from pose keypoints, normalised by
body height so near and far players are comparable.

Per-frame choices are then stabilised by `PossessionSmoother`:
  - a new holder must win `switch_frames` frames in a row
  - the holder keeps the ball through gaps up to `max_gap_frames` (dribbles,
    missed ball detections)
  - because video is processed offline, confirmed switches are back-filled so
    the change appears when it actually happened, not `switch_frames` later
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import PossessionConfig
from .types import NO_PLAYER, PlayerObs

LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = 7, 8, 9, 10


def hand_points(kps: np.ndarray, cfg: PossessionConfig) -> List[np.ndarray]:
    pts = []
    for wrist, elbow in ((LEFT_WRIST, LEFT_ELBOW), (RIGHT_WRIST, RIGHT_ELBOW)):
        wx, wy, wc = kps[wrist]
        if wc < cfg.keypoint_conf:
            continue
        p = np.array([wx, wy], dtype=float)
        ex, ey, ec = kps[elbow]
        if ec >= cfg.keypoint_conf:
            p = p + (p - np.array([ex, ey])) * cfg.hand_extension
        pts.append(p)
    return pts


def _distance_to_box(pt: np.ndarray, bbox: np.ndarray) -> float:
    x = min(max(pt[0], bbox[0]), bbox[2])
    y = min(max(pt[1], bbox[1]), bbox[3])
    return float(np.hypot(pt[0] - x, pt[1] - y))


def frame_candidate(ball_xy: np.ndarray, players: Dict[int, PlayerObs], cfg: PossessionConfig) -> int:
    """Player whose hands are closest to the ball in this frame, or NO_PLAYER."""
    if ball_xy is None or np.isnan(ball_xy[0]):
        return NO_PLAYER
    best_id, best_score = NO_PLAYER, np.inf
    for tid, p in players.items():
        hands = hand_points(p.keypoints, cfg) if (cfg.use_pose and p.keypoints is not None) else []
        if hands:
            score = min(float(np.linalg.norm(ball_xy - h)) for h in hands) / p.height
            radius = cfg.hand_radius
        else:
            score = _distance_to_box(ball_xy, p.bbox) / p.height
            radius = cfg.bbox_fallback_radius
        if score <= radius and score < best_score:
            best_id, best_score = tid, score
    return best_id


class PossessionSmoother:
    def __init__(self, switch_frames: int = 5, max_gap_frames: int = 8, backfill: bool = True):
        self.switch_frames = switch_frames
        self.max_gap_frames = max_gap_frames
        self.backfill = backfill

    def smooth(self, candidates: List[int]) -> List[int]:
        out = [NO_PLAYER] * len(candidates)
        holder, gap = NO_PLAYER, 0
        pending, pending_start, pending_count = NO_PLAYER, 0, 0

        for i, cand in enumerate(candidates):
            if cand == NO_PLAYER:
                gap += 1
                pending, pending_count = NO_PLAYER, 0
                if gap > self.max_gap_frames:
                    holder = NO_PLAYER
            elif cand == holder:
                gap = 0
                pending, pending_count = NO_PLAYER, 0
            else:
                if cand != pending:
                    pending, pending_start, pending_count = cand, i, 0
                pending_count += 1
                if pending_count >= self.switch_frames:
                    holder, gap = cand, 0
                    if self.backfill:
                        out[pending_start:i] = [holder] * (i - pending_start)
                    pending, pending_count = NO_PLAYER, 0
            out[i] = holder
        return out


def detect_possession(frames, ball_centers: np.ndarray, cfg: PossessionConfig) -> List[int]:
    candidates = [frame_candidate(ball_centers[f.index], f.players, cfg) for f in frames]
    return PossessionSmoother(cfg.switch_frames, cfg.max_gap_frames, cfg.backfill).smooth(candidates)
