"""Plain data containers passed between pipeline stages.

Every stage after detection works on these lightweight records rather than on
video frames, so a full game never has to sit in memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

NO_PLAYER = -1


@dataclass
class PlayerObs:
    """One tracked player in one frame."""
    track_id: int
    bbox: np.ndarray                      # (4,) x1, y1, x2, y2 in pixels
    conf: float
    keypoints: Optional[np.ndarray] = None  # (17, 3) COCO x, y, conf

    @property
    def foot(self) -> np.ndarray:
        x1, _, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, y2])

    @property
    def height(self) -> float:
        return float(max(self.bbox[3] - self.bbox[1], 1.0))


@dataclass
class FrameObs:
    """Everything the models saw in one frame."""
    index: int
    players: Dict[int, PlayerObs] = field(default_factory=dict)
    ball_candidates: Optional[np.ndarray] = None  # (k, 3) cx, cy, conf; None if none seen
    court_keypoints: Optional[np.ndarray] = None  # (K, 3) image x, y, conf


@dataclass
class Event:
    kind: str          # "pass" | "interception"
    frame: int
    from_player: int
    to_player: int
    team: int          # team that has the ball after the event


@dataclass
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    num_frames: int


@dataclass
class Analysis:
    """Output of the post-processing stages, ready to render or export."""
    meta: VideoMeta
    frames: List[FrameObs]
    ball_centers: np.ndarray                  # (N, 2), NaN where unknown
    teams: Dict[int, int]                     # track_id -> 0 / 1
    team_colors: List[tuple]                  # BGR per team, for drawing
    possession: List[int]                     # track_id per frame or NO_PLAYER
    events: List[Event]
    court_xy: Dict[int, np.ndarray]           # track_id -> (N, 2) metres, NaN when off-screen
    speed_kmh: Dict[int, np.ndarray]          # track_id -> (N,)
    distance_m: Dict[int, np.ndarray]         # track_id -> (N,) cumulative
    homographies: List[Optional[np.ndarray]]  # image -> court metres, per frame
