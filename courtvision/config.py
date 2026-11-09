"""Configuration as dataclasses, loadable from YAML.

Every tunable number in the pipeline lives here, so experiments are a config
change rather than a code change.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    # A YOLO *pose* model detects players and gives their skeletons in one pass.
    player_pose_model: str = "yolov8s-pose.pt"
    player_conf: float = 0.35
    # Optional fine-tuned player detector (players only, no crowd/bench). When set,
    # it supplies the tracked boxes and pose skeletons are attached by IoU.
    player_detector: Optional[str] = None
    player_class: int = 0
    pose_match_iou: float = 0.3
    # Ball: COCO "sports ball" (class 32) works out of the box; swap in a
    # fine-tuned single-class ball model for broadcast footage.
    ball_model: str = "yolov8s.pt"
    ball_class: int = 32
    # Measured on a broadcast clip with COCO yolov8s: 640px finds the ball in 56% of
    # frames vs 15% at 1280px/0.15. A low threshold is safe because the ball tracker
    # chooses among candidates by motion.
    ball_conf: float = 0.05
    ball_imgsz: int = 640
    ball_candidates: int = 5
    # Optional YOLO-pose model trained on court landmarks. Its keypoints must be
    # in the order given by court.keypoint_names.
    court_keypoint_model: Optional[str] = None
    court_keypoint_conf: float = 0.5
    imgsz: int = 1280             # broadcast players are small; 640 misses most of them
    device: Optional[str] = None  # "cuda:0", "cpu", "mps"; None = auto
    half: bool = False            # FP16 on CUDA


@dataclass
class BallConfig:
    max_speed_px: float = 60.0    # max plausible ball travel per frame (pixels)
    max_interp_gap: int = 20      # fill gaps up to this many frames


@dataclass
class TeamConfig:
    samples_per_track: int = 12   # jersey colour samples kept per track
    sample_every: int = 5         # frames between samples of the same track
    min_samples: int = 3          # tracks with fewer samples get the nearest team by colour


@dataclass
class PossessionConfig:
    use_pose: bool = True              # False = box-distance only (baseline for evaluation)
    keypoint_conf: float = 0.35
    hand_radius: float = 0.22          # wrist-ball distance / player height
    hand_extension: float = 0.25       # push wrist along the forearm to reach the palm
    bbox_fallback_radius: float = 0.12
    switch_frames: int = 5
    max_gap_frames: int = 8
    backfill: bool = True


@dataclass
class EventConfig:
    max_pass_frames: int = 45     # longest the ball can be "in the air" for a pass (~1.5 s @30fps)


@dataclass
class CourtConfig:
    calibration: Optional[str] = None  # JSON from tools/calibrate.py (fixed camera)
    keypoint_names: List[str] = field(default_factory=list)  # order for court_keypoint_model
    filter_off_court: bool = True      # drop people standing outside the court (bench, crowd)
    off_court_margin_m: float = 1.0


@dataclass
class KinematicsConfig:
    smooth_window: int = 7        # frames, centred moving average on court positions
    max_speed_kmh: float = 36.0   # faster than this is a projection glitch, not a player


@dataclass
class RenderConfig:
    minimap: bool = True
    minimap_width: int = 360
    show_speed: bool = True
    show_ids: bool = True
    event_banner_seconds: float = 1.5


@dataclass
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    teams: TeamConfig = field(default_factory=TeamConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    events: EventConfig = field(default_factory=EventConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @classmethod
    def load(cls, path: Optional[str]) -> "Config":
        cfg = cls()
        if path:
            with open(path) as f:
                _merge(cfg, yaml.safe_load(f) or {})
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(obj, data: dict) -> None:
    valid = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' in {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)
