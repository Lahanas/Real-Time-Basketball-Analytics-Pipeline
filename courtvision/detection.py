"""Model inference: players (+ skeletons), ball, and optional court keypoints.

Default mode: a YOLO *pose* model run through Ultralytics' ByteTrack, so every
track ID arrives with its 17 body keypoints attached. One model, one pass.

Fine-tuned mode (`models.player_detector`): a detector trained on players
only supplies the tracked boxes, which keeps crowd and bench out on broadcast
footage. The pose model still runs, and skeletons are attached to tracks with
one-to-one IoU matching.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import ModelConfig
from .types import FrameObs, PlayerObs


def bbox_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0.0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def attach_poses(players: Dict[int, PlayerObs], pose_boxes: np.ndarray, pose_kps: np.ndarray,
                 min_iou: float) -> None:
    """Greedy one-to-one IoU matching, best pairs first, so neighbours can't share a skeleton."""
    pairs = sorted(
        ((bbox_iou(p.bbox, pb), tid, j) for tid, p in players.items() for j, pb in enumerate(pose_boxes)),
        reverse=True,
    )
    used_t, used_p = set(), set()
    for iou, tid, j in pairs:
        if iou < min_iou:
            break
        if tid in used_t or j in used_p:
            continue
        players[tid].keypoints = pose_kps[j]
        used_t.add(tid)
        used_p.add(j)


class Detector:
    def __init__(self, cfg: ModelConfig):
        from ultralytics import YOLO  # heavy import kept out of the pure-logic modules

        self.cfg = cfg
        self.pose_model = YOLO(cfg.player_pose_model)
        self.player_model = YOLO(cfg.player_detector) if cfg.player_detector else None
        self.ball_model = YOLO(cfg.ball_model)
        self.court_model = YOLO(cfg.court_keypoint_model) if cfg.court_keypoint_model else None
        self._common = dict(imgsz=cfg.imgsz, device=cfg.device, verbose=False)
        if cfg.half:
            self._common["half"] = True

    def __call__(self, index: int, frame: np.ndarray) -> FrameObs:
        obs = FrameObs(index=index)
        obs.players = self._players_finetuned(frame) if self.player_model else self._players_pose(frame)
        obs.ball_candidates = self._ball(frame)
        if self.court_model is not None:
            obs.court_keypoints = self._court(frame)
        return obs

    @staticmethod
    def _tracked(res, keypoints: bool) -> Dict[int, PlayerObs]:
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return {}
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        kps = res.keypoints.data.cpu().numpy() if keypoints and res.keypoints is not None else None
        return {
            int(t): PlayerObs(int(t), xyxy[i], float(confs[i]), None if kps is None else kps[i])
            for i, t in enumerate(ids)
        }

    def _players_pose(self, frame) -> Dict[int, PlayerObs]:
        res = self.pose_model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                                    conf=self.cfg.player_conf, **self._common)[0]
        return self._tracked(res, keypoints=True)

    def _players_finetuned(self, frame) -> Dict[int, PlayerObs]:
        res = self.player_model.track(frame, persist=True, tracker="bytetrack.yaml",
                                      classes=[self.cfg.player_class], conf=self.cfg.player_conf,
                                      **self._common)[0]
        players = self._tracked(res, keypoints=False)
        if players:
            pose = self.pose_model.predict(frame, classes=[0], conf=0.2, **self._common)[0]
            if pose.keypoints is not None and len(pose.boxes):
                attach_poses(players, pose.boxes.xyxy.cpu().numpy(), pose.keypoints.data.cpu().numpy(),
                             self.cfg.pose_match_iou)
        return players

    def _ball(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Top-k ball candidates as (cx, cy, conf); the ball tracker picks between them."""
        kw = {**self._common, "imgsz": self.cfg.ball_imgsz}
        res = self.ball_model.predict(frame, classes=[self.cfg.ball_class], conf=self.cfg.ball_conf,
                                      max_det=self.cfg.ball_candidates, **kw)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
        return np.stack([cx, cy, conf], axis=1)

    def _court(self, frame: np.ndarray) -> Optional[np.ndarray]:
        res = self.court_model.predict(frame, conf=0.25, **self._common)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return None
        best = int(np.argmax(res.boxes.conf.cpu().numpy()))
        return res.keypoints.data.cpu().numpy()[best]
