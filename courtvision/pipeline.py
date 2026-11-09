"""End-to-end orchestration.

Two streaming passes over the video keep memory flat no matter how long the
clip is:

  pass 1  read -> detect/track players+pose, ball, court keypoints
          -> drop off-court people -> sample jersey colours
          (only small per-frame records are kept)
  post    ball cleanup, teams, possession, events, court positions, speed
  pass 2  read again -> draw overlays -> write video

Pass-1 results can be cached, so tuning any post-processing parameter
re-runs in seconds without touching a GPU.
"""
from __future__ import annotations

import os
import pickle
from typing import List, Optional

import numpy as np

from . import ball as ball_mod
from . import court
from .config import Config
from .events import detect_events
from .kinematics import court_tracks, speed_and_distance
from .possession import detect_possession
from .teams import JerseySampler, assign_teams
from .timing import StageTimer
from .types import Analysis, FrameObs
from .video import VideoWriter, iter_frames, probe


def _frame_homography(cfg: Config, static_H, obs: FrameObs):
    if static_H is not None:
        return static_H
    if obs.court_keypoints is not None and cfg.court.keypoint_names:
        return court.homography_from_keypoints(obs.court_keypoints, cfg.court.keypoint_names,
                                               cfg.models.court_keypoint_conf)
    return None


def _drop_off_court(obs: FrameObs, H, margin: float) -> None:
    if H is None or not obs.players:
        return
    ids = list(obs.players)
    xy = court.to_court(H, np.stack([obs.players[t].foot for t in ids]))
    for tid, p in zip(ids, xy):
        if not court.on_court(p, margin):
            del obs.players[tid]


def detect_pass(video: str, cfg: Config, timer: StageTimer, max_frames: Optional[int] = None,
                progress: bool = True):
    from .detection import Detector

    meta = probe(video)
    static_H = court.homography_from_calibration(cfg.court.calibration) if cfg.court.calibration else None
    detector = Detector(cfg.models)
    sampler = JerseySampler(cfg.teams)
    frames: List[FrameObs] = []
    raw_H = []
    total = min(meta.num_frames, max_frames) if max_frames else meta.num_frames

    for i, frame in iter_frames(video, max_frames):
        with timer.stage("detect + track"):
            obs = detector(i, frame)
        with timer.stage("court filter + jerseys"):
            H = _frame_homography(cfg, static_H, obs)
            if cfg.court.filter_off_court:
                _drop_off_court(obs, H, cfg.court.off_court_margin_m)
            sampler.update(i, frame, obs.players)
        frames.append(obs)
        raw_H.append(H)
        if progress and (i % 50 == 0 or i == total - 1):
            print(f"\r  detecting {i + 1}/{total}", end="", flush=True)
    if progress:
        print()
    meta.num_frames = len(frames)
    return meta, frames, dict(sampler.samples), raw_H


def post_process(meta, frames, samples, raw_H, cfg: Config, timer: StageTimer) -> Analysis:
    n = len(frames)
    with timer.stage("ball cleanup"):
        ball_centers = ball_mod.clean_ball_track([f.ball_candidates for f in frames], cfg.ball)
    with timer.stage("team assignment"):
        teams, team_colors = assign_teams(samples, cfg.teams.min_samples)
    with timer.stage("possession + events"):
        possession = detect_possession(frames, ball_centers, cfg.possession)
        events = detect_events(possession, teams, cfg.events.max_pass_frames)
    with timer.stage("court positions + speed"):
        homographies = court.fill_homographies(raw_H)
        xy = court_tracks(frames, homographies, n)
        speed, dist = {}, {}
        for tid, track in xy.items():
            speed[tid], dist[tid] = speed_and_distance(track, meta.fps, cfg.kinematics.smooth_window,
                                                       cfg.kinematics.max_speed_kmh)
    return Analysis(meta, frames, ball_centers, teams, team_colors, possession, events,
                    xy, speed, dist, homographies)


def analyze(video: str, cfg: Config, timer: Optional[StageTimer] = None, cache: Optional[str] = None,
            max_frames: Optional[int] = None) -> Analysis:
    timer = timer or StageTimer(enabled=False)
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            meta, frames, samples, raw_H = pickle.load(f)
        if max_frames:
            frames, raw_H = frames[:max_frames], raw_H[:max_frames]
            meta.num_frames = len(frames)
        print(f"  loaded detections from {cache}")
    else:
        meta, frames, samples, raw_H = detect_pass(video, cfg, timer, max_frames)
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump((meta, frames, samples, raw_H), f)
    return post_process(meta, frames, samples, raw_H, cfg, timer)


def render(analysis: Analysis, out_path: str, cfg: Config, timer: Optional[StageTimer] = None) -> None:
    from .render import Renderer

    timer = timer or StageTimer(enabled=False)
    renderer = Renderer(analysis, cfg.render)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    m = analysis.meta
    with VideoWriter(out_path, m.fps, (m.width, m.height)) as w:
        for i, frame in iter_frames(m.path, len(analysis.frames)):
            with timer.stage("render"):
                w.write(renderer.draw(frame, i))
