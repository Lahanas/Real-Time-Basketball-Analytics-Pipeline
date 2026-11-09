"""Post-processing, stats, rendering and evaluation on a synthetic clip (no models needed)."""
import numpy as np

from courtvision.config import Config
from courtvision.pipeline import post_process
from courtvision.render import Renderer
from courtvision.stats import summarize
from courtvision.types import VideoMeta
from evaluation.evaluate_possession import evaluate

from helpers import frame, player


def _synthetic():
    """Player 1 (team A) passes to player 2 (team A); player 3 (team B) steals it."""
    n, fps = 60, 30.0
    frames = []
    for i in range(n):
        a = player(1, [100, 200, 160, 380], wrist=(150, 280))
        b = player(2, [400, 200, 460, 380], wrist=(450, 280))
        c = player(3, [700, 200, 760, 380], wrist=(750, 280))
        ball = (152, 282) if i < 20 else (452, 282) if i < 40 else (752, 282)
        frames.append(frame(i, a, b, c, ball=ball))
    samples = {1: [np.array([230, 128, 128], np.float32)] * 4,
               2: [np.array([228, 128, 128], np.float32)] * 4,
               3: [np.array([60, 150, 100], np.float32)] * 4}
    # identity-ish homography: pixels / 40 = metres
    H = np.diag([1 / 40, 1 / 40, 1.0])
    meta = VideoMeta("synthetic.mp4", fps, 1280, 720, n)
    cfg = Config()
    cfg.ball.max_speed_px = 1000
    return post_process(meta, frames, samples, [H] * n, cfg, timer=_NoTimer()), cfg


class _NoTimer:
    from contextlib import nullcontext

    def stage(self, _):
        return self.nullcontext()


def test_pipeline_post_processing_and_stats():
    a, _ = _synthetic()
    assert a.teams[1] == a.teams[2] != a.teams[3]
    assert [(e.kind, e.from_player, e.to_player) for e in a.events] == [("pass", 1, 2), ("interception", 2, 3)]
    s = summarize(a, min_seconds_visible=0.5)
    by_id = {p["track_id"]: p for p in s["players"]}
    assert by_id[1]["passes_made"] == 1 and by_id[2]["passes_received"] == 1
    assert by_id[3]["interceptions"] == 1
    assert abs(sum(t["possession_pct"] for t in s["teams"]) - 100) < 0.2
    assert by_id[1]["distance_m"] == 0.0  # standing still


def test_renderer_draws_every_frame():
    a, cfg = _synthetic()
    r = Renderer(a, cfg.render)
    blank = np.zeros((720, 1280, 3), np.uint8)
    for i in (0, 20, 41, 59):
        out = r.draw(blank, i)
        assert out.shape == blank.shape and out.any()


def test_evaluation_metrics():
    N = -1
    gt = [4, 4, 4, N, N, N, 9, 9]
    pred = [4, 4, 4, 7, 7, 7, 9, 9]   # possession called mid-pass
    r = evaluate(pred, gt)
    assert r["false_pos"] == 1 and r["missed"] == 0 and r["correct"] == 5
    assert evaluate(gt, gt)["false_pos"] == 0
