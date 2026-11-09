import numpy as np

from courtvision.ball import interpolate_gaps, track_ball
from courtvision.events import detect_events, possession_spells
from courtvision.kinematics import smooth_positions, speed_and_distance
from courtvision.types import NO_PLAYER as N

from helpers import cand

nan = np.nan


# ---------------------------------------------------------------- ball

def test_impossible_jump_is_rejected():
    c = [cand((0, 0, .9)), cand((10, 0, .9)), cand((500, 500, .9)), cand((20, 0, .9))]
    out = track_ball(c, max_speed_px=30)
    assert np.isnan(out[2]).all()
    assert np.allclose(out[3], [20, 0])


def test_picks_the_candidate_that_fits_the_motion_not_the_most_confident():
    c = [cand((100, 100, .8)), cand((900, 50, .95), (112, 104, .2))]  # a confident head far away
    out = track_ball(c, max_speed_px=40)
    assert np.allclose(out[1], [112, 104])


def test_first_detection_uses_confidence():
    out = track_ball([cand((5, 5, .1), (300, 300, .7))], max_speed_px=40)
    assert np.allclose(out[0], [300, 300])


def test_gate_widens_with_elapsed_frames():
    c = [cand((0, 0, .9)), None, None, cand((80, 0, .9))]
    assert not np.isnan(track_ball(c, max_speed_px=30)[3, 0])  # 80 px in 3 frames is fine


def test_reanchors_after_long_silence():
    c = [cand((0, 0, .9))] + [None] * 5 + [cand((900, 900, .9))]
    assert not np.isnan(track_ball(c, max_speed_px=10, reanchor_after=3)[-1, 0])


def test_short_gap_interpolated_long_gap_left():
    c = np.array([[0, 0], [nan, nan], [nan, nan], [30, 0], [nan, nan], [nan, nan], [nan, nan], [70, 0]])
    out = interpolate_gaps(c, max_gap=2)
    assert np.allclose(out[1:3], [[10, 0], [20, 0]])
    assert np.isnan(out[4:7]).all()


# ---------------------------------------------------------------- events

def test_spells():
    assert possession_spells([N, 4, 4, N, 4, 9]) == [(4, 1, 2), (4, 4, 4), (9, 5, 5)]


def test_pass_and_interception():
    poss = [1] * 5 + [N] * 3 + [2] * 5 + [N] * 2 + [7] * 5
    teams = {1: 0, 2: 0, 7: 1}
    ev = detect_events(poss, teams, max_pass_frames=10)
    assert [(e.kind, e.from_player, e.to_player, e.team) for e in ev] == [
        ("pass", 1, 2, 0), ("interception", 2, 7, 1)]
    assert ev[0].frame == 8


def test_slow_change_is_not_a_pass():
    poss = [1] * 5 + [N] * 50 + [2] * 5
    assert detect_events(poss, {1: 0, 2: 0}, max_pass_frames=45) == []


def test_unknown_team_skipped():
    assert detect_events([1, 1, 2, 2], {1: 0, 2: -1}, 10) == []


# ---------------------------------------------------------------- kinematics

def test_constant_speed():
    fps = 30
    t = np.arange(90) / fps
    xy = np.stack([5.0 * t, np.full_like(t, 7.0)], axis=1)  # 5 m/s along x
    speed, dist = speed_and_distance(xy, fps, window=5, max_speed_kmh=40)
    assert abs(np.nanmedian(speed) - 18.0) < 0.1      # 5 m/s = 18 km/h
    assert abs(dist[-1] - 5.0 * t[-1]) < 0.2


def test_projection_glitch_adds_no_distance():
    xy = np.zeros((20, 2))
    xy[10] = [15, 0]          # one-frame 15 m teleport
    _, dist = speed_and_distance(xy, 30, window=1, max_speed_kmh=36)
    assert dist[-1] == 0


def test_smoothing_ignores_gaps():
    xy = np.array([[0, 0], [nan, nan], [2, 0]], dtype=float)
    out = smooth_positions(xy, 3)
    assert np.isnan(out[1, 0]) and np.allclose(out[0], [0, 0])
