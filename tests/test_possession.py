import numpy as np

from courtvision.config import PossessionConfig
from courtvision.possession import PossessionSmoother, detect_possession, frame_candidate
from courtvision.types import NO_PLAYER as N

from helpers import frame, player


def test_overlapping_boxes_ball_goes_to_the_hand_not_the_bigger_box():
    cfg = PossessionConfig()
    attacker = player(1, [100, 100, 200, 400], wrist=(190, 250))
    defender = player(2, [80, 90, 230, 420], wrist=(90, 150))  # box fully contains the ball
    players = {1: attacker, 2: defender}
    ball = np.array([195.0, 252.0])
    assert frame_candidate(ball, players, cfg) == 1

    # Box-only mode (the evaluation baseline) can't tell them apart: the ball is inside
    # both boxes, both score 0, and the winner depends only on iteration order.
    box_only = PossessionConfig(use_pose=False)
    assert frame_candidate(ball, {2: defender, 1: attacker}, box_only) == 2
    assert frame_candidate(ball, {1: attacker, 2: defender}, box_only) == 1


def test_ball_away_from_every_hand_is_loose():
    cfg = PossessionConfig()
    players = {1: player(1, [100, 100, 200, 400], wrist=(190, 250))}
    assert frame_candidate(np.array([600.0, 100.0]), players, cfg) == N


def test_unknown_ball_position_is_loose():
    players = {1: player(1, [100, 100, 200, 400], wrist=(190, 250))}
    assert frame_candidate(np.array([np.nan, np.nan]), players, PossessionConfig()) == N


def test_distance_is_relative_to_body_height():
    cfg = PossessionConfig()
    near = {1: player(1, [0, 0, 150, 300], wrist=(120, 150))}
    far = {1: player(1, [0, 0, 50, 100], wrist=(40, 50))}
    # 15% of body height below the wrist in both cases
    assert frame_candidate(np.array([120.0, 195.0]), near, cfg) == 1
    assert frame_candidate(np.array([40.0, 65.0]), far, cfg) == 1


def test_palm_is_extended_past_the_wrist():
    cfg = PossessionConfig(hand_radius=0.03)
    p = {1: player(1, [0, 0, 100, 400], wrist=(60, 200), elbow=(20, 200))}  # forearm points right
    assert frame_candidate(np.array([70.0, 200.0]), p, cfg) == 1   # at the palm
    assert frame_candidate(np.array([50.0, 200.0]), p, cfg) == N   # behind the wrist


def test_low_confidence_wrist_falls_back_to_box():
    cfg = PossessionConfig()
    p = {1: player(1, [100, 100, 200, 400], wrist=(900, 900), conf=0.05)}
    assert frame_candidate(np.array([150.0, 250.0]), p, cfg) == 1


def test_flicker_is_ignored():
    assert PossessionSmoother(3, 2).smooth([4, 4, 4, 9, 4, 4]) == [4] * 6


def test_dribble_gap_keeps_possession():
    assert PossessionSmoother(2, 3).smooth([4, 4, N, N, N, 4]) == [4] * 6


def test_long_gap_releases_the_ball():
    assert PossessionSmoother(2, 2).smooth([4, 4, N, N, N, N]) == [4, 4, 4, 4, N, N]


def test_switch_is_backfilled():
    assert PossessionSmoother(3, 2, backfill=True).smooth([4, 4, 4, 9, 9, 9, 9]) == [4, 4, 4, 9, 9, 9, 9]


def test_switch_lags_without_backfill():
    assert PossessionSmoother(3, 2, backfill=False).smooth([4, 4, 4, 9, 9, 9, 9]) == [N, N, 4, 4, 4, 9, 9]


def test_detect_possession_end_to_end():
    cfg = PossessionConfig(switch_frames=2, max_gap_frames=2)
    a = player(1, [100, 100, 200, 400], wrist=(190, 250))
    b = player(2, [300, 100, 400, 400], wrist=(310, 250))
    frames = [frame(i, a, b) for i in range(6)]
    balls = np.array([[192, 252]] * 3 + [[312, 252]] * 3, dtype=float)
    assert detect_possession(frames, balls, cfg) == [1, 1, 1, 2, 2, 2]
