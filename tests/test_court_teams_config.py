import cv2
import numpy as np
import pytest

from courtvision import court
from courtvision.config import Config, TeamConfig
from courtvision.teams import JerseySampler, assign_teams, torso_color

from helpers import player


# ---------------------------------------------------------------- court

def _fake_camera():
    """A known projective transform court-metres -> image pixels."""
    src = np.float32([[0, 0], [28.65, 0], [28.65, 15.24], [0, 15.24]])
    dst = np.float32([[300, 200], [1600, 210], [1900, 900], [20, 880]])
    return cv2.getPerspectiveTransform(src, dst)


def test_homography_round_trip():
    M = _fake_camera()
    names = ["corner_top_left", "corner_top_right", "center", "left_ft_line_top",
             "right_ft_line_bottom", "corner_bottom_left"]
    court_pts = np.float32([court.LANDMARKS[n] for n in names])
    img_pts = cv2.perspectiveTransform(court_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
    H = court.estimate_homography(img_pts, court_pts)
    back = court.to_court(H, img_pts)
    assert np.abs(back - court_pts).max() < 1e-3


def test_keypoint_homography_skips_low_confidence():
    M = _fake_camera()
    names = ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left", "center"]
    pts = cv2.perspectiveTransform(np.float32([court.LANDMARKS[n] for n in names]).reshape(-1, 1, 2), M)
    kps = np.hstack([pts.reshape(-1, 2), np.ones((5, 1))])
    kps[4] = [5, 5, 0.1]  # wrong and low confidence: must be ignored
    H = court.homography_from_keypoints(kps, names, min_conf=0.5)
    assert np.allclose(court.to_court(H, pts.reshape(-1, 2)[:1]), [[0, 0]], atol=1e-3)
    assert court.homography_from_keypoints(kps[:3], names[:3], 0.5) is None


def test_fill_homographies():
    A, B = np.eye(3), 2 * np.eye(3)
    out = court.fill_homographies([None, A, None, None, B, None])
    assert out[0] is A and out[2] is A and out[3] is A and out[5] is B


def test_on_court():
    assert court.on_court((14, 7))
    assert not court.on_court((-3, 7), margin=1.0)
    assert court.on_court((-0.5, 7), margin=1.0)


def test_minimap_draws():
    m = court.Minimap(360)
    assert m.base.shape == (m.height, 360, 3)
    assert m.px(0, 0) == (m.pad, m.pad)


# ---------------------------------------------------------------- teams

def _jersey_frame(colors):
    """One 200x100 'player' per colour, side by side on a grey court."""
    img = np.full((200, 100 * len(colors), 3), 120, np.uint8)
    for i, c in enumerate(colors):
        img[40:90, 100 * i + 30:100 * i + 70] = c
    return img


def test_torso_color_reads_the_jersey():
    img = _jersey_frame([(0, 0, 200)])
    lab = torso_color(img, np.array([0, 0, 100, 200]))
    expected = cv2.cvtColor(np.uint8([[[0, 0, 200]]]), cv2.COLOR_BGR2LAB)[0, 0]
    assert np.allclose(lab, expected, atol=2)


def test_two_teams_by_colour_lighter_is_team_0():
    white, navy = (235, 235, 235), (90, 30, 20)
    colors = [white, navy, white, navy, white, navy]
    img = _jersey_frame(colors)
    sampler = JerseySampler(TeamConfig(sample_every=1, samples_per_track=5))
    players = {i: player(i, [100 * i, 0, 100 * i + 100, 200]) for i in range(6)}
    for f in range(5):
        sampler.update(f, img, players)
    teams, team_colors = assign_teams(sampler.samples)
    assert [teams[i] for i in range(6)] == [0, 1, 0, 1, 0, 1]
    assert sum(team_colors[0]) > sum(team_colors[1])


def test_one_bad_sample_is_outvoted():
    red, blue = np.array([60, 200, 170], np.float32), np.array([40, 140, 70], np.float32)
    samples = {1: [red] * 4 + [blue], 2: [blue] * 5, 3: [red] * 5}
    teams, _ = assign_teams(samples)
    assert teams[1] == teams[3] != teams[2]


# ---------------------------------------------------------------- config

def test_config_yaml_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("possession:\n  hand_radius: 0.3\nmodels:\n  imgsz: 640\n")
    cfg = Config.load(str(p))
    assert cfg.possession.hand_radius == 0.3 and cfg.models.imgsz == 640
    assert cfg.possession.switch_frames == 5  # untouched default


def test_config_rejects_typos(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("possession:\n  hand_raduis: 0.3\n")
    with pytest.raises(KeyError):
        Config.load(str(p))
