import numpy as np

from courtvision.detection import attach_poses, bbox_iou

from helpers import player


def test_iou():
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9


def test_each_skeleton_goes_to_one_player_best_match_first():
    players = {7: player(7, [0, 0, 100, 200]), 8: player(8, [10, 0, 110, 200])}
    pose_boxes = np.array([[2, 0, 102, 200], [500, 0, 600, 200]], dtype=float)
    kps = np.stack([np.full((17, 3), 1.0), np.full((17, 3), 2.0)])
    attach_poses(players, pose_boxes, kps, min_iou=0.3)
    assert players[7].keypoints is not None and players[7].keypoints[0, 0] == 1.0
    assert players[8].keypoints is None   # its only overlap was already taken by 7
