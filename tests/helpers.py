import numpy as np

from courtvision.types import FrameObs, PlayerObs


def skeleton(wrist=None, elbow=None, conf=0.9):
    """COCO-17 keypoints with only the right arm set."""
    k = np.zeros((17, 3))
    if wrist is not None:
        k[10] = [*wrist, conf]
    if elbow is not None:
        k[8] = [*elbow, conf]
    return k


def player(tid, bbox, wrist=None, elbow=None, conf=0.9):
    kps = skeleton(wrist, elbow, conf) if wrist is not None else None
    return PlayerObs(tid, np.array(bbox, dtype=float), 0.9, kps)


def frame(i, *players, ball=None):
    f = FrameObs(index=i, players={p.track_id: p for p in players})
    if ball is not None:
        x, y = ball
        f.ball_candidates = np.array([[x, y, 0.9]], dtype=float)
    return f


def cand(*points):
    """Ball candidates for one frame: cand((x, y, conf), ...) or None."""
    return np.array(points, dtype=float) if points else None
