"""Court geometry, image -> court homography, and the minimap drawing.

Court coordinates are metres on an NBA court: x runs baseline to baseline
(0 .. 28.65), y runs sideline to sideline (0 .. 15.24), origin at the top-left
corner as seen on the minimap.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

COURT_LENGTH = 28.65
COURT_WIDTH = 15.24
HOOP_FROM_BASELINE = 1.575
PAINT_WIDTH = 4.88
PAINT_LENGTH = 5.79
FT_CIRCLE_R = 1.83
CENTER_CIRCLE_R = 1.83
THREE_R = 7.24
THREE_CORNER_FROM_SIDE = 0.91

_mid_y = COURT_WIDTH / 2
_paint_top = _mid_y - PAINT_WIDTH / 2
_paint_bot = _mid_y + PAINT_WIDTH / 2
_three_break_x = HOOP_FROM_BASELINE + float(np.sqrt(THREE_R ** 2 - (_mid_y - THREE_CORNER_FROM_SIDE) ** 2))


def _mirror(x: float) -> float:
    return COURT_LENGTH - x


# Named landmarks used for calibration and for court-keypoint models.
LANDMARKS: Dict[str, tuple] = {
    "corner_top_left": (0.0, 0.0),
    "corner_bottom_left": (0.0, COURT_WIDTH),
    "corner_top_right": (COURT_LENGTH, 0.0),
    "corner_bottom_right": (COURT_LENGTH, COURT_WIDTH),
    "halfcourt_top": (COURT_LENGTH / 2, 0.0),
    "halfcourt_bottom": (COURT_LENGTH / 2, COURT_WIDTH),
    "center": (COURT_LENGTH / 2, _mid_y),
    "left_paint_baseline_top": (0.0, _paint_top),
    "left_paint_baseline_bottom": (0.0, _paint_bot),
    "left_ft_line_top": (PAINT_LENGTH, _paint_top),
    "left_ft_line_bottom": (PAINT_LENGTH, _paint_bot),
    "left_three_corner_top": (0.0, THREE_CORNER_FROM_SIDE),
    "left_three_corner_bottom": (0.0, COURT_WIDTH - THREE_CORNER_FROM_SIDE),
    "left_three_top_of_arc": (HOOP_FROM_BASELINE + THREE_R, _mid_y),
    "right_paint_baseline_top": (COURT_LENGTH, _paint_top),
    "right_paint_baseline_bottom": (COURT_LENGTH, _paint_bot),
    "right_ft_line_top": (_mirror(PAINT_LENGTH), _paint_top),
    "right_ft_line_bottom": (_mirror(PAINT_LENGTH), _paint_bot),
    "right_three_corner_top": (COURT_LENGTH, THREE_CORNER_FROM_SIDE),
    "right_three_corner_bottom": (COURT_LENGTH, COURT_WIDTH - THREE_CORNER_FROM_SIDE),
    "right_three_top_of_arc": (_mirror(HOOP_FROM_BASELINE + THREE_R), _mid_y),
}


# ---------------------------------------------------------------- homography

def estimate_homography(image_pts: np.ndarray, court_pts: np.ndarray) -> Optional[np.ndarray]:
    """Image pixels -> court metres. Needs >= 4 non-collinear correspondences."""
    if len(image_pts) < 4:
        return None
    method = cv2.RANSAC if len(image_pts) > 4 else 0
    H, _ = cv2.findHomography(image_pts.astype(np.float32), court_pts.astype(np.float32), method, 0.5)
    return H


def homography_from_calibration(path: str) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    names = [n for n in data["points"] if n in LANDMARKS]
    img = np.array([data["points"][n] for n in names], dtype=np.float32)
    court = np.array([LANDMARKS[n] for n in names], dtype=np.float32)
    H = estimate_homography(img, court)
    if H is None:
        raise ValueError(f"{path}: need at least 4 known landmarks, got {len(names)}")
    return H


def homography_from_keypoints(kps: np.ndarray, names: Sequence[str], min_conf: float) -> Optional[np.ndarray]:
    """Per-frame homography from a court keypoint model's output."""
    img, court = [], []
    for (x, y, c), name in zip(kps, names):
        if c >= min_conf and name in LANDMARKS:
            img.append((x, y))
            court.append(LANDMARKS[name])
    if len(img) < 4:
        return None
    return estimate_homography(np.array(img), np.array(court))


def fill_homographies(hs: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
    """Frames without enough keypoints reuse the nearest earlier (else later) homography."""
    out = list(hs)
    last = None
    for i, H in enumerate(out):
        if H is not None:
            last = H
        else:
            out[i] = last
    nxt = None
    for i in range(len(out) - 1, -1, -1):
        if hs[i] is not None:
            nxt = hs[i]
        elif out[i] is None:
            out[i] = nxt
    return out


def to_court(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def on_court(xy: np.ndarray, margin: float = 0.0) -> bool:
    x, y = xy
    return -margin <= x <= COURT_LENGTH + margin and -margin <= y <= COURT_WIDTH + margin


# ---------------------------------------------------------------- minimap

class Minimap:
    """Draws the court once, then maps metres to minimap pixels."""

    BG = (64, 116, 168)        # hardwood, BGR
    PAINT = (48, 92, 140)
    LINE = (240, 240, 240)

    def __init__(self, width_px: int = 360, pad: int = 10):
        self.pad = pad
        self.scale = (width_px - 2 * pad) / COURT_LENGTH
        self.width = width_px
        self.height = int(round(COURT_WIDTH * self.scale)) + 2 * pad
        self.base = self._draw()

    def px(self, x: float, y: float) -> tuple:
        return int(round(self.pad + x * self.scale)), int(round(self.pad + y * self.scale))

    def _draw(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), np.uint8)
        img[:] = self.BG
        s, t = self.scale, max(1, int(round(self.scale * 0.08)))
        L = self.LINE

        for side in (0, 1):
            fx = (lambda x: x) if side == 0 else _mirror
            # paint
            p1, p2 = self.px(fx(0), _paint_top), self.px(fx(PAINT_LENGTH), _paint_bot)
            cv2.rectangle(img, p1, p2, self.PAINT, -1)
            cv2.rectangle(img, p1, p2, L, t)
            # free-throw circle
            cv2.circle(img, self.px(fx(PAINT_LENGTH), _mid_y), int(FT_CIRCLE_R * s), L, t)
            # corner threes
            for y in (THREE_CORNER_FROM_SIDE, COURT_WIDTH - THREE_CORNER_FROM_SIDE):
                cv2.line(img, self.px(fx(0), y), self.px(fx(_three_break_x), y), L, t)
            # three-point arc
            hoop = self.px(fx(HOOP_FROM_BASELINE), _mid_y)
            half_angle = float(np.degrees(np.arcsin((_mid_y - THREE_CORNER_FROM_SIDE) / THREE_R)))
            start, end = (-half_angle, half_angle) if side == 0 else (180 - half_angle, 180 + half_angle)
            cv2.ellipse(img, hoop, (int(THREE_R * s), int(THREE_R * s)), 0, start, end, L, t)
            # hoop
            cv2.circle(img, hoop, max(2, int(0.23 * s)), (0, 110, 255), t)

        cv2.rectangle(img, self.px(0, 0), self.px(COURT_LENGTH, COURT_WIDTH), L, t)
        cv2.line(img, self.px(COURT_LENGTH / 2, 0), self.px(COURT_LENGTH / 2, COURT_WIDTH), L, t)
        cv2.circle(img, self.px(COURT_LENGTH / 2, _mid_y), int(CENTER_CIRCLE_R * s), L, t)
        return img
