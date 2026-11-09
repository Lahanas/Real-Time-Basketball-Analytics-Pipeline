"""Overlay drawing: players, ball, possession, HUD, event banners and minimap."""
from __future__ import annotations

import cv2
import numpy as np

from .config import RenderConfig
from .court import Minimap
from .stats import team_possession_counts
from .types import NO_PLAYER, Analysis

FONT = cv2.FONT_HERSHEY_SIMPLEX
UNKNOWN_COLOR = (160, 160, 160)
BALL_COLOR = (0, 215, 255)
HOLDER_COLOR = (40, 40, 230)


def _text(img, text, org, scale=0.5, color=(255, 255, 255), thick=1, bg=None, pad=4):
    (w, h), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad // 2), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _readable_on(color) -> tuple:
    b, g, r = color
    return (20, 20, 20) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (245, 245, 245)


class Renderer:
    def __init__(self, analysis: Analysis, cfg: RenderConfig):
        self.a = analysis
        self.cfg = cfg
        self.minimap = Minimap(cfg.minimap_width) if cfg.minimap and analysis.court_xy else None
        self.team_counts = team_possession_counts(analysis.possession, analysis.teams)
        self.banner_frames = int(cfg.event_banner_seconds * analysis.meta.fps)
        self.events_by_frame = {}
        for e in analysis.events:
            self.events_by_frame.setdefault(e.frame, e)
        self.passes = np.zeros((len(analysis.possession) + 1, 2), dtype=int)
        self.steals = np.zeros_like(self.passes)
        for e in analysis.events:
            if e.team in (0, 1):
                (self.passes if e.kind == "pass" else self.steals)[e.frame + 1:, e.team] += 1

    def team_color(self, tid: int):
        t = self.a.teams.get(tid, -1)
        return self.a.team_colors[t] if t in (0, 1) else UNKNOWN_COLOR

    def draw(self, frame: np.ndarray, i: int) -> np.ndarray:
        out = frame.copy()
        obs = self.a.frames[i]
        holder = self.a.possession[i] if i < len(self.a.possession) else NO_PLAYER

        for tid, p in obs.players.items():
            self._player(out, tid, p, tid == holder, i)

        bc = self.a.ball_centers[i]
        if not np.isnan(bc[0]):
            x, y = int(bc[0]), int(bc[1])
            cv2.circle(out, (x, y), 9, BALL_COLOR, 2, cv2.LINE_AA)

        self._hud(out, i)
        self._banner(out, i)
        if self.minimap is not None:
            self._minimap(out, i, holder)
        _text(out, f"frame {i}", (10, out.shape[0] - 10), 0.45, (230, 230, 230), bg=(0, 0, 0))
        return out

    def _player(self, img, tid, p, has_ball, i):
        color = self.team_color(tid)
        x1, y1, x2, y2 = (int(v) for v in p.bbox)
        cx, w = (x1 + x2) // 2, x2 - x1
        cv2.ellipse(img, (cx, y2), (max(w // 2, 8), max(w // 6, 3)), 0, -45, 235, color, 2, cv2.LINE_AA)
        if self.cfg.show_ids:
            _text(img, str(tid), (cx - 8, y2 + 18), 0.45, _readable_on(color), bg=color, pad=3)
        if has_ball:
            tri = np.array([[cx, y1 - 6], [cx - 9, y1 - 22], [cx + 9, y1 - 22]], np.int32)
            cv2.fillPoly(img, [tri], HOLDER_COLOR)
            cv2.polylines(img, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)
        if self.cfg.show_speed and tid in self.a.speed_kmh:
            s = self.a.speed_kmh[tid][i]
            d = self.a.distance_m[tid][i]
            if not np.isnan(s):
                _text(img, f"{s:.1f} km/h  {d:.0f} m", (cx - 40, y2 + 36), 0.4, (255, 255, 255), bg=(30, 30, 30), pad=2)

    def _hud(self, img, i):
        x, y, w = 14, 14, 300
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + 92), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        _text(img, "BALL CONTROL", (x + 10, y + 20), 0.45, (200, 200, 200))
        counts = self.team_counts[i] if i < len(self.team_counts) else np.zeros(2)
        total = counts.sum()
        frac0 = counts[0] / total if total else 0.5
        bx, by, bw, bh = x + 10, y + 30, w - 20, 16
        split = bx + int(bw * frac0)
        cv2.rectangle(img, (bx, by), (split, by + bh), self.a.team_colors[0], -1)
        cv2.rectangle(img, (split, by), (bx + bw, by + bh), self.a.team_colors[1], -1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (230, 230, 230), 1)
        if total:
            _text(img, f"{100 * frac0:.0f}%", (bx + 4, by + 13), 0.42, _readable_on(self.a.team_colors[0]))
            _text(img, f"{100 * (1 - frac0):.0f}%", (bx + bw - 38, by + 13), 0.42, _readable_on(self.a.team_colors[1]))
        p, s = self.passes[i], self.steals[i]
        _text(img, f"Passes  {p[0]} - {p[1]}", (x + 10, y + 66), 0.45)
        _text(img, f"Steals  {s[0]} - {s[1]}", (x + 160, y + 66), 0.45)

    def _banner(self, img, i):
        for f in range(i, max(-1, i - self.banner_frames), -1):
            e = self.events_by_frame.get(f)
            if e is None:
                continue
            label = "PASS" if e.kind == "pass" else "INTERCEPTION"
            text = f"{label}  #{e.from_player} -> #{e.to_player}"
            (tw, _), _ = cv2.getTextSize(text, FONT, 0.8, 2)
            color = self.a.team_colors[e.team] if e.team in (0, 1) else UNKNOWN_COLOR
            _text(img, text, ((img.shape[1] - tw) // 2, 46), 0.8, _readable_on(color), 2, bg=color, pad=10)
            return

    def _minimap(self, img, i, holder):
        mm = self.minimap.base.copy()
        for tid, xy in self.a.court_xy.items():
            if tid not in self.a.frames[i].players or np.isnan(xy[i, 0]):
                continue
            p = self.minimap.px(*xy[i])
            cv2.circle(mm, p, 6, self.team_color(tid), -1, cv2.LINE_AA)
            cv2.circle(mm, p, 6, (0, 0, 0), 1, cv2.LINE_AA)
            if tid == holder:
                cv2.circle(mm, p, 10, HOLDER_COLOR, 2, cv2.LINE_AA)
        h, w = mm.shape[:2]
        H, W = img.shape[:2]
        if h + 20 > H or w + 20 > W:
            return
        y0, x0 = H - h - 14, W - w - 14
        img[y0:y0 + h, x0:x0 + w] = mm
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + w, y0 + h), (240, 240, 240), 1)
