"""Streaming video IO: frames are yielded one at a time, never collected in a list."""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from .types import VideoMeta


def probe(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    meta = VideoMeta(
        path=path,
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def iter_frames(path: str, max_frames: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(path)
    i = 0
    try:
        while max_frames is None or i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()


class VideoWriter:
    def __init__(self, path: str, fps: float, size: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if path.lower().endswith(".mp4") else "XVID"))
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        if not self._writer.isOpened():
            raise IOError(f"Cannot write video: {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
