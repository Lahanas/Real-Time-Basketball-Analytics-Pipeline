"""Per-stage wall-clock timing, synchronised with the GPU so numbers are real."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict


class StageTimer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.times: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - t0

    def report(self, num_frames: int) -> str:
        total = sum(self.times.values()) or 1e-9
        rows = [f"{'stage':<26}{'total s':>9}{'ms/frame':>10}{'share':>8}"]
        for name, s in self.times.items():
            rows.append(f"{name:<26}{s:>9.2f}{1000 * s / num_frames:>10.1f}{100 * s / total:>7.1f}%")
        rows.append(f"{'TOTAL':<26}{total:>9.2f}{1000 * total / num_frames:>10.1f}")
        rows.append(f"Throughput: {num_frames / total:.1f} FPS over {num_frames} frames")
        return "\n".join(rows)


def _sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
