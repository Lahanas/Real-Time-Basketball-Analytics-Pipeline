"""Passes and interceptions from the possession timeline.

Possession is collapsed into "spells" (contiguous frames held by one player,
ignoring short loose-ball gaps). Consecutive spells by different players are:
  - a pass, if both players are on the same team
  - an interception, if the ball changed teams
A change that takes longer than `max_pass_frames` (e.g. a shot, rebound or
dead ball in between) isn't counted as either.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .types import NO_PLAYER, Event
from .teams import UNKNOWN_TEAM


def possession_spells(possession: List[int]) -> List[Tuple[int, int, int]]:
    """(player, first_frame, last_frame) for each uninterrupted holding run."""
    spells = []
    for i, pid in enumerate(possession):
        if pid == NO_PLAYER:
            continue
        if spells and spells[-1][0] == pid and spells[-1][2] == i - 1:
            spells[-1] = (pid, spells[-1][1], i)
        else:
            spells.append((pid, i, i))
    return spells


def detect_events(possession: List[int], teams: Dict[int, int], max_pass_frames: int) -> List[Event]:
    events = []
    spells = possession_spells(possession)
    for (a, _, a_end), (b, b_start, _) in zip(spells, spells[1:]):
        if a == b or b_start - a_end > max_pass_frames:
            continue
        ta, tb = teams.get(a, UNKNOWN_TEAM), teams.get(b, UNKNOWN_TEAM)
        if UNKNOWN_TEAM in (ta, tb):
            continue
        kind = "pass" if ta == tb else "interception"
        events.append(Event(kind, b_start, a, b, tb))
    return events
