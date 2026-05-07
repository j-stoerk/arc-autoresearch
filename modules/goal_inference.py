"""
Step i — Goal Inference
Infer goal state and decompose into ordered subgoals.
Barebones: compare current grid to the hypothesised goal grid cell-by-cell.
Subgoals are individual differing cells, ordered by distance from top-left.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from modules.perception import StateRepr


@dataclass
class Subgoal:
    row:    int
    col:    int
    target: int    # target color value


class GoalInference:
    def __init__(self):
        self.goal_grid: Optional[np.ndarray] = None

    def set_goal(self, goal_grid: np.ndarray) -> None:
        """Register the target grid (provided by the environment or inferred)."""
        self.goal_grid = np.asarray(goal_grid, dtype=np.int32)

    def subgoals(self, state: StateRepr) -> list[Subgoal]:
        """Return list of cells that differ from goal, ordered top-left first."""
        if self.goal_grid is None:
            return []
        diff = np.argwhere(state.grid != self.goal_grid)
        subgoals = [
            Subgoal(row=int(r), col=int(c), target=int(self.goal_grid[r, c]))
            for r, c in diff
        ]
        return subgoals

    def goal_reached(self, state: StateRepr) -> bool:
        if self.goal_grid is None:
            return False
        return bool(np.array_equal(state.grid, self.goal_grid))
