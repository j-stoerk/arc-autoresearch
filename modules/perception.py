"""
Step vi — Perception
Parse a raw grid observation into multi-level representations:
  - pixel:   raw (H, W) int array, values 0-9
  - objects: list of connected-component dicts
  - delta:   changed cells vs previous state (or None on first call)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Object:
    color: int
    cells: list[tuple[int, int]]   # (row, col) list

    @property
    def bbox(self):
        rows = [r for r, _ in self.cells]
        cols = [c for _, c in self.cells]
        return min(rows), min(cols), max(rows), max(cols)

    @property
    def size(self):
        return len(self.cells)


@dataclass
class StateRepr:
    grid:    np.ndarray              # raw (H, W) int grid
    objects: list[Object]            # connected components
    delta:   Optional[np.ndarray]    # changed mask vs previous (H, W) bool or None


def _connected_components(grid: np.ndarray) -> list[Object]:
    """4-connected flood-fill for each non-background (non-0) cell."""
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    objects: list[Object] = []

    for r in range(H):
        for c in range(W):
            if grid[r, c] == 0 or visited[r, c]:
                continue
            color = grid[r, c]
            cells: list[tuple[int, int]] = []
            stack = [(r, c)]
            while stack:
                nr, nc = stack.pop()
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if visited[nr, nc] or grid[nr, nc] != color:
                    continue
                visited[nr, nc] = True
                cells.append((nr, nc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((nr + dr, nc + dc))
            objects.append(Object(color=color, cells=cells))

    return objects


class Perception:
    def __init__(self):
        self._prev_grid: Optional[np.ndarray] = None

    def reset(self):
        self._prev_grid = None

    def parse(self, obs: np.ndarray) -> StateRepr:
        """Convert raw gymnasium observation to StateRepr."""
        grid = np.asarray(obs, dtype=np.int32)
        objects = _connected_components(grid)
        delta = None
        if self._prev_grid is not None and self._prev_grid.shape == grid.shape:
            delta = (grid != self._prev_grid)
        self._prev_grid = grid.copy()
        return StateRepr(grid=grid, objects=objects, delta=delta)
