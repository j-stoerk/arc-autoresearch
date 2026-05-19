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

    # ------------------------------------------------------------------ #
    # State-key helpers (used by BFS in Search)                            #
    # ------------------------------------------------------------------ #

    def local_state_key(self, raw_env) -> tuple:
        """Position+tag state key for keyboard BFS."""
        game = getattr(raw_env, "_game", None)
        if game is None:
            return (id(raw_env),)

        parts = [getattr(game, "_current_level_index", 0)]
        level = getattr(game, "current_level", None)
        if level is not None:
            for s in getattr(level, "_sprites", []):
                tags = tuple(sorted(getattr(s, "tags", [])))
                parts.append((int(getattr(s, "_x", 0)), int(getattr(s, "_y", 0)), tags))
        else:
            for v in vars(game).values():
                if hasattr(v, "pixels") and hasattr(v, "_x") and hasattr(v, "_y"):
                    tags = tuple(sorted(getattr(v, "tags", [])))
                    parts.append((int(getattr(v, "_x", 0)), int(getattr(v, "_y", 0)), tags))
        return tuple(parts)

    def full_state_key(self, raw_env) -> tuple:
        """Position + pixel-hash state key — captures both movement and pixel-only changes.

        Used for games where an action changes pixel content without moving sprites
        (e.g. selection cycling), which local_state_key misses.
        """
        import numpy as np
        game = getattr(raw_env, "_game", None)
        if game is None:
            return (id(raw_env),)
        parts = [getattr(game, "_current_level_index", 0)]
        level = getattr(game, "current_level", None)
        if level is not None:
            for s in getattr(level, "_sprites", []):
                px = getattr(s, "pixels", None)
                ph = hash(np.asarray(px, dtype=np.int32).tobytes()) if px is not None else 0
                parts.append((int(getattr(s, "_x", 0)), int(getattr(s, "_y", 0)), ph))
        return tuple(parts)

    # ------------------------------------------------------------------ #
    # LeCun JEPA-inspired compressed state representation                  #
    # ------------------------------------------------------------------ #

    def filtered_local_state_key(self, raw_env, relevant_indices: list[int]) -> tuple:
        """State key using only the dynamically relevant sprites (learned encoder).

        Compresses the state representation to the minimal set of sprite indices
        that actually change during exploration — analogous to JEPA's latent
        representation that discards irrelevant scene background.
        """
        game = getattr(raw_env, "_game", None)
        if game is None:
            return (id(raw_env),)
        parts: list = [getattr(game, "_current_level_index", 0)]
        level = getattr(game, "current_level", None)
        sprites = getattr(level, "_sprites", []) if level else []
        for i in relevant_indices:
            if i < len(sprites):
                s = sprites[i]
                tags = tuple(sorted(getattr(s, "tags", [])))
                parts.append((int(getattr(s, "_x", 0)), int(getattr(s, "_y", 0)), tags))
        return tuple(parts)

    @staticmethod
    def infer_relevant_sprites(observed_keys: list[tuple]) -> list[int]:
        """From a list of observed BFS state keys, find which sprite indices vary.

        The world model LEARNS which dimensions of the state key are informative
        by observing which ones change across transitions — self-supervised
        representation learning from the game's own dynamics.
        Returns sorted list of sprite indices (0-based into level._sprites).
        """
        if len(observed_keys) < 2:
            return []
        # Each key = (level_index, sprite_0_tuple, sprite_1_tuple, ...)
        n_sprites = max((len(k) - 1 for k in observed_keys), default=0)
        if n_sprites == 0:
            return []
        relevant = []
        for i in range(n_sprites):
            pos = i + 1  # position in key tuple
            vals: set = set()
            for k in observed_keys:
                if pos < len(k):
                    vals.add(k[pos])
            if len(vals) > 1:
                relevant.append(i)
        return relevant

    def click_state_key(self, raw_env) -> tuple:
        """Pixel-hash state key for click BFS."""
        game = getattr(raw_env, "_game", None)
        if game is None:
            return ()
        level = getattr(game, "current_level", None)
        if level is None:
            return (getattr(game, "_current_level_index", 0),)
        parts = [getattr(game, "_current_level_index", 0)]
        for s in getattr(level, "_sprites", []):
            px = getattr(s, "pixels", None)
            if px is not None:
                parts.append((s._x, s._y, hash(np.asarray(px, dtype=np.int32).tobytes())))
        return tuple(parts)

    def camera_scale(self, raw_env) -> int:
        """Detect display-to-grid scale factor."""
        cam = getattr(getattr(raw_env, "_game", None), "camera", None)
        if cam is None:
            return 1
        try:
            result = cam.display_to_grid(4, 4)
            if result and result[0] > 0:
                return 4 // result[0]
        except Exception:
            pass
        return 1
