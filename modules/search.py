"""
Step iii — Search
Neural-guided beam search over the conditioned DSL sub-language.
Barebones: uniform scoring (all ops equally likely). The agent should
replace score_ops() with a learned neural prior.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import gc
import heapq
import copy
import time
from collections import deque
import numpy as np

from modules.dsl import DSLConfig, Operation
from modules.library import Library, Macro


@dataclass
class Beam:
    ops:   list[int]    # action_id sequence so far
    score: float = 0.0  # log-probability (higher = better)


class Search:
    def __init__(self, dsl: DSLConfig, library: Library):
        self.dsl     = dsl
        self.library = library

    def score_ops(
        self,
        ops: list[Operation],
        state,           # StateRepr — ignored in barebones, used by neural prior
        goal,            # goal repr — ignored in barebones
    ) -> list[float]:
        """
        Return a score per operation (higher = more promising).
        Barebones: uniform. Replace with neural prior conditioned on (state, goal).
        """
        return [op.relevance for op in ops]

    def beam_search(
        self,
        active_tags: set[str],
        state,
        goal,
        max_depth: Optional[int] = None,
    ) -> list[list[int]]:
        """
        Return top-beam_width action sequences (shortest programs first).
        Expands the conditioned DSL ops plus active macro-ops.
        """
        depth      = max_depth or self.dsl.max_depth
        beam_width = self.dsl.beam_width

        # Seed beam with single-step candidates
        cond_ops   = self.dsl.conditioned_ops(active_tags)
        scores     = self.score_ops(cond_ops, state, goal)
        beams: list[Beam] = sorted(
            [Beam(ops=[op.action_id], score=s) for op, s in zip(cond_ops, scores)],
            key=lambda b: b.score,
            reverse=True,
        )[:beam_width]

        # Also seed with macro expansions
        for macro in self.library.active_macros(active_tags):
            macro_score = sum(scores[i] for i, op in enumerate(cond_ops)
                              if op.action_id in macro.ops[:1]) / max(1, len(macro.ops))
            beams.append(Beam(ops=macro.ops[:], score=macro_score))
        beams = sorted(beams, key=lambda b: b.score, reverse=True)[:beam_width]

        all_beams = beams[:]

        # Expand depth-first up to max_depth
        for _ in range(depth - 1):
            candidates: list[Beam] = []
            for beam in beams:
                new_scores = self.score_ops(cond_ops, state, goal)
                for op, s in zip(cond_ops, new_scores):
                    candidates.append(
                        Beam(ops=beam.ops + [op.action_id], score=beam.score + s)
                    )
            beams = sorted(candidates, key=lambda b: b.score, reverse=True)[:beam_width]
            all_beams.extend(beams)

        all_beams = sorted(all_beams, key=lambda b: (len(b.ops), -b.score))
        return [b.ops for b in all_beams[:beam_width]]

    # ------------------------------------------------------------------ #
    # BFS / A* search methods                                              #
    # ------------------------------------------------------------------ #

    def local_bfs_plan(
        self,
        raw_env,
        actions,
        start_levels: int,
        node_budget: int,
        perception,
        _state_name,
        global_deadline: float = float("inf"),
    ) -> tuple[list[str] | None, int, int]:
        """A* keyboard BFS (f = g + h, h = win_score - score).

        Returns (plan, nodes_explored, unique_states).
        plan is None if no solution found within budget.
        """
        max_depth = 40

        def _game_score(env):
            g = getattr(env, "_game", None)
            return int(getattr(g, "_score", 0) or 0) if g else 0

        def _win_score(env):
            g = getattr(env, "_game", None)
            return int(getattr(g, "_win_score", 1000) or 1000) if g else 1000

        ws = _win_score(raw_env)
        ctr = 0
        h0 = ws - _game_score(raw_env)
        heap = [(h0, 0, ctr, copy.deepcopy(raw_env), [])]
        best_g: dict = {}
        nodes = 0

        gc.disable()
        try:
            while heap and nodes < node_budget and time.monotonic() < global_deadline:
                f, g, _, current, plan = heapq.heappop(heap)
                nodes += 1
                key = perception.local_state_key(current)
                if key in best_g and best_g[key] <= g:
                    continue
                best_g[key] = g
                if len(plan) >= max_depth:
                    continue

                for action in actions:
                    try:
                        nxt = copy.deepcopy(current)
                        obs = nxt.step(action)
                    except Exception:
                        continue
                    new_plan = plan + [getattr(action, "name", str(action))]
                    obs_state = _state_name(obs)
                    if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                        return new_plan, nodes, len(best_g)
                    if obs_state == "WIN":
                        return new_plan, nodes, len(best_g)
                    if obs_state == "NOT_FINISHED":
                        new_g = g + 1
                        new_key = perception.local_state_key(nxt)
                        if new_key not in best_g or best_g[new_key] > new_g:
                            new_h = ws - _game_score(nxt)
                            ctr += 1
                            heapq.heappush(heap, (new_g + new_h, new_g, ctr, nxt, new_plan))
        finally:
            gc.enable()
        return None, nodes, len(best_g)

    def click_bfs_plan(
        self,
        raw_env,
        start_levels: int,
        perception,
        _state_name,
        max_nodes: int = 500,
        time_limit: float = 5.0,
    ) -> list[dict] | None:
        """Click BFS: sprite-center probe + optional grid probe."""
        scale = perception.camera_scale(raw_env)
        game = getattr(raw_env, "_game", None)
        if game is None:
            return None
        level = getattr(game, "current_level", None)
        if level is None:
            return None

        try:
            from arcengine.enums import GameAction as _GameAction
            click_action = _GameAction.ACTION6
        except ImportError:
            return None

        init_key = perception.click_state_key(raw_env)
        candidates: list[dict] = []
        seen_xy: set[tuple[int, int]] = set()

        # Phase 1: sprite-center probing — pre-filter to initially-active positions.
        gc.disable()
        try:
            for s in getattr(level, "_sprites", []):
                data = {"x": (int(getattr(s, "_x", 0)) + 1) * scale,
                        "y": (int(getattr(s, "_y", 0)) + 1) * scale}
                xy = (data["x"], data["y"])
                if xy in seen_xy:
                    continue
                seen_xy.add(xy)
                try:
                    probe = copy.deepcopy(raw_env)
                    probe.step(click_action, data=data)
                    if perception.click_state_key(probe) != init_key:
                        candidates.append(data)
                except Exception:
                    pass
        finally:
            gc.enable()

        # Phase 2a: grid-probe fallback (pre-filtered) — extends candidates when sprite
        # centers give < 3 hits (e.g. lp85, s5i5, r11l).
        used_grid_probe = False
        sprites = getattr(level, "_sprites", [])
        max_gx, max_gy = 64, 64
        for s in sprites:
            px_s = getattr(s, "pixels", None)
            sw = px_s.shape[1] if px_s is not None and hasattr(px_s, "shape") else 8
            sh = px_s.shape[0] if px_s is not None and hasattr(px_s, "shape") else 8
            sx_end = (int(getattr(s, "_x", 0)) + sw) * scale
            sy_end = (int(getattr(s, "_y", 0)) + sh) * scale
            max_gx = max(max_gx, sx_end + 4)
            max_gy = max(max_gy, sy_end + 4)
        max_gx = min(max_gx, 256)
        max_gy = min(max_gy, 256)

        if len(candidates) < 3:
            gc.disable()
            try:
                t_probe = time.monotonic()
                for dy in range(0, max_gy, 4):
                    if time.monotonic() - t_probe > 8.0:
                        break
                    for dx in range(0, max_gx, 4):
                        if time.monotonic() - t_probe > 8.0:
                            break
                        if (dx, dy) in seen_xy:
                            continue
                        data = {"x": dx, "y": dy}
                        try:
                            probe = copy.deepcopy(raw_env)
                            probe.step(click_action, data=data)
                            if perception.click_state_key(probe) != init_key:
                                candidates.append(data)
                                seen_xy.add((dx, dy))
                        except Exception:
                            pass
            finally:
                gc.enable()
            used_grid_probe = True

        if not candidates:
            return None

        effective_limit = 30.0 if used_grid_probe else time_limit

        queue = deque([(copy.deepcopy(raw_env), [])])
        seen = {init_key}
        nodes = 0
        deadline = time.monotonic() + effective_limit

        gc.disable()
        try:
            while queue and nodes < max_nodes and time.monotonic() < deadline:
                current, path = queue.popleft()
                nodes += 1
                for data in candidates:
                    try:
                        nxt = copy.deepcopy(current)
                        obs = nxt.step(click_action, data=data)
                    except Exception:
                        continue
                    new_path = path + [data]
                    obs_state = _state_name(obs)
                    if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                        return new_path
                    if obs_state == "WIN":
                        return new_path
                    if obs_state == "NOT_FINISHED":
                        k = perception.click_state_key(nxt)
                        if k not in seen:
                            seen.add(k)
                            queue.append((nxt, new_path))
        finally:
            gc.enable()

        return None

    def update_dsl_from_bfs(self, raw_env, actions, perception) -> None:
        """DSL feedback from BFS — feeds effective actions to WorldModel.

        Actions that change game state keep preconditions=[] (always included by
        conditioned_ops). Actions that do nothing get preconditions=["blocked"]
        which is never in active_tags, so they are excluded from beam-search.
        """
        for op in self.dsl.operations:
            op.preconditions = []
        init_key = perception.local_state_key(raw_env)
        for action in actions:
            name = getattr(action, "name", "")
            if not name.startswith("ACTION"):
                continue
            try:
                action_id = int(name[6:]) - 1   # ACTION1 → id=0
                if not (0 <= action_id < 7):
                    continue
                nxt = copy.deepcopy(raw_env)
                nxt.step(action)
                new_key = perception.local_state_key(nxt)
                if new_key == init_key:
                    for op in self.dsl.operations:
                        if op.action_id == action_id:
                            op.preconditions = ["blocked"]
            except Exception:
                pass
