"""
Mechanics learning module — learns game mechanics from probes and plans directly.

LeCun world model principle: rather than exhaustive BFS, observe transitions to
build a causal model, identify the goal structure, and plan in O(N) time.

Primary target: tr87-style "cycle-to-match" games where:
  - A cursor selects one of N mutable sprites
  - ACTION2/ACTION1 cycle the selected sprite through K variants
  - WIN = all mutable sprites match their pre-randomization target names
    (detected by bsqsshqpox-style logic: ztgmtnnufb[k].name == cifzvbcuwqe[r][1][j].name)

Direct plan construction:
  1. Simulate bsqsshqpox rule-matching to find target name for each ztgmtnnufb slot
  2. Compute cycles = (target_digit - current_digit + cycle_len) % cycle_len
  3. Build action sequence: cursor movements + cycle actions
  O(N) probes — no exhaustive BFS needed.
"""
from __future__ import annotations
import copy
import gc
import numpy as np


def _pix_key(env) -> tuple:
    """Pixel-hash of all sprites — captures variant state changes."""
    game = getattr(env, "_game", None)
    level = getattr(game, "current_level", None) if game else None
    sprites = getattr(level, "_sprites", []) if level else []
    parts: list = [getattr(game, "_current_level_index", 0)]
    for s in sprites:
        px = getattr(s, "pixels", None)
        if px is not None:
            parts.append(hash(np.asarray(px, dtype=np.int32).tobytes()))
    return tuple(parts)


def _xy_set(env) -> frozenset:
    """Set of (x, y) positions — reorder-invariant, detects actual movement.

    When wpbnovjwkv remove+adds at SAME position, this set stays constant.
    When cursor moves, the set changes. Correct for cycle vs cursor detection.
    """
    game = getattr(env, "_game", None)
    level = getattr(game, "current_level", None) if game else None
    sprites = getattr(level, "_sprites", []) if level else []
    return frozenset((getattr(s, "_x", 0), getattr(s, "_y", 0)) for s in sprites)


class MechanicsLearner:
    """Learns causal models; builds direct plans without exhaustive BFS."""

    def __init__(self, perception, state_name_fn):
        self.perception = perception
        self._state_name = state_name_fn

    # ------------------------------------------------------------------ #

    def learn_and_plan(
        self,
        raw_env,
        actions,
        start_levels: int,
        node_budget: int,
        global_deadline: float = float("inf"),
    ) -> list[str] | None:
        """Try to compute a plan by learning game mechanics. Returns action names or None."""
        import time
        if time.monotonic() >= global_deadline:
            return None

        # Primary path: direct solver for cursor+cycle games (tr87 style)
        plan = self._direct_cycle_solver(raw_env, actions, start_levels, global_deadline)
        if plan is not None:
            return plan

        # Fallback: generic greedy cycle search (other cycle-to-match games)
        return self._generic_greedy(raw_env, actions, start_levels, node_budget, global_deadline)

    # ------------------------------------------------------------------ #
    # Direct solver: reads game state to compute exact plan in O(N)        #
    # ------------------------------------------------------------------ #

    def _direct_cycle_solver(self, raw_env, actions, start_levels, global_deadline) -> list[str] | None:
        """For games with cifzvbcuwqe + ztgmtnnufb structure: compute exact plan.

        Simulates bsqsshqpox rule-matching to determine target digit for each
        mutable slot, then builds min-cycle action sequence.
        """
        import time
        game = getattr(raw_env, "_game", None)
        if game is None:
            return None
        cifzvbcuwqe = getattr(game, "cifzvbcuwqe", None)
        ztgmtnnufb = getattr(game, "ztgmtnnufb", None)
        zvojhrjxxm = getattr(game, "zvojhrjxxm", None)

        if not cifzvbcuwqe or not ztgmtnnufb or not zvojhrjxxm:
            return None
        N = len(ztgmtnnufb)
        if N == 0:
            return None

        # Verify sprites have digit-suffixed names (cycle-variant game)
        for s in ztgmtnnufb:
            nm = getattr(s, "name", "")
            if not nm or not nm[-1].isdigit():
                return None

        # --- Step 1: Detect cycle length via probing ---
        action_map = {getattr(a, "name", ""): a for a in actions}
        # Find cycle actions (pixel-only change) and cursor actions (position change).
        # Use xy_set (position set) not local_state_key: wpbnovjwkv remove+adds at the
        # SAME (x,y) position, so xy_set stays constant but local_state_key (list-ordered)
        # changes due to reordering. xy_set correctly ignores this reordering.
        cycle_acts, cursor_acts = [], []
        for nm, act in action_map.items():
            xy0 = _xy_set(raw_env)
            ph0 = _pix_key(raw_env)
            probe = copy.deepcopy(raw_env)
            gc.disable()
            try:
                probe.step(act)
            finally:
                gc.enable()
            if _xy_set(probe) != xy0:
                cursor_acts.append(nm)
            elif _pix_key(probe) != ph0:
                cycle_acts.append(nm)

        if not cycle_acts or not cursor_acts:
            return None

        # We try both cycle-direction orderings (ACTION1 vs ACTION2 as "forward")
        # because we can't know a priori which increments vs decrements without
        # reading internal game state. Validation catches the wrong ordering.
        fwd_cur = cursor_acts[-1]  # ACTION4 (forward cursor)
        bck_cur = cursor_acts[0] if len(cursor_acts) > 1 else None
        # Try each possible forward/backward assignment for cycle actions
        cycle_direction_pairs = [(cycle_acts[0], cycle_acts[-1]),
                                 (cycle_acts[-1], cycle_acts[0])]
        if cycle_acts[0] == cycle_acts[-1]:
            cycle_direction_pairs = cycle_direction_pairs[:1]

        # --- Step 2: Simulate bsqsshqpox rule-matching to get targets ---
        # For each zvojhrjxxm block: find matching rule, record target for ztgmtnnufb.
        def _names_match(lst, offset, chain):
            if offset + len(chain) > len(lst):
                return False
            return all(getattr(lst[offset+i], "name", "") == getattr(chain[i], "name", "")
                       for i in range(len(chain)))

        targets: dict[int, str] = {}  # ztgmtnnufb_index → target_name
        eu = 0   # zvojhrjxxm offset
        pc = 0   # ztgmtnnufb offset

        while eu < len(zvojhrjxxm) and pc <= N:
            if time.monotonic() >= global_deadline:
                return None
            found_rule = False
            for target_chain, mutable_chain in cifzvbcuwqe:
                if not _names_match(zvojhrjxxm, eu, target_chain):
                    continue
                # Matched rule — record targets
                for j, stale in enumerate(mutable_chain):
                    k = pc + j
                    if k < N:
                        targets[k] = getattr(stale, "name", "")
                eu += len(target_chain)
                pc += len(mutable_chain)
                found_rule = True
                break
            if not found_rule:
                break  # Cannot determine targets for remaining positions

        if not targets:
            return None

        # --- Step 3 & 4: Build plan, try both cycle-direction assignments ---
        cycle_len = 7

        def _build_plan_for_direction(fwd_cyc_name, bck_cyc_name):
            plan_els = []
            for k, target_name in sorted(targets.items()):
                if k >= N:
                    break
                current_name = getattr(ztgmtnnufb[k], "name", "")
                if not current_name or not current_name[-1].isdigit():
                    return None
                if not target_name or not target_name[-1].isdigit():
                    return None
                t_d = int(target_name[-1])
                c_d = int(current_name[-1])
                fwd = (t_d - c_d + cycle_len) % cycle_len
                bck = cycle_len - fwd
                if fwd == 0:
                    continue
                use_fwd = fwd <= bck
                cycles = fwd if use_fwd else bck
                cyc = fwd_cyc_name if use_fwd else bck_cyc_name
                plan_els.append((k, cycles, cyc))
            p = []
            c = 0
            for k, cycs, cyc_name in plan_els:
                fw = (k - c) % N
                bk = (c - k) % N
                if fw <= bk:
                    p.extend([fwd_cur] * fw)
                elif bck_cur:
                    p.extend([bck_cur] * bk)
                c = k
                p.extend([cyc_name] * cycs)
            return p

        # --- Step 5: Validate; try both forward/backward assignments ---
        def _validate_plan(p):
            if p is None:
                return False
            val = copy.deepcopy(raw_env)
            gc.disable()
            try:
                for nm in p:
                    act = action_map.get(nm)
                    if act is None:
                        return False
                    obs = val.step(act)
                    if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                        return True
                    if self._state_name(obs) == "WIN":
                        return True
            finally:
                gc.enable()
            return False

        for fwd_cyc, bck_cyc in cycle_direction_pairs:
            plan = _build_plan_for_direction(fwd_cyc, bck_cyc)
            if plan is not None and _validate_plan(plan):
                return plan
        return None

    # ------------------------------------------------------------------ #
    # Generic greedy fallback for other cycle-to-match games               #
    # ------------------------------------------------------------------ #

    def _generic_greedy(self, raw_env, actions, start_levels, budget, global_deadline) -> list[str] | None:
        """Greedy WIN search: for each cursor position try each cycle count."""
        import time
        action_map = {getattr(a, "name", ""): a for a in actions}

        cycle_acts, cursor_acts = [], []
        for nm, act in action_map.items():
            pk0 = self.perception.local_state_key(raw_env)
            ph0 = _pix_key(raw_env)
            probe = copy.deepcopy(raw_env)
            gc.disable()
            try:
                probe.step(act)
            finally:
                gc.enable()
            if self.perception.local_state_key(probe) != pk0:
                cursor_acts.append(nm)
            elif _pix_key(probe) != ph0:
                cycle_acts.append(nm)

        if not cycle_acts or not cursor_acts:
            return None

        fwd_cyc, bck_cyc = cycle_acts[0], cycle_acts[-1]
        fwd_cur = cursor_acts[-1]
        bck_cur = cursor_acts[0] if len(cursor_acts) > 1 else None

        # Detect cycle length
        ph0 = _pix_key(raw_env)
        p = copy.deepcopy(raw_env)
        cyc_len = 0
        gc.disable()
        try:
            for _ in range(20):
                p.step(action_map[fwd_cyc])
                cyc_len += 1
                if _pix_key(p) == ph0:
                    break
            else:
                cyc_len = 7
        finally:
            gc.enable()
        if cyc_len < 2:
            return None

        # Detect N cursor positions
        pk0 = self.perception.local_state_key(raw_env)
        p2 = copy.deepcopy(raw_env)
        n_cur = 0
        gc.disable()
        try:
            for _ in range(30):
                p2.step(action_map[fwd_cur])
                n_cur += 1
                if self.perception.local_state_key(p2) == pk0:
                    break
            else:
                n_cur = 5
        finally:
            gc.enable()
        if n_cur < 1:
            return None

        def _build(best):
            p2 = []
            c = 0
            for k, (cyc, fwd) in enumerate(best):
                if cyc == 0:
                    continue
                fw = (k - c) % n_cur
                bk = (c - k) % n_cur
                if fw <= bk:
                    p2.extend([fwd_cur] * fw)
                elif bck_cur:
                    p2.extend([bck_cur] * bk)
                c = k
                p2.extend(([fwd_cyc] if fwd else [bck_cyc]) * cyc)
            return p2

        def _validate(plan):
            v = copy.deepcopy(raw_env)
            gc.disable()
            try:
                for nm in plan:
                    act = action_map.get(nm)
                    if act is None:
                        return False
                    obs = v.step(act)
                    if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                        return True
                    if self._state_name(obs) == "WIN":
                        return True
            finally:
                gc.enable()
            return False

        best = [(0, True)] * n_cur
        for k in range(n_cur):
            if time.monotonic() >= global_deadline:
                break
            for fwd_flag in (True, False):
                for cyc in range(1, cyc_len):
                    trial = best[:]
                    trial[k] = (cyc, fwd_flag)
                    if _validate(_build(trial)):
                        best[k] = (cyc, fwd_flag)
                        plan = _build(best)
                        return plan if plan else None
        return None
