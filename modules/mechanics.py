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

        # Secondary path: direct solver for piece-placement games (re86 style)
        plan = self._piece_placement_solver(raw_env, actions, start_levels, global_deadline)
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
    # Piece-placement direct solver (re86 style)                           #
    # ------------------------------------------------------------------ #

    def _piece_placement_solver(self, raw_env, actions, start_levels, global_deadline) -> list[str] | None:
        """Direct solver for games where a selector action cycles which piece is active,
        and movement actions (ACTION1-4) position the active piece on target canvas.

        Algorithm:
        1. Classify actions: selector (pix-only change) vs mover (xy change).
        2. Identify step size and direction→action mapping.
        3. Build target canvas from non-moving sprites with meaningful pixels.
        4. For each piece (cycling through selections): template-match against target.
        5. Build plan: move current piece to target, then switch to next.
        6. Validate on deepcopy.
        """
        import time
        import numpy as np

        if time.monotonic() >= global_deadline:
            return None

        action_map = {getattr(a, "name", ""): a for a in actions}
        game = getattr(raw_env, "_game", None)
        if game is None:
            return None
        level = getattr(game, "current_level", None)
        if level is None:
            return None

        # Step 1: classify actions
        selector_acts = []   # pix changes, xy same
        mover_acts: list[tuple[str, int, int]] = []  # (name, dx, dy)

        xy0 = _xy_set(raw_env)
        ph0 = _pix_key(raw_env)

        for nm, act in action_map.items():
            probe = copy.deepcopy(raw_env)
            gc.disable()
            try:
                probe.step(act)
            finally:
                gc.enable()
            xy1 = _xy_set(probe)
            ph1 = _pix_key(probe)
            if xy1 != xy0:
                # Position change: find which sprite moved and by how much
                sprites0 = {(getattr(s, "_x", 0), getattr(s, "_y", 0)) for s in getattr(level, "_sprites", [])}
                level1 = getattr(probe._game, "current_level", None)
                sprites1 = {(getattr(s, "_x", 0), getattr(s, "_y", 0)) for s in getattr(level1, "_sprites", [])} if level1 else set()
                moved = sprites1 - sprites0
                gone = sprites0 - sprites1
                if moved and gone:
                    # One sprite moved: delta = (new - old)
                    new_pos = next(iter(moved))
                    old_pos = next(iter(gone))
                    dx = new_pos[0] - old_pos[0]
                    dy = new_pos[1] - old_pos[1]
                    mover_acts.append((nm, dx, dy))
            elif ph1 != ph0:
                selector_acts.append(nm)

        if not selector_acts or not mover_acts:
            return None

        # Need at least 2 perpendicular mover directions
        directions = {(dx, dy) for _, dx, dy in mover_acts}
        if len(directions) < 2:
            return None

        # Step 2: find step size (all movers must use same step size)
        step_sizes = {max(abs(dx), abs(dy)) for _, dx, dy in mover_acts}
        if len(step_sizes) != 1:
            return None
        step = next(iter(step_sizes))
        if step == 0:
            return None

        # Build action→(dx,dy) lookup
        move_to_act: dict[tuple[int, int], str] = {(dx, dy): nm for nm, dx, dy in mover_acts}

        # Step 3: identify all sprite indices that ever move (across all selections).
        # Only sprites that NEVER move with any selection are "fixed target" sprites.
        all_sprites = list(getattr(level, "_sprites", []))
        if not all_sprites:
            return None

        first_mover_nm, first_dx, first_dy = mover_acts[0]

        # Probe with current selection to find first moved sprite
        test_probe = copy.deepcopy(raw_env)
        gc.disable()
        try:
            test_probe.step(action_map[first_mover_nm])
        finally:
            gc.enable()
        test_level = getattr(test_probe._game, "current_level", None)
        test_sprites = list(getattr(test_level, "_sprites", [])) if test_level else []

        moved_sprite_idx = None
        for i, (s0, s1) in enumerate(zip(all_sprites, test_sprites)):
            if getattr(s0, "_x", 0) != getattr(s1, "_x", 0) or getattr(s0, "_y", 0) != getattr(s1, "_y", 0):
                moved_sprite_idx = i
                break
        if moved_sprite_idx is None:
            return None

        # Cycle through all selections and collect all sprite indices that ever move
        ever_moved_indices: set[int] = {moved_sprite_idx}
        sel_nm = selector_acts[0]
        cycle_env = raw_env
        max_pieces = min(20, len(all_sprites))  # don't cycle forever
        for _ in range(max_pieces):
            if time.monotonic() >= global_deadline:
                return None
            next_sel = copy.deepcopy(cycle_env)
            gc.disable()
            try:
                next_sel.step(action_map[sel_nm])
            finally:
                gc.enable()
            cycle_env = next_sel
            # Probe movement with this selection
            move_probe = copy.deepcopy(cycle_env)
            gc.disable()
            try:
                move_probe.step(action_map[first_mover_nm])
            finally:
                gc.enable()
            cycle_level = getattr(cycle_env._game, "current_level", None)
            move_level = getattr(move_probe._game, "current_level", None)
            cycle_sprites = list(getattr(cycle_level, "_sprites", [])) if cycle_level else []
            move_sprites = list(getattr(move_level, "_sprites", [])) if move_level else []
            for i, (s0, s1) in enumerate(zip(cycle_sprites, move_sprites)):
                if getattr(s0, "_x", 0) != getattr(s1, "_x", 0) or getattr(s0, "_y", 0) != getattr(s1, "_y", 0):
                    if i in ever_moved_indices:
                        # Back to first piece — stop cycling
                        break
                    ever_moved_indices.add(i)
                    break
            else:
                break  # no sprite moved → this selection is a no-op, stop

        n_pieces = len(ever_moved_indices)

        # Step 4: build target canvas from sprites that NEVER move (fixed)
        # Use the initial test_sprites comparison to find which sprites are fixed.
        fixed_sprites = []
        for i, s in enumerate(all_sprites):
            if i in ever_moved_indices:
                continue  # this sprite is a piece — skip
            px = getattr(s, "pixels", None)
            if px is None:
                continue
            pxa = np.asarray(px, dtype=np.int32)
            if not np.any((pxa != -1) & (pxa != 4)):
                continue
            fixed_sprites.append(s)

        if not fixed_sprites:
            return None

        # Build expected canvas from fixed target sprites
        # target_canvas[r, c] = expected color at canvas position (r, c)
        canvas_size = 64
        target_canvas = np.full((canvas_size, canvas_size), -1, dtype=np.int32)
        for fs in fixed_sprites:
            fsx, fsy = getattr(fs, "_x", 0), getattr(fs, "_y", 0)
            fpx = np.asarray(getattr(fs, "pixels", None), dtype=np.int32)
            if fpx is None:
                continue
            fh, fw = fpx.shape
            for ri in range(fh):
                for ci in range(fw):
                    v = fpx[ri, ci]
                    if v != -1 and v != 4:
                        cr, cc = fsy + ri, fsx + ci
                        if 0 <= cr < canvas_size and 0 <= cc < canvas_size:
                            target_canvas[cr, cc] = v

        if not np.any(target_canvas != -1):
            return None

        # Step 5: for each piece (cycle through selections), template match against target
        piece_plans: list[tuple[int, int, int]] = []  # (selector_presses, dx_total, dy_total)

        cur_env = raw_env

        for sel_idx in range(n_pieces):  # exactly n_pieces iterations
            if time.monotonic() >= global_deadline:
                return None

            # Find current selected piece by probing a movement
            probe_move = copy.deepcopy(cur_env)
            sel_nm = selector_acts[0]
            gc.disable()
            try:
                probe_move.step(action_map[first_mover_nm])
            finally:
                gc.enable()
            probe_level = getattr(probe_move._game, "current_level", None)
            probe_sprites = list(getattr(probe_level, "_sprites", [])) if probe_level else []
            cur_sprites = list(getattr(getattr(cur_env._game, "current_level", None), "_sprites", []))

            # Find which sprite moved
            cur_moved_idx = None
            for i, (s0, s1) in enumerate(zip(cur_sprites, probe_sprites)):
                if getattr(s0, "_x", 0) != getattr(s1, "_x", 0) or getattr(s0, "_y", 0) != getattr(s1, "_y", 0):
                    cur_moved_idx = i
                    break
            if cur_moved_idx is None:
                break

            piece = cur_sprites[cur_moved_idx]
            px0 = getattr(piece, "_x", 0)
            py0 = getattr(piece, "_y", 0)
            ppix = np.asarray(getattr(piece, "pixels", None), dtype=np.int32)
            if ppix is None:
                break

            # Get piece's dominant color (the pieces' non-transparent pixels)
            piece_colors = ppix[ppix != -1]
            piece_colors_uniq = np.unique(piece_colors)
            # Use only the dominant non-zero color for matching
            piece_color = None
            for c in piece_colors_uniq:
                if c > 0:
                    piece_color = int(c)
                    break
            if piece_color is None:
                # All zeros — piece is transparent selection marker, skip
                # Advance to next selection
                next_probe = copy.deepcopy(cur_env)
                gc.disable()
                try:
                    next_probe.step(action_map[sel_nm])
                finally:
                    gc.enable()
                cur_env = next_probe
                if sel_idx > 0 and cur_moved_idx == moved_sprite_idx:
                    break  # cycled back to first piece
                continue

            # Template match: find (tx, ty) such that for all non-transparent piece pixels (ri,ci),
            # target_canvas[ty+ri, tx+ci] == piece_color
            piece_nontrans = np.argwhere(ppix == piece_color)
            if len(piece_nontrans) == 0:
                break

            target_check = np.argwhere(target_canvas == piece_color)
            if len(target_check) == 0:
                break

            # For each possible anchor: first piece pixel at each target position
            ph, pw = ppix.shape
            target_tx, target_ty = None, None
            for (tr, tc) in target_check:
                for (pr, pc) in piece_nontrans:
                    ty_cand = int(tr) - int(pr)
                    tx_cand = int(tc) - int(pc)
                    # Check ALL piece pixels match target at this offset
                    ok = True
                    for (ppr, ppc) in piece_nontrans:
                        cr, cc = ty_cand + ppr, tx_cand + ppc
                        if not (0 <= cr < canvas_size and 0 <= cc < canvas_size):
                            # allow off-canvas (clipping)
                            continue
                        if target_canvas[cr, cc] != -1 and target_canvas[cr, cc] != piece_color:
                            ok = False
                            break
                    if ok:
                        # Verify: all target pixels of this color ARE covered by piece
                        for (ttr, ttc) in target_check:
                            pr_idx = ttr - ty_cand
                            pc_idx = ttc - tx_cand
                            if 0 <= pr_idx < ph and 0 <= pc_idx < pw:
                                if ppix[pr_idx, pc_idx] != piece_color:
                                    ok = False
                                    break
                        if ok:
                            target_tx = tx_cand
                            target_ty = ty_cand
                            break
                if target_tx is not None:
                    break

            if target_tx is None or target_ty is None:
                break

            # Check reachability: (target - current) divisible by step
            dx_total = target_tx - px0
            dy_total = target_ty - py0
            if dx_total % step != 0 or dy_total % step != 0:
                break

            piece_plans.append((sel_idx, dx_total, dy_total))

            # Advance to next selection (not needed on last iteration)
            if sel_idx < n_pieces - 1:
                next_probe = copy.deepcopy(cur_env)
                gc.disable()
                try:
                    next_probe.step(action_map[sel_nm])
                finally:
                    gc.enable()
                cur_env = next_probe

        if not piece_plans:
            return None

        # Step 6: build plan
        def _move_actions(dx_total: int, dy_total: int) -> list[str] | None:
            """Convert total displacement into action name sequence."""
            moves: list[str] = []
            # x displacement
            if dx_total != 0:
                dx_sign = step if dx_total > 0 else -step
                act = move_to_act.get((dx_sign, 0))
                if act is None:
                    return None
                n = abs(dx_total) // step
                if n * step != abs(dx_total):
                    return None
                moves.extend([act] * n)
            # y displacement
            if dy_total != 0:
                dy_sign = step if dy_total > 0 else -step
                act = move_to_act.get((0, dy_sign))
                if act is None:
                    return None
                n = abs(dy_total) // step
                if n * step != abs(dy_total):
                    return None
                moves.extend([act] * n)
            return moves

        plan: list[str] = []
        sel_nm = selector_acts[0]
        for sel_idx, dx_total, dy_total in piece_plans:
            # sel_idx selector presses to advance to this piece
            plan.extend([sel_nm] * sel_idx)
            moves = _move_actions(dx_total, dy_total)
            if moves is None:
                return None
            plan.extend(moves)

        if not plan:
            return None

        # Step 7: validate
        val = copy.deepcopy(raw_env)
        gc.disable()
        try:
            for nm in plan:
                act = action_map.get(nm)
                if act is None:
                    return None
                obs = val.step(act)
                if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                    return plan
                if self._state_name(obs) == "WIN":
                    return plan
        finally:
            gc.enable()
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
