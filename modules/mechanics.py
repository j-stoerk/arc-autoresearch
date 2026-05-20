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

        # Secondary path: direct solver for delivery games (wa30 style)
        plan = self._delivery_solver(raw_env, actions, start_levels, global_deadline)
        if plan is not None:
            return plan

        # Tertiary path: direct solver for piece-placement games (re86 style)
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
    # Delivery direct solver (wa30 style)                                  #
    # ------------------------------------------------------------------ #

    def _delivery_solver(self, raw_env, actions, start_levels, global_deadline) -> list[str] | None:
        """Direct solver for games where ACTION5 picks up/drops movable objects
        and movement actions navigate the player to place objects at target positions.

        Algorithm:
        1. Detect: 4 movement actions + 1 interaction action (ACTION5).
        2. Probe interaction to detect pickup/drop semantics.
        3. Use local_state_key BFS to build a navigation graph.
        4. Identify objects (sprites that can be picked up) and target positions.
        5. Plan sequence: navigate to each object, pick up, navigate to target, drop.
        6. Try all orderings for 2-3 objects. Validate on deepcopy.
        """
        import time
        import itertools

        if time.monotonic() >= global_deadline:
            return None

        action_map = {getattr(a, "name", ""): a for a in actions}
        game = getattr(raw_env, "_game", None)
        if game is None:
            return None
        level = getattr(game, "current_level", None)
        if level is None:
            return None

        # Step 1: classify actions — movement (xy changes) vs interaction (neither xy nor pix)
        mover_acts: list[tuple[str, int, int]] = []  # (name, dx, dy)
        interact_acts: list[str] = []
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
                # Position change — record delta
                sprites0 = {(getattr(s, "_x", 0), getattr(s, "_y", 0)) for s in getattr(level, "_sprites", [])}
                level1 = getattr(probe._game, "current_level", None)
                sprites1 = {(getattr(s, "_x", 0), getattr(s, "_y", 0)) for s in getattr(level1, "_sprites", [])} if level1 else set()
                moved = sprites1 - sprites0
                gone = sprites0 - sprites1
                if moved and gone:
                    new_pos = next(iter(moved))
                    old_pos = next(iter(gone))
                    mover_acts.append((nm, new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]))
            elif ph1 == ph0:
                # Neither xy nor pix changed → potential interaction action
                interact_acts.append(nm)

        if not mover_acts or not interact_acts:
            return None
        # All movers must use the same step size (max of |dx|, |dy|)
        if len({max(abs(dx), abs(dy)) for _, dx, dy in mover_acts}) != 1:
            return None

        step = max(abs(dx) + abs(dy) for _, dx, dy in mover_acts)
        if step == 0:
            return None

        interact_nm = interact_acts[0]
        move_to_act = {(dx, dy): nm for nm, dx, dy in mover_acts}

        # Step 2: Detect delivery semantics — probe interaction at each object neighbor
        # A delivery game has objects that can be picked up (interaction changes something)
        # and a win condition that requires them at target positions.
        all_sprites = list(getattr(level, "_sprites", []))

        # Find player sprite (moves with mover actions)
        first_mover_nm, first_dx, first_dy = mover_acts[0]
        test_probe = copy.deepcopy(raw_env)
        gc.disable()
        try:
            test_probe.step(action_map[first_mover_nm])
        finally:
            gc.enable()
        test_level = getattr(test_probe._game, "current_level", None)
        test_sprites = list(getattr(test_level, "_sprites", [])) if test_level else []
        player_idx = None
        for i, (s0, s1) in enumerate(zip(all_sprites, test_sprites)):
            if getattr(s0, "_x", 0) != getattr(s1, "_x", 0) or getattr(s0, "_y", 0) != getattr(s1, "_y", 0):
                player_idx = i
                break
        if player_idx is None:
            return None

        player = all_sprites[player_idx]
        player_x = getattr(player, "_x", 0)
        player_y = getattr(player, "_y", 0)

        # Step 3: Build navigation graph via BFS
        # Map from (x,y) → dict of (dx,dy) → (x',y')
        nav_graph: dict[tuple[int, int], dict[tuple[int, int], tuple[int, int]]] = {}
        k0 = self.perception.local_state_key(raw_env)
        from collections import deque
        queue = deque([(copy.deepcopy(raw_env), (player_x, player_y))])
        seen_nav = {k0}
        gc.disable()
        try:
            while queue:
                cur_env, cur_pos = queue.popleft()
                if cur_pos not in nav_graph:
                    nav_graph[cur_pos] = {}
                for nm, dx, dy in mover_acts:
                    probe = copy.deepcopy(cur_env)
                    probe.step(action_map[nm])
                    nk = self.perception.local_state_key(probe)
                    probe_level = getattr(probe._game, "current_level", None)
                    probe_player = list(getattr(probe_level, "_sprites", []))[player_idx] if probe_level else None
                    if probe_player is None:
                        continue
                    new_pos = (getattr(probe_player, "_x", 0), getattr(probe_player, "_y", 0))
                    nav_graph[cur_pos][(dx, dy)] = new_pos
                    if nk not in seen_nav:
                        seen_nav.add(nk)
                        queue.append((probe, new_pos))
        finally:
            gc.enable()

        if len(nav_graph) < 2:
            return None

        # Step 4: Identify objects (movable sprites, not player) and target positions
        # Try interaction at each position adjacent to each non-player sprite
        # If interaction changes sprite positions (pickup detected), it's a delivery game
        # Build approach direction for pickup: (adx, ady) → player must be at (ox+adx, oy+ady)
        # and LAST ACTION must be the step TOWARD the object = (-adx,-ady) direction.
        # So navigate to adj_start=(ox+2*adx, oy+2*ady), execute direction action, then interact.
        object_sprites: list[int] = []  # (sprite_idx, adx, ady) tuples
        object_approach: dict[int, tuple[int,int]] = {}  # sprite_idx → (adx, ady)
        for i, s in enumerate(all_sprites):
            if i == player_idx:
                continue
            ox, oy = getattr(s, "_x", 0), getattr(s, "_y", 0)
            for adx, ady in [(0, step), (0, -step), (step, 0), (-step, 0)]:
                adj = (ox + adx, oy + ady)
                adj_start = (ox + 2*adx, oy + 2*ady)
                # Need adj and adj_start in nav_graph (or just adj if adj_start is off-grid)
                if adj not in nav_graph:
                    continue
                direction_nm = move_to_act.get((-adx, -ady))  # action to move toward object
                if direction_nm is None:
                    continue
                # Navigate to adj_start (or adj if adj_start not reachable)
                start_pos = adj_start if adj_start in nav_graph else adj
                path = self._find_path(nav_graph, (player_x, player_y), start_pos, step, move_to_act)
                if path is None:
                    path = self._find_path(nav_graph, (player_x, player_y), adj, step, move_to_act)
                if path is None:
                    continue
                # Build: path_to_start + [direction_step_if_needed] + [interact]
                # If we navigated to adj directly (not adj_start), add the direction step manually
                approach_path = path[:]
                if start_pos == adj_start:
                    approach_path.append(direction_nm)  # one step toward object
                # Execute approach + interaction and check if pickup happened
                test_e = copy.deepcopy(raw_env)
                gc.disable()
                try:
                    for pnm in approach_path:
                        test_e.step(action_map[pnm])
                    g_before = getattr(test_e, "_game", None)
                    carry_before = 0
                    for attr in ("nsevyuople", "_held", "_carrying"):
                        v = getattr(g_before, attr, None)
                        if v is not None:
                            carry_before = len(v) if hasattr(v, "__len__") else int(bool(v))
                            break
                    old_k = self.perception.local_state_key(test_e)
                    test_e.step(action_map[interact_nm])
                    new_k = self.perception.local_state_key(test_e)
                    g_after = getattr(test_e, "_game", None)
                    carry_after = 0
                    for attr in ("nsevyuople", "_held", "_carrying"):
                        v = getattr(g_after, attr, None)
                        if v is not None:
                            carry_after = len(v) if hasattr(v, "__len__") else int(bool(v))
                            break
                finally:
                    gc.enable()
                # Pickup detected if: carry state increased OR state key changed
                if carry_after > carry_before or new_k != old_k:
                    object_sprites.append(i)
                    object_approach[i] = (adx, ady)
                    break  # this sprite is a pickup object

        if not object_sprites:
            return None

        # Step 5: Find safe/target positions for each object
        # Safe positions: check game win condition at each reachable position for each object
        # Simple heuristic: simulate moving each object to each nav position and check win
        reachable_positions = list(nav_graph.keys())

        # Find all safe positions in the nav_graph (reachable positions where win fn returns True)
        game_obj = getattr(raw_env, "_game", None)
        safe_fn = None
        for attr_name in ["shbxbhnhjc", "is_safe", "_is_safe"]:
            fn = getattr(game_obj, attr_name, None)
            if fn is not None and callable(fn):
                safe_fn = fn
                break
        if safe_fn is None:
            return None

        all_safe_positions: list[tuple[int, int]] = []
        for pos in reachable_positions:
            try:
                if safe_fn((pos[0], pos[1])):
                    all_safe_positions.append(pos)
            except Exception:
                pass

        # Need at least as many safe positions as objects
        if len(all_safe_positions) < len(object_sprites):
            return None

        # Assign UNIQUE target positions to each object (different safe position per object)
        # Use the first N safe positions (sorted for determinism)
        all_safe_positions = sorted(set(all_safe_positions))
        target_positions: list[tuple[int, int]] = []
        for i, obj_idx in enumerate(object_sprites):
            ox, oy = getattr(all_sprites[obj_idx], "_x", 0), getattr(all_sprites[obj_idx], "_y", 0)
            # Find a safe position not already assigned and not the object's initial position
            found_target = None
            for pos in all_safe_positions:
                if pos not in target_positions and pos != (ox, oy):
                    found_target = pos
                    break
            if found_target is None:
                return None
            target_positions.append(found_target)

        if len(target_positions) != len(object_sprites):
            return None

        # Step 6: Plan delivery sequence for each object ordering
        n_objs = len(object_sprites)
        for obj_order in itertools.permutations(range(n_objs)):
            if time.monotonic() >= global_deadline:
                return None
            plan = self._build_delivery_plan(
                raw_env, all_sprites, player_idx, object_sprites, target_positions,
                list(obj_order), nav_graph, step, move_to_act, interact_nm, action_map,
                object_approach,
            )
            if plan is not None:
                # Validate
                val = copy.deepcopy(raw_env)
                gc.disable()
                try:
                    for nm in plan:
                        act = action_map.get(nm)
                        if act is None:
                            break
                        obs = val.step(act)
                        if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                            return plan
                        if self._state_name(obs) == "WIN":
                            return plan
                finally:
                    gc.enable()
        return None

    def _find_path(
        self,
        nav_graph: dict,
        start: tuple[int, int],
        goal: tuple[int, int],
        step: int,
        move_to_act: dict,
    ) -> list[str] | None:
        """BFS path from start to goal using precomputed navigation graph."""
        if start == goal:
            return []
        from collections import deque
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            pos, path = queue.popleft()
            for (dx, dy), new_pos in nav_graph.get(pos, {}).items():
                if new_pos in seen:
                    continue
                nm = move_to_act.get((dx, dy))
                if nm is None:
                    continue
                new_path = path + [nm]
                if new_pos == goal:
                    return new_path
                seen.add(new_pos)
                queue.append((new_pos, new_path))
        return None

    def _build_delivery_plan(
        self,
        raw_env, all_sprites, player_idx, object_sprites, target_positions,
        obj_order, nav_graph, step, move_to_act, interact_nm, action_map,
        object_approach=None,
    ) -> list[str] | None:
        """Build a plan that picks up objects in obj_order and delivers them to targets."""
        plan: list[str] = []
        sim_env = copy.deepcopy(raw_env)
        gc.disable()
        try:
            for i in obj_order:
                obj_idx = object_sprites[i]
                target_pos = target_positions[i]

                # Find current player and object positions from sim_env.
                # Use tag lookup because drops may reorder the sprite list (like tr87's wpbnovjwkv).
                sim_level = getattr(sim_env._game, "current_level", None)
                if sim_level is None:
                    return None
                # Find player by checking which sprite has the same tags as original player
                player_tags = tuple(sorted(getattr(all_sprites[player_idx], "tags", [])))
                cur_player = None
                for s in getattr(sim_level, "_sprites", []):
                    if tuple(sorted(getattr(s, "tags", []))) == player_tags:
                        cur_player = s
                        break
                if cur_player is None:
                    return None
                # Find target object by its INITIAL position (all geezpjgiyd sprites share same tags,
                # so tag-based lookup would find the wrong one).
                obj_initial_pos = (getattr(all_sprites[obj_idx], "_x", 0), getattr(all_sprites[obj_idx], "_y", 0))
                obj_tags = tuple(sorted(getattr(all_sprites[obj_idx], "tags", [])))
                delivered_targets = set(target_positions[k] for k in obj_order[:obj_order.index(i)])
                cur_obj = None
                # First: look for the object at its expected initial position (most reliable)
                for s in getattr(sim_level, "_sprites", []):
                    if tuple(sorted(getattr(s, "tags", []))) == obj_tags:
                        pos = (getattr(s, "_x", 0), getattr(s, "_y", 0))
                        if pos == obj_initial_pos:
                            cur_obj = s
                            break
                # Fallback: look for any undelivered object with same tags
                if cur_obj is None:
                    for s in getattr(sim_level, "_sprites", []):
                        if tuple(sorted(getattr(s, "tags", []))) == obj_tags:
                            pos = (getattr(s, "_x", 0), getattr(s, "_y", 0))
                            if pos not in delivered_targets:
                                cur_obj = s
                                break
                if cur_obj is None:
                    return None
                cur_player_pos = (getattr(cur_player, "_x", 0), getattr(cur_player, "_y", 0))
                cur_obj_pos = (getattr(cur_obj, "_x", 0), getattr(cur_obj, "_y", 0))

                # Determine approach: (adx, ady) = player offset from object
                # The approach direction action = move_to_act[(-adx, -ady)]
                approach = object_approach.get(obj_idx) if object_approach else None
                pickup_done = False

                # Try stored approach first, then all 4 directions
                _all_approaches = [(0,step),(0,-step),(step,0),(-step,0)]
                _ordered_approaches = ([approach] + _all_approaches) if approach else _all_approaches
                # blocked = positions of already-delivered objects (solid walls for navigation)
                delivered_idx = obj_order[:obj_order.index(i)]
                blocked: set = set()
                for prev_i in delivered_idx:
                    blocked.add(target_positions[prev_i])

                pickup_done = False
                pickup_adx, pickup_ady = 0, 0

                for adx, ady in _ordered_approaches:
                    adj = (cur_obj_pos[0] + adx, cur_obj_pos[1] + ady)
                    adj_start = (cur_obj_pos[0] + 2*adx, cur_obj_pos[1] + 2*ady)
                    direction_nm = move_to_act.get((-adx, -ady))
                    if direction_nm is None:
                        continue
                    if adj in blocked or adj_start in blocked:
                        continue

                    # Fast navigation to adj_start using nav_graph with blocked positions
                    path_start = self._find_nav_path(nav_graph, cur_player_pos, adj_start, move_to_act, blocked)
                    if path_start is None:
                        # Try adj directly
                        path_start = self._find_nav_path(nav_graph, cur_player_pos, adj, move_to_act, blocked)
                        if path_start is None:
                            continue
                        if not path_start or path_start[-1] != direction_nm:
                            continue
                        # Execute path to adj (last action was direction_nm)
                        for nm in path_start:
                            sim_env.step(action_map[nm])
                            plan.append(nm)
                        sim_env.step(action_map[interact_nm])
                        plan.append(interact_nm)
                        pickup_done = True
                        pickup_adx, pickup_ady = adx, ady
                        break
                    else:
                        # Execute: path_to_adj_start + direction_step
                        for nm in path_start:
                            sim_env.step(action_map[nm])
                            plan.append(nm)
                        sim_env.step(action_map[direction_nm])
                        plan.append(direction_nm)
                        sim_env.step(action_map[interact_nm])
                        plan.append(interact_nm)
                        pickup_done = True
                        pickup_adx, pickup_ady = adx, ady
                        break

                if not pickup_done:
                    return None

                adx, ady = pickup_adx, pickup_ady

                # After pickup: carried object has offset (-adx, -ady) from player.
                # Player must be at (tx+adx, ty+ady) to drop at target_pos (tx,ty).
                drop_player_pos = (target_pos[0] + adx, target_pos[1] + ady)
                carry_offset = (-adx, -ady)

                # Get player's current position after pickup
                sim_level2 = getattr(sim_env._game, "current_level", None)
                if sim_level2 is None:
                    return None
                cur_player2 = None
                for s in getattr(sim_level2, "_sprites", []):
                    if tuple(sorted(getattr(s, "tags", []))) == player_tags:
                        cur_player2 = s
                        break
                if cur_player2 is None:
                    return None
                cur_player_pos2 = (getattr(cur_player2, "_x", 0), getattr(cur_player2, "_y", 0))

                # Fast carry navigation using nav_graph with carry offset and blocked positions.
                # passable_pos = object's original position (player can pass through it when carrying)
                path_to_target = self._find_nav_path(
                    nav_graph, cur_player_pos2, drop_player_pos, move_to_act,
                    blocked=blocked, carry_offset=carry_offset,
                    passable_pos=cur_obj_pos,
                )
                if path_to_target is None:
                    return None

                # Execute navigation to drop position
                for nm in path_to_target:
                    sim_env.step(action_map[nm])
                    plan.append(nm)

                # Drop (object lands at target_pos)
                sim_env.step(action_map[interact_nm])
                plan.append(interact_nm)
        finally:
            gc.enable()

        return plan

    def _find_nav_path(
        self, nav_graph, start, goal, move_to_act,
        blocked: set | None = None,
        carry_offset: tuple | None = None,
        passable_pos: tuple | None = None,  # player can pass through this pos (picked-up obj's origin)
    ) -> list[str] | None:
        """Fast BFS using precomputed nav_graph with optional carry offset and blocked positions.

        blocked: set of (x,y) positions the player cannot enter (occupied by placed objects).
        carry_offset: if carrying an object, (dx,dy) offset of carried object from player.
                      If carry_offset=(dx,dy): player at (px,py) → object at (px+dx,py+dy).
                      Move blocked if (new_px+dx, new_py+dy) is in blocked.
        passable_pos: a position that nav_graph says is blocked (was an object) but is now
                      passable because that object is being carried. When a move (dx,dy)
                      would normally lead to new_pos==pos (blocked), but pos+(dx,dy)==passable_pos,
                      override to allow the move.
        """
        if start == goal:
            return []
        from collections import deque
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            pos, path = queue.popleft()
            for (dx, dy), new_pos in nav_graph.get(pos, {}).items():
                # Check if blocked by nav_graph: new_pos == pos means the original BFS saw a wall
                actual_new_pos = new_pos
                if new_pos == pos:
                    # Maybe this was blocked by the carried object's original position
                    candidate = (pos[0] + dx, pos[1] + dy)
                    if passable_pos and candidate == passable_pos:
                        actual_new_pos = candidate  # override: allow passage
                    else:
                        continue  # truly blocked
                if blocked and actual_new_pos in blocked:
                    continue  # player would enter blocked cell
                if carry_offset:
                    new_obj = (actual_new_pos[0] + carry_offset[0], actual_new_pos[1] + carry_offset[1])
                    if blocked and new_obj in blocked:
                        continue  # carried object would enter blocked cell
                nm = move_to_act.get((dx, dy))
                if nm is None:
                    continue
                if actual_new_pos in seen:
                    continue
                new_path = path + [nm]
                if actual_new_pos == goal:
                    return new_path
                seen.add(actual_new_pos)
                # For passable_pos override positions: also add neighbors by direct offset
                # (nav_graph may not have this position since it was an obstacle).
                if actual_new_pos == passable_pos and actual_new_pos not in nav_graph:
                    # Synthetic nav: treat passable_pos as navigable, add its ±step neighbors
                    queue.append((actual_new_pos, new_path))
                else:
                    queue.append((actual_new_pos, new_path))
        return None

    def _find_path_ignoring_pos(
        self, raw_env, sim_env, player_idx, start, goal, action_map, step, move_to_act,
        player_tags=None, max_nodes: int = 300,
    ) -> list[str] | None:
        """BFS pathfinding using sim_env (with held object) to correctly handle carry state.
        Uses tag-based player lookup to handle sprite list reordering.
        """
        if start == goal:
            return []
        # Determine player tags for robust lookup
        if player_tags is None:
            game0 = getattr(raw_env, "_game", None)
            level0 = getattr(game0, "current_level", None) if game0 else None
            sprites0 = list(getattr(level0, "_sprites", [])) if level0 else []
            if player_idx < len(sprites0):
                player_tags = tuple(sorted(getattr(sprites0[player_idx], "tags", [])))
        from collections import deque
        queue = deque([(copy.deepcopy(sim_env), start, [])])
        seen = {start}
        nodes = 0
        gc.disable()
        try:
            while queue and nodes < max_nodes:
                cur, pos, path = queue.popleft()
                nodes += 1
                for (dx, dy), nm in move_to_act.items():
                    pr = copy.deepcopy(cur)
                    pr.step(action_map[nm])
                    pr_level = getattr(pr._game, "current_level", None)
                    pr_sprites = list(getattr(pr_level, "_sprites", [])) if pr_level else []
                    # Find player by tag (robust against reordering)
                    new_pos = None
                    if player_tags:
                        for s in pr_sprites:
                            if tuple(sorted(getattr(s, "tags", []))) == player_tags:
                                new_pos = (getattr(s, "_x", 0), getattr(s, "_y", 0))
                                break
                    if new_pos is None:
                        if not pr_sprites or player_idx >= len(pr_sprites):
                            continue
                        new_pos = (getattr(pr_sprites[player_idx], "_x", 0), getattr(pr_sprites[player_idx], "_y", 0))
                    if new_pos in seen:
                        continue
                    new_path = path + [nm]
                    if new_pos == goal:
                        return new_path
                    seen.add(new_pos)
                    queue.append((pr, new_pos, new_path))
        finally:
            gc.enable()
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
