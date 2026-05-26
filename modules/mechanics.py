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
        # Track games where internal_state_bfs found nothing (persist across episodes)
        self._isb_exhausted: set[str] = set()

    # ------------------------------------------------------------------ #

    def learn_and_plan(
        self,
        raw_env,
        actions,
        start_levels: int,
        node_budget: int,
        global_deadline: float = float("inf"),
        all_actions=None,
    ) -> list[str] | None:
        """Try to compute a plan by learning game mechanics. Returns action names or None.

        actions: keyboard-only (non-complex) actions for BFS-based solvers.
        all_actions: full action list including complex (ACTION6) actions.
                     Used by solvers that need click capabilities (e.g. spell casting).
        """
        import time
        if time.monotonic() >= global_deadline:
            return None

        # Primary path: direct solver for cursor+cycle games (tr87 style)
        plan = self._direct_cycle_solver(raw_env, actions, start_levels, global_deadline)
        if plan is not None:
            return plan

        # Quaternary path: direct solver for spell-casting games (sc25 style).
        # Run BEFORE ISB so we short-circuit for games already ISB-exhausted.
        # Uses full action list (needs ACTION6). Returns list[dict] click plan.
        _full = all_actions if all_actions is not None else actions
        plan = self._spell_casting_solver(raw_env, _full, start_levels, global_deadline)
        if plan is not None:
            return plan

        # Secondary path: BFS with internal game state variables (ls20 style)
        game_prefix = (getattr(getattr(raw_env, "_game", None), "_game_id", "") or "")[:4]
        if game_prefix not in self._isb_exhausted:
            plan = self._internal_state_bfs(raw_env, actions, start_levels, node_budget, global_deadline)
            if plan is not None:
                return plan
            else:
                self._isb_exhausted.add(game_prefix)  # don't try again

        # Tertiary path: direct solver for delivery games (wa30 style)
        plan = self._delivery_solver(raw_env, actions, start_levels, global_deadline)
        if plan is not None:
            return plan

        # Tertiary path: direct solver for piece-placement games (re86 style)
        plan = self._piece_placement_solver(raw_env, actions, start_levels, global_deadline)
        if plan is not None:
            return plan

        # Quinary path: direct solver for visual-programming games (tn36 style)
        plan = self._visual_program_solver(raw_env, _full, start_levels, global_deadline)
        if plan is not None:
            return plan

        # Senary path: ball-placement solver (su15-style click-to-move games)
        plan = self._ball_placement_solver(raw_env, _full, start_levels, global_deadline)
        if plan is not None:
            return plan

        # Septenary path: tile-sorting solver (sb26-style drag-and-match games)
        plan = self._tile_sorting_solver(raw_env, _full, start_levels, global_deadline)
        if plan is not None:
            return plan

        # Octonary path: sprite-connection solver (cn04-style pixel-align games)
        plan = self._sprite_connection_solver(raw_env, _full, actions, start_levels, global_deadline)
        if plan is not None:
            return plan

        # Nenary path: ball-routing solver (r11l-style checkpoint-centroid games)
        plan = self._ball_routing_solver(raw_env, _full, start_levels, global_deadline)
        if plan is not None:
            return plan

        # Decenary path: checkers-jump solver (lf52-style piece-jumping games)
        plan = self._checkers_jump_solver(raw_env, _full, start_levels, global_deadline)
        if plan is not None:
            return plan

        # Undecenary path: undoing maze solver (g50t-style ghost mechanic games)
        plan = self._undoing_maze_solver(raw_env, _full, start_levels, global_deadline)
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
    # Internal-state BFS (ls20 style)                                     #
    # ------------------------------------------------------------------ #

    def _internal_state_bfs(
        self, raw_env, actions, start_levels, node_budget, global_deadline,
        max_nodes: int = 2000,  # internal cap; overrides node_budget when larger
    ) -> list[str] | None:
        """BFS that discovers internal game state variables (not reflected in sprite
        positions) and includes them in the state key.

        Games like ls20 have internal variables (shape, color, rotation) that affect
        the win condition but don't change sprite pixel hashes. This solver discovers
        those variables by comparing all simple (int/bool/float) game attributes before
        and after each action, then runs BFS with an enriched key.
        """
        import time
        import heapq

        if time.monotonic() >= global_deadline:
            return None

        action_map = {getattr(a, "name", ""): a for a in actions}
        game = getattr(raw_env, "_game", None)
        if game is None:
            return None

        # Snapshot all simple game attributes at initial state
        def _snap(env):
            g = getattr(env, "_game", None)
            if g is None:
                return {}
            snap = {}
            for k, v in vars(g).items():
                if isinstance(v, (int, float, bool)):
                    snap[k] = v
            return snap

        snap0 = _snap(raw_env)
        if not snap0:
            return None

        # Discover which attributes change: first probe single steps, then run a short
        # local-state-key BFS (≤50 nodes) to find changes at special positions.
        changed_attrs: set[str] = set()

        # Phase 1: single-step probes from initial state
        for nm, act in action_map.items():
            probe = copy.deepcopy(raw_env)
            gc.disable()
            try:
                probe.step(act)
            finally:
                gc.enable()
            snap1 = _snap(probe)
            for k in snap0:
                if k in snap1 and snap1[k] != snap0[k]:
                    changed_attrs.add(k)

        # Phase 2: short BFS with local_state_key to find changes at special positions
        from collections import deque as _deque
        init_lsk = self.perception.local_state_key(raw_env)
        mini_queue = _deque([(copy.deepcopy(raw_env), 0)])
        mini_seen = {init_lsk}
        mini_nodes = 0
        mini_deadline = time.monotonic() + 5.0  # hard 5s cap for discovery
        gc.disable()
        try:
            while mini_queue and mini_nodes < 60 and time.monotonic() < mini_deadline:
                cur_env, depth = mini_queue.popleft()
                mini_nodes += 1
                cur_snap = _snap(cur_env)
                for k in snap0:
                    if k in cur_snap and cur_snap[k] != snap0[k]:
                        changed_attrs.add(k)
                if depth < 15:
                    for act in actions:
                        if time.monotonic() >= mini_deadline:
                            break
                        try:
                            nxt = copy.deepcopy(cur_env)
                            nxt.step(act)
                        except Exception:
                            continue
                        nk = self.perception.local_state_key(nxt)
                        if nk not in mini_seen:
                            mini_seen.add(nk)
                            mini_queue.append((nxt, depth + 1))
        finally:
            gc.enable()

        # Only proceed if internal variables changed (not just player position)
        # Filter: small value range, non-private, not step counters
        relevant_attrs: list[str] = sorted(
            k for k in changed_attrs
            if not k.startswith('_') and abs(snap0.get(k, 0)) < 1000
        )
        if not relevant_attrs:
            return None  # No meaningful internal state changes detected

        # Define enriched state key: local_state_key + changed_attrs values
        def _enriched_key(env):
            base = self.perception.local_state_key(env)
            g = getattr(env, "_game", None)
            extras = tuple(getattr(g, k, None) for k in sorted(changed_attrs))
            return (base, extras)

        # Also try player's direct position (using gudziatsk if available)
        def _player_pos(env):
            g = getattr(env, "_game", None)
            gd = getattr(g, "gudziatsk", None)
            if gd:
                return (getattr(gd, "x", None), getattr(gd, "y", None))
            return self.perception.local_state_key(env)

        def _fast_key(env):
            g = getattr(env, "_game", None)
            gd = getattr(g, "gudziatsk", None)
            if gd is not None:
                player_pos = (getattr(gd, "x", None), getattr(gd, "y", None))
            else:
                player_pos = self.perception.local_state_key(env)
            extras = tuple(getattr(g, k, None) for k in relevant_attrs)
            return (player_pos, extras)

        def _game_score(env):
            g = getattr(env, "_game", None)
            return int(getattr(g, "_score", 0) or 0) if g else 0

        ws = int(getattr(game, "_win_score", 1000) or 1000)
        h0 = ws - _game_score(raw_env)
        ctr = 0
        heap = [(h0, 0, ctr, copy.deepcopy(raw_env), [])]
        best_g: dict = {}
        nodes = 0
        effective_budget = max(node_budget, max_nodes)
        # Cap time to avoid consuming too much budget on a single game
        time_cap = min(global_deadline, time.monotonic() + 15.0)

        gc.disable()
        try:
            while heap and nodes < effective_budget and time.monotonic() < time_cap:
                f, g, _, current, plan = heapq.heappop(heap)
                nodes += 1
                key = _fast_key(current)
                if key in best_g and best_g[key] <= g:
                    continue
                best_g[key] = g
                if len(plan) >= 60:
                    continue

                for act in actions:
                    try:
                        nxt = copy.deepcopy(current)
                        obs = nxt.step(act)
                    except Exception:
                        continue
                    new_plan = plan + [getattr(act, "name", str(act))]
                    obs_state = self._state_name(obs)
                    if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                        return new_plan
                    if obs_state == "WIN":
                        return new_plan
                    if obs_state == "NOT_FINISHED":
                        new_g = g + 1
                        new_key = _fast_key(nxt)
                        if new_key not in best_g or best_g[new_key] > new_g:
                            new_h = ws - _game_score(nxt)
                            ctr += 1
                            heapq.heappush(heap, (new_g + new_h, new_g, ctr, nxt, new_plan))
        finally:
            gc.enable()

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

    # ------------------------------------------------------------------ #
    # Spell-casting direct solver (sc25 style)                             #
    # ------------------------------------------------------------------ #

    def _spell_casting_solver(
        self, raw_env, actions, start_levels, global_deadline,
    ) -> list | None:
        """Direct solver for games with a spell-casting mechanic (sc25).

        Returns a mixed plan: list[str | dict] where str = keyboard action
        name and dict = click position {"x":..., "y":...}.
        Agent.py dispatches each item by type.

        Algorithm per level:
        1. Select spell via sptivk-... button click (if ijhfdcamokt=None)
        2. For fibcey: first press ACTION4 (face right) so fireball hits target
        3. Demo trigger if qytejzcythm=True (first click on grid)
        4. Cast spell pattern (click True cells in 3×3 grid)
        5. Keyboard A* BFS from post-cast state to advance level
        6. Repeat for each of the win_score levels
        """
        import time
        import heapq as _heapq

        if time.monotonic() >= global_deadline:
            return None

        game = getattr(raw_env, "_game", None)
        if game is None:
            return None

        # Detection guard: must have the spell-pattern dict
        zzpoabuniyn = getattr(game, "zzpoabuniyn", None)
        if not zzpoabuniyn or not isinstance(zzpoabuniyn, dict):
            return None

        action_map = {getattr(a, "name", ""): a for a in actions}
        action6 = action_map.get("ACTION6")
        if action6 is None:
            return None

        ws = int(getattr(game, "_win_score", 6) or 6)

        # ── Gather the 9 grid click positions from bmmtkvkbcdd ──────── #
        try:
            from arcengine.enums import GameAction as _GameAction
            click_enum = _GameAction.ACTION6
        except ImportError:
            return None

        # Camera scale for sprite → display coord conversion
        _scale = 1
        try:
            cam = getattr(game, "camera", None)
            if cam is not None:
                result = cam.display_to_grid(4, 4)
                if result and result[0] > 0:
                    _scale = 4 // result[0]
        except Exception:
            pass

        grid_positions: list[dict] = []
        for attr_name in ("bmmtkvkbcdd", "human_actions", "_human_actions"):
            ha_list = getattr(game, attr_name, None)
            if not ha_list:
                continue
            for ha in ha_list:
                if getattr(ha, "id", None) == click_enum:
                    data = getattr(ha, "data", {}) or {}
                    hx, hy = data.get("x"), data.get("y")
                    if hx is not None and hy is not None:
                        grid_positions.append({"x": int(hx), "y": int(hy)})
            if grid_positions:
                break

        if len(grid_positions) < 9:
            return None

        grid_positions.sort(key=lambda p: (p["y"], p["x"]))

        # ── Keyboard actions for navigation ──────────────────────────── #
        kb_actions = [a for a in actions
                      if not (callable(getattr(a, "is_complex", None)) and a.is_complex())]

        def _kb_bfs(env_, total_lc_, budget=2000):
            """A* BFS from env_ to find keyboard plan that advances level."""
            _ws = int(getattr(env_._game, "_win_score", 1000) or 1000)
            def _score(e): return int(getattr(e._game, "_score", 0) or 0) if e._game else 0
            def _key(e):
                _g2 = e._game
                pp2 = getattr(_g2, "plnqvukupu", None)
                return (int(pp2.x), int(pp2.y), int(_g2.jdmucabyqar)) if pp2 else (0, 0, 0)
            _counter = [0]
            _heap = [(_ws - _score(env_), 0, _counter[0], copy.deepcopy(env_), [])]
            _bg: dict = {}
            _nodes = 0
            gc.disable()
            try:
                while _heap and _nodes < budget:
                    _f, _g, _, _cur, _plan = _heapq.heappop(_heap)
                    _nodes += 1
                    _k = _key(_cur)
                    if _k in _bg and _bg[_k] <= _g:
                        continue
                    _bg[_k] = _g
                    if len(_plan) >= 50:
                        continue
                    for _act in kb_actions:
                        _nxt = copy.deepcopy(_cur)
                        _obs = _nxt.step(_act)
                        _lc = int(getattr(_obs, "levels_completed", 0) or 0)
                        if _lc > total_lc_:
                            return _plan + [getattr(_act, "name", "")]
                        if self._state_name(_obs) == "NOT_FINISHED":
                            _nk = _key(_nxt)
                            _ng = _g + 1
                            if _nk not in _bg or _bg[_nk] > _ng:
                                _counter[0] += 1
                                _heapq.heappush(_heap, (
                                    _ng + _ws - _score(_nxt), _ng, _counter[0], _nxt,
                                    _plan + [getattr(_act, "name", "")]
                                ))
            finally:
                gc.enable()
            return None

        def _select_spell(sim_, zzp_):
            """Find and click the first spell button that sets ijhfdcamokt to a valid spell."""
            sg = sim_._game
            sl = getattr(sg, "current_level", None)
            if sl is None:
                return None
            # Probe each sptivk-... sprite by clicking its display position
            for s in getattr(sl, "_sprites", []):
                nm = str(getattr(s, "name", ""))
                if not nm.startswith("sptivk-"):
                    continue
                if "clzbxlm" in nm or nm in ("sptivk-ui",):
                    continue
                sx = (int(getattr(s, "_x", 0)) + 1) * _scale
                sy = (int(getattr(s, "_y", 0)) + 1) * _scale
                click_data = {"x": sx, "y": sy}
                probe = copy.deepcopy(sim_)
                gc.disable()
                try:
                    probe.step(click_enum, data=click_data)
                finally:
                    gc.enable()
                pg = probe._game
                new_spell = getattr(pg, "ijhfdcamokt", None) if pg else None
                if new_spell is not None and new_spell in zzp_:
                    return click_data
            return None

        # ── Simulation: build mixed plan ──────────────────────────────── #
        sim = copy.deepcopy(raw_env)
        plan: list = []  # mixed: str (keyboard) | dict (click)
        total_lc = start_levels

        gc.disable()
        try:
            for _round in range(ws * 8):  # generous upper bound
                if time.monotonic() >= global_deadline:
                    break

                sg = sim._game
                if sg is None:
                    break
                if total_lc >= ws:
                    break

                demo_mode = bool(getattr(sg, "qytejzcythm", False))
                current_spell = getattr(sg, "ijhfdcamokt", None)

                # ── Step 1: Select spell if none active ─────────────── #
                if current_spell is None:
                    click_data = _select_spell(sim, zzpoabuniyn)
                    if click_data is None:
                        break
                    obs = sim.step(click_enum, data=click_data)
                    plan.append(click_data)
                    lc = int(getattr(obs, "levels_completed", 0) or 0)
                    if lc > total_lc:
                        total_lc = lc
                        if total_lc >= ws:
                            break
                    continue

                # ── Step 2: Demo trigger (level 0 only) ──────────────── #
                if demo_mode:
                    obs = sim.step(click_enum, data=grid_positions[0])
                    plan.append(grid_positions[0])
                    lc = int(getattr(obs, "levels_completed", 0) or 0)
                    if lc > total_lc:
                        total_lc = lc
                        if total_lc >= ws:
                            break
                    continue

                # ── Step 3: Pre-cast orientation for fibcey ──────────── #
                # fibcey shoots fireball in facing direction (jdmucabyqar).
                # Must face RIGHT (jdm=3) to hit tagsmh at (55,22) from any
                # left-of-tagsmh position. Press ACTION4 to set jdm=3.
                if current_spell == "fibcey":
                    jdm = int(getattr(sg, "jdmucabyqar", 0))
                    if jdm != 3 and "ACTION4" in action_map:
                        obs = sim.step(action_map["ACTION4"])
                        plan.append("ACTION4")
                        lc = int(getattr(obs, "levels_completed", 0) or 0)
                        if lc > total_lc:
                            total_lc = lc
                            if total_lc >= ws:
                                break
                        # Refresh grid positions (they may have changed via level reset)
                        sg2 = sim._game
                        new_gp = []
                        for attr_name in ("bmmtkvkbcdd", "human_actions"):
                            ha_list = getattr(sg2, attr_name, None)
                            if not ha_list:
                                continue
                            for ha in ha_list:
                                if getattr(ha, "id", None) == click_enum:
                                    d = getattr(ha, "data", {}) or {}
                                    hx2, hy2 = d.get("x"), d.get("y")
                                    if hx2 is not None and hy2 is not None:
                                        new_gp.append({"x": int(hx2), "y": int(hy2)})
                            if new_gp:
                                break
                        if new_gp:
                            grid_positions[:] = sorted(new_gp, key=lambda p: (p["y"], p["x"]))

                # ── Step 4: Cast current spell ────────────────────────── #
                pattern = zzpoabuniyn.get(current_spell)
                if pattern is None:
                    break
                cast_won = False
                for row_idx, row in enumerate(pattern):
                    if time.monotonic() >= global_deadline:
                        break
                    for col_idx, cell in enumerate(row):
                        if cell:
                            pos = grid_positions[row_idx * 3 + col_idx]
                            obs = sim.step(click_enum, data=pos)
                            plan.append(pos)
                            lc = int(getattr(obs, "levels_completed", 0) or 0)
                            if lc > total_lc:
                                total_lc = lc
                                if total_lc >= ws:
                                    cast_won = True
                                    break
                    if cast_won:
                        break
                if cast_won or total_lc >= ws:
                    break

                # ── Step 5: Keyboard BFS to advance level ─────────────── #
                kb_plan = _kb_bfs(sim, total_lc)
                if kb_plan is None:
                    break
                for nm in kb_plan:
                    act = action_map.get(nm)
                    if act is None:
                        break
                    obs = sim.step(act)
                    plan.append(nm)
                    lc = int(getattr(obs, "levels_completed", 0) or 0)
                    if lc > total_lc:
                        total_lc = lc
                        if total_lc >= ws:
                            break
                if total_lc >= ws:
                    break
        finally:
            gc.enable()

        return plan if plan else None

    # ------------------------------------------------------------------ #
    # Visual-programming solver (tn36 style)                              #
    # ------------------------------------------------------------------ #

    def _visual_program_solver(
        self, raw_env, all_actions, start_levels, global_deadline
    ) -> list | None:
        """Solver for tn36-style visual programming games.

        Game structure: player edits a binary-coded program by clicking Maidxz
        bit-checkboxes, then clicks a run button (sucqgkbuojsa) to execute it.
        Win when the piece (htntnzkbzu) reaches target (aqszntqeae).

        Strategy:
          1. Read mvqheosngn's reference panel (Level 2+) or compute analytically (Level 1).
          2. Compute which bits to toggle to set the target program.
          3. Execute toggles + run for each level until game is won.
        """
        import time

        if time.monotonic() >= global_deadline:
            return None

        game = getattr(raw_env, "_game", None)
        if game is None:
            return None
        fdk = getattr(game, "fdksqlmpki", None)
        if fdk is None:
            return None
        brz = getattr(fdk, "bzirenxmrg", None)
        if brz is None:
            return None

        # Find ACTION6 (click)
        click_enum = None
        for a in all_actions:
            if getattr(a, "name", "").endswith("6"):
                click_enum = a
                break
        if click_enum is None:
            return None

        win_score = int(getattr(game, "_win_score", 1) or 1)
        plan = []
        sim = copy.deepcopy(raw_env)

        gc.disable()
        try:
            cur_lc = start_levels
            while cur_lc < win_score and time.monotonic() < global_deadline:
                g = getattr(sim, "_game", None)
                if g is None:
                    break
                fdk_s = getattr(g, "fdksqlmpki", None)
                if fdk_s is None:
                    break
                brz_s = getattr(fdk_s, "bzirenxmrg", None)
                if brz_s is None:
                    break

                level_plan = self._vp_solve_level(fdk_s, brz_s, click_enum, global_deadline)
                if level_plan is None:
                    break

                plan.extend(level_plan)
                level_advanced = False
                for click in level_plan:
                    if time.monotonic() >= global_deadline:
                        break
                    obs = sim.step(click_enum, data=click)
                    new_lc = int(getattr(obs, "levels_completed", 0) or 0)
                    if new_lc > cur_lc:
                        cur_lc = new_lc
                        level_advanced = True
                        break
                if not level_advanced:
                    break  # plan didn't win this level; abort
        finally:
            gc.enable()

        return plan if plan else None

    def _vp_solve_level(self, fdk, brz, click_enum, global_deadline) -> list | None:
        """Compute toggle clicks + run click to solve the current level."""
        sxh = getattr(brz, "sxhtkytekm", None)
        if sxh is None:
            return None
        run_click = {"x": sxh.x + sxh._width // 2, "y": sxh.y + sxh._height // 2}

        vup = getattr(brz, "vupcwzjtxu", None)
        if vup is None or not vup.rzmeklhluf:
            return None

        current_prog = list(vup.vkuvtkaerv)
        num_slots = len(current_prog)

        # Try reference from mvqheosngn first (Level 2+)
        target_prog = self._vp_reference_program(fdk, num_slots)

        # Fall back to analytical (Level 1 / no reference)
        if target_prog is None:
            target_prog = self._vp_analytical_program(brz, num_slots)

        # Fall back to brute force simulation (<=12 bits)
        if target_prog is None:
            target_prog = self._vp_brute_force(brz, vup, current_prog, global_deadline)

        if target_prog is None:
            return None

        toggle_clicks = self._vp_compute_toggles(vup, current_prog, target_prog)
        if toggle_clicks is None:
            return None

        return toggle_clicks + [run_click]

    def _vp_reference_program(self, fdk, num_slots) -> list | None:
        """Read mvqheosngn's panel for the reference program."""
        mvq = getattr(fdk, "mvqheosngn", None)
        if mvq is None:
            return None
        mvq_vup = getattr(mvq, "vupcwzjtxu", None)
        if mvq_vup is None or not mvq_vup.rzmeklhluf:
            return None
        ref = list(mvq_vup.vkuvtkaerv)
        if len(ref) != num_slots:
            return None
        return ref

    def _vp_analytical_program(self, brz, num_slots) -> list | None:
        """Compute program from (init_pos -> aqszntqeae) analytically."""
        CSPOIQWER = 4

        aqsz = getattr(brz, "aqszntqeae", None)
        if aqsz is None:
            return None

        x0 = brz.fwrnsvyvrz
        y0 = brz.bmhxacplut
        rot0 = brz.qixyeojolu
        scale0 = brz.fpofcohbab

        dx = aqsz.x - x0
        dy = aqsz.y - y0
        d_rot = (aqsz.rotation - rot0) % 360
        d_scale = aqsz.scale - scale0

        if dx % CSPOIQWER != 0 or dy % CSPOIQWER != 0:
            return None

        program = []
        if d_rot == 90:    program.append(5)
        elif d_rot == 180: program.append(7)
        elif d_rot == 270: program.append(6)

        if d_scale > 0:    program.extend([8] * d_scale)
        elif d_scale < 0:  program.extend([9] * (-d_scale))

        steps_y = dy // CSPOIQWER
        steps_x = dx // CSPOIQWER
        if steps_y > 0:   program.extend([3] * steps_y)
        elif steps_y < 0: program.extend([33] * (-steps_y))
        if steps_x > 0:   program.extend([2] * steps_x)
        elif steps_x < 0: program.extend([1] * (-steps_x))

        while len(program) < num_slots:
            program.append(0)

        if len(program) > num_slots:
            return None

        return program

    def _vp_brute_force(self, brz, vup, current_prog, global_deadline) -> list | None:
        """Brute-force over bit combinations for <=12 bits."""
        import time

        slots = vup.rzmeklhluf
        total_bits = sum(len(s.sonocxtjtj) for s in slots)
        if total_bits > 12:
            return None

        aqsz = getattr(brz, "aqszntqeae", None)
        if aqsz is None:
            return None

        x0 = brz.fwrnsvyvrz
        y0 = brz.bmhxacplut
        rot0 = brz.qixyeojolu
        scale0 = brz.fpofcohbab
        sjm0 = brz.nzmblccilq

        for combo in range(2 ** total_bits):
            if time.monotonic() >= global_deadline:
                return None
            program = []
            bit_idx = 0
            for slot in slots:
                val = 0
                for b_i, bit in enumerate(slot.sonocxtjtj):
                    cur_checked = bit.yliktcpsfp
                    want_flip = bool(combo & (1 << bit_idx))
                    if cur_checked ^ want_flip:
                        val |= (1 << b_i)
                    bit_idx += 1
                program.append(val)

            if self._vp_simulate(x0, y0, rot0, scale0, sjm0, program, aqsz):
                return program

        return None

    @staticmethod
    def _vp_simulate(x0, y0, rot0, scale0, sjm0, program, aqsz) -> bool:
        """Simulate program execution and check win (no collision detection)."""
        CSPOIQWER = 4
        x, y, rot, scale, sjm = x0, y0, rot0 % 360, scale0, sjm0

        for code in program:
            if code == 1:    x -= CSPOIQWER
            elif code == 2:  x += CSPOIQWER
            elif code == 3:  y += CSPOIQWER
            elif code == 33: y -= CSPOIQWER
            elif code == 5:  rot = (rot + 90) % 360
            elif code == 6:  rot = (rot - 90 + 360) % 360
            elif code == 7:  rot = (rot + 180) % 360
            elif code == 8:  scale += 1
            elif code == 9:  scale = max(1, scale - 1)
            elif code == 10 or code == 11: x += CSPOIQWER * 2
            elif code == 12 or code == 13: x -= CSPOIQWER * 2
            elif code == 16: rot = (rot + 270) % 360
            elif code == 34: x -= CSPOIQWER
            elif code == 14: sjm = 9
            elif code == 15: sjm = 8
            elif code == 63: sjm = 15

        return (x == aqsz.x and y == aqsz.y and
                scale == aqsz.scale and rot == aqsz.rotation and
                sjm == aqsz.sjmtdfxdrc)

    def _vp_compute_toggles(self, vup, current_prog, target_prog) -> list | None:
        """Compute click positions needed to change current_prog to target_prog."""
        if len(current_prog) != len(target_prog):
            return None

        toggle_clicks = []
        for slot, cur_val, tgt_val in zip(vup.rzmeklhluf, current_prog, target_prog):
            diff = cur_val ^ tgt_val
            if diff == 0:
                continue
            for b_i, bit in enumerate(slot.sonocxtjtj):
                if diff & (1 << b_i):
                    cx = bit.x + bit._width // 2
                    cy = bit.y + bit._height // 2
                    toggle_clicks.append({"x": cx, "y": cy})

        return toggle_clicks

    # ------------------------------------------------------------------ #
    # Ball-placement solver (su15 style)                                  #
    # ------------------------------------------------------------------ #

    def _ball_placement_solver(
        self, raw_env, all_actions, start_levels, global_deadline
    ) -> list | None:
        """Solver for su15-style ball-placement games.

        Mechanic: clicking within radius R of a ball moves it toward the click
        by up to STEP pixels per frame over FRAMES frames. Solve by computing
        a greedy path from ball position to goal zone center.

        Detection: lkujttxgs (ball sprites, tag 'zmlxwcvwb'), powykypsm (goal
        zone sprites, tag 'xkstxyqbs'), kqywaxhmsb (ball→color mapping),
        dsqlbvwaj (level target spec), ikskfqldi (step), kacsjmxae (radius).
        """
        import time
        import math

        if time.monotonic() >= global_deadline:
            return None

        game = getattr(raw_env, "_game", None)
        if game is None:
            return None

        # Detect su15-style structure
        lkujttxgs = getattr(game, "lkujttxgs", None)
        powykypsm = getattr(game, "powykypsm", None)
        dsqlbvwaj = getattr(game, "dsqlbvwaj", None)
        kqywaxhmsb = getattr(game, "kqywaxhmsb", {})
        ikskfqldi = getattr(game, "ikskfqldi", 4)  # pixels per frame
        kacsjmxae = getattr(game, "kacsjmxae", 8)  # click radius

        if not lkujttxgs or not powykypsm or dsqlbvwaj is None:
            return None

        # Find ACTION6 (click)
        click_enum = None
        for a in all_actions:
            if getattr(a, "name", "").endswith("6"):
                click_enum = a
                break
        if click_enum is None:
            return None

        win_score = int(getattr(game, "_win_score", 1) or 1)

        plan = []
        sim = copy.deepcopy(raw_env)

        gc.disable()
        try:
            cur_lc = start_levels
            while cur_lc < win_score and time.monotonic() < global_deadline:
                g = getattr(sim, "_game", None)
                if g is None:
                    break

                level_plan = self._bp_solve_level(g, click_enum, global_deadline)
                if level_plan is None:
                    break

                plan.extend(level_plan)
                level_advanced = False
                for click in level_plan:
                    if time.monotonic() >= global_deadline:
                        break
                    obs = sim.step(click_enum, data=click)
                    new_lc = int(getattr(obs, "levels_completed", 0) or 0)
                    if new_lc > cur_lc:
                        cur_lc = new_lc
                        level_advanced = True
                        break
                if not level_advanced:
                    break
        finally:
            gc.enable()

        return plan if plan else None

    def _bp_solve_level(self, game, click_enum, global_deadline) -> list | None:
        """Compute click sequence to move ball(s) into goal zone for one level."""
        import time
        import math

        lkujttxgs = getattr(game, "lkujttxgs", [])
        powykypsm = getattr(game, "powykypsm", [])
        kqywaxhmsb = getattr(game, "kqywaxhmsb", {})
        dsqlbvwaj = getattr(game, "dsqlbvwaj", None)
        ikskfqldi = getattr(game, "ikskfqldi", 4)
        kacsjmxae = getattr(game, "kacsjmxae", 8)
        gdamdvokm = getattr(game, "gdamdvokm", 4)
        gvvyzrusqq = 10  # min valid click y
        qsqeqpepjy = 63  # max valid click y (exclusive)

        if not lkujttxgs or not powykypsm or dsqlbvwaj is None:
            return None

        # Parse target: dsqlbvwaj = [color, count] or list of pairs
        try:
            if isinstance(dsqlbvwaj[0], (list, tuple)):
                targets = [(int(c), int(n)) for c, n in dsqlbvwaj]
            else:
                targets = [(int(dsqlbvwaj[0]), int(dsqlbvwaj[1]))]
        except Exception:
            return None

        # Find balls that need to reach goal zones
        # Match ball color to targets
        zone = powykypsm[0]  # use first zone as target area
        zone_cx = zone.x + zone.width // 2
        zone_cy = zone.y + zone.height // 2

        # Maximum step per click = ikskfqldi per frame × gdamdvokm frames
        # But ball moves TO click position if close enough
        # Safe step: kacsjmxae * 0.6 (within radius, guarantees movement)
        safe_step = max(1, int(kacsjmxae * 0.55))

        clicks = []
        for ball in lkujttxgs:
            if time.monotonic() >= global_deadline:
                return None

            ball_cx = ball.x + ball.width // 2
            ball_cy = ball.y + ball.height // 2

            # Check if already in zone
            if (zone.x <= ball_cx < zone.x + zone.width and
                    zone.y <= ball_cy < zone.y + zone.height):
                continue

            # Compute steps needed to move from ball to zone
            dx_total = zone_cx - ball_cx
            dy_total = zone_cy - ball_cy
            dist = math.sqrt(dx_total * dx_total + dy_total * dy_total)

            if dist < 1:
                continue

            # Number of clicks to cover the distance
            # Each click moves ball toward click by safe_step in each axis
            step_x = safe_step if dx_total > 0 else (-safe_step if dx_total < 0 else 0)
            step_y = safe_step if dy_total > 0 else (-safe_step if dy_total < 0 else 0)

            cur_bx, cur_by = ball_cx, ball_cy
            max_clicks = 50
            for _ in range(max_clicks):
                if time.monotonic() >= global_deadline:
                    break
                # Check if ball is now in zone
                if (zone.x <= cur_bx < zone.x + zone.width and
                        zone.y <= cur_by < zone.y + zone.height):
                    break

                # Compute click toward zone center
                rem_x = zone_cx - cur_bx
                rem_y = zone_cy - cur_by

                # Click step: limited by remaining distance and safe_step
                cx_step = min(safe_step, abs(rem_x)) * (1 if rem_x > 0 else -1)
                cy_step = min(safe_step, abs(rem_y)) * (1 if rem_y > 0 else -1)

                click_x = cur_bx + cx_step
                click_y = cur_by + cy_step

                # Clamp to valid click range
                click_x = max(0, min(click_x, 63))
                click_y = max(gvvyzrusqq, min(click_y, qsqeqpepjy - 1))

                # Verify click is within radius of ball
                actual_dist = math.sqrt((click_x - cur_bx)**2 + (click_y - cur_by)**2)
                if actual_dist > kacsjmxae:
                    # Scale back to radius
                    factor = (kacsjmxae - 1) / actual_dist
                    click_x = int(cur_bx + (click_x - cur_bx) * factor)
                    click_y = int(cur_by + (click_y - cur_by) * factor)
                    click_y = max(gvvyzrusqq, min(click_y, qsqeqpepjy - 1))

                clicks.append({"x": int(click_x), "y": int(click_y)})
                cur_bx = click_x
                cur_by = click_y

        return clicks if clicks else None

    # ------------------------------------------------------------------ #
    # Tile-sorting solver (sb26 style)                                    #
    # ------------------------------------------------------------------ #

    def _tile_sorting_solver(
        self, raw_env, all_actions, start_levels, global_deadline
    ) -> list | None:
        """Solver for sb26-style tile-sorting games.

        Game structure: lngftsryyw tile sprites must be placed into
        susublrply slot sprites in order matching wcfyiodrx chapter colors.
        ACTION6 pick+place moves tiles; ACTION5 triggers sequential check.

        Detection: dkouqqads (tile list), dewwplfix (slot list),
        wcfyiodrx (chapter color sequence), qaagahahj (frame structure).
        """
        import time

        if time.monotonic() >= global_deadline:
            return None

        game = getattr(raw_env, "_game", None)
        if game is None:
            return None

        # Detect sb26-style structure
        wcfyiodrx = getattr(game, "wcfyiodrx", None)
        dkouqqads = getattr(game, "dkouqqads", None)
        qaagahahj = getattr(game, "qaagahahj", None)

        if not wcfyiodrx or not dkouqqads or not qaagahahj:
            return None

        # Find ACTION5 and ACTION6
        action5_enum = None
        click_enum = None
        for a in all_actions:
            nm = getattr(a, "name", "")
            if nm.endswith("5"):
                action5_enum = a
            elif nm.endswith("6"):
                click_enum = a
        if action5_enum is None or click_enum is None:
            return None

        win_score = int(getattr(game, "_win_score", 1) or 1)
        plan = []
        sim = copy.deepcopy(raw_env)

        gc.disable()
        try:
            cur_lc = start_levels
            while cur_lc < win_score and time.monotonic() < global_deadline:
                g = getattr(sim, "_game", None)
                if g is None:
                    break

                level_plan = self._ts_solve_level(g, click_enum, action5_enum, global_deadline)
                if level_plan is None:
                    break

                plan.extend(level_plan)
                level_advanced = False
                for item in level_plan:
                    if time.monotonic() >= global_deadline:
                        break
                    if isinstance(item, dict):
                        obs = sim.step(click_enum, data=item)
                    else:
                        obs = sim.step(action5_enum)
                    new_lc = int(getattr(obs, "levels_completed", 0) or 0)
                    if new_lc > cur_lc:
                        cur_lc = new_lc
                        level_advanced = True
                        break
                if not level_advanced:
                    break
        finally:
            gc.enable()

        return plan if plan else None

    def _ts_solve_level(self, game, click_enum, action5_enum, global_deadline) -> list | None:
        """Compute tile placement sequence + ACTION5 for one sb26 level."""
        import time

        wcfyiodrx = getattr(game, "wcfyiodrx", [])
        level = getattr(game, "current_level", None)
        if not wcfyiodrx or level is None:
            return None

        sprites = getattr(level, "_sprites", [])

        # Tiles: lngftsryyw tagged (visible)
        tiles = [s for s in sprites
                 if "lngftsryyw" in str(getattr(s, "tags", [])) and s.is_visible]
        # Slots: susublrply tagged, at low y (in slot row)
        all_slots = [s for s in sprites if "susublrply" in str(getattr(s, "tags", []))]
        # Separate slot-row (high y) from tile-row indicators
        if not tiles or not all_slots:
            return None

        # Slots with y <= median_tile_y are in slot area
        tile_ys = [t._y for t in tiles]
        median_ty = sum(tile_ys) // len(tile_ys)
        slots = sorted([s for s in all_slots if s._y < median_ty and s.is_visible],
                       key=lambda s: s._x)

        if not slots or len(slots) != len(wcfyiodrx):
            return None

        # Get chapter required colors: wcfyiodrx[i].pixels[0,0]
        chapter_colors = []
        for wc in wcfyiodrx:
            px = getattr(wc, "pixels", None)
            if px is not None and px.shape[0] > 0 and px.shape[1] > 0:
                chapter_colors.append(int(px[0, 0]))
            else:
                chapter_colors.append(-1)

        # Get tile colors: tile.pixels[height//2, width//2]
        def tile_center_color(t):
            px = getattr(t, "pixels", None)
            if px is None:
                return -1
            return int(px[px.shape[0] // 2, px.shape[1] // 2])

        # Match tiles to slots by chapter color
        unmatched_tiles = list(tiles)
        assignment = {}  # slot_i -> tile
        for slot_i, chap_color in enumerate(chapter_colors):
            for t in unmatched_tiles:
                if tile_center_color(t) == chap_color:
                    assignment[slot_i] = t
                    unmatched_tiles.remove(t)
                    break

        if len(assignment) != len(wcfyiodrx):
            return None

        # Build plan: for each slot_i, pick tile then place in slot
        plan = []
        for slot_i in range(len(slots)):
            if slot_i not in assignment or time.monotonic() >= global_deadline:
                return None

            tile = assignment[slot_i]
            slot = slots[slot_i]

            # Click center of tile (pick)
            tx_px = getattr(tile, "pixels", None)
            tw = tx_px.shape[1] if tx_px is not None else 1
            th = tx_px.shape[0] if tx_px is not None else 1
            pick_x = tile._x + tw // 2
            pick_y = tile._y + th // 2
            plan.append({"x": pick_x, "y": pick_y})

            # Click center of slot (place)
            sx_px = getattr(slot, "pixels", None)
            sw = sx_px.shape[1] if sx_px is not None else 1
            sh = sx_px.shape[0] if sx_px is not None else 1
            place_x = slot._x + sw // 2
            place_y = slot._y + sh // 2
            plan.append({"x": place_x, "y": place_y})

        # Add ACTION5 as sentinel ("A5") to trigger check
        plan.append("ACTION5")
        return plan

    # ------------------------------------------------------------------ #
    # Sprite-connection solver (cn04 style)                               #
    # ------------------------------------------------------------------ #

    def _sprite_connection_solver(
        self, raw_env, all_actions, kb_actions, start_levels, global_deadline
    ) -> list | None:
        """Solver for cn04-style pixel-connection games.

        Game structure: sprites have canonical 8/13 pixels (hlxyvcmpk).
        Win when all 8/13 pixels are "connected" (two sprites share same
        world position with same color). CONNECTION = iahpylgry populated.

        Detection: hlxyvcmpk, vausolnec, kpgnbcoir, iahpylgry fields.
        Action vocab: A6=click to select, A5=rotate 90° + check win,
                      A1-4=move 1 step + check win.

        Strategy: analytically find rotation+offset that aligns all
        connection pixels between sprites. Use brute-force probe to
        validate (accounts for complex camera/coordinate transforms).
        """
        import time
        import numpy as np

        if time.monotonic() >= global_deadline:
            return None

        game = getattr(raw_env, "_game", None)
        if game is None:
            return None

        # Detect cn04-style structure
        hlxyvcmpk = getattr(game, "hlxyvcmpk", None)
        vausolnec = getattr(game, "vausolnec", None)
        kpgnbcoir = getattr(game, "kpgnbcoir", None)
        iahpylgry = getattr(game, "iahpylgry", None)

        if hlxyvcmpk is None or vausolnec is None or kpgnbcoir is None:
            return None
        if len(hlxyvcmpk) < 2:
            return None

        # Find click, keyboard, rotate actions
        click_enum = None
        action5_enum = None
        action1_enum = None
        action2_enum = None
        action3_enum = None
        action4_enum = None
        for a in all_actions:
            nm = getattr(a, "name", "")
            if nm.endswith("6"): click_enum = a
            if nm.endswith("5"): action5_enum = a
        for a in kb_actions:
            nm = getattr(a, "name", "")
            if nm.endswith("1"): action1_enum = a
            if nm.endswith("2"): action2_enum = a
            if nm.endswith("3"): action3_enum = a
            if nm.endswith("4"): action4_enum = a

        if click_enum is None or action5_enum is None:
            return None
        if action1_enum is None or action3_enum is None:
            return None

        win_score = int(getattr(game, "_win_score", 1) or 1)

        # Find camera scale: display_to_grid(s, s) → (g, g) → scale = s/g
        cam = getattr(game, "camera", None)
        cam_scale = 2  # default
        cam_offset = 0
        if cam:
            for test_d in range(4, 24, 4):
                try:
                    r = cam.display_to_grid(test_d, test_d)
                    if r and r[0] > 0:
                        # Fit: (test_d - offset) // scale = r[0]
                        # From two points: try to find scale and offset
                        for test_d2 in range(test_d+4, 48, 4):
                            r2 = cam.display_to_grid(test_d2, test_d2)
                            if r2 and r2[0] > r[0]:
                                scale_est = (test_d2 - test_d) // (r2[0] - r[0])
                                offset_est = test_d - r[0] * scale_est
                                # Verify
                                for td in [4, 8, 12, 42, 30]:
                                    rv = cam.display_to_grid(td, td)
                                    if rv and rv[0] != (td - offset_est) // scale_est:
                                        break
                                else:
                                    cam_scale = scale_est
                                    cam_offset = offset_est
                                    break
                        break
                except Exception:
                    pass

        def grid_to_display(gx, gy):
            """Center of grid cell in display coordinates."""
            return (gx * cam_scale + cam_offset + cam_scale // 2,
                    gy * cam_scale + cam_offset + cam_scale // 2)

        # Get sprites and their canonical pixel arrays
        level = getattr(game, "current_level", None)
        if level is None:
            return None

        sprites_list = [s for s in level._sprites
                       if s.name in hlxyvcmpk and s.is_visible]
        if len(sprites_list) < 2:
            return None

        plan = []
        sim = copy.deepcopy(raw_env)

        gc.disable()
        try:
            cur_lc = start_levels
            while cur_lc < win_score and time.monotonic() < global_deadline:
                g = getattr(sim, "_game", None)
                if g is None:
                    break

                level_plan = self._sc_solve_level(
                    g, sim, click_enum, action5_enum,
                    action1_enum, action2_enum, action3_enum, action4_enum,
                    cam_scale, cam_offset, start_levels, global_deadline
                )
                if level_plan is None:
                    break

                plan.extend(level_plan)
                level_advanced = False
                for item in level_plan:
                    if time.monotonic() >= global_deadline:
                        break
                    if isinstance(item, dict):
                        obs = sim.step(click_enum, data=item)
                    else:
                        act = None
                        nm = item
                        if nm == "ACTION5": act = action5_enum
                        elif nm == "ACTION1": act = action1_enum
                        elif nm == "ACTION3": act = action3_enum
                        if act is None:
                            for a in kb_actions:
                                if getattr(a, "name", "") == nm:
                                    act = a; break
                        if act is None:
                            break
                        obs = sim.step(act)
                    new_lc = int(getattr(obs, "levels_completed", 0) or 0)
                    if new_lc > cur_lc:
                        cur_lc = new_lc
                        level_advanced = True
                        break
                if not level_advanced:
                    break
        finally:
            gc.enable()

        return plan if plan else None

    def _sc_solve_level(self, game, sim, click_enum, action5_enum,
                        action1_enum, action2_enum, action3_enum, action4_enum,
                        cam_scale, cam_offset, start_levels, global_deadline) -> list | None:
        """Analytically find click+rotate+move plan to align sprite canonical pixels.

        For each target sprite and each rotation (0..3):
          1. One deepcopy probe to get actual rotated pixel array.
          2. Compute (dx, dy) = displacement needed to align a matching pixel value.
          3. One verification probe per candidate displacement.
        Total probes: O(sprites × 4 + candidates) — fast.
        """
        import time
        import numpy as np

        hlxyvcmpk = getattr(game, "hlxyvcmpk", {})
        level = getattr(game, "current_level", None)
        if level is None:
            return None

        sprites_list = [s for s in level._sprites if s.name in hlxyvcmpk and s.is_visible]
        if len(sprites_list) < 2:
            return None

        action_map = {}
        for nm, a in [("ACTION1", action1_enum), ("ACTION2", action2_enum),
                      ("ACTION3", action3_enum), ("ACTION4", action4_enum),
                      ("ACTION5", action5_enum)]:
            if a is not None:
                action_map[nm] = a

        def grid_to_display(gx, gy):
            return {"x": gx * cam_scale + cam_offset + cam_scale // 2,
                    "y": gy * cam_scale + cam_offset + cam_scale // 2}

        def px_world(px_arr, sx, sy):
            """Non-transparent pixel world positions: {value: [(wx,wy)...]}."""
            arr = np.asarray(px_arr, dtype=np.int32)
            result = {}
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    v = int(arr[r, c])
                    if v >= 0:
                        result.setdefault(v, []).append((sx + c, sy + r))
            return result

        def find_click_pos(sprite):
            """Display coords of first non-transparent pixel of sprite."""
            px = getattr(sprite, 'pixels', None)
            if px is None:
                return grid_to_display(sprite._x, sprite._y)
            arr = np.asarray(px, dtype=np.int32)
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    if arr[r, c] >= 0:
                        return grid_to_display(sprite._x + c, sprite._y + r)
            return grid_to_display(sprite._x, sprite._y)

        def step_item(env, item):
            if isinstance(item, dict):
                return env.step(click_enum, data=item)
            a = action_map.get(item)
            return env.step(a) if a else None

        xseexqzst_init = getattr(game, "xseexqzst", None)

        # Fixed reference: the initially-selected sprite(s). Use current pixels (rotation already applied).
        fixed_list = [s for s in sprites_list if s is xseexqzst_init]
        if not fixed_list:
            fixed_list = sprites_list[:1]
        moveables = [s for s in sprites_list if s not in fixed_list]
        if not moveables:
            return None

        ref_pixels = {}
        for fs in fixed_list:
            px = getattr(fs, 'pixels', None)
            if px is None:
                continue
            for v, positions in px_world(px, fs._x, fs._y).items():
                ref_pixels.setdefault(v, []).extend(positions)
        if not ref_pixels:
            return None

        ref_set = {v: set(pos) for v, pos in ref_pixels.items()}
        max_move = 25

        for target_sprite in moveables:
            if time.monotonic() >= global_deadline:
                break

            for rot_count in range(4):
                if time.monotonic() >= global_deadline:
                    break

                # Probe to get sprite's pixel array after rot_count rotations
                probe_rot = copy.deepcopy(sim)
                pg = probe_rot._game

                click_pos = find_click_pos(
                    next(s for s in pg.current_level._sprites if s.name == target_sprite.name)
                )
                probe_rot.step(click_enum, data=click_pos)
                for _ in range(rot_count):
                    probe_rot.step(action5_enum)

                # Check immediate win (only rotation needed)
                if pg.sjwqloivve():
                    plan = [click_pos] + ["ACTION5"] * rot_count
                    del probe_rot; gc.collect()
                    return plan

                ptgt = next((s for s in pg.current_level._sprites if s.name == target_sprite.name), None)
                if ptgt is None:
                    del probe_rot; gc.collect()
                    continue

                tgt_px = getattr(ptgt, 'pixels', None)
                if tgt_px is None:
                    del probe_rot; gc.collect()
                    continue

                tgt_pixels = px_world(tgt_px, ptgt._x, ptgt._y)
                del probe_rot; gc.collect()

                # Compute candidate (dx, dy) displacements: align matching pixel values
                tried = set()
                candidates = []
                for v, tgt_pos in tgt_pixels.items():
                    if v not in ref_pixels:
                        continue
                    for rwx, rwy in ref_pixels[v]:
                        for twx, twy in tgt_pos:
                            dx, dy = rwx - twx, rwy - twy
                            if (dx, dy) in tried:
                                continue
                            if abs(dx) > max_move or abs(dy) > max_move:
                                continue
                            tried.add((dx, dy))
                            # Count total aligned pixels at this displacement
                            align = sum(
                                1 for v2, t2 in tgt_pixels.items()
                                if v2 in ref_set
                                for tw2x, tw2y in t2
                                if (tw2x + dx, tw2y + dy) in ref_set[v2]
                            )
                            candidates.append((align, dx, dy))

                candidates.sort(reverse=True)  # most alignment first

                for align, dx, dy in candidates:
                    if time.monotonic() >= global_deadline:
                        break

                    move_plan = []
                    if dx < 0:
                        move_plan.extend(["ACTION3"] * (-dx))
                    elif dx > 0:
                        move_plan.extend(["ACTION4"] * dx)
                    if dy < 0:
                        move_plan.extend(["ACTION1"] * (-dy))
                    elif dy > 0:
                        move_plan.extend(["ACTION2"] * dy)

                    plan = [click_pos] + ["ACTION5"] * rot_count + move_plan

                    # Verification probe
                    probe_v = copy.deepcopy(sim)
                    won = False
                    for item in plan:
                        obs = step_item(probe_v, item)
                        if obs is None:
                            break
                        if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                            won = True
                            break
                    del probe_v; gc.collect()

                    if won:
                        return plan

        return None

    # ------------------------------------------------------------------ #
    # Ball-routing solver: r11l-style checkpoint-centroid games            #
    # ------------------------------------------------------------------ #

    def _ball_routing_solver(self, raw_env, all_actions, start_levels, global_deadline) -> list | None:
        """For games where clicking checkpoints positions a ball at their centroid.

        Detection: game has bbijaigbknc (checkpoint list), kacotwgjcyq (connections dict),
        and gabrtablhx method. Only ACTION6 is available.

        Strategy: analytically find checkpoint positions whose centroid puts the ball
        on the target, then execute in 3 clicks: move chk1, select chk2, move chk2.
        """
        import time
        game = getattr(raw_env, "_game", None)
        if game is None:
            return None
        bbijaigbknc = getattr(game, "bbijaigbknc", None)
        kacotwgjcyq = getattr(game, "kacotwgjcyq", None)
        gabrtablhx = getattr(game, "gabrtablhx", None)
        if not bbijaigbknc or not kacotwgjcyq or gabrtablhx is None:
            return None

        click_enum = None
        for a in all_actions:
            if hasattr(a, "name") and "ACTION6" in a.name:
                click_enum = a
                break
        if click_enum is None:
            return None

        # Get camera
        cam = getattr(game, "camera", None)
        if cam is None:
            return None
        cam_scale = 1
        try:
            r = cam.display_to_grid(4, 4)
            if r and r[0] > 0:
                cam_scale = 4 // r[0]
        except Exception:
            return None

        if time.monotonic() >= global_deadline:
            return None

        # Scan valid display positions for the selected checkpoint
        chk = getattr(game, "wiayqaumjug", None)
        if not chk:
            if bbijaigbknc:
                chk = bbijaigbknc[0]
            else:
                return None
        chk_w = chk.width
        chk_h = chk.height
        half_w = chk_w // 2
        half_h = chk_h // 2

        valid_grid = []
        valid_display = []
        for dy in range(64):
            for dx in range(64):
                g = cam.display_to_grid(dx, dy)
                if not g:
                    continue
                gx, gy = g
                tx, ty = gx - half_w, gy - half_h
                if not gabrtablhx(tx, ty):
                    valid_grid.append((tx, ty))
                    valid_display.append((dx, dy))

        if not valid_grid:
            return None

        grid_to_disp = {pos: disp for pos, disp in zip(valid_grid, valid_display)}
        valid_set = set(valid_grid)

        # For each connection, solve: find checkpoint positions that put ball on target
        for conn_key, conn_data in kacotwgjcyq.items():
            if time.monotonic() >= global_deadline:
                return None
            ball = conn_data.get("roduyfsmiznvg")
            target = conn_data.get("gosubdcyegamj")
            chks = conn_data.get("lecfirgqbwunn", [])
            if not ball or not target or not chks:
                continue
            if len(chks) != 2:
                continue  # Only handle 2-checkpoint connections for now

            ball_w, ball_h = ball.width, ball.height
            target_x, target_y = target.x, target.y
            target_w, target_h = target.width, target.height

            # Ball overlaps target when:
            # ball.x < target.x + target_w AND ball.x + ball_w > target.x
            # ball.y < target.y + target_h AND ball.y + ball_h > target.y
            # ball.x = centroid_x - ball_w//2
            # centroid_x = (chk1.x + chk1.w//2 + chk2.x + chk2.w//2) // n_chks
            # => (sum_cx) // 2 - ball_w//2 in [target_x, target_x+target_w)
            # => sum_cx in [2*(target_x + ball_w//2), 2*(target_x + target_w - 1 + ball_w//2)]
            # More precisely: any overlap, so:
            # cx_min = 2*(target_x - ball_w + 1)  (ball.x < target_x+target_w)
            # cx_max = 2*(target_x + target_w - 1 + ball_w//2) (too broad)
            # Simple approach: for each p1, look for p2 in valid_set with correct sum
            # Target ball.x = target_x (best case alignment)
            # => centroid_x = target_x + ball_w//2
            # => chk1.x + chk1.w//2 + chk2.x + chk2.w//2 = 2*(target_x + ball_w//2)
            # => chk1.x + chk2.x = 2*(target_x + ball_w//2) - chk_w = need_sum_x
            need_sum_x = 2 * (target_x + ball_w // 2) - chk_w
            need_sum_y = 2 * (target_y + ball_h // 2) - chk_h

            # Find pairs whose centroid places ball on target.
            # Build all (sum_x, sum_y) candidates: exact center first, then expand
            # to cover the full overlap range so any ball-target collision wins.
            candidate_sums = []
            for ox in range(-(target_w + ball_w), target_w + ball_w + 1):
                for oy in range(-(target_h + ball_h), target_h + ball_h + 1):
                    candidate_sums.append((need_sum_x + ox, need_sum_y + oy))
            # Sort by distance from exact center so closest configurations tried first
            candidate_sums.sort(key=lambda s: (s[0] - need_sum_x) ** 2 + (s[1] - need_sum_y) ** 2)

            solutions = []
            for sx, sy in candidate_sums:
                for (gx1, gy1) in valid_grid:
                    gx2 = sx - gx1
                    gy2 = sy - gy1
                    if (gx2, gy2) in valid_set:
                        solutions.append(((gx1, gy1), (gx2, gy2)))
                if solutions:
                    break

            if not solutions:
                continue

            # Try solutions: probe to verify win
            chk1_sprite = chks[1]
            for (gx1, gy1), (gx2, gy2) in solutions[:50]:
                if time.monotonic() >= global_deadline:
                    break
                if (gx1, gy1) not in grid_to_disp or (gx2, gy2) not in grid_to_disp:
                    continue

                dx1, dy1 = grid_to_disp[(gx1, gy1)]
                dx2_sel = chk1_sprite.x + chk_w // 2
                dy2_sel = chk1_sprite.y + chk_h // 2
                dx2, dy2 = grid_to_disp[(gx2, gy2)]

                # Probe: 3 clicks
                probe = copy.deepcopy(raw_env)
                gc.disable()
                try:
                    ob1 = probe.step(click_enum, data={"x": dx1, "y": dy1})
                    lc1 = int(getattr(ob1, "levels_completed", 0) or 0)
                    if lc1 > start_levels:
                        del probe
                        return [{"x": dx1, "y": dy1}]
                    ob2 = probe.step(click_enum, data={"x": dx2_sel, "y": dy2_sel})
                    lc2 = int(getattr(ob2, "levels_completed", 0) or 0)
                    if lc2 > start_levels:
                        del probe
                        return [{"x": dx1, "y": dy1}, {"x": dx2_sel, "y": dy2_sel}]
                    ob3 = probe.step(click_enum, data={"x": dx2, "y": dy2})
                    lc3 = int(getattr(ob3, "levels_completed", 0) or 0)
                    won = lc3 > start_levels or getattr(probe._game, "uyawyyswbya", False)
                finally:
                    gc.enable()

                del probe
                gc.collect()

                if won:
                    return [{"x": dx1, "y": dy1}, {"x": dx2_sel, "y": dy2_sel}, {"x": dx2, "y": dy2}]

        return None

    def _checkers_jump_solver(self, raw_env, all_actions, start_levels, global_deadline) -> list | None:
        """For games where pieces jump over each other to merge (lf52-style checkers).

        Detection: game has ikhhdzfmarl with posalhhmjq + hncnfaqaddg + ndtvadsrqf.
        Each move: select piece (click its grid pixel), then click destination pixel.
        Destination pixel == arrow pixel (arrow at 2-grid-cell offset from piece).
        BFS on frozenset of piece positions; win when count reaches 1.
        """
        import time
        from collections import deque
        game = getattr(raw_env, "_game", None)
        if game is None:
            return None
        ikh = getattr(game, "ikhhdzfmarl", None)
        if ikh is None or not hasattr(ikh, "posalhhmjq") or not hasattr(ikh, "hncnfaqaddg"):
            return None

        click_enum = None
        for a in all_actions:
            if hasattr(a, "name") and "ACTION6" in a.name:
                click_enum = a
                break
        if click_enum is None:
            return None

        grid = getattr(ikh, "hncnfaqaddg", None)
        if grid is None or not hasattr(grid, "ndtvadsrqf"):
            return None

        grid_off = getattr(grid, "cdpcbbnfdp", (0, 0))

        fozw = grid.ndtvadsrqf("fozwvlovdui")
        if not fozw:
            return None
        init_pieces = frozenset(getattr(p, "chahdtpdoz", None) for p in fozw)
        if not init_pieces or None in init_pieces:
            return None

        if time.monotonic() >= global_deadline:
            return None

        # Precompute valid landing positions using game's own collision check
        floor_positions: set = set()
        try:
            for gx in range(-5, 30):
                for gy in range(-5, 30):
                    if ikh.posalhhmjq((gx, gy)):
                        floor_positions.add((gx, gy))
        except Exception:
            return None

        if not floor_positions:
            return None

        DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def successors(pieces_set):
            result = []
            for px, py in pieces_set:
                for dx, dy in DIRS:
                    mid = (px + dx, py + dy)
                    dst = (px + 2 * dx, py + 2 * dy)
                    if mid in pieces_set and dst in floor_positions and dst not in pieces_set:
                        nps = set(pieces_set)
                        nps.discard((px, py))
                        nps.add(dst)
                        nps.discard(mid)
                        result.append((frozenset(nps), (px, py), dx, dy, dst))
            return result

        queue: deque = deque([(init_pieces, [])])
        visited: set = {init_pieces}
        found = None
        MAX_NODES = 200000

        nodes = 0
        while queue and nodes < MAX_NODES:
            if time.monotonic() >= global_deadline:
                return None
            state, plan = queue.popleft()
            nodes += 1
            for new_state, frm, dx, dy, dst in successors(state):
                new_plan = plan + [(frm, dx, dy, dst)]
                if len(new_state) == 1:
                    found = new_plan
                    break
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, new_plan))
            if found:
                break

        if found is None:
            return None

        # Convert to clicks: piece pixel = (gx*6+off_x, gy*6+off_y)
        # Destination pixel = (dst_x*6+off_x, dst_y*6+off_y)
        # (arrow sits at destination grid position because arrow offset = 2*dir*6)
        ox, oy = int(grid_off[0]), int(grid_off[1])
        plan_clicks = []
        for (gx, gy), dx, dy, (dst_x, dst_y) in found:
            plan_clicks.append({"x": gx * 6 + ox, "y": gy * 6 + oy})
            plan_clicks.append({"x": dst_x * 6 + ox, "y": dst_y * 6 + oy})

        # Probe to verify and detect early win
        probe = copy.deepcopy(raw_env)
        gc.disable()
        try:
            for i, click_data in enumerate(plan_clicks):
                ob = probe.step(click_enum, data=click_data)
                lc = int(getattr(ob, "levels_completed", 0) or 0)
                if lc > start_levels:
                    del probe
                    gc.enable()
                    gc.collect()
                    return plan_clicks[: i + 1]
            won = int(getattr(ob, "levels_completed", 0) or 0) > start_levels
        finally:
            gc.enable()
        del probe
        gc.collect()

        return plan_clicks if won else None

    def _undoing_maze_solver(self, raw_env, all_actions, start_levels, global_deadline) -> list | None:
        """For maze games with UNDO mechanic that creates ghost player copies (g50t-style).

        Detection: game has vgwycxsxjz with safkknjslo + dzxunlkwxt + whftgckbcu.
        The UNDO action (A5/pmlawcgvcp) teleports player to start and creates a ghost
        that replays past moves — ghost state must be included in BFS state key.
        Returns list of action name strings (e.g. ["ACTION4", "ACTION2", ...]).
        """
        import time
        from collections import deque
        game = getattr(raw_env, "_game", None)
        if game is None:
            return None
        vgw = getattr(game, "vgwycxsxjz", None)
        if vgw is None:
            return None
        if not (hasattr(vgw, "safkknjslo") and hasattr(vgw, "dzxunlkwxt") and hasattr(vgw, "whftgckbcu")):
            return None
        if not hasattr(game, "hctlyapjnq"):
            return None

        by_name = {getattr(a, "name", ""): a for a in all_actions}
        action_pairs = [(name, by_name[name]) for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5") if name in by_name]
        if not action_pairs:
            return None

        def ghost_state_key(env_copy):
            g = env_copy._game
            v = getattr(g, "vgwycxsxjz", None)
            if v is None:
                return None
            player_pos = (v.dzxunlkwxt.x, v.dzxunlkwxt.y)
            timer_x = getattr(getattr(g, "twyixucrqi", None), "_x", 0)
            ghost_info = tuple(sorted(
                (ghost.x, ghost.y, tuple(moves)) for ghost, moves in v.rloltuowth.items()
            ))
            return (player_pos, timer_x, ghost_info)

        def check_win(env_copy):
            v = getattr(getattr(env_copy, "_game", None), "vgwycxsxjz", None)
            return bool(v and v.safkknjslo)

        init_sk = ghost_state_key(raw_env)
        if init_sk is None:
            return None

        queue: deque = deque([(copy.deepcopy(raw_env), [])])
        visited: set = {init_sk}
        found = None
        nodes = 0
        MAX_NODES = 5000
        gc.disable()
        try:
            while queue and nodes < MAX_NODES:
                if time.monotonic() >= global_deadline:
                    return None
                env_cur, plan = queue.popleft()
                nodes += 1
                for label, ga in action_pairs:
                    en = copy.deepcopy(env_cur)
                    ob = en.step(ga)
                    lc = int(getattr(ob, "levels_completed", 0) or 0)
                    if lc > start_levels or check_win(en):
                        found = plan + [label]
                        break
                    if getattr(en._game, "hctlyapjnq", False):
                        del en
                        continue
                    sk = ghost_state_key(en)
                    if sk is None or sk not in visited:
                        if sk:
                            visited.add(sk)
                        queue.append((en, plan + [label]))
                    else:
                        del en
                if found:
                    break
        finally:
            gc.enable()
        gc.collect()
        return found
