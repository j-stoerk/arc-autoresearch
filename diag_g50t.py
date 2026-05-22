"""Targeted diagnostic for g50t: understand win condition, timer, and state space."""
import gc
import copy
import time
import heapq
from collections import deque
import numpy as np
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "g50t" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)
    level = getattr(game, "current_level", None)
    sprites = getattr(level, "_sprites", []) if level else []

    print(f"=== g50t diagnostic ===")
    print(f"n_sprites={len(sprites)}")

    # Game state
    vgwy = getattr(game, "vgwycxsxjz", None)
    print(f"\nGame state:")
    print(f"  _win_score={game._win_score}")
    print(f"  ucorwtereb={game.ucorwtereb}")
    if vgwy:
        dzx = getattr(vgwy, "dzxunlkwxt", None)  # player
        whf = getattr(vgwy, "whftgckbcu", None)  # target
        print(f"  player (dzxunlkwxt): ({dzx.x},{dzx.y})" if dzx else "  player: None")
        print(f"  target (whftgckbcu): ({whf.x},{whf.y})" if whf else "  target: None")
        print(f"  win_pos: ({whf.x+1},{whf.y+1})" if whf else "  win_pos: ?")
        print(f"  safkknjslo (win): {vgwy.safkknjslo}")
        print(f"  zvuxrhnlcb (lose): {vgwy.zvuxrhnlcb}")

    # Timer
    twy = getattr(game, "twyixucrqi", None)
    if twy:
        print(f"  timer: pos=({twy.x},{twy.y}) width={twy.width}")
        print(f"  max_moves approx: {2 * (1 + twy.width)} (timer moves -1 every 2 actions)")

    # Sprites
    print("\n--- sprites ---")
    for i, s in enumerate(sprites[:25]):
        nm = str(getattr(s, "name", ""))
        tg = str(getattr(s, "tags", []))
        print(f"  [{i}] name={nm!r} tags={tg} pos=({s._x},{s._y})")

    # State key
    def state_key(env_):
        g_ = getattr(env_, "_game", None)
        l_ = getattr(g_, "current_level", None)
        sps_ = getattr(l_, "_sprites", []) if l_ else []
        return tuple((s._x, s._y) for s in sps_)

    def dist_to_win(env_):
        g_ = getattr(env_, "_game", None)
        v_ = getattr(g_, "vgwycxsxjz", None)
        if v_ is None: return 999
        d = v_.dzxunlkwxt
        w = v_.whftgckbcu
        if d is None or w is None: return 999
        return abs(d.x - (w.x + 1)) + abs(d.y - (w.y + 1))

    try:
        from arcengine.enums import GameAction as _GameAction
        kb_actions = [_GameAction.ACTION1, _GameAction.ACTION2, _GameAction.ACTION3, _GameAction.ACTION4]
        all_kb = [_GameAction.ACTION1, _GameAction.ACTION2, _GameAction.ACTION3, _GameAction.ACTION4, _GameAction.ACTION5]

        print(f"\nInitial dist to win: {dist_to_win(raw_env)}")
        print(f"Initial state key: {state_key(raw_env)}")

        # Greedy search: try to minimize dist to win
        print("\n--- Greedy search (minimize dist to win) ---")
        probe = copy.deepcopy(raw_env)
        path = []
        cur_dist = dist_to_win(probe)
        for step in range(30):
            best_act = None
            best_dist = cur_dist
            for act in kb_actions:
                p2 = copy.deepcopy(probe)
                obs2 = p2.step(act)
                if _state_name(obs2) == "WIN" or int(getattr(obs2,"levels_completed",0) or 0) > 0:
                    print(f"WIN at step {step}!")
                    break
                d = dist_to_win(p2)
                if d < best_dist:
                    best_dist = d
                    best_act = act
            else:
                if best_act:
                    obs2 = probe.step(best_act)
                    path.append(best_act.name)
                    cur_dist = best_dist
                    vg = probe._game.vgwycxsxjz
                    print(f"  step {step}: {best_act.name} -> dist={best_dist} player=({vg.dzxunlkwxt.x},{vg.dzxunlkwxt.y})")
                else:
                    print(f"  step {step}: no improvement, stuck at dist={cur_dist}")
                    break
                continue
            break

        # A* BFS towards target
        print("\n--- A* BFS (up to 10000 nodes) ---")
        init_key = state_key(raw_env)
        init_d = dist_to_win(raw_env)
        # heap: (f, counter, env, path)
        heap = [(init_d, 0, copy.deepcopy(raw_env), [])]
        seen = {init_key}
        nodes = 0
        counter = 0
        found = False
        t0 = time.monotonic()
        gc.disable()
        try:
            while heap and nodes < 10000 and time.monotonic() - t0 < 60:
                f, _, cur, path2 = heapq.heappop(heap)
                nodes += 1
                for act in kb_actions:
                    try:
                        nxt = copy.deepcopy(cur)
                        obs2 = nxt.step(act)
                        if _state_name(obs2) == "WIN" or int(getattr(obs2,"levels_completed",0) or 0) > 0:
                            print(f"WIN at node {nodes}! path length={len(path2)+1}")
                            print(f"  Path: {path2+[act.name]}")
                            found = True
                            break
                        k = state_key(nxt)
                        if k not in seen:
                            seen.add(k)
                            d = dist_to_win(nxt)
                            g = len(path2) + 1
                            counter += 1
                            heapq.heappush(heap, (g + d, counter, nxt, path2 + [act.name]))
                    except: pass
                if found: break
        finally:
            gc.enable()
        print(f"A* done: nodes={nodes} unique={len(seen)} t={time.monotonic()-t0:.1f}s found={found}")
        if not found:
            print(f"Best dist reached: min f={heap[0][0] if heap else '?'}")

        # Count reachable states without timer constraint
        print("\n--- Pure BFS (no timer, count all reachable states) ---")
        seen_pure = {init_key}
        q_pure = deque([copy.deepcopy(raw_env)])
        nodes_pure = 0
        t0p = time.monotonic()
        gc.disable()
        try:
            while q_pure and nodes_pure < 5000 and time.monotonic() - t0p < 20:
                cur = q_pure.popleft()
                nodes_pure += 1
                for act in kb_actions:
                    try:
                        nxt = copy.deepcopy(cur)
                        obs2 = nxt.step(act)
                        if _state_name(obs2) == "WIN" or int(getattr(obs2,"levels_completed",0) or 0) > 0:
                            print(f"WIN at node {nodes_pure}!")
                            break
                        k = state_key(nxt)
                        if k not in seen_pure:
                            seen_pure.add(k)
                            q_pure.append(nxt)
                    except: pass
        finally:
            gc.enable()
        print(f"Pure BFS done: nodes={nodes_pure} unique={len(seen_pure)} t={time.monotonic()-t0p:.1f}s queue_left={len(q_pure)}")

    except Exception as e:
        import traceback
        traceback.print_exc()

    break
print("Done.")
