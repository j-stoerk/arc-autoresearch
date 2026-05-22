"""dc22 deeper analysis: try path-tracking BFS and click sequences."""
import gc
import copy
import time
from collections import deque
import numpy as np
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "dc22" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)
    level = getattr(game, "current_level", None)
    sprites = getattr(level, "_sprites", []) if level else []

    print(f"=== dc22 deeper analysis ===  win_score={game._win_score}")

    try:
        from arcengine.enums import GameAction as _GameAction
        kb_actions = [_GameAction.ACTION1, _GameAction.ACTION2, _GameAction.ACTION3, _GameAction.ACTION4]
        click_action = _GameAction.ACTION6

        # Scale
        cam = getattr(game, "camera", None)
        scale = 1
        if cam:
            try:
                r = cam.display_to_grid(4, 4)
                if r and r[0] > 0: scale = 4 // r[0]
            except: pass
        print(f"scale={scale}")

        # Click positions for sys_click sprites
        sys_clicks = []
        for s in sprites:
            if "sys_click" in str(getattr(s, "tags", [])):
                px = getattr(s, "pixels", None)
                pw = px.shape[1] if px is not None and hasattr(px, "shape") else 1
                ph = px.shape[0] if px is not None and hasattr(px, "shape") else 1
                cx = (int(s._x) + pw // 2) * scale
                cy = (int(s._y) + ph // 2) * scale
                sys_clicks.append({"x": cx, "y": cy, "name": str(getattr(s, "name", ""))})
        print(f"Sys_click sprites: {sys_clicks}")

        # BFS including PATH as part of state (visiting nodes matters)
        # State = (position of all sprites, visited_set)
        def pos_key(env_):
            g_ = getattr(env_, "_game", None)
            l_ = getattr(g_, "current_level", None)
            sps_ = getattr(l_, "_sprites", []) if l_ else []
            return tuple((s._x, s._y) for s in sps_)

        # Try: what happens if we click buezna sprites?
        print("\n--- test sys_click actions ---")
        for sc in sys_clicks:
            probe = copy.deepcopy(raw_env)
            obs2 = probe.step(click_action, data=sc)
            state = _state_name(obs2)
            lc = int(getattr(obs2, "levels_completed", 0) or 0)
            sc2 = probe._game._score
            pos_after = pos_key(probe)
            print(f"  click {sc['name']} at ({sc['x']},{sc['y']}): state={state} score={sc2} lc={lc}")
            # Check game state vars
            g2 = probe._game
            for av in ["guspipewt", "fadccmsnb", "fjiyimenq", "dxcfrrcpp", "scshqquvb"]:
                print(f"    {av}={getattr(g2, av, '?')}")

        # BFS with path tracking: each position can be visited at most once per path
        print("\n--- path-tracking BFS (positions × visited) ---")
        init_pos = pos_key(raw_env)
        init_state = (init_pos, frozenset([init_pos]))  # (current, visited_set)
        queue = deque([(copy.deepcopy(raw_env), [], frozenset([init_pos]))])
        seen = {init_state}
        nodes = 0
        found = False
        t0 = time.monotonic()
        gc.disable()
        try:
            while queue and nodes < 500 and time.monotonic() - t0 < 30:
                cur, path, visited = queue.popleft()
                nodes += 1
                for act in kb_actions:
                    try:
                        nxt = copy.deepcopy(cur)
                        obs2 = nxt.step(act)
                        if _state_name(obs2) == "WIN" or int(getattr(obs2,"levels_completed",0) or 0) > 0:
                            print(f"WIN at node {nodes}! path={path+[act.name]}")
                            found = True; break
                        pos = pos_key(nxt)
                        new_visited = visited | {pos}
                        state = (pos, frozenset(new_visited))
                        if state not in seen:
                            seen.add(state)
                            queue.append((nxt, path+[act.name], new_visited))
                    except: pass
                if found: break
        finally:
            gc.enable()
        print(f"Path-tracking BFS: nodes={nodes} unique={(len(seen))} t={time.monotonic()-t0:.1f}s found={found} queue={len(queue)}")

        # Check if clicking sys_click opens up new keyboard states
        print("\n--- BFS after initial sys_click ---")
        for sc in sys_clicks:
            probe = copy.deepcopy(raw_env)
            probe.step(click_action, data=sc)
            init_pos2 = pos_key(probe)
            q2 = deque([(copy.deepcopy(probe), [])])
            seen2 = {init_pos2}
            nodes2 = 0
            found2 = False
            gc.disable()
            try:
                while q2 and nodes2 < 100:
                    cur2, p2 = q2.popleft()
                    nodes2 += 1
                    for act in kb_actions:
                        try:
                            nxt2 = copy.deepcopy(cur2)
                            obs2 = nxt2.step(act)
                            if _state_name(obs2) == "WIN" or int(getattr(obs2,"levels_completed",0) or 0) > 0:
                                print(f"  WIN after clicking {sc['name']}! path={p2+[act.name]}")
                                found2 = True; break
                            k2 = pos_key(nxt2)
                            if k2 not in seen2:
                                seen2.add(k2); q2.append((nxt2, p2+[act.name]))
                        except: pass
                    if found2: break
            finally:
                gc.enable()
            print(f"  After click {sc['name']}: {len(seen2)} states  found={found2}")

        # Look at ujotjblwn object (jktvoccigf)
        print("\n--- ujotjblwn (jktvoccigf) state ---")
        ujot = getattr(game, "ujotjblwn", None)
        if ujot:
            for av, vv in sorted(vars(ujot).items()):
                if av.startswith("__"): continue
                if callable(vv): continue
                s = str(vv)
                if len(s) > 80: s = s[:80] + "..."
                print(f"  .{av}: {type(vv).__name__} = {s}")
            print(f"  ykevdpbntc(): {ujot.ykevdpbntc()}")

    except Exception as e:
        import traceback; traceback.print_exc()
    break
print("Done.")
