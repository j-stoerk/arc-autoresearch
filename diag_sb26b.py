"""
sb26: BFS with game path state key (buvfjfmpp, pmygakdvy, ppsxsxiod).
The game is a tree-navigation puzzle: click nodes to follow the correct path.
"""
import gc
import copy
import time
from collections import deque
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "sb26" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)
    level = getattr(game, "current_level", None)
    sprites = level._sprites if level else []

    print(f"=== sb26 path-state BFS ===  win_score={game._win_score}")

    try:
        from arcengine.enums import GameAction as _GameAction
        action5 = _GameAction.ACTION5
        action6 = _GameAction.ACTION6
        action7 = _GameAction.ACTION7
        start_lc = int(getattr(obs, "levels_completed", 0) or 0)

        # Scale
        cam = getattr(game, "camera", None)
        scale = 1
        if cam:
            try:
                r = cam.display_to_grid(4, 4)
                if r and r[0] > 0: scale = 4 // r[0]
            except: pass

        # Better state key: path position + sprite pixel key
        def state_key(env_):
            g_ = getattr(env_, "_game", None)
            if g_ is None: return ()
            # Path state
            buvf = tuple((id(f), i) for f, i in getattr(g_, "buvfjfmpp", []))
            pmyg = getattr(g_, "pmygakdvy", 0)
            ppsx = getattr(g_, "ppsxsxiod", False)
            sjcu = getattr(g_, "sjcuorclg", 0)
            artf = getattr(g_, "artsfnufc", 0)
            # Sprite content key (node pixels)
            l_ = getattr(g_, "current_level", None)
            sps_ = getattr(l_, "_sprites", []) if l_ else []
            px_key = tuple((s._x, s._y) for s in sps_[:20])
            return (buvf, pmyg, ppsx, sjcu, artf, px_key)

        # Find all sys_click sprites
        sys_clicks = []
        seen_xy = set()
        for s in sprites:
            if "sys_click" in str(getattr(s, "tags", [])):
                px = getattr(s, "pixels", None)
                pw = px.shape[1] if px is not None and hasattr(px, "shape") else 1
                ph = px.shape[0] if px is not None and hasattr(px, "shape") else 1
                cx = (int(s._x) + pw // 2) * scale
                cy = (int(s._y) + ph // 2) * scale
                if (cx, cy) not in seen_xy:
                    seen_xy.add((cx, cy))
                    sys_clicks.append({"x": cx, "y": cy, "name": str(getattr(s, "name", ""))})
        print(f"Sys-click positions: {len(sys_clicks)}")
        for sc in sys_clicks:
            print(f"  {sc['name']!r} ({sc['x']},{sc['y']})")

        # Show tree structure
        g = raw_env._game
        print(f"\nTree structure:")
        print(f"  buvfjfmpp: {[(f.name, i) for f, i in g.buvfjfmpp]}")
        print(f"  qaagahahj: {[f.name for f in g.qaagahahj]}")
        print(f"  pmygakdvy: {g.pmygakdvy}")
        print(f"  wcfyiodrx len={len(g.wcfyiodrx)}")
        if g.wcfyiodrx:
            wc = g.wcfyiodrx[0]
            px = getattr(wc, "pixels", None)
            print(f"  wcfyiodrx[0]: name={wc.name} pos=({wc.x},{wc.y}) center_px={px[1,1] if px is not None else '?'}")
        print(f"  rzbeqaiky:")
        for frame, nodes in g.rzbeqaiky.items():
            print(f"    frame {frame.name}: {[n.name for n in nodes]}")

        # BFS with good state key
        print("\n--- Path-state BFS (500 nodes, 60s) ---")
        init_key = state_key(raw_env)
        actions = []
        for sc in sys_clicks:
            actions.append({"action": action6, "data": sc, "name": f"click{sc['x']},{sc['y']}"})
        actions.append({"action": action5, "data": None, "name": "A5"})

        queue = deque([(copy.deepcopy(raw_env), [])])
        seen = {init_key}
        nodes = 0
        found = False
        t0 = time.monotonic()
        gc.disable()
        try:
            while queue and nodes < 500 and time.monotonic() - t0 < 60:
                cur, path = queue.popleft()
                nodes += 1
                for act in actions:
                    try:
                        nxt = copy.deepcopy(cur)
                        if act["data"] is not None:
                            obs2 = nxt.step(act["action"], data=act["data"])
                        else:
                            obs2 = nxt.step(act["action"])
                        lc = int(getattr(obs2, "levels_completed", 0) or 0)
                        if lc > start_lc or _state_name(obs2) == "WIN":
                            print(f"\nWIN! path (len={len(path)+1}): {path+[act['name']]}")
                            found = True; break
                        k = state_key(nxt)
                        if k not in seen:
                            seen.add(k)
                            queue.append((nxt, path + [act["name"]]))
                    except Exception as e:
                        pass
                if found: break
        finally:
            gc.enable()
        print(f"BFS: nodes={nodes} unique={len(seen)} t={time.monotonic()-t0:.1f}s found={found}")

        # Show state after each click
        if not found:
            print("\n--- State after each click ---")
            for sc in sys_clicks:
                probe = copy.deepcopy(raw_env)
                obs2 = probe.step(action6, data=sc)
                g2 = probe._game
                print(f"  click {sc['name']!r}: pmygakdvy={g2.pmygakdvy} buvfjfmpp={[(f.name,i) for f,i in g2.buvfjfmpp]} ppsxsxiod={g2.ppsxsxiod} modqnpqfi={g2.modqnpqfi}")

    except Exception as e:
        import traceback; traceback.print_exc()
    break
print("Done.")
