"""sb26: ACTION5 builds tree, then navigate with clicks."""
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

    print("=== sb26: ACTION5 → navigate ===")

    try:
        from arcengine.enums import GameAction as _GameAction
        action5 = _GameAction.ACTION5
        action6 = _GameAction.ACTION6
        start_lc = int(getattr(obs, "levels_completed", 0) or 0)

        # First: ACTION5 to build tree
        probe_a5 = copy.deepcopy(raw_env)
        obs_a5 = probe_a5.step(action5)
        g5 = probe_a5._game
        print(f"After ACTION5: lc={int(getattr(obs_a5,'levels_completed',0) or 0)} state={_state_name(obs_a5)}")
        print(f"  buvfjfmpp: {[(f.name,i) for f,i in g5.buvfjfmpp]}")
        print(f"  pmygakdvy: {g5.pmygakdvy}")
        print(f"  sjcuorclg: {g5.sjcuorclg}")
        print(f"  rzbeqaiky:")
        for frame, nodes in g5.rzbeqaiky.items():
            print(f"    frame {frame.name}: {[n.name for n in nodes]}")
        print(f"  wcfyiodrx[0] center: {g5.wcfyiodrx[0].pixels[1,1] if g5.wcfyiodrx else '?'}")

        # Find sys_click positions
        scale = 1
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

        # After ACTION5: test each click
        print(f"\n--- After ACTION5: test clicks ---")
        for sc in sys_clicks:
            probe = copy.deepcopy(probe_a5)
            obs2 = probe.step(action6, data=sc)
            lc = int(getattr(obs2, "levels_completed", 0) or 0)
            state = _state_name(obs2)
            g2 = probe._game
            print(f"  click {sc['name']!r} ({sc['x']},{sc['y']}): lc={lc} state={state} pmyg={g2.pmygakdvy} buvf={[(f.name,i) for f,i in g2.buvfjfmpp]} modq={g2.modqnpqfi}")

        # BFS starting from after ACTION5
        print(f"\n--- BFS after ACTION5 (200 nodes) ---")
        def path_key(env_):
            g_ = getattr(env_, "_game", None)
            if g_ is None: return ()
            buvf = tuple((f.name, i) for f, i in getattr(g_, "buvfjfmpp", []))
            pmyg = getattr(g_, "pmygakdvy", 0)
            ppsx = getattr(g_, "ppsxsxiod", False)
            sjcu = getattr(g_, "sjcuorclg", 0)
            artf = getattr(g_, "artsfnufc", 0)
            modq = getattr(g_, "modqnpqfi", 0)
            l_ = getattr(g_, "current_level", None)
            sps_ = getattr(l_, "_sprites", []) if l_ else []
            px_key = tuple((s._x, s._y) for s in sps_[:15])
            return (buvf, pmyg, ppsx, sjcu, artf, modq, px_key)

        init_k = path_key(probe_a5)
        queue = deque([(copy.deepcopy(probe_a5), ["A5"])])
        seen = {init_k}
        nodes = 0
        found = False
        t0 = time.monotonic()
        gc.disable()
        all_acts = []
        for sc in sys_clicks:
            all_acts.append({"action": action6, "data": sc, "name": f"c{sc['x']},{sc['y']}"})
        all_acts.append({"action": action5, "data": None, "name": "A5"})

        try:
            while queue and nodes < 200 and time.monotonic() - t0 < 30:
                cur, path = queue.popleft()
                nodes += 1
                for act in all_acts:
                    try:
                        nxt = copy.deepcopy(cur)
                        if act["data"] is not None:
                            obs2 = nxt.step(act["action"], data=act["data"])
                        else:
                            obs2 = nxt.step(act["action"])
                        lc = int(getattr(obs2, "levels_completed", 0) or 0)
                        if lc > start_lc or _state_name(obs2) == "WIN":
                            print(f"\nWIN! path: {path+[act['name']]}")
                            found = True; break
                        k = path_key(nxt)
                        if k not in seen:
                            seen.add(k)
                            queue.append((nxt, path+[act["name"]]))
                    except: pass
                if found: break
        finally:
            gc.enable()

        print(f"BFS: nodes={nodes} unique={len(seen)} t={time.monotonic()-t0:.1f}s found={found}")

        # Try: ACTION5 to build, then analyze lngftsryyw vs susublrply nodes
        print("\n--- Understand node types ---")
        probe5 = copy.deepcopy(probe_a5)
        for sc in sys_clicks:
            probe5b = copy.deepcopy(probe_a5)
            obs2 = probe5b.step(action6, data=sc)
            g2 = probe5b._game
            # Check what node was selected
            buvf = g2.buvfjfmpp
            if buvf:
                frame, idx = buvf[-1]
                nodes_list = g2.rzbeqaiky.get(frame, [])
                if idx < len(nodes_list):
                    node = nodes_list[idx]
                    print(f"  click {sc['name']!r}: current node={node.name} pixels={node.pixels[1,1] if node.pixels is not None else '?'}")
                else:
                    print(f"  click {sc['name']!r}: idx={idx} out of range")
            else:
                print(f"  click {sc['name']!r}: buvfjfmpp empty")

    except Exception as e:
        import traceback; traceback.print_exc()
    break
print("Done.")
