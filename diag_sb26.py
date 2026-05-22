"""
sb26: First-principles analysis.
- Available actions: [5, 6, 7]
- ACTION5, 6, 7 only
- Find controllable objects, true action vocabulary, state latent vars
"""
import gc
import copy
import time
import numpy as np
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "sb26" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)
    level = getattr(game, "current_level", None)
    sprites = level._sprites if level else []

    print(f"=== sb26 first-principles analysis ===")
    print(f"n_sprites={len(sprites)}  win_score={game._win_score}")

    # Object discovery: sprites by tag
    print("\n--- Sprites ---")
    for i, s in enumerate(sprites[:30]):
        nm = str(getattr(s, "name", ""))
        tg = str(getattr(s, "tags", []))
        px = getattr(s, "pixels", None)
        pw = px.shape[1] if px is not None and hasattr(px, "shape") else 0
        ph = px.shape[0] if px is not None and hasattr(px, "shape") else 0
        print(f"  [{i}] {nm!r} tags={tg} pos=({s._x},{s._y}) size={pw}x{ph}")

    # State variables
    print("\n--- Game state variables ---")
    for av, vv in sorted(vars(game).items()):
        if av.startswith("__"): continue
        if callable(vv): continue
        sv = str(vv)
        if len(sv) > 70: sv = sv[:70] + "..."
        print(f"  .{av}: {type(vv).__name__} = {sv}")

    try:
        from arcengine.enums import GameAction as _GameAction
        action5 = _GameAction.ACTION5
        action6 = _GameAction.ACTION6
        action7 = _GameAction.ACTION7
        start_lc = int(getattr(obs, "levels_completed", 0) or 0)

        # Camera scale
        cam = getattr(game, "camera", None)
        scale = 1
        if cam:
            try:
                r = cam.display_to_grid(4, 4)
                if r and r[0] > 0: scale = 4 // r[0]
            except: pass
        print(f"\nscale={scale}")

        # Action vocabulary discovery: what changes with each action type
        print("\n--- Action effects ---")
        for aname, aval in [("ACTION5", action5), ("ACTION7", action7)]:
            probe = copy.deepcopy(raw_env)
            obs2 = probe.step(aval)
            state = _state_name(obs2)
            lc = int(getattr(obs2, "levels_completed", 0) or 0)
            g2 = probe._game
            # Find changed vars
            changed = []
            for av, vv in sorted(vars(g2).items()):
                if av.startswith("__") or callable(vv): continue
                orig = getattr(game, av, None)
                if str(orig) != str(vv):
                    changed.append(f"{av}: {str(orig)[:20]}->{str(vv)[:20]}")
            print(f"  {aname}: state={state} lc={lc} changes={changed[:5]}")

        # For ACTION6: need click coords
        # Find sys_click sprites
        sys_clicks = []
        for s in sprites:
            if "sys_click" in str(getattr(s, "tags", [])):
                px = getattr(s, "pixels", None)
                pw = px.shape[1] if px is not None and hasattr(px, "shape") else 1
                ph = px.shape[0] if px is not None and hasattr(px, "shape") else 1
                cx = (int(s._x) + pw // 2) * scale
                cy = (int(s._y) + ph // 2) * scale
                sys_clicks.append({"x": cx, "y": cy, "name": str(getattr(s, "name", ""))})
        print(f"\nSys_click positions ({len(sys_clicks)}): {sys_clicks[:10]}")

        # Test each sys_click
        print("\n--- Click effects ---")
        state_vars_before = {av: str(vv) for av, vv in vars(game).items()
                             if not av.startswith("__") and not callable(vv)}
        for sc in sys_clicks[:15]:
            probe = copy.deepcopy(raw_env)
            obs2 = probe.step(action6, data=sc)
            state = _state_name(obs2)
            lc = int(getattr(obs2, "levels_completed", 0) or 0)
            g2 = probe._game
            # Find changed vars
            changed = []
            for av, vv in sorted(vars(g2).items()):
                if av.startswith("__") or callable(vv): continue
                orig = state_vars_before.get(av)
                if orig and str(vv) != orig:
                    changed.append(av)
            sc2 = g2._score
            print(f"  click {sc['name']!r} ({sc['x']},{sc['y']}): state={state} lc={lc} score={sc2} changed={changed[:4]}")

        # Try ACTION5 then ACTION6 combinations
        print("\n--- ACTION5 + click ---")
        probe = copy.deepcopy(raw_env)
        probe.step(action5)
        g5 = probe._game
        for av in ["sjcuorclg", "incrguxqwfjtial_energy", "ppsxsxiod", "modqnpqfi"]:
            print(f"  after ACTION5: {av}={getattr(g5, av, '?')}")
        for sc in sys_clicks[:5]:
            p2 = copy.deepcopy(probe)
            obs2 = p2.step(action6, data=sc)
            lc = int(getattr(obs2, "levels_completed", 0) or 0)
            print(f"  A5 + click {sc['name']!r}: lc={lc} state={_state_name(obs2)}")

        # Explore what "lngftsryyw" and "pkpgflvjel" and "quhhhthrri" tags mean
        print("\n--- Tagged sprite groups ---")
        for tag in ["pkpgflvjel", "lngftsryyw", "quhhhthrri", "susublrply"]:
            tagged = [s for s in sprites if tag in str(getattr(s, "tags", []))]
            print(f"  tag={tag!r}: {len(tagged)} sprites")
            for s in tagged[:3]:
                print(f"    {str(getattr(s,'name',''))!r} pos=({s._x},{s._y}) visible={s.is_visible}")

        # State key analysis: what determines unique game state?
        def state_key(env_):
            g_ = getattr(env_, "_game", None)
            return (
                getattr(g_, "_score", 0),
                getattr(g_, "sjcuorclg", 0),  # energy remaining
                getattr(g_, "artsfnufc", 0),   # some counter
                getattr(g_, "modqnpqfi", 0),   # step counter
                getattr(g_, "ppsxsxiod", False), # flag
            )

        print(f"\nInitial state key: {state_key(raw_env)}")

        # Quick BFS: what states are reachable?
        from collections import deque
        queue = deque([(copy.deepcopy(raw_env), [])])
        seen = {state_key(raw_env)}
        nodes = 0
        found = False
        t0 = time.monotonic()
        gc.disable()
        all_actions = [action5, action7]
        for sc in sys_clicks:
            all_actions.append({"action": action6, "data": sc})
        try:
            while queue and nodes < 200 and time.monotonic() - t0 < 20:
                cur, path = queue.popleft()
                nodes += 1
                for act in all_actions:
                    try:
                        nxt = copy.deepcopy(cur)
                        if isinstance(act, dict):
                            obs2 = nxt.step(act["action"], data=act["data"])
                        else:
                            obs2 = nxt.step(act)
                        if _state_name(obs2) == "WIN" or int(getattr(obs2,"levels_completed",0) or 0) > start_lc:
                            lc = int(getattr(obs2,"levels_completed",0) or 0)
                            print(f"\nWIN! path={path+[str(act)]} lc={lc}")
                            found = True; break
                        k = state_key(nxt)
                        if k not in seen:
                            seen.add(k)
                            queue.append((nxt, path + [str(act)]))
                    except: pass
                if found: break
        finally:
            gc.enable()
        print(f"\nBFS done: nodes={nodes} unique={len(seen)} t={time.monotonic()-t0:.1f}s found={found}")

    except Exception as e:
        import traceback; traceback.print_exc()
    break
print("Done.")
