"""Deep diagnostic for dc22: understand why only 9 unique states with no win."""
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

    print(f"=== dc22 diagnostic ===")
    print(f"n_sprites={len(sprites)}")

    # Get scale
    cam = getattr(game, "camera", None)
    scale = 1
    if cam:
        try:
            r = cam.display_to_grid(4, 4)
            if r and r[0] > 0: scale = 4 // r[0]
        except: pass
    print(f"scale={scale}")

    # Dump game vars
    print("\n--- game vars ---")
    for attr_name, attr_val in sorted(vars(game).items()):
        if attr_name.startswith("__"): continue
        if callable(attr_val) and not isinstance(attr_val, (list, tuple)): continue
        s = str(attr_val)
        if len(s) > 80: s = s[:80] + "..."
        print(f"  game.{attr_name}: {type(attr_val).__name__} = {s}")

    # Sprites
    print("\n--- sprites ---")
    for i, s in enumerate(sprites[:30]):
        nm = str(getattr(s, "name", ""))
        tg = str(getattr(s, "tags", []))
        px = getattr(s, "pixels", None)
        pw = px.shape[1] if px is not None and hasattr(px, "shape") else "?"
        ph = px.shape[0] if px is not None and hasattr(px, "shape") else "?"
        print(f"  [{i}] name={nm!r} tags={tg} pos=({s._x},{s._y}) size={pw}x{ph}")

    # Actions available
    try:
        from arcengine.enums import GameAction as _GameAction
        avail = [ga for ga in _GameAction if hasattr(raw_env, "step")]
        # Try each action
        print(f"\n--- available actions & state changes ---")
        for aname in ["ACTION1","ACTION2","ACTION3","ACTION4","ACTION5","ACTION6"]:
            try:
                ga = getattr(_GameAction, aname)
                probe = copy.deepcopy(raw_env)
                obs2 = probe.step(ga)
                state = _state_name(obs2)
                lc = int(getattr(obs2, "levels_completed", 0) or 0)
                g2 = probe._game
                score = g2._score
                level2 = g2._current_level_index
                # Check sprite changes
                sps2 = getattr(g2.current_level, "_sprites", [])
                changed = False
                for s, s2 in zip(sprites[:30], sps2[:30]):
                    if s._x != s2._x or s._y != s2._y:
                        changed = True; break
                print(f"  {aname}: state={state} score={score} lc={lc} level={level2} sprite_moved={changed}")
            except Exception as e:
                print(f"  {aname}: ERROR {e}")
    except ImportError:
        print("No GameAction"); break

    # BFS to explore all states
    print("\n--- BFS (keyboard actions, 200 nodes) ---")
    try:
        from arcengine.enums import GameAction as _GameAction
        kb_actions = [_GameAction.ACTION1, _GameAction.ACTION2, _GameAction.ACTION3, _GameAction.ACTION4, _GameAction.ACTION5]

        def state_key(env_):
            g_ = getattr(env_, "_game", None)
            l_ = getattr(g_, "current_level", None)
            sps_ = getattr(l_, "_sprites", []) if l_ else []
            parts = [getattr(g_, "_current_level_index", 0), getattr(g_, "_score", 0)]
            for s_ in sps_:
                parts.append((s_._x, s_._y))
            return tuple(parts)

        init_key = state_key(raw_env)
        queue = deque([(copy.deepcopy(raw_env), [])])
        seen = {init_key}
        nodes = 0
        found_win = False
        t0 = time.monotonic()
        gc.disable()
        try:
            while queue and nodes < 200:
                cur, path = queue.popleft()
                nodes += 1
                for act in kb_actions:
                    try:
                        nxt = copy.deepcopy(cur)
                        obs2 = nxt.step(act)
                        if _state_name(obs2) == "WIN" or int(getattr(obs2, "levels_completed", 0) or 0) > 0:
                            print(f"WIN at node {nodes}! path length={len(path)+1}")
                            found_win = True
                            break
                        k = state_key(nxt)
                        if k not in seen:
                            seen.add(k)
                            queue.append((nxt, path + [act.name]))
                    except: pass
                if found_win: break
        finally:
            gc.enable()

        print(f"BFS done: nodes={nodes} unique_states={len(seen)} t={time.monotonic()-t0:.1f}s found_win={found_win}")
        print(f"Queue remaining: {len(queue)}")

        # What do the unique states look like?
        print(f"\nSample states (first 20 visited):")
        # Re-run BFS collecting states
        visited_states = [init_key]
        q2 = deque([(copy.deepcopy(raw_env), [])])
        s2 = {init_key}
        gc.disable()
        try:
            while q2 and len(visited_states) < 20:
                cur, path = q2.popleft()
                for act in kb_actions:
                    try:
                        nxt = copy.deepcopy(cur)
                        obs2 = nxt.step(act)
                        k = state_key(nxt)
                        if k not in s2:
                            s2.add(k)
                            visited_states.append(k)
                            q2.append((nxt, path + [act.name]))
                    except: pass
        finally:
            gc.enable()

        for i, st in enumerate(visited_states[:10]):
            print(f"  [{i}] {st}")

    except Exception as e:
        print(f"BFS error: {e}")

    # Read game class source
    print("\n--- game class methods ---")
    cls = type(game)
    for mname in sorted(dir(cls)):
        if mname.startswith("__"): continue
        meth = getattr(cls, mname, None)
        if callable(meth):
            print(f"  {mname}")

    break
print("Done.")
