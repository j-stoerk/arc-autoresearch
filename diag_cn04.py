"""Diagnostic for cn04: understand why 193 states with no win."""
import gc
import copy
import time
from collections import deque
import numpy as np
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "cn04" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)
    level = getattr(game, "current_level", None)
    sprites = getattr(level, "_sprites", []) if level else []

    print(f"=== cn04 diagnostic ===  win_score={game._win_score}")
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

    # Game vars
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

    # Actions
    try:
        from arcengine.enums import GameAction as _GameAction
        print("\n--- actions ---")
        for aname in ["ACTION1","ACTION2","ACTION3","ACTION4","ACTION5","ACTION6"]:
            try:
                ga = getattr(_GameAction, aname)
                probe = copy.deepcopy(raw_env)
                obs2 = probe.step(ga)
                state = _state_name(obs2)
                lc = int(getattr(obs2, "levels_completed", 0) or 0)
                g2 = probe._game
                score = g2._score
                # Check changes
                sps2 = getattr(g2.current_level, "_sprites", [])
                moved = any(s._x != s2._x or s._y != s2._y for s, s2 in zip(sprites, sps2))
                print(f"  {aname}: state={state} score={score} lc={lc} sprite_moved={moved}")
            except Exception as e:
                print(f"  {aname}: ERROR {e}")
    except ImportError:
        pass

    # BFS with pixel-hash state key (richer state)
    print("\n--- BFS with pixel hash key (500 nodes) ---")
    try:
        from arcengine.enums import GameAction as _GameAction
        kb_actions = [_GameAction.ACTION1, _GameAction.ACTION2, _GameAction.ACTION3, _GameAction.ACTION4, _GameAction.ACTION5]

        def pix_hash_key(env_):
            g_ = getattr(env_, "_game", None)
            l_ = getattr(g_, "current_level", None)
            sps_ = getattr(l_, "_sprites", []) if l_ else []
            parts = [getattr(g_, "_current_level_index", 0), getattr(g_, "_score", 0)]
            for s_ in sps_:
                px = getattr(s_, "pixels", None)
                if px is not None:
                    parts.append(hash(np.asarray(px, dtype=np.int32).tobytes()))
                else:
                    parts.append((s_._x, s_._y))
            return tuple(parts)

        def pos_key(env_):
            g_ = getattr(env_, "_game", None)
            l_ = getattr(g_, "current_level", None)
            sps_ = getattr(l_, "_sprites", []) if l_ else []
            parts = [getattr(g_, "_current_level_index", 0), getattr(g_, "_score", 0)]
            for s_ in sps_:
                parts.append((s_._x, s_._y))
            return tuple(parts)

        init_pix = pix_hash_key(raw_env)
        init_pos = pos_key(raw_env)
        queue = deque([(copy.deepcopy(raw_env), [])])
        seen_pix = {init_pix}
        seen_pos = {init_pos}
        nodes = 0
        found_win = False
        t0 = time.monotonic()
        gc.disable()
        try:
            while queue and nodes < 500 and time.monotonic() - t0 < 30:
                cur, path = queue.popleft()
                nodes += 1
                for act in kb_actions:
                    try:
                        nxt = copy.deepcopy(cur)
                        obs2 = nxt.step(act)
                        if _state_name(obs2) == "WIN" or int(getattr(obs2, "levels_completed", 0) or 0) > 0:
                            print(f"WIN at node {nodes}! path={path+[act.name]}")
                            found_win = True
                            break
                        pk = pix_hash_key(nxt)
                        if pk not in seen_pix:
                            seen_pix.add(pk)
                            seen_pos.add(pos_key(nxt))
                            queue.append((nxt, path + [act.name]))
                    except: pass
                if found_win: break
        finally:
            gc.enable()

        print(f"BFS done: nodes={nodes} pos_states={len(seen_pos)} pix_states={len(seen_pix)} t={time.monotonic()-t0:.1f}s found_win={found_win}")
        print(f"Queue remaining: {len(queue)}")

        # Show which sprites change
        print("\nAnalyzing which sprites change across states...")
        all_states = list(seen_pix)[:10]

        # Score analysis: does score change with movement?
        probe_sc = copy.deepcopy(raw_env)
        for act in [_GameAction.ACTION1, _GameAction.ACTION2, _GameAction.ACTION3, _GameAction.ACTION4,
                    _GameAction.ACTION1, _GameAction.ACTION2, _GameAction.ACTION3]:
            obs2 = probe_sc.step(act)
            sc = probe_sc._game._score
            lc = int(getattr(obs2, "levels_completed", 0) or 0)
            print(f"  {act.name}: score={sc} lc={lc}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"BFS error: {e}")

    break
print("Done.")
