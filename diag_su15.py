"""Quick diagnostic for su15: what happens with grid clicks?"""
import gc
import copy
import time
from collections import deque
import numpy as np
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "su15" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)
    level = getattr(game, "current_level", None)
    sprites = getattr(level, "_sprites", []) if level else []

    print(f"=== su15 quick diagnostic ===")
    print(f"n_sprites={len(sprites)}  win_score={game._win_score}")
    print(f"available_actions=[6,7]")

    # Print game vars
    print("\n--- game vars (non-trivial) ---")
    for av, vv in sorted(vars(game).items()):
        if av.startswith("__"): continue
        if callable(vv): continue
        if isinstance(vv, (np.ndarray,)):
            print(f"  .{av}: ndarray shape={vv.shape}")
        elif isinstance(vv, (list, dict, set)) and len(str(vv)) > 20:
            print(f"  .{av}: {type(vv).__name__} len={len(vv)}")
        elif isinstance(vv, (int, float, bool, str)):
            print(f"  .{av}: {type(vv).__name__} = {str(vv)[:60]}")

    # Generate grid click positions (from on_set_level)
    grid_clicks = []
    for i in range(16):
        for j in range(14):
            y = 10 + j * 4
            grid_clicks.append({"x": i * 4, "y": y})
    print(f"\nGrid clicks: {len(grid_clicks)} positions (x=0..60 step4, y=10..62 step4)")

    try:
        from arcengine.enums import GameAction as _GameAction
        click6 = _GameAction.ACTION6
        click7 = _GameAction.ACTION7

        # State key
        def state_key(env_):
            g_ = getattr(env_, "_game", None)
            l_ = getattr(g_, "current_level", None)
            sps_ = getattr(l_, "_sprites", []) if l_ else []
            return tuple((s._x, s._y, hash(np.asarray(s.pixels, dtype=np.int32).tobytes()) if s.pixels is not None else 0) for s in sps_)

        # First: check what a single grid click does
        init_key = state_key(raw_env)
        print("\n--- test first 5 grid clicks ---")
        for cp in grid_clicks[:5]:
            try:
                probe = copy.deepcopy(raw_env)
                obs2 = probe.step(click6, data=cp)
                k = state_key(probe)
                changed = k != init_key
                sc = probe._game._score
                state = _state_name(obs2)
                lc = int(getattr(obs2, "levels_completed", 0) or 0)
                print(f"  click({cp['x']},{cp['y']}): changed={changed} score={sc} lc={lc} state={state}")
            except Exception as e:
                print(f"  click({cp['x']},{cp['y']}): ERROR {e}")

        # Count how many grid positions change state
        print("\n--- counting active grid positions ---")
        active = []
        t0 = time.monotonic()
        gc.disable()
        try:
            for cp in grid_clicks:
                try:
                    probe = copy.deepcopy(raw_env)
                    obs2 = probe.step(click6, data=cp)
                    k = state_key(probe)
                    if k != init_key:
                        active.append(cp)
                        sc = probe._game._score
                        lc = int(getattr(obs2, "levels_completed", 0) or 0)
                        if lc > 0 or _state_name(obs2) == "WIN":
                            print(f"WIN at ({cp['x']},{cp['y']})!")
                except: pass
        finally:
            gc.enable()
        print(f"Active clicks: {len(active)} of {len(grid_clicks)} in {time.monotonic()-t0:.1f}s")
        print(f"Positions: {[(c['x'],c['y']) for c in active[:20]]}")

        # BFS with active clicks
        print(f"\n--- BFS with {len(active)} active clicks (200 nodes) ---")
        if active:
            queue = deque([(copy.deepcopy(raw_env), [])])
            seen = {init_key}
            nodes = 0
            found = False
            t0 = time.monotonic()
            gc.disable()
            try:
                while queue and nodes < 200 and time.monotonic() - t0 < 30:
                    cur, path = queue.popleft()
                    nodes += 1
                    for cp in active:
                        try:
                            nxt = copy.deepcopy(cur)
                            obs2 = nxt.step(click6, data=cp)
                            if _state_name(obs2) == "WIN" or int(getattr(obs2,"levels_completed",0) or 0) > 0:
                                print(f"WIN at node {nodes}! path len={len(path)+1}"); found=True; break
                            k = state_key(nxt)
                            if k not in seen:
                                seen.add(k)
                                queue.append((nxt, path + [cp]))
                        except: pass
                    if found: break
            finally:
                gc.enable()
            exhausted = len(queue) == 0
            print(f"BFS done: nodes={nodes} unique={len(seen)} t={time.monotonic()-t0:.1f}s found={found} exhausted={exhausted}")

    except Exception as e:
        import traceback; traceback.print_exc()
    break
print("Done.")
