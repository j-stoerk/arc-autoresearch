"""Diagnostic for r11l click BFS: show scale, bbox, candidate counts, state counts."""
import gc
import copy
import time
from collections import deque
import numpy as np
from prepare import episode_iterator, _state_name

def click_state_key(raw_env):
    game = getattr(raw_env, "_game", None)
    if game is None:
        return ()
    level = getattr(game, "current_level", None)
    if level is None:
        return (getattr(game, "_current_level_index", 0),)
    parts = [getattr(game, "_current_level_index", 0)]
    for s in getattr(level, "_sprites", []):
        px = getattr(s, "pixels", None)
        if px is not None:
            parts.append((s._x, s._y, hash(np.asarray(px, dtype=np.int32).tobytes())))
    return tuple(parts)

def get_scale(raw_env):
    cam = getattr(getattr(raw_env, "_game", None), "camera", None)
    if cam is None:
        return 1
    try:
        result = cam.display_to_grid(4, 4)
        if result and result[0] > 0:
            return 4 // result[0]
    except Exception:
        pass
    return 1

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "r11l" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    if raw_env is None:
        print("No raw_env")
        break

    scale = get_scale(raw_env)
    game = getattr(raw_env, "_game", None)
    level = getattr(game, "current_level", None)
    sprites = getattr(level, "_sprites", []) if level else []

    print(f"game_id={game_id}  scale={scale}  n_sprites={len(sprites)}")

    # Print sprite positions
    for i, s in enumerate(sprites[:20]):
        px = getattr(s, "pixels", None)
        pw = px.shape[1] if px is not None and hasattr(px, "shape") else "?"
        ph = px.shape[0] if px is not None and hasattr(px, "shape") else "?"
        print(f"  sprite[{i}] grid=({s._x},{s._y}) display=({(s._x+1)*scale},{(s._y+1)*scale}) pixels={pw}x{ph}")

    # Compute bbox using full sprite extent (position + pixel size)
    max_gx, max_gy = 64, 64
    for s in sprites:
        px_s = getattr(s, "pixels", None)
        sw = px_s.shape[1] if px_s is not None and hasattr(px_s, "shape") else 8
        sh = px_s.shape[0] if px_s is not None and hasattr(px_s, "shape") else 8
        sx_end = (int(getattr(s, "_x", 0)) + sw) * scale
        sy_end = (int(getattr(s, "_y", 0)) + sh) * scale
        max_gx = max(max_gx, sx_end + 4)
        max_gy = max(max_gy, sy_end + 4)
    max_gx = min(max_gx, 256)
    max_gy = min(max_gy, 256)
    print(f"bbox: max_gx={max_gx}  max_gy={max_gy}")

    try:
        from arcengine.enums import GameAction as _GameAction
        click_action = _GameAction.ACTION6
    except ImportError:
        print("No click action!")
        break

    init_key = click_state_key(raw_env)
    start_levels = int(getattr(env.last_obs, "levels_completed", 0) or 0)

    # Phase 1: sprite-center pre-filter
    candidates = []
    seen_xy = set()
    gc.disable()
    try:
        for s in sprites:
            data = {"x": (int(getattr(s, "_x", 0)) + 1) * scale,
                    "y": (int(getattr(s, "_y", 0)) + 1) * scale}
            xy = (data["x"], data["y"])
            if xy in seen_xy:
                continue
            seen_xy.add(xy)
            try:
                probe = copy.deepcopy(raw_env)
                probe.step(click_action, data=data)
                if click_state_key(probe) != init_key:
                    candidates.append(data)
            except Exception:
                pass
    finally:
        gc.enable()
    print(f"Phase 1 (sprite-center): {len(candidates)} active candidates")

    # Phase 2a: pre-filtered grid probe (extended bbox)
    if len(candidates) < 3:
        gc.disable()
        try:
            t_probe = time.monotonic()
            for dy in range(0, max_gy, 4):
                if time.monotonic() - t_probe > 8.0:
                    break
                for dx in range(0, max_gx, 4):
                    if time.monotonic() - t_probe > 8.0:
                        break
                    if (dx, dy) in seen_xy:
                        continue
                    data = {"x": dx, "y": dy}
                    try:
                        probe = copy.deepcopy(raw_env)
                        probe.step(click_action, data=data)
                        if click_state_key(probe) != init_key:
                            candidates.append(data)
                            seen_xy.add((dx, dy))
                    except Exception:
                        pass
        finally:
            gc.enable()
    print(f"Phase 2a (grid probe 0-{max_gx}x0-{max_gy} step=4): {len(candidates)} total candidates")
    print(f"  Positions: {[(d['x'],d['y']) for d in candidates]}")

    # Phase 2a BFS (pre-filtered)
    queue = deque([(copy.deepcopy(raw_env), [])])
    seen_bfs = {init_key}
    nodes = 0
    deadline = time.monotonic() + 30.0
    t_bfs = time.monotonic()
    gc.disable()
    try:
        while queue and nodes < 500 and time.monotonic() < deadline:
            current, path = queue.popleft()
            nodes += 1
            for data in candidates:
                try:
                    nxt = copy.deepcopy(current)
                    obs = nxt.step(click_action, data=data)
                except Exception:
                    continue
                if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                    print(f"SOLVED by Phase 2a BFS at node {nodes}! plan={path+[data]}")
                    break
                if _state_name(obs) == "WIN":
                    print(f"SOLVED (WIN) by Phase 2a BFS at node {nodes}!")
                    break
                if _state_name(obs) == "NOT_FINISHED":
                    k = click_state_key(nxt)
                    if k not in seen_bfs:
                        seen_bfs.add(k)
                        queue.append((nxt, path + [data]))
    finally:
        gc.enable()
    exhausted = len(queue) == 0
    print(f"Phase 2a BFS: nodes={nodes} unique_states={len(seen_bfs)} t={time.monotonic()-t_bfs:.1f}s exhausted={exhausted}")

    # Discovery phase: probe from each visited state to find dynamic positions
    print(f"\nDiscovery phase: probing from {len(candidates)} candidates, 21 visited states...")
    visited_envs = []
    queue_d = deque([(copy.deepcopy(raw_env), [])])
    seen_d = {init_key}
    gc.disable()
    try:
        while queue_d and len(visited_envs) < 50:
            current, path = queue_d.popleft()
            visited_envs.append(current)
            for data in candidates:
                try:
                    nxt = copy.deepcopy(current)
                    obs = nxt.step(click_action, data=data)
                except Exception:
                    continue
                k = click_state_key(nxt)
                if k not in seen_d:
                    seen_d.add(k)
                    queue_d.append((nxt, path + [data]))
    finally:
        gc.enable()
    print(f"  Collected {len(visited_envs)} visited states")

    known_xy = set((d["x"], d["y"]) for d in candidates)
    new_xy = set()
    t_disc = time.monotonic()
    gc.disable()
    try:
        for i, env_copy in enumerate(visited_envs):
            if time.monotonic() - t_disc > 60.0:
                print(f"  Time budget hit at state {i}")
                break
            cur_key = click_state_key(env_copy)
            for dy in range(0, max_gy, 4):
                if time.monotonic() - t_disc > 60.0:
                    break
                for dx in range(0, max_gx, 4):
                    if (dx, dy) in known_xy or (dx, dy) in new_xy:
                        continue
                    try:
                        probe = copy.deepcopy(env_copy)
                        probe.step(click_action, data={"x": dx, "y": dy})
                        if click_state_key(probe) != cur_key:
                            new_xy.add((dx, dy))
                    except Exception:
                        pass
    finally:
        gc.enable()
    print(f"  Discovery: found {len(new_xy)} NEW dynamic positions in {time.monotonic()-t_disc:.1f}s")
    print(f"  New positions: {sorted(new_xy)[:20]}")

    if new_xy:
        dyn_cands = [{"x": dx, "y": dy} for dx, dy in sorted(known_xy | new_xy)]
        print(f"\nPhase 2b BFS with {len(dyn_cands)} candidates (step=4, bbox={max_gx}x{max_gy})")
        queue2 = deque([(copy.deepcopy(raw_env), [])])
        seen2 = {init_key}
        nodes2 = 0
        deadline2 = time.monotonic() + 90.0
        t2 = time.monotonic()
        gc.disable()
        try:
            while queue2 and nodes2 < 500 and time.monotonic() < deadline2:
                current, path = queue2.popleft()
                nodes2 += 1
                for data in dyn_cands:
                    try:
                        nxt = copy.deepcopy(current)
                        obs = nxt.step(click_action, data=data)
                    except Exception:
                        continue
                    if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                        print(f"SOLVED at node {nodes2}! path={path+[data]}")
                        break
                    if _state_name(obs) == "WIN":
                        print(f"SOLVED WIN at node {nodes2}!")
                        break
                    if _state_name(obs) == "NOT_FINISHED":
                        k = click_state_key(nxt)
                        if k not in seen2:
                            seen2.add(k)
                            queue2.append((nxt, path + [data]))
        finally:
            gc.enable()
        exhausted2 = len(queue2) == 0
        print(f"Phase 2b BFS: nodes={nodes2} unique={len(seen2)} t={time.monotonic()-t2:.1f}s exhausted={exhausted2}")
    else:
        print("No dynamic positions found — r11l has no dynamic click sequences")

    # Scan ALL vars of game for ActionInput-like lists
    game_obj = getattr(raw_env, "_game", None)
    print("\nGame vars (non-trivial):")
    for attr_name, attr_val in sorted(vars(game_obj).items()):
        if isinstance(attr_val, (list, tuple)) and len(attr_val) > 0:
            sample = [x for x in attr_val[:5] if x is not None]
            has_xy = any(hasattr(x, "x") and hasattr(x, "y") for x in sample)
            print(f"  .{attr_name}: {type(attr_val).__name__}[{len(attr_val)}] has_xy={has_xy} sample={sample[:2]}")
        elif not callable(attr_val) and not attr_name.startswith("__") and not isinstance(attr_val, type):
            print(f"  .{attr_name}: {type(attr_val).__name__} = {str(attr_val)[:60]}")

    break  # only process r11l
print("Done.")
