"""Deep diagnostic for tn36: understand win condition and action mechanics."""
import gc
import copy
import time
from collections import deque
import numpy as np
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "tn36" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    if raw_env is None:
        print("No raw_env"); break

    game = getattr(raw_env, "_game", None)
    level = getattr(game, "current_level", None)
    sprites = getattr(level, "_sprites", []) if level else []

    print(f"=== tn36 diagnostic ===")
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

    # Dump all game-level vars
    print("\n--- game vars ---")
    for attr_name, attr_val in sorted(vars(game).items()):
        if attr_name.startswith("__"): continue
        if callable(attr_val) and not isinstance(attr_val, (list, tuple)): continue
        s = str(attr_val)
        if len(s) > 80: s = s[:80] + "..."
        print(f"  game.{attr_name}: {type(attr_val).__name__} = {s}")

    print("\n--- level vars ---")
    for attr_name, attr_val in sorted(vars(level).items()):
        if attr_name.startswith("__"): continue
        if callable(attr_val) and not isinstance(attr_val, (list, tuple)): continue
        s = str(attr_val)
        if len(s) > 80: s = s[:80] + "..."
        print(f"  level.{attr_name}: {type(attr_val).__name__} = {s}")

    # Print all sprites with full info
    print("\n--- sprites ---")
    for i, s in enumerate(sprites):
        nm = str(getattr(s, "name", ""))
        tg = str(getattr(s, "tags", []))
        px = getattr(s, "pixels", None)
        pw = px.shape[1] if px is not None and hasattr(px, "shape") else "?"
        ph = px.shape[0] if px is not None and hasattr(px, "shape") else "?"
        print(f"  [{i}] name={nm!r} tags={tg} pos=({s._x},{s._y}) size={pw}x{ph}")
        # Dump sprite vars
        for av, vv in sorted(vars(s).items()):
            if av.startswith("__") or av in ("pixels", "name", "tags", "_x", "_y"): continue
            if callable(vv): continue
            sv = str(vv)
            if len(sv) > 60: sv = sv[:60] + "..."
            print(f"      .{av} = {sv}")

    # Find htntnzkbzu
    print("\n--- searching for htntnzkbzu ---")
    htn = None
    for s in sprites:
        nm = str(getattr(s, "name", ""))
        if "htntnzkbzu" in nm or "htn" in nm.lower():
            htn = s
            print(f"  FOUND: {nm!r} pos=({s._x},{s._y})")
    if htn is None:
        # Search game attrs
        for av, vv in vars(game).items():
            if "htn" in av.lower():
                print(f"  game.{av} = {vv}")

    # Find vupcwzjtxu (interceptor?)
    print("\n--- searching for vupcwzjtxu ---")
    for av, vv in vars(game).items():
        if "vup" in av.lower() or "vupc" in av.lower():
            print(f"  game.{av}: {type(vv).__name__} = {str(vv)[:80]}")
    for s in sprites:
        nm = str(getattr(s, "name", ""))
        if "vupc" in nm.lower():
            print(f"  sprite: {nm!r} pos=({s._x},{s._y})")

    # Try clicking every sprite and monitor htntnzkbzu.y changes
    print("\n--- click each sprite, monitor htn.y ---")

    def get_htn_y(raw_):
        g_ = getattr(raw_, "_game", None)
        l_ = getattr(g_, "current_level", None)
        sps_ = getattr(l_, "_sprites", []) if l_ else []
        for s_ in sps_:
            nm_ = str(getattr(s_, "name", ""))
            if "htntnzkbzu" in nm_:
                return s_._y
        return None

    def get_score_info(raw_):
        g_ = getattr(raw_, "_game", None)
        sc_ = getattr(g_, "_score", None)
        ws_ = getattr(g_, "_win_score", None)
        lc_ = getattr(g_, "_current_level_index", 0)
        return sc_, ws_, lc_

    try:
        from arcengine.enums import GameAction as _GameAction
        click_action = _GameAction.ACTION6
        kb_actions = {
            "ACTION1": _GameAction.ACTION1,
            "ACTION2": _GameAction.ACTION2,
            "ACTION3": _GameAction.ACTION3,
            "ACTION4": _GameAction.ACTION4,
            "ACTION5": _GameAction.ACTION5,
        }
    except ImportError:
        print("No GameAction!"); break

    init_htn_y = get_htn_y(raw_env)
    init_score, init_ws, init_lc = get_score_info(raw_env)
    print(f"Initial: htn.y={init_htn_y} score={init_score} win_score={init_ws} lc={init_lc}")

    # Try each keyboard action
    print("\n--- keyboard actions ---")
    for aname, aval in kb_actions.items():
        try:
            probe = copy.deepcopy(raw_env)
            obs2 = probe.step(aval)
            hy = get_htn_y(probe)
            sc, ws, lc = get_score_info(probe)
            state = _state_name(obs2)
            print(f"  {aname}: htn.y={hy} score={sc} state={state}")
        except Exception as e:
            print(f"  {aname}: ERROR {e}")

    # Try clicking each sprite center
    print("\n--- click each sprite center ---")
    for i, s in enumerate(sprites[:30]):
        cx = (int(getattr(s, "_x", 0)) + 1) * scale
        cy = (int(getattr(s, "_y", 0)) + 1) * scale
        try:
            probe = copy.deepcopy(raw_env)
            obs2 = probe.step(click_action, data={"x": cx, "y": cy})
            hy = get_htn_y(probe)
            sc, ws, lc = get_score_info(probe)
            state = _state_name(obs2)
            nm = str(getattr(s, "name", "?"))[:20]
            print(f"  sprite[{i}] {nm!r} ({cx},{cy}): htn.y={hy} score={sc} state={state}")
        except Exception as e:
            nm = str(getattr(s, "name", "?"))[:20]
            print(f"  sprite[{i}] {nm!r}: ERROR {e}")

    # Look for miytdaqzei (action items) - how are they populated?
    print("\n--- miytdaqzei analysis ---")
    miy = getattr(game, "miytdaqzei", None)
    print(f"  miytdaqzei type={type(miy).__name__} len={len(miy) if miy is not None else 'None'}")

    # Check all game methods for setup/init
    print("\n--- game class methods ---")
    cls = type(game)
    for mname in sorted(dir(cls)):
        if mname.startswith("__"): continue
        meth = getattr(cls, mname, None)
        if callable(meth):
            print(f"  {mname}")

    # After clicking, what changes in game state?
    print("\n--- full state diff after one click ---")
    init_vars = {}
    for av, vv in vars(game).items():
        if not callable(vv):
            try: init_vars[av] = copy.deepcopy(vv)
            except: init_vars[av] = str(vv)

    # Click the first candidate
    if sprites:
        s0 = sprites[0]
        cx0 = (int(getattr(s0, "_x", 0)) + 1) * scale
        cy0 = (int(getattr(s0, "_y", 0)) + 1) * scale
        probe2 = copy.deepcopy(raw_env)
        probe2.step(click_action, data={"x": cx0, "y": cy0})
        game2 = getattr(probe2, "_game", None)
        for av, vv in vars(game2).items():
            if not callable(vv):
                old = init_vars.get(av)
                try: new = copy.deepcopy(vv)
                except: new = str(vv)
                if str(old) != str(new):
                    print(f"  CHANGED: {av}: {str(old)[:40]} -> {str(new)[:40]}")

    # Greedy score search - see how many improvements we can get
    print("\n--- greedy score search (up to 20 clicks) ---")
    cur_env = copy.deepcopy(raw_env)
    cur_score, _, _ = get_score_info(cur_env)
    path = []
    for step in range(20):
        best_sc = cur_score
        best_s = None
        best_data = None
        for s in sprites:
            cx = (int(getattr(s, "_x", 0)) + 1) * scale
            cy = (int(getattr(s, "_y", 0)) + 1) * scale
            try:
                probe = copy.deepcopy(cur_env)
                obs2 = probe.step(click_action, data={"x": cx, "y": cy})
                sc, _, _ = get_score_info(probe)
                if sc is not None and (best_sc is None or sc > best_sc):
                    best_sc = sc
                    best_s = str(getattr(s, "name", "?"))[:20]
                    best_data = {"x": cx, "y": cy}
            except: pass
        if best_data is None:
            print(f"  step {step}: no improvement found")
            break
        cur_env.step(click_action, data=best_data)
        path.append((best_s, best_data, best_sc))
        cur_score = best_sc
        sc, ws, lc = get_score_info(cur_env)
        hy = get_htn_y(cur_env)
        print(f"  step {step}: click {best_s!r} at {best_data} → score={sc}/{ws} htn.y={hy}")
        if sc is not None and ws is not None and sc >= ws:
            print("  ** SOLVED! **"); break

    break
print("Done.")
