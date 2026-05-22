"""cn04: try click-then-move BFS. Click selects a sprite, then keyboard moves it."""
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

    print(f"=== cn04 BFS with clicks ===")
    print(f"n_sprites={len(sprites)}  win_score={game._win_score}")

    # Sprite click positions
    click_positions = []
    for i, s in enumerate(sprites):
        nm = str(getattr(s, "name", ""))
        tg = str(getattr(s, "tags", []))
        px = getattr(s, "pixels", None)
        pw = px.shape[1] if px is not None and hasattr(px, "shape") else 1
        ph = px.shape[0] if px is not None and hasattr(px, "shape") else 1
        cx = int(s._x) + pw // 2
        cy = int(s._y) + ph // 2
        print(f"  [{i}] {nm!r} tags={tg} pos=({s._x},{s._y}) size={pw}x{ph} click=({cx},{cy})")
        if "sys_click" in tg:
            click_positions.append({"x": cx, "y": cy, "sprite": nm})

    print(f"\nClick positions: {click_positions}")

    # State includes position + selected sprite
    def state_key(env_):
        g_ = getattr(env_, "_game", None)
        l_ = getattr(g_, "current_level", None)
        sps_ = getattr(l_, "_sprites", []) if l_ else []
        sel = getattr(g_, "xseexqzst", None)
        sel_name = sel.name if sel else None
        return (tuple((s._x, s._y) for s in sps_), sel_name)

    try:
        from arcengine.enums import GameAction as _GameAction
        click_action = _GameAction.ACTION6
        kb_actions = [_GameAction.ACTION1, _GameAction.ACTION2, _GameAction.ACTION3, _GameAction.ACTION4]

        # Check initial state
        g0 = raw_env._game
        sel0 = getattr(g0, "xseexqzst", None)
        print(f"Initial selected: {sel0.name if sel0 else None}")

        # Check what clicking on each sprite does
        print("\n--- test clicks ---")
        for cp in click_positions:
            probe = copy.deepcopy(raw_env)
            try:
                obs2 = probe.step(click_action, data=cp)
                sel2 = getattr(probe._game, "xseexqzst", None)
                sc2 = probe._game._score
                state2 = _state_name(obs2)
                print(f"  click ({cp['x']},{cp['y']}): state={state2} score={sc2} selected={sel2.name if sel2 else None}")
                # Check sprite positions
                for j, s2 in enumerate(probe._game.current_level._sprites):
                    nm2 = str(getattr(s2, "name", ""))
                    print(f"    sprite[{j}] {nm2!r} pos=({s2._x},{s2._y})")
            except Exception as e:
                print(f"  click ({cp['x']},{cp['y']}): ERROR {e}")

        # BFS with clicks + keyboard
        print("\n--- BFS with click+keyboard (500 nodes) ---")
        init_key = state_key(raw_env)
        queue = deque([(copy.deepcopy(raw_env), [])])
        seen = {init_key}
        nodes = 0
        found = False
        t0 = time.monotonic()
        gc.disable()
        try:
            while queue and nodes < 500 and time.monotonic() - t0 < 30:
                cur, path = queue.popleft()
                nodes += 1

                # Try keyboard actions
                for act in kb_actions:
                    try:
                        nxt = copy.deepcopy(cur)
                        obs2 = nxt.step(act)
                        if _state_name(obs2) == "WIN" or int(getattr(obs2,"levels_completed",0) or 0) > 0:
                            print(f"WIN! path={path+[act.name]}")
                            found = True; break
                        k = state_key(nxt)
                        if k not in seen:
                            seen.add(k)
                            queue.append((nxt, path + [act.name]))
                    except: pass
                if found: break

                # Try click actions
                for cp in click_positions:
                    try:
                        nxt = copy.deepcopy(cur)
                        obs2 = nxt.step(click_action, data=cp)
                        if _state_name(obs2) == "WIN" or int(getattr(obs2,"levels_completed",0) or 0) > 0:
                            print(f"WIN! path={path + ['click(%d,%d)' % (cp['x'], cp['y'])]}"); found=True; break
                        k = state_key(nxt)
                        if k not in seen:
                            seen.add(k)
                            queue.append((nxt, path + [f'click({cp["x"]},{cp["y"]})']))
                    except Exception as e:
                        pass
                if found: break
        finally:
            gc.enable()

        print(f"BFS done: nodes={nodes} unique={len(seen)} t={time.monotonic()-t0:.1f}s found={found}")
        print(f"Queue left: {len(queue)}")

        # What does game's win condition look like?
        print("\n--- game win condition vars ---")
        for av, vv in sorted(vars(game).items()):
            if av.startswith("__"): continue
            if callable(vv) and not isinstance(vv, (list, tuple)): continue
            s = str(vv)
            if len(s) > 60: s = s[:60] + "..."
            print(f"  .{av}: {type(vv).__name__} = {s}")

        # What is the target configuration?
        print("\n--- level data ---")
        try:
            print(f"  MaxSteps: {level.get_data('MaxSteps')}")
            print(f"  GreyMasking: {level.get_data('GreyMasking')}")
            print(f"  BackgroundColour: {level.get_data('BackgroundColour')}")
        except: pass

        # What does the game compare? Check ztpxqonhr (bool=True)
        print(f"\nztpxqonhr={game.ztpxqonhr}")
        print(f"rqolqpqwo={game.rqolqpqwo}")
        print(f"spcewphwy={game.spcewphwy}")

    except Exception as e:
        import traceback; traceback.print_exc()
    break
print("Done.")
