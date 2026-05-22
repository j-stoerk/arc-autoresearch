"""su15: brute-force click combinations + ACTION7 (launch)."""
import gc
import copy
import time
import numpy as np
from collections import deque
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "su15" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)

    print(f"=== su15 click+ACTION7 brute force ===")
    print(f"ball at: {game.yxpvimvli}")
    print(f"target: {game.nscnqkkvg}")
    print(f"dsqlbvwaj: {game.dsqlbvwaj}")
    print(f"steps: {game.current_level.get_data('steps')}")

    try:
        from arcengine.enums import GameAction as _GameAction
        click6 = _GameAction.ACTION6
        click7 = _GameAction.ACTION7

        # Active click positions from previous diagnostic
        active = [(0,54),(0,58),(0,62),(4,50),(4,54),(4,58),(4,62),(8,54),(8,58),(8,62),(12,58),(12,62)]

        # Try: click each active position THEN ACTION7
        print("\n--- single click + ACTION7 ---")
        for cx, cy in active:
            probe = copy.deepcopy(raw_env)
            probe.step(click6, data={"x": cx, "y": cy})
            obs2 = probe.step(click7)
            sc = probe._game._score
            lc = int(getattr(obs2, "levels_completed", 0) or 0)
            state = _state_name(obs2)
            print(f"  ({cx},{cy}) -> ACTION7: state={state} score={sc} lc={lc}")
            if lc > 0 or state == "WIN":
                print("  *** WIN! ***"); break

        # Try: ACTION7 alone (no click first)
        probe = copy.deepcopy(raw_env)
        obs2 = probe.step(click7)
        print(f"\nACTION7 alone: state={_state_name(obs2)} score={probe._game._score}")

        # Try: double click + ACTION7
        print("\n--- double click + ACTION7 (all pairs) ---")
        found = False
        for i, (cx1, cy1) in enumerate(active):
            for j, (cx2, cy2) in enumerate(active):
                probe = copy.deepcopy(raw_env)
                probe.step(click6, data={"x": cx1, "y": cy1})
                probe.step(click6, data={"x": cx2, "y": cy2})
                obs2 = probe.step(click7)
                sc = probe._game._score
                lc = int(getattr(obs2, "levels_completed", 0) or 0)
                state = _state_name(obs2)
                if lc > 0 or state == "WIN":
                    print(f"  WIN! ({cx1},{cy1}) + ({cx2},{cy2}) + ACTION7")
                    found = True
                    break
            if found: break
        if not found:
            print("  No win with 2-click combinations")

        # Try: BFS with all grid positions (not just active) + ACTION7
        print("\n--- extended grid clicks + ACTION7 ---")
        # Try all grid positions (x=0-60, y=10-62 step 4) + ACTION7
        best_score = -1
        best_combo = None
        for cx in range(0, 64, 4):
            for cy in range(0, 64, 4):
                probe = copy.deepcopy(raw_env)
                try:
                    probe.step(click6, data={"x": cx, "y": cy})
                    obs2 = probe.step(click7)
                    sc = probe._game._score
                    lc = int(getattr(obs2, "levels_completed", 0) or 0)
                    state = _state_name(obs2)
                    if sc > best_score:
                        best_score = sc
                        best_combo = (cx, cy)
                    if lc > 0 or state == "WIN":
                        print(f"  WIN! click ({cx},{cy}) + ACTION7")
                        found = True; break
                except: pass
            if found: break
        print(f"  Best score with 1 click + ACTION7: {best_score} at {best_combo}")

        # What is bicnaxoxq? (level_index=0 special sprite)
        print(f"\nbicnaxoxq={game.bicnaxoxq}")
        if game.bicnaxoxq:
            b = game.bicnaxoxq
            print(f"  pos=({b.x},{b.y}) tag={b.tags}")
            # Try clicking it
            probe = copy.deepcopy(raw_env)
            probe.step(click6, data={"x": b.x + b.width//2, "y": b.y + b.height//2})
            obs2 = probe.step(click7)
            print(f"  click bicnaxoxq + ACTION7: score={probe._game._score} lc={int(getattr(obs2,'levels_completed',0) or 0)}")

    except Exception as e:
        import traceback; traceback.print_exc()
    break
print("Done.")
