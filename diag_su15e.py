"""
su15: Verify the ball-movement mechanic and compute solution.

Discovery: pkrdtzfrth moves balls IN xxkedcuzq toward click position.
Ball at (4,59), zone at (44,11,9x9). Click (+4,-4) each step moves ball diagonally.
11 clicks should move ball from (4,59) to (48,15) which IS in zone.
"""
import copy
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "su15" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)

    print("=== su15: ball-movement mechanic test ===")

    try:
        from arcengine.enums import GameAction as _GameAction
        click6 = _GameAction.ACTION6
        start_lc = int(getattr(obs, "levels_completed", 0) or 0)

        # Compute solution: 11 diagonal clicks (+4,-4) from (4,59)
        ball_x, ball_y = 4, 59  # initial ball center
        clicks = []
        for k in range(1, 13):  # try up to 12 clicks
            cx = ball_x + 4
            cy = ball_y - 4
            clicks.append({"x": cx, "y": cy, "_ball_after": (cx, cy)})
            ball_x, ball_y = cx, cy

        print(f"Solution clicks: {len(clicks)} total")
        for i, c in enumerate(clicks):
            print(f"  click {i+1}: ({c['x']},{c['y']}) -> ball at {c['_ball_after']}")

        # Execute and verify step by step
        probe = copy.deepcopy(raw_env)
        goal_found = False
        for i, c in enumerate(clicks):
            obs2 = probe.step(click6, data={"x": c["x"], "y": c["y"]})
            lc = int(getattr(obs2, "levels_completed", 0) or 0)
            state = _state_name(obs2)
            g = probe._game
            ball = g.lkujttxgs[0] if g.lkujttxgs else None
            ball_pos = (ball.x, ball.y) if ball else "?"
            print(f"  Click {i+1} ({c['x']},{c['y']}): ball={ball_pos} lc={lc} state={state}")
            if lc > start_lc or state == "WIN":
                print(f"  *** LEVEL ADVANCED! lc={lc} at click {i+1} ***")
                goal_found = True
                break
            if ball_pos == "?" or (isinstance(ball_pos, tuple) and ball_pos[0] > 44):
                print(f"  Ball in zone area!")

        if not goal_found:
            # Also try 1-step approach: click DIRECTLY at goal zone center
            print("\n--- Direct click test (at goal zone) ---")
            probe2 = copy.deepcopy(raw_env)
            zone_cx = 44 + 4  # zone center x
            zone_cy = 11 + 4  # zone center y
            obs2 = probe2.step(click6, data={"x": zone_cx, "y": zone_cy})
            g2 = probe2._game
            ball2 = g2.lkujttxgs[0] if g2.lkujttxgs else None
            ball_pos2 = (ball2.x, ball2.y) if ball2 else "?"
            print(f"  click ({zone_cx},{zone_cy}): ball={ball_pos2} lc={int(getattr(obs2,'levels_completed',0) or 0)}")

            # Try from initial position: does ball move?
            print("\n--- Single step movement test ---")
            probe3 = copy.deepcopy(raw_env)
            obs3 = probe3.step(click6, data={"x": 8, "y": 55})
            g3 = probe3._game
            ball3 = g3.lkujttxgs[0] if g3.lkujttxgs else None
            pos3 = (ball3.x, ball3.y) if ball3 else "?"
            print(f"  click (8,55): ball={pos3}")

            # Check if movement happened
            ball_orig = game.lkujttxgs[0] if game.lkujttxgs else None
            orig_pos = (ball_orig.x, ball_orig.y) if ball_orig else "?"
            print(f"  original: {orig_pos}, after click: {pos3}")
            if pos3 != orig_pos:
                print(f"  *** BALL MOVED! from {orig_pos} to {pos3} ***")
            else:
                print(f"  Ball did not move. Checking xxkedcuzq...")
                # Try clicking exactly at ball center
                probe4 = copy.deepcopy(raw_env)
                bx, by = orig_pos if isinstance(orig_pos, tuple) else (3,58)
                center_x = bx + 1  # center of 3x3 sprite
                center_y = by + 1
                print(f"  Ball at ({bx},{by}), center ({center_x},{center_y})")
                obs4 = probe4.step(click6, data={"x": center_x, "y": center_y})
                g4 = probe4._game
                ball4 = g4.lkujttxgs[0] if g4.lkujttxgs else None
                pos4 = (ball4.x, ball4.y) if ball4 else "?"
                print(f"  click ({center_x},{center_y}): ball={pos4}")

    except Exception as e:
        import traceback; traceback.print_exc()
    break
print("Done.")
