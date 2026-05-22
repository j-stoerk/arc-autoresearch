"""
First-principles physics game solver for su15.

Discovery phase:
  - Identify controllable objects (balls), their positions, and targets
  - Identify action vocabulary: click(cx,cy) sets trajectory, ACTION7 launches
  - Fit: try all (cx,cy) parameter values, observe landing position

Search phase:
  - Find which (cx,cy) causes ball to land at target
"""
import gc
import copy
import time
import numpy as np
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "su15" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)

    print("=== su15: first-principles physics solver ===")

    try:
        from arcengine.enums import GameAction as _GameAction
        click6 = _GameAction.ACTION6
        click7 = _GameAction.ACTION7

        # --- DISCOVERY: object state ---
        ball_pos = game.yxpvimvli.copy() if game.yxpvimvli is not None else None
        targets = game.nscnqkkvg.copy() if game.nscnqkkvg else []
        ball_colors = game.oxgxfichu.copy() if game.oxgxfichu is not None else []
        print(f"Ball positions: {ball_pos}")
        print(f"Targets: {targets}")
        print(f"Ball colors: {ball_colors}")
        print(f"Max steps: {game.current_level.get_data('steps')}")

        # Canvas analysis: find passable cells (zeros in sdrzkrjcp)
        canvas = game.sdrzkrjcp
        if canvas is not None:
            zeros = np.argwhere(canvas == 0)
            print(f"Passable cells: {len(zeros)} at {zeros[:10].tolist()}...")

        # Canvas for target visualization
        zdoy = game.zdoyfxons
        if zdoy is not None:
            zdoy_zeros = np.argwhere(zdoy == 0)
            print(f"zdoyfxons zeros: {len(zdoy_zeros)}")

        # --- DISCOVERY: action vocabulary ---
        # Pre-generated click positions (from on_set_level)
        grid_clicks = []
        for i in range(16):
            for j in range(14):
                y = 10 + j * 4
                grid_clicks.append({"x": i * 4, "y": y})
        print(f"\nAction vocabulary: {len(grid_clicks)} click positions + ACTION7")

        # --- PROBE: try each click + ACTION7, observe ball landing ---
        print("\n--- Probing all click positions + ACTION7 ---")
        landing_map = {}  # (cx,cy) -> final_ball_pos
        t0 = time.monotonic()
        gc.disable()
        found_win = False
        start_lc = int(getattr(obs, "levels_completed", 0) or 0)
        try:
            for cp in grid_clicks:
                probe = copy.deepcopy(raw_env)
                # Apply click to set direction/trajectory
                probe.step(click6, data=cp)
                # Launch
                obs2 = probe.step(click7)

                # Read final ball position
                g2 = probe._game
                final_pos = g2.yxpvimvli.copy() if g2.yxpvimvli is not None else None
                final_score = g2._score
                final_lc = int(getattr(obs2, "levels_completed", 0) or 0)
                state = _state_name(obs2)

                key = (cp["x"], cp["y"])
                landing_map[key] = {
                    "pos": final_pos.tolist() if final_pos is not None else None,
                    "score": final_score,
                    "lc": final_lc,
                    "state": state,
                }

                if final_lc > start_lc or state == "WIN":
                    print(f"  WIN! click({cp['x']},{cp['y']}) -> lc={final_lc}")
                    found_win = True
                    break

                # Show score improvements
                if final_score > 0:
                    print(f"  Score! click({cp['x']},{cp['y']}) -> score={final_score} pos={final_pos}")
        finally:
            gc.enable()

        print(f"Probed {len(landing_map)} positions in {time.monotonic()-t0:.1f}s found_win={found_win}")

        # Analyze landing distribution
        unique_landings = {}
        for key, val in landing_map.items():
            pos_key = str(val["pos"])
            if pos_key not in unique_landings:
                unique_landings[pos_key] = []
            unique_landings[pos_key].append(key)

        print(f"\nUnique landing positions: {len(unique_landings)}")
        for pos, clicks in list(unique_landings.items())[:10]:
            print(f"  {pos}: clicked from {clicks[:3]}")

        # Find if any landing is near target
        if targets and ball_pos is not None:
            for target_tuple in targets:
                tag, color, tx, ty = target_tuple
                print(f"\nTarget: {tag} color={color} at ({tx},{ty})")
                for (cx, cy), val in landing_map.items():
                    pos = val["pos"]
                    if pos:
                        for p in pos:
                            if abs(p[0]-tx) <= 1 and abs(p[1]-ty) <= 1:
                                print(f"  Near target! click({cx},{cy}) -> ball at {p}")

        # Also try: click near ball position and above target
        print("\n--- Targeted clicks near ball ---")
        bx, by = ball_pos[0] if ball_pos is not None and len(ball_pos) > 0 else (0, 0)
        for dx in range(-8, 9, 2):
            for dy in range(-8, 9, 2):
                cx = int(bx) + dx
                cy = int(by) + dy
                if cx < 0 or cy < 0 or cx >= 64 or cy >= 64:
                    continue
                probe = copy.deepcopy(raw_env)
                try:
                    probe.step(click6, data={"x": cx, "y": cy})
                    obs2 = probe.step(click7)
                    g2 = probe._game
                    final_pos = g2.yxpvimvli
                    final_lc = int(getattr(obs2, "levels_completed", 0) or 0)
                    if final_lc > start_lc:
                        print(f"  WIN! click({cx},{cy}) near ball")
                        found_win = True
                    elif final_pos is not None and final_pos.tolist() != ball_pos.tolist():
                        print(f"  Ball moved: click({cx},{cy}) -> {final_pos.tolist()}")
                except Exception as e:
                    pass

        if not found_win:
            print("\nNo win found with click+ACTION7 approach.")
            print("Investigating action sequence further...")

            # Try: multiple clicks before ACTION7
            print("\n--- 2-click chains + ACTION7 ---")
            # Find clicks that move ball to different positions
            moving_clicks = [(k, v["pos"]) for k, v in landing_map.items()
                            if v["pos"] != ball_pos.tolist() if ball_pos is not None]
            print(f"Clicks that move ball: {len(moving_clicks)}")
            for (cx, cy), pos in moving_clicks[:5]:
                print(f"  click({cx},{cy}) -> ball at {pos}")

    except Exception as e:
        import traceback; traceback.print_exc()
    break
print("Done.")
