"""
su15: precise object-centric approach.
- Identify: bicnaxoxq at (8,52) is the key object for level 0
- Try clicking it at exact position (not pre-generated grid)
- Try all positions around ball (4,59) at fine resolution
- Read canvas changes to track ball trajectory
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

    print("=== su15: precise object-centric probe ===")

    try:
        from arcengine.enums import GameAction as _GameAction
        click6 = _GameAction.ACTION6
        click7 = _GameAction.ACTION7

        start_lc = int(getattr(obs, "levels_completed", 0) or 0)

        # Object discovery: find ALL sprites with tags
        level = game.current_level
        sprites = level._sprites
        print(f"Sprites ({len(sprites)}):")
        for i, s in enumerate(sprites[:20]):
            nm = str(getattr(s, "name", ""))
            tg = str(getattr(s, "tags", []))
            px = getattr(s, "pixels", None)
            w = px.shape[1] if px is not None and hasattr(px, "shape") else 0
            h = px.shape[0] if px is not None and hasattr(px, "shape") else 0
            print(f"  [{i}] {nm!r} tags={tg} pos=({s._x},{s._y}) size={w}x{h}")

        # bicnaxoxq specific positions
        bicn = game.bicnaxoxq
        if bicn:
            print(f"\nbicnaxoxq: pos=({bicn.x},{bicn.y}) size={bicn.width}x{bicn.height}")
            # Try all positions within bicnaxoxq
            for cx in range(bicn.x, bicn.x + bicn.width):
                for cy in range(bicn.y, bicn.y + bicn.height):
                    probe = copy.deepcopy(raw_env)
                    probe.step(click6, data={"x": cx, "y": cy})
                    obs2 = probe.step(click7)
                    lc = int(getattr(obs2, "levels_completed", 0) or 0)
                    sc = probe._game._score
                    state = _state_name(obs2)
                    pos = probe._game.yxpvimvli
                    if lc > start_lc or state == "WIN":
                        print(f"  WIN! click({cx},{cy}) -> lc={lc}")
                    elif sc > 0:
                        print(f"  Score! click({cx},{cy}) -> sc={sc}")
                    elif pos is not None and pos.tolist() != game.yxpvimvli.tolist():
                        print(f"  Ball moved! click({cx},{cy}) -> {pos.tolist()}")
            print(f"  Probed bicnaxoxq area")

        # Try clicking at ball position itself
        ball_pos = game.yxpvimvli
        if ball_pos is not None:
            bx, by = int(ball_pos[0][0]), int(ball_pos[0][1])
            print(f"\nBall at grid ({bx},{by})")

            # Check canvas state at ball position
            canvas = game.sdrzkrjcp
            zdoy = game.zdoyfxons
            print(f"Canvas at ball pos: sdrzkrjcp[{by}][{bx}]={canvas[by][bx] if canvas is not None else '?'}")
            print(f"Canvas at target: zdoyfxons[58][3]={zdoy[58][3] if zdoy is not None else '?'}")

            # What are the zero positions in canvas (passable)?
            if canvas is not None:
                zeros = np.argwhere(canvas == 0)
                print(f"\nPassable positions (canvas zeros): {len(zeros)}")
                # Group by column
                for row, col in zeros[:20]:
                    print(f"  canvas[{row}][{col}]=0 -> grid({col},{row})")

        # Try clicking at ALL positions (0-63 × 0-63) to find ball movement
        print("\n--- Full position sweep (fine grid) ---")
        ball_start = game.yxpvimvli.tolist() if game.yxpvimvli is not None else None
        best_dist = 999
        best_click = None
        gc.disable()
        try:
            for cy in range(0, 64):
                for cx in range(0, 64):
                    probe = copy.deepcopy(raw_env)
                    try:
                        probe.step(click6, data={"x": cx, "y": cy})
                        obs2 = probe.step(click7)
                        lc = int(getattr(obs2, "levels_completed", 0) or 0)
                        state = _state_name(obs2)
                        if lc > start_lc or state == "WIN":
                            print(f"WIN! click({cx},{cy})")
                            break
                        pos = probe._game.yxpvimvli
                        if pos is not None and pos.tolist() != ball_start:
                            px2, py2 = pos[0][0], pos[0][1]
                            # Distance to target (3,58)
                            dist = abs(px2-3) + abs(py2-58)
                            if dist < best_dist:
                                best_dist = dist
                                best_click = (cx, cy, pos.tolist())
                                print(f"  Better ball pos: click({cx},{cy}) -> {pos.tolist()} dist={dist}")
                    except: pass
        finally:
            gc.enable()

        print(f"\nBest: dist={best_dist} click={best_click}")

        # Also read canvas change after click
        print("\n--- Canvas changes after click ---")
        probe = copy.deepcopy(raw_env)
        probe.step(click6, data={"x": 8, "y": 52})
        canvas_after = probe._game.sdrzkrjcp
        if canvas is not None and canvas_after is not None:
            diff = canvas_after - canvas
            changed = np.argwhere(diff != 0)
            print(f"Canvas changes at (8,52): {len(changed)} pixels changed")
            if len(changed) > 0:
                print(f"  Changed positions: {changed[:10].tolist()}")

        # What happens to qbykqfzrj (target sprites)?
        print("\n--- qbykqfzrj (target sprites) ---")
        for s in game.qbykqfzrj:
            print(f"  pos=({s.x},{s.y}) pixels={s.pixels[:2,:2] if s.pixels is not None else '?'}")

    except Exception as e:
        import traceback; traceback.print_exc()
    break
print("Done.")
