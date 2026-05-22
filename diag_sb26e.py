"""
sb26: Place all tiles first, then ACTION5 to trigger check.
Correct order: match tile color to chapter color at each slot position.
Tiles: [14,15,9,11] at positions [(17,56),(25,56),(33,56),(41,56)]
Chapters: [9,14,11,15] -> need tiles [2,0,3,1] at slots [0,1,2,3]
"""
import copy
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "sb26" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)
    level = getattr(game, "current_level", None)
    sprites = level._sprites if level else []

    print("=== sb26: place-all-then-A5 ===")

    try:
        from arcengine.enums import GameAction as _GameAction
        action5 = _GameAction.ACTION5
        action6 = _GameAction.ACTION6
        start_lc = int(getattr(obs, "levels_completed", 0) or 0)

        # Get tile and slot positions/colors
        tiles = [s for s in sprites if "lngftsryyw" in str(getattr(s, "tags", []))]
        slots_y27 = [s for s in sprites if "susublrply" in str(getattr(s, "tags", []))
                     and s._y <= 40 and s.is_visible]

        tile_info = [(t._x, t._y,
                      t.pixels[t.pixels.shape[0]//2, t.pixels.shape[1]//2] if t.pixels is not None else -1,
                      t.pixels.shape[1]//2, t.pixels.shape[0]//2)
                     for t in tiles]
        slot_info = [(s._x, s._y,
                      s.pixels.shape[1]//2, s.pixels.shape[0]//2)
                     for s in slots_y27]
        chapter_colors = [wc.pixels[0,0] if wc.pixels is not None else -1 for wc in game.wcfyiodrx]

        print(f"Tiles: {[(t[0],t[1],t[2]) for t in tile_info]}")
        print(f"Slots: {[(s[0],s[1]) for s in slot_info]}")
        print(f"Chapter colors: {chapter_colors}")

        # Match tile to slot based on chapter color
        # Chapter i at slot i needs tile with color = chapter_colors[i]
        tile_color = [t[2] for t in tile_info]
        assignment = {}  # slot_i -> tile_i
        for slot_i, chap_color in enumerate(chapter_colors):
            for tile_i, tc in enumerate(tile_color):
                if tc == chap_color:
                    assignment[slot_i] = tile_i
                    break

        print(f"\nAssignment: slot_i -> tile_i: {assignment}")

        # Try: place all tiles, then ACTION5
        probe = copy.deepcopy(raw_env)
        total_actions = 0

        for slot_i in range(len(slot_info)):
            if slot_i not in assignment:
                print(f"  No tile for slot {slot_i}!")
                break
            tile_i = assignment[slot_i]
            tx, ty, tc, thx, thy = tile_info[tile_i]
            sx, sy, shx, shy = slot_info[slot_i]

            pick_x, pick_y = tx + thx, ty + thy
            place_x, place_y = sx + shx, sy + shy

            print(f"  Slot {slot_i}: pick tile {tile_i} (color={tc}) at ({pick_x},{pick_y}) -> slot ({place_x},{place_y})")

            # Pick up tile
            obs1 = probe.step(action6, data={"x": pick_x, "y": pick_y})
            total_actions += 1
            print(f"    After pick: lqcskynzr={probe._game.lqcskynzr.name if probe._game.lqcskynzr else None} lc={int(getattr(obs1,'levels_completed',0) or 0)}")

            # Place in slot
            obs2 = probe.step(action6, data={"x": place_x, "y": place_y})
            total_actions += 1
            lc = int(getattr(obs2, "levels_completed", 0) or 0)
            print(f"    After place: lc={lc} sjcuorclg={probe._game.sjcuorclg} artsfnufc={probe._game.artsfnufc}")
            if lc > start_lc:
                print("  *** LEVEL WIN! ***"); break

        # Now ACTION5 to trigger sequential check
        print("\n--- ACTION5 to trigger check ---")
        obs_a5 = probe.step(action5)
        total_actions += 1
        lc = int(getattr(obs_a5, "levels_completed", 0) or 0)
        g2 = probe._game
        print(f"After A5: lc={lc} state={_state_name(obs_a5)} pmygakdvy={g2.pmygakdvy}")

        if lc > start_lc:
            print(f"*** WIN! Total actions = {total_actions} ***")
        else:
            print(f"No win. pmygakdvy={g2.pmygakdvy}")

            # Try A5 → place → check?
            print("\n--- Alternative: A5 first, then place ---")
            probe2 = copy.deepcopy(raw_env)
            probe2.step(action5)

            for slot_i in range(len(slot_info)):
                if slot_i not in assignment:
                    break
                tile_i = assignment[slot_i]
                tx, ty, tc, thx, thy = tile_info[tile_i]
                sx, sy, shx, shy = slot_info[slot_i]
                pick_x, pick_y = tx + thx, ty + thy
                place_x, place_y = sx + shx, sy + shy

                probe2.step(action6, data={"x": pick_x, "y": pick_y})
                obs2 = probe2.step(action6, data={"x": place_x, "y": place_y})
                lc = int(getattr(obs2, "levels_completed", 0) or 0)
                print(f"  After place slot {slot_i}: lc={lc} pmygakdvy={probe2._game.pmygakdvy}")
                if lc > start_lc:
                    print("  *** WIN! ***"); break

            # Try multiple A5 → place cycles
            print("\n--- A5 + place + A5 + place + ... ---")
            probe3 = copy.deepcopy(raw_env)
            for slot_i in range(len(slot_info)):
                if slot_i not in assignment:
                    break
                tile_i = assignment[slot_i]
                tx, ty, tc, thx, thy = tile_info[tile_i]
                sx, sy, shx, shy = slot_info[slot_i]
                pick_x, pick_y = tx + thx, ty + thy
                place_x, place_y = sx + shx, sy + shy

                probe3.step(action5)
                probe3.step(action6, data={"x": pick_x, "y": pick_y})
                obs2 = probe3.step(action6, data={"x": place_x, "y": place_y})
                lc = int(getattr(obs2, "levels_completed", 0) or 0)
                print(f"  A5+pick+place slot {slot_i}: lc={lc} pmygakdvy={probe3._game.pmygakdvy}")
                if lc > start_lc:
                    print("  *** WIN! ***"); break

    except Exception as e:
        import traceback; traceback.print_exc()
    break
print("Done.")
