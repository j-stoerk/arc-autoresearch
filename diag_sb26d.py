"""
sb26: Tile-sorting puzzle.
- Pick up lngftsryyw tile (click to select)
- Place in susublrply slot (click to place)
- Match tile color to chapter color
- ACTION5 activates the puzzle, ACTION7 undoes
"""
import gc
import copy
import time
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "sb26" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)
    level = getattr(game, "current_level", None)
    sprites = level._sprites if level else []

    print("=== sb26: tile-sorting puzzle ===")

    try:
        from arcengine.enums import GameAction as _GameAction
        action5 = _GameAction.ACTION5
        action6 = _GameAction.ACTION6
        start_lc = int(getattr(obs, "levels_completed", 0) or 0)

        # Find the constant evrmzyfopo (separator y between tiles and slots)
        # From code: "lqcskynzr.y > evrmzyfopo and xyygiwqeyp.y > evrmzyfopo"
        # suggests evrmzyfopo separates two groups

        # Read tile colors and slot positions
        print("\n--- lngftsryyw tiles ---")
        tiles = [s for s in sprites if "lngftsryyw" in str(getattr(s, "tags", []))]
        for s in tiles:
            px = getattr(s, "pixels", None)
            center = px[px.shape[0]//2, px.shape[1]//2] if px is not None else "?"
            print(f"  pos=({s._x},{s._y}) visible={s.is_visible} center={center}")

        print("\n--- susublrply slots ---")
        slots = [s for s in sprites if "susublrply" in str(getattr(s, "tags", []))]
        for s in slots:
            px = getattr(s, "pixels", None)
            center = px[px.shape[0]//2, px.shape[1]//2] if px is not None else "?"
            print(f"  pos=({s._x},{s._y}) visible={s.is_visible} center={center}")

        print("\n--- wcfyiodrx chapters ---")
        for i, wc in enumerate(game.wcfyiodrx):
            px = getattr(wc, "pixels", None)
            if px is not None:
                print(f"  [{i}] pos=({wc.x},{wc.y}) topleft={px[0,0]} center={px[px.shape[0]//2,px.shape[1]//2]}")

        # Try: ACTION5 → pick first tile → place in first slot
        print("\n--- Try: A5, pick_tile, place_slot ---")
        probe = copy.deepcopy(raw_env)
        obs5 = probe.step(action5)
        print(f"After A5: lc={int(getattr(obs5,'levels_completed',0) or 0)}")

        # Now: click on first tile to pick it up
        if tiles:
            tile = tiles[0]
            cx = tile._x + tile.pixels.shape[1] // 2
            cy = tile._y + tile.pixels.shape[0] // 2
            print(f"Picking up tile at ({cx},{cy}) color={tile.pixels[1,1] if tile.pixels is not None else '?'}")
            obs_pick = probe.step(action6, data={"x": cx, "y": cy})
            g2 = probe._game
            print(f"After pick: lqcskynzr={g2.lqcskynzr.name if g2.lqcskynzr else None}")

            # Now: click on first slot to place it
            if slots:
                slot = slots[0]
                sx = slot._x + slot.pixels.shape[1] // 2
                sy = slot._y + slot.pixels.shape[0] // 2
                print(f"Placing in slot at ({sx},{sy})")
                obs_place = probe.step(action6, data={"x": sx, "y": sy})
                g3 = probe._game
                lc = int(getattr(obs_place, "levels_completed", 0) or 0)
                print(f"After place: lc={lc} state={_state_name(obs_place)} pmygakdvy={g3.pmygakdvy} sjcuorclg={g3.sjcuorclg}")

        # Try all 24 permutations (4 tiles into 4 slots)
        print("\n--- Try all color-matched placements ---")
        from itertools import permutations

        # Get tile and slot colors
        tile_data = [(t._x, t._y, t.pixels[1,1] if t.pixels is not None else -1) for t in tiles]
        slot_data = [(s._x, s._y) for s in slots]
        chapter_colors = [wc.pixels[0,0] if wc.pixels is not None else -1 for wc in game.wcfyiodrx]
        print(f"Tile colors (center): {[td[2] for td in tile_data]}")
        print(f"Chapter colors (topleft): {chapter_colors}")

        best_lc = 0
        best_seq = None
        t0 = time.monotonic()

        for perm in permutations(range(len(tiles))):
            if time.monotonic() - t0 > 30:
                break
            # perm[i] = which tile goes in slot i
            probe = copy.deepcopy(raw_env)
            probe.step(action5)

            seq = []
            won = False
            for slot_i, tile_i in enumerate(perm):
                if tile_i >= len(tile_data) or slot_i >= len(slot_data):
                    break
                tile_x, tile_y, tile_color = tile_data[tile_i]
                slot_x, slot_y = slot_data[slot_i]
                tile_cx = tile_x + (tiles[tile_i].pixels.shape[1] // 2 if tiles[tile_i].pixels is not None else 1)
                tile_cy = tile_y + (tiles[tile_i].pixels.shape[0] // 2 if tiles[tile_i].pixels is not None else 1)
                slot_cx = slot_x + (slots[slot_i].pixels.shape[1] // 2 if slots[slot_i].pixels is not None else 1)
                slot_cy = slot_y + (slots[slot_i].pixels.shape[0] // 2 if slots[slot_i].pixels is not None else 1)

                # Pick up tile
                probe.step(action6, data={"x": tile_cx, "y": tile_cy})
                # Place in slot
                obs2 = probe.step(action6, data={"x": slot_cx, "y": slot_cy})
                seq.extend([(tile_cx, tile_cy), (slot_cx, slot_cy)])

                lc = int(getattr(obs2, "levels_completed", 0) or 0)
                if lc > start_lc:
                    print(f"\nWIN! perm={perm} at slot {slot_i}")
                    won = True; break
                if probe._game._score > 0:
                    if probe._game._score > best_lc:
                        best_lc = probe._game._score
                        best_seq = (perm, list(seq))

            if won:
                best_seq = (perm, seq); break

        if best_seq:
            perm, seq = best_seq
            print(f"Best score: {best_lc} with perm={perm}")
        else:
            print("No improvement found")

        # Show what ACTION5 does exactly: is it needed?
        print("\n--- Test: Just pick+place without A5 ---")
        probe = copy.deepcopy(raw_env)
        if tiles and slots:
            tile = tiles[0]
            slot = slots[0]
            tx = tile._x + (tile.pixels.shape[1]//2 if tile.pixels is not None else 1)
            ty = tile._y + (tile.pixels.shape[0]//2 if tile.pixels is not None else 1)
            sx = slot._x + (slot.pixels.shape[1]//2 if slot.pixels is not None else 1)
            sy = slot._y + (slot.pixels.shape[0]//2 if slot.pixels is not None else 1)
            probe.step(action6, data={"x": tx, "y": ty})
            obs2 = probe.step(action6, data={"x": sx, "y": sy})
            g2 = probe._game
            print(f"Without A5: lc={int(getattr(obs2,'levels_completed',0) or 0)} artsfnufc={g2.artsfnufc} sjcuorclg={g2.sjcuorclg}")

    except Exception as e:
        import traceback; traceback.print_exc()
    break
print("Done.")
