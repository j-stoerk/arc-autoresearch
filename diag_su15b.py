"""su15: check dsqlbvwaj and canvas arrays to understand the puzzle."""
import copy
import numpy as np
from prepare import episode_iterator

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "su15" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)

    print(f"=== su15 puzzle spec ===")

    # dsqlbvwaj (level data 'xkstxyqbs')
    d = getattr(game, "dsqlbvwaj", None)
    print(f"\ndsqlbvwaj type={type(d).__name__} len={len(d) if d is not None else None}")
    if d:
        print(f"  first item: {d[0]}")
        for i, item in enumerate(d[:5]):
            print(f"  [{i}]: {item}")

    # Canvas state
    sc = getattr(game, "sdrzkrjcp", None)
    zd = getattr(game, "zdoyfxons", None)
    if sc is not None:
        nz = np.count_nonzero(sc)
        print(f"\nsdrzkrjcp: {sc.shape} nonzero={nz}")
    if zd is not None:
        nz2 = np.count_nonzero(zd)
        print(f"zdoyfxons: {zd.shape} nonzero={nz2}")

    # nscnqkkvg
    n = getattr(game, "nscnqkkvg", None)
    if n:
        print(f"\nnscnqkkvg len={len(n)}")
        for i, row in enumerate(n[:3]):
            print(f"  [{i}]: {row}")

    # Level data
    level = getattr(game, "current_level", None)
    if level:
        for key in ["xkstxyqbs", "steps", "GreyMasking"]:
            try:
                val = level.get_data(key)
                print(f"\nlevel.get_data({key!r}): {val}")
            except: pass

    # oxgxfichu, yxpvimvli
    ox = getattr(game, "oxgxfichu", None)
    yx = getattr(game, "yxpvimvli", None)
    if ox is not None: print(f"\noxgxfichu={ox}")
    if yx is not None: print(f"yxpvimvli={yx}")

    # Try ACTION7 (submit) and see what changes
    from arcengine.enums import GameAction as _GameAction
    probe = copy.deepcopy(raw_env)
    obs2 = probe.step(_GameAction.ACTION7)
    from prepare import _state_name
    print(f"\nACTION7: state={_state_name(obs2)} score={probe._game._score} lc={int(getattr(obs2,'levels_completed',0) or 0)}")
    print(f"  qygchysnh={probe._game.qygchysnh}")
    print(f"  vsfwpngmx={probe._game.vsfwpngmx}")
    print(f"  oicctzexh={probe._game.oicctzexh}")

    break
print("Done.")
