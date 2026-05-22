"""Debug exactly what happens when we click run on tn36."""
import copy
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "tn36" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)
    fdk = game.fdksqlmpki
    brz = fdk.bzirenxmrg

    print(f"Initial obs: levels_completed={getattr(obs, 'levels_completed', None)}")
    print(f"  game._current_level_index={game._current_level_index}")
    print(f"  game._win_score={game._win_score}")
    print(f"  brz.htntnzkbzu pos=({brz.htntnzkbzu.x},{brz.htntnzkbzu.y})")
    print(f"  brz.aqszntqeae pos=({brz.aqszntqeae.x},{brz.aqszntqeae.y}) if not None else None")
    print(f"  brz.vklyonlcrw={brz.vklyonlcrw}")
    print(f"  program={fdk.bzirenxmrg.vupcwzjtxu.vkuvtkaerv}")

    try:
        from arcengine.enums import GameAction as _GameAction
        click_action = _GameAction.ACTION6
    except ImportError:
        break

    # Try combo=0 (just click run, no toggles)
    run_btn = {'x': 36, 'y': 55}
    probe0 = copy.deepcopy(raw_env)
    obs0 = probe0.step(click_action, data=run_btn)
    lc0 = int(getattr(obs0, "levels_completed", 0) or 0)
    g0 = probe0._game
    fdk0 = g0.fdksqlmpki
    brz0 = fdk0.bzirenxmrg
    print(f"\nCombo=0 (just run): lc={lc0} state={_state_name(obs0)}")
    print(f"  level_idx={g0._current_level_index} score={g0._score}")
    print(f"  brz.htntnzkbzu pos=({brz0.htntnzkbzu.x},{brz0.htntnzkbzu.y})")
    print(f"  brz.vklyonlcrw={brz0.vklyonlcrw}")
    print(f"  nyhaiggftp={g0.nyhaiggftp} pgualuszrs={g0.pgualuszrs}")

    # Try combo=4 (toggle slot1 bit0, then run)
    probe4 = copy.deepcopy(raw_env)
    obs4a = probe4.step(click_action, data={"x": 26, "y": 42})
    g4 = probe4._game
    fdk4 = g4.fdksqlmpki
    brz4 = fdk4.bzirenxmrg
    print(f"\nAfter toggle (26,42): lc={getattr(obs4a,'levels_completed',None)}")
    print(f"  program now={fdk4.bzirenxmrg.vupcwzjtxu.vkuvtkaerv}")
    print(f"  brz.htntnzkbzu pos=({brz4.htntnzkbzu.x},{brz4.htntnzkbzu.y})")

    obs4b = probe4.step(click_action, data=run_btn)
    lc4 = int(getattr(obs4b, "levels_completed", 0) or 0)
    print(f"\nCombo=4 (toggle+run): lc={lc4} state={_state_name(obs4b)}")
    print(f"  level_idx={g4._current_level_index} score={g4._score}")
    print(f"  brz.htntnzkbzu pos=({brz4.htntnzkbzu.x},{brz4.htntnzkbzu.y})")
    print(f"  brz.vklyonlcrw={brz4.vklyonlcrw}")
    print(f"  nyhaiggftp={g4.nyhaiggftp} pgualuszrs={g4.pgualuszrs}")

    # Try the program [3,3,3,3,3] explicitly
    # Need to toggle: slot1 both, slot3 both, slot4 both
    probe5 = copy.deepcopy(raw_env)
    clicks = [
        {"x": 26, "y": 42},  # slot1 bit0
        {"x": 26, "y": 45},  # slot1 bit1
        {"x": 36, "y": 42},  # slot3 bit0
        {"x": 36, "y": 45},  # slot3 bit1
        {"x": 41, "y": 42},  # slot4 bit0
        {"x": 41, "y": 45},  # slot4 bit1
    ]
    for c in clicks:
        probe5.step(click_action, data=c)
    g5 = probe5._game
    print(f"\nAfter 6 toggles: program={g5.fdksqlmpki.bzirenxmrg.vupcwzjtxu.vkuvtkaerv}")
    obs5 = probe5.step(click_action, data=run_btn)
    lc5 = int(getattr(obs5, "levels_completed", 0) or 0)
    print(f"After run: lc={lc5} state={_state_name(obs5)}")
    print(f"  level_idx={g5._current_level_index}")
    print(f"  brz.htntnzkbzu pos=({g5.fdksqlmpki.bzirenxmrg.htntnzkbzu.x},{g5.fdksqlmpki.bzirenxmrg.htntnzkbzu.y})")
    print(f"  brz.vklyonlcrw={g5.fdksqlmpki.bzirenxmrg.vklyonlcrw}")

    break
print("Done.")
