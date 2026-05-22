"""Check if tn36 win requires multiple steps to process, and verify the 6-toggle solution."""
import copy
from prepare import episode_iterator, _state_name

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "tn36" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)

    try:
        from arcengine.enums import GameAction as _GameAction
        click_action = _GameAction.ACTION6
        noop_action = _GameAction.ACTION1  # keyboard action (no-op for click game)
    except ImportError:
        break

    run_btn = {'x': 36, 'y': 55}

    # Solution: 6 toggles to set [3,3,3,3,3] then run
    toggle_clicks = [
        {"x": 26, "y": 42},  # slot1 bit0
        {"x": 26, "y": 45},  # slot1 bit1
        {"x": 36, "y": 42},  # slot3 bit0
        {"x": 36, "y": 45},  # slot3 bit1
        {"x": 41, "y": 42},  # slot4 bit0
        {"x": 41, "y": 45},  # slot4 bit1
    ]

    probe = copy.deepcopy(raw_env)
    for c in toggle_clicks:
        probe.step(click_action, data=c)

    g = probe._game
    print(f"After 6 toggles: program={g.fdksqlmpki.bzirenxmrg.vupcwzjtxu.vkuvtkaerv}")
    print(f"  timer_count={g.lmkazecqdh.lmkazecqdh}  nyhaiggftp={g.nyhaiggftp}")

    # Now click run and check state after EACH action call
    obs_run = probe.step(click_action, data=run_btn)
    g2 = probe._game
    lc = getattr(obs_run, "levels_completed", None)
    print(f"After run click: lc={lc} state={_state_name(obs_run)}")
    print(f"  level_idx={g2._current_level_index} nyhaiggftp={g2.nyhaiggftp}")
    print(f"  queue_len={len(g2.fdksqlmpki.pxbksnibsu)}")
    print(f"  htn_pos=({g2.fdksqlmpki.bzirenxmrg.htntnzkbzu.x},{g2.fdksqlmpki.bzirenxmrg.htntnzkbzu.y})")

    # Try ACTION1 (non-click) to drain any remaining state
    for i in range(10):
        obs_step = probe.step(noop_action)
        g3 = probe._game
        lc2 = getattr(obs_step, "levels_completed", None)
        state = _state_name(obs_step)
        print(f"  step {i+1}: lc={lc2} state={state} idx={g3._current_level_index} nyhaiggftp={g3.nyhaiggftp}")
        if lc2 is not None and lc2 > 0:
            print(f"  *** LEVEL ADVANCED TO {lc2} ***")
            break
        if state == "WIN":
            print("  *** WIN! ***")
            break

    # Also try: click run MULTIPLE times (for action-based steps)
    print("\n--- Testing multi-click run ---")
    probe2 = copy.deepcopy(raw_env)
    for c in toggle_clicks:
        probe2.step(click_action, data=c)

    for i in range(15):
        obs_i = probe2.step(click_action, data=run_btn)
        g_i = probe2._game
        lc_i = getattr(obs_i, "levels_completed", None)
        state_i = _state_name(obs_i)
        try:
            htn = g_i.fdksqlmpki.bzirenxmrg.htntnzkbzu
            pos = (htn.x, htn.y)
        except:
            pos = "?"
        print(f"  click {i+1}: lc={lc_i} state={state_i} idx={g_i._current_level_index} htn={pos} nyhaiggftp={g_i.nyhaiggftp}")
        if lc_i is not None and lc_i > 0:
            print(f"  *** WIN at click {i+1}! ***")
            break
        if state_i == "WIN":
            break

    break
print("Done.")
