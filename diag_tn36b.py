"""
tn36 targeted diagnostic: read program state, simulate win condition, find solution.

Game structure:
- bzirenxmrg wraps plljmx_4 (right panel, visible)
- Program encoded in Maidxz bits in 'takfnb' panel
- Run by clicking sucqgkbuojsa button
- Win when htntnzkbzu.x/y/scale/rotation/sjmtdfxdrc matches aqszntqeae
"""
import gc
import copy
import time
from prepare import episode_iterator, _state_name

def get_game_state(raw_env):
    g = getattr(raw_env, "_game", None)
    if g is None: return None
    fdk = getattr(g, "fdksqlmpki", None)
    if fdk is None: return None
    return fdk

def read_program(fdk):
    """Read the current program from bzirenxmrg's dalucpicjf."""
    brz = fdk.bzirenxmrg
    vup = brz.vupcwzjtxu  # dalucpicjf
    prog = list(vup.vkuvtkaerv)
    return prog

def get_target(fdk):
    """Get target position/state for bzirenxmrg."""
    brz = fdk.bzirenxmrg
    aqsz = brz.aqszntqeae
    if aqsz is None:
        return None
    return {
        'x': aqsz.x, 'y': aqsz.y,
        'scale': aqsz.scale, 'rotation': aqsz.rotation,
        'sjmtdfxdrc': aqsz.sjmtdfxdrc,
    }

def get_current(fdk):
    """Get current position/state of bzirenxmrg's htntnzkbzu."""
    brz = fdk.bzirenxmrg
    htn = brz.htntnzkbzu
    return {
        'x': htn.x, 'y': htn.y,
        'scale': htn.scale, 'rotation': htn.rotation,
        'sjmtdfxdrc': htn.sjmtdfxdrc,
    }

def get_panel_info(fdk):
    """Get Maidxz bit positions for editing bzirenxmrg's program."""
    brz = fdk.bzirenxmrg
    vup = brz.vupcwzjtxu
    slots = []
    for slot in vup.rzmeklhluf:
        bits = []
        for bit in slot.sonocxtjtj:
            bits.append({
                'x': bit.x, 'y': bit.y,
                'w': bit.width, 'h': bit.height,
                'checked': bit.yliktcpsfp,
                'center_x': bit.x + bit.width // 2,
                'center_y': bit.y + bit.height // 2,
            })
        slots.append({'bits': bits, 'value': slot.qaeirkuwro})
    return slots

def get_run_button(fdk):
    """Get sucqgkbuojsa (run button) click position."""
    brz = fdk.bzirenxmrg
    sxh = brz.sxhtkytekm
    if sxh is None: return None
    return {'x': sxh.x + sxh.width // 2, 'y': sxh.y + sxh.height // 2}

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "tn36" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    if raw_env is None:
        print("No raw_env"); break

    game = getattr(raw_env, "_game", None)
    fdk = get_game_state(raw_env)
    lc_init = int(getattr(obs, "levels_completed", 0) or 0)

    print(f"=== tn36 solver ===  level_index={game._current_level_index}  win_score={game._win_score}")
    print(f"Levels completed: {lc_init}")

    prog = read_program(fdk)
    target = get_target(fdk)
    current = get_current(fdk)
    slots = get_panel_info(fdk)
    run_btn = get_run_button(fdk)

    print(f"\nInitial program: {prog}")
    print(f"Current pos: {current}")
    print(f"Target pos:  {target}")
    print(f"\nRun button: {run_btn}")

    print(f"\nPanel slots ({len(slots)} slots):")
    for i, s in enumerate(slots):
        print(f"  slot[{i}] val={s['value']} bits:", end="")
        for b in s['bits']:
            print(f" ({b['center_x']},{b['center_y']})={'ON' if b['checked'] else 'off'}", end="")
        print()

    try:
        from arcengine.enums import GameAction as _GameAction
        click_action = _GameAction.ACTION6
    except ImportError:
        print("No GameAction!"); break

    # BRUTE FORCE: try all combinations of bit-toggling to find winning program
    # For each slot, each bit can be toggled. Total combos = 2^(total_bits)
    total_bits = sum(len(s['bits']) for s in slots)
    print(f"\nTotal bits: {total_bits}, combos: {2**total_bits}")

    best_solution = None
    t0 = time.monotonic()

    for combo in range(2**total_bits):
        if time.monotonic() - t0 > 30.0:
            print("Time limit hit")
            break

        probe = copy.deepcopy(raw_env)
        click_seq = []

        # Apply toggles
        bit_idx = 0
        for s_i, s in enumerate(slots):
            for b_i, b in enumerate(s['bits']):
                if combo & (1 << bit_idx):
                    # Toggle this bit by clicking its center
                    cx = b['center_x']
                    cy = b['center_y']
                    probe.step(click_action, data={"x": cx, "y": cy})
                    click_seq.append({"x": cx, "y": cy})
                bit_idx += 1

        # Click run button
        if run_btn:
            obs2 = probe.step(click_action, data=run_btn)
            lc = int(getattr(obs2, "levels_completed", 0) or 0)
            if lc > lc_init:
                best_solution = (combo, click_seq)
                print(f"\nSOLUTION FOUND! combo={combo:0{total_bits}b}  {len(click_seq)+1} clicks")
                print(f"  Toggle clicks: {click_seq}")
                print(f"  Run click: {run_btn}")
                break

    if best_solution is None:
        print("\nNo solution found in brute force.")
        # Check what program gives current state
        if target is not None:
            # Determine target from first principles: what's needed
            dx = target['x'] - current['x']
            dy = target['y'] - current['y']
            print(f"Need to move: dx={dx}, dy={dy}")
            print(f"CSPOIQWER=4, so dx/4={dx//4}, dy/4={dy//4}")
        else:
            print("No target (aqszntqeae is None)")

    # Also check what happens after solution (does level increment?)
    if best_solution:
        print("\n--- Simulating full level completion ---")
        probe2 = copy.deepcopy(raw_env)
        combo, toggle_clicks = best_solution
        for c in toggle_clicks:
            probe2.step(click_action, data=c)
        obs2 = probe2.step(click_action, data=run_btn)
        lc = int(getattr(obs2, "levels_completed", 0) or 0)
        state = _state_name(obs2)
        print(f"After solution: lc={lc} state={state}")
        fdk2 = get_game_state(probe2)
        if fdk2:
            print(f"bzirenxmrg.vklyonlcrw={fdk2.bzirenxmrg.vklyonlcrw}")

    break
print("Done.")
