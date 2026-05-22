"""g50t: check rloltuowth, kgvnkyaimw, uwxkstolmf for level 0."""
import copy
from prepare import episode_iterator

for spec, env, obs in episode_iterator():
    game_id = str(spec.task_id)
    if "g50t" not in game_id.lower():
        continue

    raw_env = getattr(env, "raw_env", None)
    game = getattr(raw_env, "_game", None)
    vgwy = getattr(game, "vgwycxsxjz", None)

    print("=== g50t level 0 state ===")
    print(f"player: ({vgwy.dzxunlkwxt.x},{vgwy.dzxunlkwxt.y})")
    print(f"target: ({vgwy.whftgckbcu.x},{vgwy.whftgckbcu.y}) -> win at ({vgwy.whftgckbcu.x+1},{vgwy.whftgckbcu.y+1})")

    print(f"\nrloltuowth (pre-programmed movers): {len(vgwy.rloltuowth)} entries")
    for k, v in vgwy.rloltuowth.items():
        print(f"  entity at ({k.x},{k.y}) path: {v}")

    print(f"\nkgvnkyaimw (pushable boxes): {len(vgwy.kgvnkyaimw)} entries")
    for k, v in vgwy.kgvnkyaimw.items():
        print(f"  entity at ({k.x},{k.y}) path: {v}")

    print(f"\nuwxkstolmf (walls/kjrcloicja): {len(vgwy.uwxkstolmf)} entries")
    for w in vgwy.uwxkstolmf:
        print(f"  wall at ({w.x},{w.y}) visible={w.is_visible}")

    print(f"\nhamayflsib (interactive tiles): {len(vgwy.hamayflsib)} entries")
    for h in vgwy.hamayflsib:
        print(f"  tile at ({h.x},{h.y})")

    print(f"\ndrofvwhbxb (checkpoints): {len(vgwy.drofvwhbxb)} entries")
    for i, d in enumerate(vgwy.drofvwhbxb):
        print(f"  checkpoint[{i}] at ({d.x},{d.y})")

    print(f"\nareahjypvy (move history): {vgwy.areahjypvy}")
    print(f"rlazdofsxb (checkpoint idx): {vgwy.rlazdofsxb}")
    print(f"zmqoxwsfgh (maze start): ({vgwy.zmqoxwsfgh.x},{vgwy.zmqoxwsfgh.y})")

    # Try moving and see what happens to rloltuowth entities
    from arcengine.enums import GameAction as _GameAction
    probe = copy.deepcopy(raw_env)
    probe.step(_GameAction.ACTION2)  # move down
    g2 = probe._game
    v2 = g2.vgwycxsxjz
    print(f"\nAfter ACTION2 (down):")
    print(f"  player: ({v2.dzxunlkwxt.x},{v2.dzxunlkwxt.y})")
    print(f"  walls after move: {[(w.x,w.y) for w in v2.uwxkstolmf]}")
    print(f"  areahjypvy: {v2.areahjypvy}")
    print(f"  kgvnkyaimw paths: {[(k.x,k.y,v) for k,v in v2.kgvnkyaimw.items()]}")

    break
print("Done.")
