"""Episode runner — plan execution and full solve dispatch."""
import time
from prepare import EpisodeResult, TIME_BUDGET, _state_name

try:
    from arcengine.enums import GameAction as _GameAction
    _CLICK = _GameAction.ACTION6
except ImportError:
    _CLICK = None


def _exec(raw_env, plan, by_name, t_start, start_levels, env):
    """Execute mixed plan (list[str|dict]). Mutates env. Returns goal_reached."""
    for item in plan:
        if time.monotonic() - t_start > TIME_BUDGET or env.actions_taken >= env.action_budget:
            break
        if isinstance(item, dict):
            if _CLICK is None:
                break
            obs = raw_env.step(_CLICK, data=item)
        else:
            act = by_name.get(item)
            if act is None:
                break
            obs = raw_env.step(act)
        env.actions_taken += 1
        env.last_obs = obs
        if int(getattr(obs, "levels_completed", 0) or 0) > start_levels or _state_name(obs) == "WIN":
            return True
    return False


def solve_episode(spec, env, t_start, eval_start, world, search, mechanics, perception):
    """Full episode solver: cache → keyboard BFS → MechanicsLearner → click BFS."""
    raw_env = getattr(env, "raw_env", None)
    if raw_env is None:
        return EpisodeResult(spec.task_id, env.actions_taken, False, 0.0)

    actions = list(getattr(raw_env, "action_space", []) or [])
    if not actions:
        return EpisodeResult(spec.task_id, env.actions_taken, False, 0.0)

    game_id      = getattr(getattr(raw_env, "_game", None), "_game_id", "") or ""
    start_levels = int(getattr(env.last_obs, "levels_completed", 0) or 0)
    by_name      = {getattr(a, "name", ""): a for a in actions}
    deadline     = eval_start + TIME_BUDGET - 10
    plan_nodes   = world.get_node_budget(game_id)

    def run(plan):
        return _exec(raw_env, plan, by_name, t_start, start_levels, env)

    def result(goal):
        return EpisodeResult(spec.task_id, env.actions_taken, goal, 0.0)

    def mech(sacts, budget=50):
        if world.is_isb_exhausted(game_id):
            mechanics._isb_exhausted = {game_id[:4]}
        else:
            mechanics._isb_exhausted = set()
        p = mechanics.learn_and_plan(
            raw_env, sacts, start_levels,
            node_budget=budget, global_deadline=deadline, all_actions=actions,
        )
        if game_id[:4] in mechanics._isb_exhausted and not world.is_isb_exhausted(game_id):
            world.mark_isb_exhausted(game_id)
        return p

    def click_bfs():
        has_complex = any(callable(getattr(a, "is_complex", None)) and a.is_complex() for a in actions)
        if not has_complex or world.is_click_exhausted(game_id):
            return None
        return search.click_bfs_plan(
            raw_env, start_levels, perception, _state_name,
            on_phase2b_exhausted=world.mark_click_exhausted, game_id=game_id,
        )

    # Cached plan
    cached = world.get_cached_plan(game_id)
    if cached is not None and run(cached.get("plan", [])):
        return result(True)

    simple = [a for a in actions if not (callable(getattr(a, "is_complex", None)) and a.is_complex())]

    # Click-only game
    if not simple:
        p = click_bfs()
        if p is not None:
            goal = run(p)
            if goal:
                world.cache_plan(game_id, "click", p)
            return result(goal)
        p = mech([], 50)
        if p is not None:
            goal = run(p)
            if goal:
                world.cache_plan(game_id, "click", p)
            return result(goal)
        return result(False)

    # Keyboard game
    plan, nodes, unique = search.local_bfs_plan(
        raw_env, simple, start_levels, plan_nodes, perception, _state_name,
    )
    world.record_game_result(game_id, nodes, plan is not None, unique)

    if plan is None:
        p = mech(simple, max(50, plan_nodes))
        if p is not None:
            goal = run(p)
            if goal:
                world.cache_plan(game_id, "click", p)
            return result(goal)
        p = click_bfs()
        if p is not None:
            goal = run(p)
            if goal:
                world.cache_plan(game_id, "click", p)
            return result(goal)
        return result(False)

    goal = run(plan)
    if goal:
        world.cache_plan(game_id, "keyboard", plan)
    return result(goal)
