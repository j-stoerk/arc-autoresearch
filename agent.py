"""
agent.py — ARC-AGI-3 autoresearch agent.
Analogous to train.py in the original autoresearch repo.

This is the ONLY file the autonomous research agent modifies.
All module imports, hyperparameters, and the run_episode() method
are fair game. The evaluation harness in env_setup.py is fixed.

Architecture (9 steps):
  i.   GoalInference   — infer goal and decompose into subgoals
  ii.  HypothesisSet   — maintain candidate mechanic hypotheses
  iii. DSLConfig        — world-model-conditioned operation vocabulary
       Library          — macro-ops learned via program compression
       Search           — neural-guided beam search
  iv.  Verifier         — symbolic constraint checking + ranking
  v.   ActionExecutor   — execute verified program against environment
  vi.  Perception       — multi-level state representation
  vii. Memory           — episodic store + retrieval
  viii.WorldModel        — symbolic rule revision + DSL co-evolution
  ix.  LoopController   — belief revision + replan decisions

Usage:
    uv run agent.py > run.log 2>&1
"""

import time
import numpy as np

from prepare import (
    evaluate_rhae,
    episode_iterator,
    EpisodeResult,
    TIME_BUDGET,
    NUM_PUBLIC_TASKS,
    _state_name,
)

from modules.perception    import Perception
from modules.goal_inference import GoalInference
from modules.hypothesis    import HypothesisSet, Hypothesis
from modules.dsl           import DSLConfig, Operation
from modules.library       import Library
from modules.search        import Search
from modules.verifier      import Verifier
from modules.action        import ActionExecutor
from modules.memory        import Memory, Episode
from modules.world_model   import WorldModel
from modules.loop          import LoopController
from modules.mechanics     import MechanicsLearner

# ---------------------------------------------------------------------------
# Hyperparameters (agent modifies these)
# ---------------------------------------------------------------------------

BEAM_WIDTH       = 7       # search beam width
MAX_DEPTH        = 4       # max program length per plan
MAX_REPLANS      = 8       # max replanning cycles per episode
STALL_THRESHOLD  = 4       # steps without grid change before replan
MDL_THRESHOLD    = 0.05    # world model MDL gate
MIN_RELEVANCE    = 0.1     # DSL op suppression threshold
MEMORY_SIZE      = 64      # max stored episodes

# Seed hypothesis rules (action_id must match real ARC-AGI-3 gymnasium action space)
SEED_HYPOTHESES: list[dict] = [
    {"name": "select_and_move",  "action_id": 3,  "type": "move"},
    {"name": "select_and_color", "action_id": 7,  "type": "recolor"},
    {"name": "rotate",           "action_id": 10, "type": "rotate"},
    {"name": "fill_region",      "action_id": 14, "type": "fill"},
]

# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class Agent:
    """
    Stateful agent that persists memory and world model across episodes
    within a single experiment run. Module instances are shared.
    """

    def __init__(self):
        self.dsl        = DSLConfig(beam_width=BEAM_WIDTH, max_depth=MAX_DEPTH,
                                    min_relevance=MIN_RELEVANCE)
        self.dsl.operations = [
            Operation("arc_action_1", 0, preconditions=[], relevance=1.0),
            Operation("arc_action_2", 1, preconditions=[], relevance=1.0),
            Operation("arc_action_3", 2, preconditions=[], relevance=1.0),
            Operation("arc_action_4", 3, preconditions=[], relevance=1.0),
            Operation("arc_action_5", 4, preconditions=[], relevance=1.0),
            Operation("arc_action_6", 5, preconditions=[], relevance=1.0),
            Operation("arc_action_7", 6, preconditions=[], relevance=1.0),
        ]
        self.library    = Library(min_freq=2)
        self.search     = Search(dsl=self.dsl, library=self.library)
        self.verifier   = Verifier(dsl=self.dsl)
        self.memory     = Memory(max_episodes=MEMORY_SIZE)
        self.world      = WorldModel(mdl_threshold=MDL_THRESHOLD)

        self.perception = Perception()
        self.goal_inf   = GoalInference()
        self.hyp_set    = HypothesisSet()
        self.executor   = ActionExecutor()
        self.loop       = LoopController(
            max_replans=MAX_REPLANS,
            stall_threshold=STALL_THRESHOLD,
        )
        self.mechanics  = MechanicsLearner(self.perception, _state_name)

        # Seed hypotheses
        self.hyp_set.seed(SEED_HYPOTHESES)

        # Set on first episode call; used to enforce global TIME_BUDGET in BFS.
        self._eval_start: float = 0.0

    # ------------------------------------------------------------------ #
    # Core episode loop                                                    #
    # ------------------------------------------------------------------ #

    def run_episode(self, spec, env, initial_obs: np.ndarray) -> EpisodeResult:
        """
        Run one full episode. Called by evaluate_rhae() in env_setup.py.
        Returns EpisodeResult with actions_taken and goal_reached.
        """
        t_start = time.monotonic()
        if self._eval_start == 0.0:
            self._eval_start = t_start

        # ── Step vi: parse initial state ─────────────────────────────── #
        self.perception.reset()
        self.executor.reset()
        self.loop.reset()

        state = self.perception.parse(initial_obs)

        # Retrieve similar past episodes from memory
        similar = self.memory.retrieve(state.grid, top_k=3)

        # ── Step i: infer goal ────────────────────────────────────────── #
        goal_grid = self._infer_goal(env, state, similar)
        if goal_grid is not None:
            self.goal_inf.set_goal(goal_grid)
        else:
            planned = self._run_local_search_plan(spec, env, state, t_start)
            if planned is not None:
                return planned
            # BFS and click BFS both failed; beam search has never solved any game and
            # burns 8-14s per episode on animation-heavy games. Return immediately.
            ep = Episode(
                task_id=spec.task_id,
                trajectory=[r.action_id for r in self.executor.history],
                outcome=False,
                rhae=0.0,
                fingerprint=initial_obs.flatten().astype(float),
            )
            self.memory.store(ep)
            return EpisodeResult(spec.task_id, env.actions_taken, False, 0.0)

        goal_reached = self.goal_inf.goal_reached(state)
        program_done = False

        while not goal_reached:
            # Time-budget guard
            if time.monotonic() - t_start > TIME_BUDGET:
                break

            # Action budget guard
            if self.loop.budget_exhausted(env.actions_taken, env.action_budget):
                break

            # ── Step ix: decide whether to replan ────────────────────── #
            if not self.loop.should_replan(state, goal_reached, program_done):
                if program_done:
                    break   # out of replans — give up

            # ── Step ii: update hypotheses ───────────────────────────── #
            # (done incrementally inside the program execution loop below)

            # ── Step iii: DSL conditioning + search ──────────────────── #
            self.world.update_from_world_model(self.dsl)
            active_tags = self.world.active_tags
            programs = self.search.beam_search(active_tags, state, goal_grid)

            # ── Step iv: verify and rank programs ────────────────────── #
            ranked = self.verifier.rank(programs)
            if not ranked:
                program_done = True
                continue

            best_program = ranked[self.loop.state.replans % len(ranked)]

            # ── Step v: execute program ───────────────────────────────── #
            records = self.executor.execute_program(best_program, env)
            program_done = True

            if not records:
                break
            if any(rec.reward > 0.0 for rec in records):
                goal_reached = True
                break

            # ── Step vi: parse new state ──────────────────────────────── #
            last_obs  = records[-1].obs
            new_state = self.perception.parse(last_obs)

            # ── Step ii: update hypotheses from observed transitions ───── #
            for rec in records:
                self.hyp_set.update(state, rec.action_id, new_state)

            # ── Step viii: update world model ─────────────────────────── #
            goal_reached = self.goal_inf.goal_reached(new_state)
            for rec in records:
                self.world.update(state, rec.action_id, new_state, goal_reached)

            state = new_state
            self.loop.record_step()

        # ── Step iii: library compression ────────────────────────────── #
        if goal_reached and self.executor.history:
            traj = [r.action_id for r in self.executor.history]
            self.library.add_trajectory(traj)
            self.library.compress(self.world.active_tags)

        # ── Step vii: store episode in memory ────────────────────────── #
        traj = [r.action_id for r in self.executor.history]
        ep   = Episode(
            task_id     = spec.task_id,
            trajectory  = traj,
            outcome     = goal_reached,
            rhae        = 0.0,   # filled by evaluate_rhae()
            fingerprint = initial_obs.flatten().astype(float),
        )
        self.memory.store(ep)

        return EpisodeResult(
            task_id       = spec.task_id,
            actions_taken = env.actions_taken,
            goal_reached  = goal_reached,
            rhae          = 0.0,   # filled by evaluate_rhae()
        )

    # ------------------------------------------------------------------ #
    # Goal inference helper                                                #
    # ------------------------------------------------------------------ #

    def _infer_goal(self, env, state, similar_episodes) -> "np.ndarray | None":
        """
        Attempt to obtain the goal grid.
        Priority: (1) env info dict, (2) similar memory episodes (placeholder),
        (3) return None (GoalInference will yield no subgoals).
        """
        # Many ARC envs expose the target grid directly via env.spec.env_spec
        try:
            goal_raw = env.spec.env_spec.get("target")
            if goal_raw is not None:
                return np.asarray(goal_raw, dtype=np.int32)
        except Exception:
            pass
        return None

    def _run_local_search_plan(self, spec, env, state, t_start: float) -> EpisodeResult | None:
        raw_env = getattr(env, "raw_env", None)
        actions = list(getattr(raw_env, "action_space", []) or [])
        if raw_env is None or not actions:
            return None

        game_id = getattr(getattr(raw_env, "_game", None), "_game_id", "") or ""
        start_levels = int(getattr(env.last_obs, "levels_completed", 0) or 0)

        # Fast path: replay cached winning plan — skip BFS entirely for solved games.
        cached = self.world.get_cached_plan(game_id)
        if cached is not None:
            plan_type = cached.get("type", "keyboard")
            plan = cached.get("plan", [])
            goal_reached = False
            by_name = {getattr(a, "name", ""): a for a in actions}
            if plan_type == "keyboard":
                for name in plan:
                    if time.monotonic() - t_start > TIME_BUDGET:
                        break
                    if self.loop.budget_exhausted(env.actions_taken, env.action_budget):
                        break
                    action = by_name.get(name)
                    if action is None:
                        break
                    obs = raw_env.step(action)
                    env.actions_taken += 1
                    env.last_obs = obs
                    if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                        goal_reached = True
                        break
            else:  # click
                try:
                    from arcengine.enums import GameAction as _GameAction
                    click_action = _GameAction.ACTION6
                    for data in plan:
                        if time.monotonic() - t_start > TIME_BUDGET:
                            break
                        if self.loop.budget_exhausted(env.actions_taken, env.action_budget):
                            break
                        obs = raw_env.step(click_action, data=data)
                        env.actions_taken += 1
                        env.last_obs = obs
                        if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                            goal_reached = True
                            break
                        if _state_name(obs) == "WIN":
                            goal_reached = True
                            break
                except ImportError:
                    goal_reached = False
            if goal_reached:
                ep = Episode(
                    task_id=spec.task_id, trajectory=[], outcome=True,
                    rhae=0.0, fingerprint=state.grid.flatten().astype(float),
                )
                self.memory.store(ep)
                return EpisodeResult(spec.task_id, env.actions_taken, True, 0.0)
            # Cached plan failed: fall through to BFS (game state may have changed)

        simple_actions = [
            action for action in actions
            if not (callable(getattr(action, "is_complex", None)) and action.is_complex())
        ]

        # Phase 2: click BFS for pure click games (no keyboard simple actions)
        if not simple_actions:
            has_complex = any(
                callable(getattr(a, "is_complex", None)) and a.is_complex()
                for a in actions
            )
            if has_complex:
                if self.world.is_click_exhausted(game_id):
                    click_plan = None
                else:
                    click_plan = self.search.click_bfs_plan(
                        raw_env, start_levels, self.perception, _state_name,
                        on_phase2b_exhausted=self.world.mark_click_exhausted,
                        game_id=game_id,
                    )
                if click_plan is not None:
                    try:
                        from arcengine.enums import GameAction as _GameAction
                        click_action = _GameAction.ACTION6
                    except ImportError:
                        click_action = None
                    goal_reached = False
                    if click_action is not None:
                        for data in click_plan:
                            if time.monotonic() - t_start > TIME_BUDGET:
                                break
                            if self.loop.budget_exhausted(env.actions_taken, env.action_budget):
                                break
                            obs = raw_env.step(click_action, data=data)
                            env.actions_taken += 1
                            env.last_obs = obs
                            if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                                goal_reached = True
                                break
                            if _state_name(obs) == "WIN":
                                goal_reached = True
                                break
                    if goal_reached:
                        self.world.cache_plan(game_id, "click", click_plan)
                    ep = Episode(
                        task_id=spec.task_id,
                        trajectory=[],
                        outcome=goal_reached,
                        rhae=0.0,
                        fingerprint=state.grid.flatten().astype(float),
                    )
                    self.memory.store(ep)
                    return EpisodeResult(spec.task_id, env.actions_taken, goal_reached, 0.0)
            return None

        # Phase 1: keyboard A* BFS — budget determined by WorldModel (adaptive + learned)
        plan_nodes = self.world.get_node_budget(game_id)

        plan, nodes_explored, unique_states = self.search.local_bfs_plan(
            raw_env, simple_actions, start_levels, plan_nodes,
            self.perception, _state_name,
        )
        self.world.record_game_result(game_id, nodes_explored, plan is not None, unique_states)

        if plan is None:
            # BFS found no plan. Try MechanicsLearner: detects cycle semantics from
            # observed transitions and builds a direct plan in O(N) — returns None
            # immediately for games without cycle structure (cheap probe).
            mech_plan = self.mechanics.learn_and_plan(
                raw_env, simple_actions, start_levels,
                node_budget=max(50, plan_nodes),
                global_deadline=self._eval_start + TIME_BUDGET - 10,
            )
            # Note: relevant_full_key_bfs_plan is available in search module for
            # future use when games with hidden carry-state are identified.

            if mech_plan is not None:
                by_name = {getattr(a, "name", ""): a for a in actions}
                goal_reached = False
                for name in mech_plan:
                    if time.monotonic() - t_start > TIME_BUDGET:
                        break
                    if self.loop.budget_exhausted(env.actions_taken, env.action_budget):
                        break
                    action = by_name.get(name)
                    if action is None:
                        break
                    obs = raw_env.step(action)
                    env.actions_taken += 1
                    env.last_obs = obs
                    if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                        goal_reached = True
                        break
                    if _state_name(obs) == "WIN":
                        goal_reached = True
                        break
                if goal_reached:
                    self.world.cache_plan(game_id, "keyboard", mech_plan)
                ep = Episode(
                    task_id=spec.task_id, trajectory=[], outcome=goal_reached,
                    rhae=0.0, fingerprint=state.grid.flatten().astype(float),
                )
                self.memory.store(ep)
                return EpisodeResult(spec.task_id, env.actions_taken, goal_reached, 0.0)

        if plan is None:
            # Keyboard BFS found nothing. For mixed games (keyboard + click), also try
            # click BFS — some games have keyboard actions that change no state but
            # require ACTION6 clicks to progress (e.g. sc25, dc22, ka59).
            has_complex = any(
                callable(getattr(a, "is_complex", None)) and a.is_complex()
                for a in actions
            )
            if has_complex and not self.world.is_click_exhausted(game_id):
                click_plan = self.search.click_bfs_plan(
                    raw_env, start_levels, self.perception, _state_name,
                    on_phase2b_exhausted=self.world.mark_click_exhausted,
                    game_id=game_id,
                )
                if click_plan is not None:
                    try:
                        from arcengine.enums import GameAction as _GameAction
                        click_action = _GameAction.ACTION6
                    except ImportError:
                        click_action = None
                    goal_reached = False
                    if click_action is not None:
                        for data in click_plan:
                            if time.monotonic() - t_start > TIME_BUDGET:
                                break
                            if self.loop.budget_exhausted(env.actions_taken, env.action_budget):
                                break
                            obs = raw_env.step(click_action, data=data)
                            env.actions_taken += 1
                            env.last_obs = obs
                            if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                                goal_reached = True
                                break
                            if _state_name(obs) == "WIN":
                                goal_reached = True
                                break
                    if goal_reached:
                        self.world.cache_plan(game_id, "click", click_plan)
                    ep = Episode(
                        task_id=spec.task_id,
                        trajectory=[],
                        outcome=goal_reached,
                        rhae=0.0,
                        fingerprint=state.grid.flatten().astype(float),
                    )
                    self.memory.store(ep)
                    return EpisodeResult(spec.task_id, env.actions_taken, goal_reached, 0.0)
            # Feed effective-action observations into WorldModel so
            # the beam-search fallback focuses on actions that actually change state.
            self.search.update_dsl_from_bfs(raw_env, simple_actions, self.perception)
            return None

        by_name = {getattr(action, "name", ""): action for action in actions}
        goal_reached = False
        for name in plan:
            if time.monotonic() - t_start > TIME_BUDGET:
                break
            if self.loop.budget_exhausted(env.actions_taken, env.action_budget):
                break
            action = by_name.get(name)
            if action is None:
                break
            obs = raw_env.step(action)
            env.actions_taken += 1
            env.last_obs = obs
            if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                goal_reached = True
                break
            if _state_name(obs) == "WIN":
                goal_reached = True
                break

        if goal_reached:
            self.world.cache_plan(game_id, "keyboard", plan)
        ep = Episode(
            task_id=spec.task_id,
            trajectory=[],
            outcome=goal_reached,
            rhae=0.0,
            fingerprint=state.grid.flatten().astype(float),
        )
        self.memory.store(ep)
        return EpisodeResult(spec.task_id, env.actions_taken, goal_reached, 0.0)


# ---------------------------------------------------------------------------
# Entry point — mirrors train.py output format
# ---------------------------------------------------------------------------

def main():
    t0     = time.monotonic()
    agent  = Agent()

    print("Starting ARC-AGI-3 autoresearch run...")
    print(f"Hyperparameters:")
    print(f"  BEAM_WIDTH       = {BEAM_WIDTH}")
    print(f"  MAX_DEPTH        = {MAX_DEPTH}")
    print(f"  MAX_REPLANS      = {MAX_REPLANS}")
    print(f"  STALL_THRESHOLD  = {STALL_THRESHOLD}")
    print(f"  MDL_THRESHOLD    = {MDL_THRESHOLD}")
    print(f"  MIN_RELEVANCE    = {MIN_RELEVANCE}")
    print(f"  MEMORY_SIZE      = {MEMORY_SIZE}")
    print()

    metrics = evaluate_rhae(agent)

    elapsed = time.monotonic() - t0

    print()
    print("---")
    print(f"rhae_score:        {metrics['rhae_score']:.6f}")
    print(f"avg_actions:       {metrics['avg_actions']:.1f}")
    print(f"goal_reached_pct:  {metrics['goal_reached_pct']:.4f}")
    print(f"episodes_run:      {metrics['episodes_run']}")
    print(f"total_seconds:     {elapsed:.1f}")


if __name__ == "__main__":
    main()
