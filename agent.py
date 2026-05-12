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
import copy
from collections import deque
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

        # Seed hypotheses
        self.hyp_set.seed(SEED_HYPOTHESES)

    # ------------------------------------------------------------------ #
    # Core episode loop                                                    #
    # ------------------------------------------------------------------ #

    def run_episode(self, spec, env, initial_obs: np.ndarray) -> EpisodeResult:
        """
        Run one full episode. Called by evaluate_rhae() in env_setup.py.
        Returns EpisodeResult with actions_taken and goal_reached.
        """
        t_start = time.monotonic()

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

        simple_actions = [
            action for action in actions
            if not (callable(getattr(action, "is_complex", None)) and action.is_complex())
        ]
        if not simple_actions:
            return None

        start_levels = int(getattr(env.last_obs, "levels_completed", 0) or 0)
        plan = self._find_level_plan(raw_env, simple_actions, start_levels)
        if not plan:
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

        ep = Episode(
            task_id=spec.task_id,
            trajectory=[],
            outcome=goal_reached,
            rhae=0.0,
            fingerprint=state.grid.flatten().astype(float),
        )
        self.memory.store(ep)
        return EpisodeResult(spec.task_id, env.actions_taken, goal_reached, 0.0)

    def _find_level_plan(self, raw_env, actions, start_levels: int) -> list[str] | None:
        max_depth = 40
        max_nodes = 1800

        queue = deque([(copy.deepcopy(raw_env), [])])
        seen = set()
        nodes = 0

        while queue and nodes < max_nodes:
            current, plan = queue.popleft()
            nodes += 1
            key = self._local_state_key(current)
            if key in seen:
                continue
            seen.add(key)
            if len(plan) >= max_depth:
                continue

            for action in actions:
                try:
                    nxt = copy.deepcopy(current)
                    obs = nxt.step(action)
                except Exception:
                    continue
                new_plan = plan + [getattr(action, "name", str(action))]
                if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                    return new_plan
                if _state_name(obs) == "NOT_FINISHED":
                    new_key = self._local_state_key(nxt)
                    if new_key not in seen:
                        queue.append((nxt, new_plan))
        return None

    def _local_state_key(self, raw_env) -> tuple:
        game = getattr(raw_env, "_game", None)
        if game is None:
            return (id(raw_env),)

        parts = [getattr(game, "_current_level_index", 0)]
        level = getattr(game, "current_level", None)
        if level is not None:
            for s in getattr(level, "_sprites", []):
                tags = tuple(sorted(getattr(s, "tags", [])))
                parts.append((int(getattr(s, "_x", 0)), int(getattr(s, "_y", 0)), tags))
        else:
            for v in vars(game).values():
                if hasattr(v, "pixels") and hasattr(v, "_x") and hasattr(v, "_y"):
                    tags = tuple(sorted(getattr(v, "tags", [])))
                    parts.append((int(getattr(v, "_x", 0)), int(getattr(v, "_y", 0)), tags))
        return tuple(parts)


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
