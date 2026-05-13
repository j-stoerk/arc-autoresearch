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

        simple_actions = [
            action for action in actions
            if not (callable(getattr(action, "is_complex", None)) and action.is_complex())
        ]

        start_levels = int(getattr(env.last_obs, "levels_completed", 0) or 0)

        # Phase 2: click BFS for pure click games (no keyboard simple actions)
        if not simple_actions:
            has_complex = any(
                callable(getattr(a, "is_complex", None)) and a.is_complex()
                for a in actions
            )
            if has_complex:
                click_plan = self._find_click_plan(raw_env, start_levels)
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

        # Phase 1: keyboard BFS
        # Game-specific node budgets: sk48 solution is at ~3368 nodes; tr87/g50t/wa30
        # are unsolvable by keyboard BFS alone, capped early to stay within 300s budget.
        # ls20/re86 are also unsolvable by keyboard alone; cap to save ~15s budget.
        game_id = getattr(getattr(raw_env, "_game", None), "_game_id", "") or ""
        if game_id.startswith("sk48"):
            plan_nodes = 3500
        elif game_id.startswith(("tr87", "g50t", "wa30", "ls20", "re86")):
            plan_nodes = 200   # confirmed unsolvable; cap to save budget
        else:
            plan_nodes = 1800

        plan = self._find_level_plan(raw_env, simple_actions, start_levels, plan_nodes)
        if plan is None:
            # BFS found nothing: feed effective-action observations into WorldModel so
            # the beam-search fallback focuses on actions that actually change state.
            self._update_dsl_from_bfs(raw_env, simple_actions)
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

    def _find_level_plan(self, raw_env, actions, start_levels: int, max_nodes: int = 1800,
                         global_deadline: float = float("inf")) -> list[str] | None:
        max_depth = 40

        queue = deque([(copy.deepcopy(raw_env), [])])
        seen = set()
        nodes = 0

        while queue and nodes < max_nodes and time.monotonic() < global_deadline:
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
                obs_state = _state_name(obs)
                if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                    return new_plan
                if obs_state == "WIN":
                    return new_plan
                if obs_state == "NOT_FINISHED":
                    new_key = self._local_state_key(nxt)
                    if new_key not in seen:
                        queue.append((nxt, new_plan))
        return None

    def _update_dsl_from_bfs(self, raw_env, actions) -> None:
        """Feed action-effectiveness into DSL preconditions (WorldModel step viii).

        Actions that change game state keep preconditions=[] (always included by
        conditioned_ops). Actions that do nothing get preconditions=["blocked"]
        which is never in active_tags, so they are excluded from beam-search.
        This survives the update_from_world_model call which only resets relevance,
        not preconditions.
        """
        # Reset all ops to unconstrained first
        for op in self.dsl.operations:
            op.preconditions = []
        init_key = self._local_state_key(raw_env)
        for action in actions:
            name = getattr(action, "name", "")
            if not name.startswith("ACTION"):
                continue
            try:
                action_id = int(name[6:]) - 1   # ACTION1 → id=0
                if not (0 <= action_id < 7):
                    continue
                nxt = copy.deepcopy(raw_env)
                nxt.step(action)
                new_key = self._local_state_key(nxt)
                if new_key == init_key:
                    for op in self.dsl.operations:
                        if op.action_id == action_id:
                            op.preconditions = ["blocked"]  # excluded from conditioned_ops
            except Exception:
                pass

    def _get_camera_scale(self, raw_env) -> int:
        cam = getattr(getattr(raw_env, "_game", None), "camera", None)
        if cam is None:
            return 1
        try:
            result = cam.display_to_grid(4, 4)
            if result and result[0] > 0:
                return 4 // result[0]
        except Exception:
            pass
        return 1

    def _click_state_key(self, raw_env) -> tuple:
        game = getattr(raw_env, "_game", None)
        if game is None:
            return ()
        level = getattr(game, "current_level", None)
        if level is None:
            return (getattr(game, "_current_level_index", 0),)
        parts = [getattr(game, "_current_level_index", 0)]
        for s in getattr(level, "_sprites", []):
            px = getattr(s, "pixels", None)
            if px is not None:
                parts.append((s._x, s._y, hash(np.asarray(px, dtype=np.int32).tobytes())))
        return tuple(parts)

    def _find_click_plan(
        self, raw_env, start_levels: int,
        max_nodes: int = 500, time_limit: float = 5.0,
    ) -> list[dict] | None:
        scale = self._get_camera_scale(raw_env)
        game = getattr(raw_env, "_game", None)
        if game is None:
            return None
        level = getattr(game, "current_level", None)
        if level is None:
            return None

        try:
            from arcengine.enums import GameAction as _GameAction
            click_action = _GameAction.ACTION6
        except ImportError:
            return None

        # Phase 1: sprite-center probing — fast filter for interactive positions.
        init_key = self._click_state_key(raw_env)
        candidates = []
        for s in getattr(level, "_sprites", []):
            data = {"x": (int(getattr(s, "_x", 0)) + 1) * scale,
                    "y": (int(getattr(s, "_y", 0)) + 1) * scale}
            try:
                probe = copy.deepcopy(raw_env)
                probe.step(click_action, data=data)
                if self._click_state_key(probe) != init_key:
                    candidates.append(data)
            except Exception:
                pass

        # Phase 2: grid-probe fallback — when sprite centers miss the hotspots
        # (e.g. games where click targets aren't at sprite boundaries).
        # Triggers when < 3 sprite-center candidates found.
        used_grid_probe = False
        if len(candidates) < 3:
            seen_xy = {(d["x"], d["y"]) for d in candidates}
            t_probe = time.monotonic()
            for dy in range(0, 64, 4):
                if len(candidates) >= 30 or time.monotonic() - t_probe > 6.0:
                    break
                for dx in range(0, 64, 4):
                    if len(candidates) >= 30:
                        break
                    if (dx, dy) in seen_xy:
                        continue
                    data = {"x": dx, "y": dy}
                    try:
                        probe = copy.deepcopy(raw_env)
                        probe.step(click_action, data=data)
                        if self._click_state_key(probe) != init_key:
                            candidates.append(data)
                            seen_xy.add((dx, dy))
                    except Exception:
                        pass
            used_grid_probe = True

        if not candidates:
            return None

        # Extend time limit when grid probe was used (deeper search needed for s5i5-like games).
        effective_limit = 12.0 if used_grid_probe else time_limit

        from collections import deque
        queue = deque([(copy.deepcopy(raw_env), [])])
        seen = {self._click_state_key(raw_env)}
        nodes = 0
        deadline = time.monotonic() + effective_limit

        while queue and nodes < max_nodes and time.monotonic() < deadline:
            current, path = queue.popleft()
            nodes += 1
            for data in candidates:
                try:
                    nxt = copy.deepcopy(current)
                    obs = nxt.step(click_action, data=data)
                except Exception:
                    continue
                new_path = path + [data]
                obs_state = _state_name(obs)
                if int(getattr(obs, "levels_completed", 0) or 0) > start_levels:
                    return new_path
                if obs_state == "WIN":
                    return new_path
                if obs_state == "NOT_FINISHED":
                    k = self._click_state_key(nxt)
                    if k not in seen:
                        seen.add(k)
                        queue.append((nxt, new_path))
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
