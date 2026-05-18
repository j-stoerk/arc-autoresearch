"""
Step viii — World Model
Maintains symbolic rules (geometry, physics, objectness priors).
Each rule has a weight and a scope tag.
After each transition, rules are confirmed/weakened/abduced.
MDL criterion gates new rule acceptance.
World model exposes active_tags for DSL conditioning.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import os
import json


@dataclass
class Rule:
    name:        str
    scope:       str             # e.g. "open_space", "enclosed", "any"
    weight:      float = 1.0    # in [0, 1]; 0 = effectively dead
    confirmed:   int   = 0
    contradicted: int  = 0

    @property
    def tag(self) -> str:
        return self.name


# Default rule set — agent should expand or prune these
DEFAULT_RULES: list[dict] = [
    {"name": "movable",     "scope": "any",        "weight": 0.5},
    {"name": "recolorable", "scope": "any",        "weight": 0.5},
    {"name": "rotatable",   "scope": "any",        "weight": 0.3},
    {"name": "mirrorable",  "scope": "any",        "weight": 0.3},
    {"name": "fillable",    "scope": "enclosed",   "weight": 0.2},
    {"name": "gravity",     "scope": "open_space", "weight": 0.1},
]


class WorldModel:
    # Static starting budgets (from exhaustive analysis; can be tightened by learning)
    _STATIC_BUDGETS: dict[str, int] = {
        "sk48": 3500,   # solution at ~2381 with A*; 3500 for safety margin
        "tr87": 200, "g50t": 200, "wa30": 200, "ls20": 200, "re86": 200,
        "cn04": 300, "ka59": 100, "dc22": 20,
    }
    _FALLBACK_BUDGET = 1800
    _BUDGET_FILE = "world_model_budgets.json"
    _PLAN_CACHE_FILE = "world_model_plans.json"
    _EXHAUSTED_FILE = "world_model_exhausted.json"
    # LeCun-inspired: learned encoder abstractions (relevant sprite indices per game).
    # Keys = 4-char game prefix; values = list[int] of sprite indices that change state.
    _ABSTRACTIONS_FILE = "world_model_abstractions.json"

    def __init__(self, mdl_threshold: float = 0.05):
        self.mdl_threshold = mdl_threshold    # agent-tunable
        self.rules: list[Rule] = [
            Rule(**r) for r in DEFAULT_RULES
        ]
        self.game_budgets: dict[str, int] = dict(self._STATIC_BUDGETS)
        self._load_budgets()
        # Plan cache: maps game_id_prefix → {'type': 'keyboard'|'click', 'plan': [...]}
        # Enables instant replay for solved games — skip BFS entirely on future runs.
        self.plan_cache: dict[str, dict] = {}
        self._load_plans()
        # Games where click BFS (including Phase 2b) fully exhausted the state space.
        # Stored as 4-char prefix set; checked before running click BFS to skip instantly.
        self.exhausted_click_games: set[str] = set()
        self._load_exhausted()
        # Maps game_id_prefix → list of sprite indices that vary during BFS exploration.
        # Learned online; used to build compressed state keys for efficient planning.
        self.sprite_abstractions: dict[str, list[int]] = {}
        self._load_abstractions()

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    @property
    def active_tags(self) -> set[str]:
        """Tags whose rules have weight >= 0.5 (considered confirmed)."""
        return {r.tag for r in self.rules if r.weight >= 0.5}

    def update(self, state_before, action_id: int, state_after, goal_reached: bool) -> None:
        """
        Symbolic rule revision after one transition.
        Barebones heuristic: if goal_reached, confirm all active rules slightly;
        otherwise, weaken the rule most associated with the taken action.
        """
        if goal_reached:
            for r in self.rules:
                r.weight = min(1.0, r.weight + 0.05)
                r.confirmed += 1
        else:
            # Weaken rules that apply only to the action taken
            for r in self.rules:
                if r.scope != "any":
                    r.weight = max(0.0, r.weight - 0.02)
                    r.contradicted += 1

        self._abduce(state_before, action_id, state_after)
        self._prune()

    def update_from_world_model(self, dsl) -> None:
        """
        Propagate rule weights back into DSL operation relevance scores.
        Called after each world model update so DSL conditioning stays fresh.
        """
        tag_weights = {r.tag: r.weight for r in self.rules}
        for op in dsl.operations:
            if op.preconditions:
                op.relevance = min(
                    tag_weights.get(tag, 0.5) for tag in op.preconditions
                )
            else:
                op.relevance = 1.0

    # ------------------------------------------------------------------ #
    # Adaptive node budgets (learning mechanism)                           #
    # ------------------------------------------------------------------ #

    def get_node_budget(self, game_id: str) -> int:
        """Return adaptive node budget for this game (static default + learned)."""
        for prefix, budget in self.game_budgets.items():
            if game_id.startswith(prefix):
                return budget
        return self._FALLBACK_BUDGET

    def record_game_result(
        self,
        game_id: str,
        nodes_explored: int,
        solved: bool,
        unique_states: int,
    ) -> None:
        """Update node budget based on observed outcome (learning)."""
        prefix = game_id[:4]
        if solved:
            # Tighten to just above where solution was found
            new_budget = min(nodes_explored + 100, 4000)
        else:
            # Exhausted: cap at exhaustion point + margin
            new_budget = min(unique_states + 50, 2000)
        current = self.game_budgets.get(prefix, self._FALLBACK_BUDGET)
        if new_budget < current:
            self.game_budgets[prefix] = new_budget
            self._save_budgets()

    def _load_budgets(self) -> None:
        try:
            with open(self._BUDGET_FILE) as f:
                saved = json.load(f)
            for k, v in saved.items():
                # Only load if LOWER than static default (only tighten, never loosen)
                if k not in self.game_budgets or v < self.game_budgets[k]:
                    self.game_budgets[k] = v
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_budgets(self) -> None:
        try:
            with open(self._BUDGET_FILE, "w") as f:
                json.dump(self.game_budgets, f)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Plan cache: instant replay for solved games                          #
    # ------------------------------------------------------------------ #

    def get_cached_plan(self, game_id: str) -> dict | None:
        """Return cached winning plan for this game, or None if unknown."""
        for prefix, plan_info in self.plan_cache.items():
            if game_id.startswith(prefix):
                return plan_info
        return None

    def cache_plan(self, game_id: str, plan_type: str, plan: list) -> None:
        """Store a winning plan so it can be replayed without BFS."""
        prefix = game_id[:4]
        self.plan_cache[prefix] = {"type": plan_type, "plan": plan}
        self._save_plans()

    def _load_plans(self) -> None:
        try:
            with open(self._PLAN_CACHE_FILE) as f:
                self.plan_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_plans(self) -> None:
        try:
            with open(self._PLAN_CACHE_FILE, "w") as f:
                json.dump(self.plan_cache, f)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Exhausted click games: skip BFS for games with no reachable solution #
    # ------------------------------------------------------------------ #

    def is_click_exhausted(self, game_id: str) -> bool:
        """True if click BFS + Phase 2b already exhausted for this game."""
        prefix = game_id[:4]
        return prefix in self.exhausted_click_games

    def mark_click_exhausted(self, game_id: str) -> None:
        """Record that Phase 2b exhausted the full state space — never try again."""
        prefix = game_id[:4]
        if prefix not in self.exhausted_click_games:
            self.exhausted_click_games.add(prefix)
            self._save_exhausted()

    def _load_exhausted(self) -> None:
        try:
            with open(self._EXHAUSTED_FILE) as f:
                data = json.load(f)
            self.exhausted_click_games = set(data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_exhausted(self) -> None:
        try:
            with open(self._EXHAUSTED_FILE, "w") as f:
                json.dump(sorted(self.exhausted_click_games), f)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Learned state abstraction (LeCun JEPA-inspired encoder)              #
    # The world model learns which sprite indices are dynamically relevant  #
    # by observing which ones change during BFS exploration.                #
    # Compressed keys eliminate redundant background-sprite dimensions,    #
    # dramatically shrinking the effective search space.                    #
    # ------------------------------------------------------------------ #

    def get_sprite_abstraction(self, game_id: str) -> list[int] | None:
        """Return cached relevant sprite indices for this game, or None if unknown."""
        return self.sprite_abstractions.get(game_id[:4])

    def cache_sprite_abstraction(self, game_id: str, indices: list[int]) -> None:
        """Store learned relevant-sprite indices for this game."""
        prefix = game_id[:4]
        if prefix not in self.sprite_abstractions:
            self.sprite_abstractions[prefix] = indices
            self._save_abstractions()

    def _load_abstractions(self) -> None:
        try:
            with open(self._ABSTRACTIONS_FILE) as f:
                saved = json.load(f)
            self.sprite_abstractions = {k: v for k, v in saved.items()}
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save_abstractions(self) -> None:
        try:
            with open(self._ABSTRACTIONS_FILE, "w") as f:
                json.dump(self.sprite_abstractions, f)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Internal                                                              #
    # ------------------------------------------------------------------ #

    def _abduce(self, state_before, action_id: int, state_after) -> None:
        """
        Abduction: if state actually changed, hypothesise a new rule.
        MDL gate: only add if it would compress the description.
        Barebones — checks if grid changed at all.
        """
        try:
            import numpy as np
            changed = not np.array_equal(state_before.grid, state_after.grid)
        except Exception:
            changed = False

        if changed:
            candidate_name = f"effect_of_{action_id}"
            existing = {r.name for r in self.rules}
            if candidate_name not in existing:
                # MDL gate: only add if description length decreases
                dl_before = self._description_length()
                new_rule = Rule(name=candidate_name, scope="any", weight=0.3)
                self.rules.append(new_rule)
                dl_after = self._description_length()
                if dl_after >= dl_before + self.mdl_threshold:
                    self.rules.pop()   # reject — does not compress

    def _description_length(self) -> float:
        """
        Proxy MDL: sum of -log2(weight) for all rules with weight > 0.
        Fewer, stronger rules → shorter description.
        """
        total = 0.0
        for r in self.rules:
            if r.weight > 0:
                total += -math.log2(r.weight + 1e-9)
        return total

    def _prune(self) -> None:
        """Remove rules with weight < 0.05 (effectively dead)."""
        self.rules = [r for r in self.rules if r.weight >= 0.05]
