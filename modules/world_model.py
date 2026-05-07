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
    def __init__(self, mdl_threshold: float = 0.05):
        self.mdl_threshold = mdl_threshold    # agent-tunable
        self.rules: list[Rule] = [
            Rule(**r) for r in DEFAULT_RULES
        ]

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
