"""
Step ii — Hypothesis
Maintain a ranked set of candidate mechanic hypotheses.
Each hypothesis is a dict of string → value describing a suspected rule.
Scored by demo-consistency: how many observed transitions it explains.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hypothesis:
    name:        str
    rule:        dict[str, Any]      # e.g. {"type": "move", "direction": "right"}
    score:       float = 0.0         # higher = more evidence
    confirmed:   int   = 0           # number of transitions it correctly predicted
    contradicted: int  = 0           # number of contradictions

    def __repr__(self):
        return f"Hypothesis({self.name!r}, score={self.score:.3f})"


class HypothesisSet:
    """Ranked list of candidate hypotheses; supports update from observations."""

    def __init__(self, max_hypotheses: int = 16):
        self.max_hypotheses = max_hypotheses
        self.hypotheses: list[Hypothesis] = []

    def seed(self, candidates: list[dict]) -> None:
        """Seed with a list of rule dicts (agent-configurable at startup)."""
        self.hypotheses = [
            Hypothesis(name=h.get("name", f"h{i}"), rule=h)
            for i, h in enumerate(candidates)
        ]

    def update(self, state_before, action: int, state_after) -> None:
        """
        Lightweight compatibility check: confirm/contradict each hypothesis.
        Barebones: a hypothesis is 'confirmed' if the action matches its rule type.
        """
        for h in self.hypotheses:
            rule_action = h.rule.get("action_id")
            if rule_action is not None:
                if rule_action == action:
                    h.confirmed += 1
                    h.score += 0.1
                else:
                    h.contradicted += 1
                    h.score = max(0.0, h.score - 0.05)

        self._prune()

    def top(self, n: int = 1) -> list[Hypothesis]:
        sorted_h = sorted(self.hypotheses, key=lambda h: h.score, reverse=True)
        return sorted_h[:n]

    def _prune(self):
        self.hypotheses = sorted(
            self.hypotheses, key=lambda h: h.score, reverse=True
        )[: self.max_hypotheses]

    def add(self, h: Hypothesis) -> None:
        self.hypotheses.append(h)
        self._prune()
