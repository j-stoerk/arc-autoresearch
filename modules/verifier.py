"""
Step iv — Verifier
Symbolic constraint checking: given a candidate program (list of action_ids)
and the current world-model constraints, return a score and validity flag.
Barebones: checks that no action violates a known world-model prohibition.
"""
from __future__ import annotations
from dataclasses import dataclass
from modules.dsl import DSLConfig


@dataclass
class VerificationResult:
    valid:   bool
    score:   float
    reason:  str = ""


class Verifier:
    def __init__(self, dsl: DSLConfig):
        self.dsl             = dsl
        self.prohibited_ids: set[int] = set()   # updated by world model

    def verify(self, program: list[int]) -> VerificationResult:
        """
        Check a candidate action sequence.
        Returns invalid if any action is prohibited; otherwise score by length
        (shorter programs are better — parsimony).
        """
        for action_id in program:
            if action_id in self.prohibited_ids:
                op = self.dsl.op_by_id(action_id)
                name = op.name if op else str(action_id)
                return VerificationResult(valid=False, score=0.0,
                                          reason=f"prohibited op: {name}")
        # Shorter programs score higher
        score = 1.0 / (len(program) + 1)
        return VerificationResult(valid=True, score=score)

    def rank(self, programs: list[list[int]]) -> list[list[int]]:
        """Return programs sorted by verification score (best first), dropping invalids."""
        scored = [
            (prog, self.verify(prog))
            for prog in programs
        ]
        valid  = [(p, v) for p, v in scored if v.valid]
        return [p for p, _ in sorted(valid, key=lambda x: x[1].score, reverse=True)]
