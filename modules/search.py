"""
Step iii — Search
Neural-guided beam search over the conditioned DSL sub-language.
Barebones: uniform scoring (all ops equally likely). The agent should
replace score_ops() with a learned neural prior.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from modules.dsl import DSLConfig, Operation
from modules.library import Library, Macro


@dataclass
class Beam:
    ops:   list[int]    # action_id sequence so far
    score: float = 0.0  # log-probability (higher = better)


class Search:
    def __init__(self, dsl: DSLConfig, library: Library):
        self.dsl     = dsl
        self.library = library

    def score_ops(
        self,
        ops: list[Operation],
        state,           # StateRepr — ignored in barebones, used by neural prior
        goal,            # goal repr — ignored in barebones
    ) -> list[float]:
        """
        Return a score per operation (higher = more promising).
        Barebones: uniform. Replace with neural prior conditioned on (state, goal).
        """
        return [op.relevance for op in ops]

    def beam_search(
        self,
        active_tags: set[str],
        state,
        goal,
        max_depth: Optional[int] = None,
    ) -> list[list[int]]:
        """
        Return top-beam_width action sequences (shortest programs first).
        Expands the conditioned DSL ops plus active macro-ops.
        """
        depth      = max_depth or self.dsl.max_depth
        beam_width = self.dsl.beam_width

        # Seed beam with single-step candidates
        cond_ops   = self.dsl.conditioned_ops(active_tags)
        scores     = self.score_ops(cond_ops, state, goal)
        beams: list[Beam] = sorted(
            [Beam(ops=[op.action_id], score=s) for op, s in zip(cond_ops, scores)],
            key=lambda b: b.score,
            reverse=True,
        )[:beam_width]

        # Also seed with macro expansions
        for macro in self.library.active_macros(active_tags):
            macro_score = sum(scores[i] for i, op in enumerate(cond_ops)
                              if op.action_id in macro.ops[:1]) / max(1, len(macro.ops))
            beams.append(Beam(ops=macro.ops[:], score=macro_score))
        beams = sorted(beams, key=lambda b: b.score, reverse=True)[:beam_width]

        # Expand depth-first up to max_depth
        for _ in range(depth - 1):
            candidates: list[Beam] = []
            for beam in beams:
                new_scores = self.score_ops(cond_ops, state, goal)
                for op, s in zip(cond_ops, new_scores):
                    candidates.append(
                        Beam(ops=beam.ops + [op.action_id], score=beam.score + s)
                    )
            beams = sorted(candidates, key=lambda b: b.score, reverse=True)[:beam_width]

        return [b.ops for b in beams]
