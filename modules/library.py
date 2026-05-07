"""
Step iii — Library
Macro-operations learned via program compression (DreamCoder / STITCH style).
Each macro is a named sequence of DSL action_ids with an acquisition context tag.
Active only when the current world-model state matches the acquisition context.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional


@dataclass
class Macro:
    name:               str
    ops:                list[int]        # sequence of action_ids
    acquisition_tags:   set[str]        # world-model tags at acquisition time
    use_count:          int = 0
    success_count:      int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.use_count)


class Library:
    """Stores and retrieves macro-operations. Compression mined after each episode."""

    def __init__(self, min_freq: int = 2):
        self.macros:   list[Macro] = []
        self.min_freq  = min_freq            # agent-tunable
        self._corpus:  list[list[int]] = []  # successful program trajectories

    def add_trajectory(self, ops: list[int]) -> None:
        """Record a successful action sequence for later compression."""
        self._corpus.append(ops)

    def compress(self, active_tags: set[str]) -> None:
        """
        Mine recurring sub-sequences of length 2-3 from the corpus.
        Promote those appearing >= min_freq times as named macros.
        Barebones: bigram + trigram frequency counting.
        """
        ngram_counts: Counter = Counter()
        for traj in self._corpus:
            for n in (2, 3):
                for i in range(len(traj) - n + 1):
                    ngram_counts[tuple(traj[i:i + n])] += 1

        existing_ops = {tuple(m.ops) for m in self.macros}
        for ngram, count in ngram_counts.items():
            if count >= self.min_freq and ngram not in existing_ops:
                macro = Macro(
                    name=f"macro_{len(self.macros)}",
                    ops=list(ngram),
                    acquisition_tags=set(active_tags),
                )
                self.macros.append(macro)

    def active_macros(self, active_tags: set[str]) -> list[Macro]:
        """Return macros whose acquisition context is compatible with current tags."""
        return [
            m for m in self.macros
            if m.acquisition_tags.issubset(active_tags)
        ]

    def record_outcome(self, macro_name: str, success: bool) -> None:
        for m in self.macros:
            if m.name == macro_name:
                m.use_count += 1
                if success:
                    m.success_count += 1
