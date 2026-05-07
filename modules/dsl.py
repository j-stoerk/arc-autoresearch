"""
Step iii — DSL
Defines the operation set and the world-model-conditioned projection
that filters the full vocabulary to a task-relevant sub-language.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Operation:
    name:         str
    action_id:    int                        # gymnasium discrete action index
    preconditions: list[str] = field(default_factory=list)  # required world-model tags
    relevance:    float = 1.0               # updated by world model


# Barebones default vocabulary — replace action_ids with real ARC-AGI-3 values
DEFAULT_OPS: list[Operation] = [
    Operation("move_up",       0, preconditions=["movable"]),
    Operation("move_down",     1, preconditions=["movable"]),
    Operation("move_left",     2, preconditions=["movable"]),
    Operation("move_right",    3, preconditions=["movable"]),
    Operation("select",        4, preconditions=[]),
    Operation("deselect",      5, preconditions=[]),
    Operation("recolor_0",     6, preconditions=["recolorable"]),
    Operation("recolor_1",     7, preconditions=["recolorable"]),
    Operation("recolor_2",     8, preconditions=["recolorable"]),
    Operation("recolor_3",     9, preconditions=["recolorable"]),
    Operation("rotate_cw",    10, preconditions=["rotatable"]),
    Operation("rotate_ccw",   11, preconditions=["rotatable"]),
    Operation("mirror_h",     12, preconditions=["mirrorable"]),
    Operation("mirror_v",     13, preconditions=["mirrorable"]),
    Operation("fill",         14, preconditions=["fillable"]),
    Operation("delete",       15, preconditions=[]),
    Operation("copy",         16, preconditions=[]),
    Operation("paste",        17, preconditions=[]),
    Operation("noop",         18, preconditions=[]),
]


@dataclass
class DSLConfig:
    operations:    list[Operation] = field(default_factory=lambda: list(DEFAULT_OPS))
    beam_width:    int   = 5      # agent-tunable
    max_depth:     int   = 7      # agent-tunable
    min_relevance: float = 0.1   # ops below this are suppressed

    def conditioned_ops(self, active_tags: set[str]) -> list[Operation]:
        """
        World-model-conditioned projection: return ops whose preconditions
        are all satisfied by active_tags AND whose relevance >= min_relevance.
        """
        result = []
        for op in self.operations:
            if op.relevance < self.min_relevance:
                continue
            if all(tag in active_tags for tag in op.preconditions):
                result.append(op)
        return sorted(result, key=lambda o: o.relevance, reverse=True)

    def op_by_id(self, action_id: int) -> Optional[Operation]:
        for op in self.operations:
            if op.action_id == action_id:
                return op
        return None
