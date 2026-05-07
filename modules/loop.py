"""
Step ix — Belief Revision and Loop Control
Decides when to re-infer goals, re-search, or give up within a single episode.
Tracks budget usage and convergence signals.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LoopState:
    steps_taken:    int   = 0
    replans:        int   = 0
    stalled_steps:  int   = 0    # steps without grid change
    last_grid_hash: int   = 0


class LoopController:
    def __init__(
        self,
        max_replans:     int   = 10,   # agent-tunable
        stall_threshold: int   = 5,    # agent-tunable
    ):
        self.max_replans     = max_replans
        self.stall_threshold = stall_threshold
        self.state           = LoopState()

    def reset(self) -> None:
        self.state = LoopState()

    def should_replan(self, state_repr, goal_reached: bool, program_exhausted: bool) -> bool:
        """
        Return True if the agent should re-run goal inference + search.
        Triggers on: program exhausted, stall detected, or goal not reached.
        """
        if goal_reached:
            return False
        if self.state.replans >= self.max_replans:
            return False

        grid_hash = hash(state_repr.grid.tobytes())
        if grid_hash == self.state.last_grid_hash:
            self.state.stalled_steps += 1
        else:
            self.state.stalled_steps = 0
        self.state.last_grid_hash = grid_hash

        if program_exhausted or self.state.stalled_steps >= self.stall_threshold:
            self.state.replans += 1
            return True

        return False

    def budget_exhausted(self, actions_taken: int, action_budget: int) -> bool:
        return actions_taken >= action_budget

    def record_step(self) -> None:
        self.state.steps_taken += 1
