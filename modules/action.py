"""
Step v — Action Execution
Execute a verified program (list of action_ids) against the environment,
one step at a time, collecting (obs, reward, done) tuples.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class StepRecord:
    action_id: int
    obs:       np.ndarray
    reward:    float
    done:      bool
    info:      dict


class ActionExecutor:
    def __init__(self):
        self.history: list[StepRecord] = []

    def reset(self):
        self.history.clear()

    def execute_program(self, program: list[int], env) -> list[StepRecord]:
        """
        Step through each action in the program.
        Stops early if done=True or ActionBudgetExceeded is raised.
        Returns list of step records.
        """
        records: list[StepRecord] = []
        for action_id in program:
            try:
                obs, reward, done, info = env.step(action_id)
                record = StepRecord(
                    action_id=action_id,
                    obs=np.asarray(obs),
                    reward=float(reward),
                    done=bool(done),
                    info=info if info else {},
                )
                records.append(record)
                self.history.append(record)
                if done:
                    break
            except Exception:
                break
        return records

    def last_obs(self) -> Optional[np.ndarray]:
        return self.history[-1].obs if self.history else None
