"""
agent.py — ARC-AGI-3 autoresearch agent (thin orchestrator).
Usage:
    uv run agent.py > run.log 2>&1
"""
import time

from prepare            import evaluate_rhae, _state_name
from modules.perception import Perception
from modules.world_model import WorldModel
from modules.search     import Search
from modules.mechanics  import MechanicsLearner
from modules.runner     import solve_episode


class Agent:
    def __init__(self):
        self.perception = Perception()
        self.world      = WorldModel()
        self.search     = Search()
        self.mechanics  = MechanicsLearner(self.perception, _state_name)
        self._eval_start = 0.0

    def run_episode(self, spec, env, initial_obs):
        t_start = time.monotonic()
        if self._eval_start == 0.0:
            self._eval_start = t_start
        return solve_episode(spec, env, t_start, self._eval_start,
                             self.world, self.search, self.mechanics, self.perception)


def main():
    t0 = time.monotonic()
    metrics = evaluate_rhae(Agent())
    elapsed = time.monotonic() - t0
    print(f"rhae_score:    {metrics['rhae_score']:.6f}")
    print(f"avg_actions:   {metrics['avg_actions']:.1f}")
    print(f"total_seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
