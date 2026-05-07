"""
Step vii — Episodic Memory
Store and retrieve past episode trajectories.
Retrieval: cosine similarity on a flat grid fingerprint.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Episode:
    task_id:    str
    trajectory: list[int]      # sequence of action_ids taken
    outcome:    bool           # goal reached?
    rhae:       float
    fingerprint: np.ndarray    # flattened initial grid (for retrieval)


class Memory:
    def __init__(self, max_episodes: int = 128):
        self.max_episodes  = max_episodes
        self.episodes:     list[Episode] = []

    def store(self, episode: Episode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def retrieve(self, query_grid: np.ndarray, top_k: int = 3) -> list[Episode]:
        """Return top-k episodes whose initial grids are most similar to query."""
        if not self.episodes:
            return []
        query = query_grid.flatten().astype(float)
        q_norm = np.linalg.norm(query) + 1e-9

        def sim(ep: Episode) -> float:
            fp = ep.fingerprint.astype(float)
            return float(np.dot(query, fp) / (q_norm * (np.linalg.norm(fp) + 1e-9)))

        scored = sorted(self.episodes, key=sim, reverse=True)
        return scored[:top_k]

    def successful_trajectories(self) -> list[list[int]]:
        return [ep.trajectory for ep in self.episodes if ep.outcome]
