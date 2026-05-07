"""
ARC-AGI-3 setup and fixed evaluation harness for autoresearch experiments.

Usage:
    uv run prepare.py

This creates/updates `.cache/arc3/` with the public environment manifest used by
the experiment harness. The agent imports this module at runtime; do not change
the scoring code during experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify during experiments)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300
NUM_PUBLIC_TASKS = 25
RHAE_EXPONENT = 2.0
DEFAULT_ACTION_BUDGET = 100
GRID_SIZE = 64

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache" / "arc3"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

# Public examples from the ARC-AGI-3 docs/changelog. `prepare.py` refreshes this
# list from the toolkit/API when possible, but these make offline setup stable.
FALLBACK_PUBLIC_GAMES = [
    "ls20",
    "ft09",
    "vc33",
    "tn36",
    "m0r0",
    "r11l",
    "tu93",
    "sc25",
    "ar25",
    "dc22",
    "cn04",
    "sp80",
    "su15",
    "re86",
    "ka59",
    "s5i5",
    "sk48",
]


@dataclass
class TaskSpec:
    task_id: str
    game_id: str
    seed: int
    level_index: int = 1
    human_actions: int = 20


@dataclass
class EpisodeResult:
    task_id: str
    actions_taken: int
    goal_reached: bool
    rhae: float = 0.0


class ActionBudgetExceeded(RuntimeError):
    pass


class ARCEnvAdapter:
    """Small compatibility layer for the current agent's gym-like executor."""

    def __init__(self, raw_env: Any, spec: TaskSpec):
        self.raw_env = raw_env
        self.spec = spec
        self.actions_taken = 0
        self.action_budget = max(1, 5 * spec.human_actions)
        self.last_obs = None
        self.spec_adapter = type(
            "SpecAdapter",
            (),
            {"env_spec": {}, "task_id": spec.task_id, "game_id": spec.game_id},
        )()

    @property
    def spec(self):
        return self._spec_adapter

    @spec.setter
    def spec(self, value: TaskSpec) -> None:
        self.task_spec = value
        self._spec_adapter = type(
            "SpecAdapter",
            (),
            {"env_spec": {}, "task_id": value.task_id, "game_id": value.game_id},
        )()

    def reset(self) -> np.ndarray:
        obs = self.raw_env.reset()
        self.actions_taken = 0
        self.last_obs = obs
        return normalize_grid(env_to_grid(self.raw_env, obs))

    def step(self, action_id: int):
        if self.actions_taken >= self.action_budget:
            raise ActionBudgetExceeded(self.task_spec.task_id)

        action, data = self._map_action(action_id)
        obs = self.raw_env.step(action, data=data)
        self.actions_taken += 1
        self.last_obs = obs

        grid = normalize_grid(env_to_grid(self.raw_env, obs))
        state_name = _state_name(obs)
        done = state_name in {"WIN", "GAME_OVER", "LOSE", "LOSS"}
        reward = 1.0 if state_name == "WIN" else 0.0
        info = {
            "state": state_name,
            "levels_completed": getattr(obs, "levels_completed", None),
            "score": getattr(obs, "score", None),
        }
        return grid, reward, done, info

    def _map_action(self, action_id: int):
        actions = list(getattr(self.raw_env, "action_space", []) or [])
        if not actions:
            raise RuntimeError(f"No actions available for {self.task_spec.game_id}")

        action = _action_by_id(actions, action_id)
        data = {}
        is_complex = getattr(action, "is_complex", None)
        if callable(is_complex) and is_complex():
            # The current DSL emits only discrete ids. Use a deterministic probe
            # point so complex actions remain legal and reproducible.
            data = {"x": (action_id * 17) % GRID_SIZE, "y": (action_id * 31) % GRID_SIZE}
        return action, data


def _action_by_id(actions: list[Any], action_id: int):
    wanted_names = (f"ACTION{action_id}", f"ACTION{action_id + 1}")
    for wanted in wanted_names:
        for action in actions:
            if getattr(action, "name", "") == wanted:
                return action
    return actions[action_id % len(actions)]


def _state_name(obs: Any) -> str:
    state = getattr(obs, "state", None)
    return getattr(state, "name", str(state or "")).upper()


def frame_to_grid(frame: Any) -> np.ndarray:
    """Extract a 2D integer grid from toolkit frame objects or raw JSON."""
    if frame is None:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

    if isinstance(frame, np.ndarray):
        arr = frame
    elif isinstance(frame, list):
        arr = np.asarray(frame)
    elif isinstance(frame, dict):
        arr = _grid_from_mapping(frame)
    else:
        for name in ("grid", "frame", "screen", "board", "cells", "data"):
            if hasattr(frame, name):
                return frame_to_grid(getattr(frame, name))
        if hasattr(frame, "model_dump"):
            return frame_to_grid(frame.model_dump())
        if hasattr(frame, "__dict__"):
            return frame_to_grid(vars(frame))
        raise ValueError(f"Cannot extract grid from frame type {type(frame)!r}")

    arr = np.asarray(arr, dtype=np.int32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grid, got shape {arr.shape}")
    return normalize_grid(arr)


def _grid_from_mapping(data: dict[str, Any]) -> np.ndarray:
    for key in ("grid", "frame", "screen", "board", "cells", "data"):
        value = data.get(key)
        if isinstance(value, (list, tuple, np.ndarray, dict)):
            try:
                return frame_to_grid(value)
            except Exception:
                pass
    for value in data.values():
        if isinstance(value, (list, tuple, np.ndarray, dict)):
            try:
                return frame_to_grid(value)
            except Exception:
                pass
    raise ValueError("No grid-like field found in frame mapping")


def env_to_grid(raw_env: Any, frame: Any = None) -> np.ndarray:
    """Render the local ARC game sprite state into a compact integer grid."""
    game = getattr(raw_env, "_game", None)
    if game is not None:
        try:
            return _grid_from_game(game)
        except Exception:
            pass
    return frame_to_grid(frame)


def _grid_from_game(game: Any) -> np.ndarray:
    sprites = []
    for value in getattr(game, "__dict__", {}).values():
        if hasattr(value, "pixels") and hasattr(value, "_x") and hasattr(value, "_y"):
            sprites.append(value)

    if not sprites:
        raise ValueError("No sprites found on ARC game instance")

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    sprites = sorted(sprites, key=lambda s: int(getattr(s, "_layer", 0)))
    for sprite in sprites:
        pixels = np.asarray(sprite.pixels, dtype=np.int32)
        if pixels.ndim != 2:
            continue
        y = int(getattr(sprite, "_y", 0))
        x = int(getattr(sprite, "_x", 0))
        h, w = pixels.shape
        r0, c0 = max(0, y), max(0, x)
        r1, c1 = min(GRID_SIZE, y + h), min(GRID_SIZE, x + w)
        if r0 >= r1 or c0 >= c1:
            continue
        sr0, sc0 = r0 - y, c0 - x
        patch = pixels[sr0 : sr0 + (r1 - r0), sc0 : sc0 + (c1 - c0)]
        mask = patch >= 0
        grid[r0:r1, c0:c1][mask] = patch[mask]
    return grid


def normalize_grid(grid: Any) -> np.ndarray:
    """Return a fixed 64x64 integer observation, padding/cropping as needed."""
    arr = np.asarray(grid, dtype=np.int32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    rows = min(GRID_SIZE, arr.shape[0])
    cols = min(GRID_SIZE, arr.shape[1])
    out[:rows, :cols] = arr[:rows, :cols]
    return out


def _import_arcade():
    try:
        import arc_agi
        from arc_agi import OperationMode
    except ImportError as exc:
        raise RuntimeError(
            "ARC-AGI-3 toolkit is not installed. Run `uv sync` after the project "
            "dependency is set to `arc-agi`, then run `uv run prepare.py`."
        ) from exc
    return arc_agi, OperationMode


def _make_arcade(operation_mode: str = "OFFLINE"):
    arc_agi, OperationMode = _import_arcade()
    mode = getattr(OperationMode, operation_mode)
    return arc_agi.Arcade(operation_mode=mode, environments_dir=str(CACHE_DIR))


def _env_id(env_info: Any) -> str:
    if isinstance(env_info, str):
        return env_info
    if isinstance(env_info, dict):
        return str(env_info.get("game_id") or env_info.get("id") or env_info.get("name"))
    return str(getattr(env_info, "game_id", None) or getattr(env_info, "id", None) or env_info)


def discover_public_games(allow_online: bool = True) -> list[str]:
    modes = ["NORMAL", "OFFLINE"] if allow_online else ["OFFLINE"]
    for mode in modes:
        try:
            arc = _make_arcade(mode)
            envs = [_env_id(env) for env in arc.get_environments()]
            envs = [env for env in envs if env and env != "None"]
            if envs:
                return sorted(dict.fromkeys(envs))[:NUM_PUBLIC_TASKS]
        except Exception as exc:
            print(f"Discovery via ARC toolkit {mode} failed: {exc}")

    if len(FALLBACK_PUBLIC_GAMES) >= NUM_PUBLIC_TASKS:
        return FALLBACK_PUBLIC_GAMES[:NUM_PUBLIC_TASKS]
    return FALLBACK_PUBLIC_GAMES[:]


def write_manifest(game_ids: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "task_id": f"{game_id}:{seed}",
            "game_id": game_id,
            "seed": seed,
            "level_index": i + 1,
            "human_actions": 20,
        }
        for i, (game_id, seed) in enumerate((gid, 0) for gid in game_ids[:NUM_PUBLIC_TASKS])
    ]
    MANIFEST_PATH.write_text(json.dumps({"tasks": tasks}, indent=2) + "\n", encoding="utf-8")
    for task in tasks:
        task_path = CACHE_DIR / f"{task['game_id']}.json"
        task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")


def load_task_specs() -> list[TaskSpec]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Missing .cache/arc3/manifest.json. Run `uv run prepare.py`.")
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [TaskSpec(**row) for row in payload.get("tasks", [])][:NUM_PUBLIC_TASKS]


def episode_iterator() -> Iterator[tuple[TaskSpec, ARCEnvAdapter, np.ndarray]]:
    specs = load_task_specs()
    if len(specs) < NUM_PUBLIC_TASKS:
        raise RuntimeError(f"Expected {NUM_PUBLIC_TASKS} tasks, found {len(specs)}")

    try:
        arc = _make_arcade("OFFLINE")
    except Exception:
        arc = _make_arcade("NORMAL")
    for spec in specs:
        raw_env = arc.make(spec.game_id, seed=spec.seed)
        if raw_env is None:
            arc = _make_arcade("NORMAL")
            raw_env = arc.make(spec.game_id, seed=spec.seed)
        if raw_env is None:
            raise RuntimeError(f"Failed to create ARC environment {spec.game_id}")
        env = ARCEnvAdapter(raw_env, spec)
        yield spec, env, env.reset()


def evaluate_rhae(agent) -> dict[str, float]:
    """Fixed RHAE evaluation over the prepared public ARC-AGI-3 environments."""
    started = time.monotonic()
    results: list[EpisodeResult] = []

    for spec, env, initial_obs in episode_iterator():
        if time.monotonic() - started > TIME_BUDGET:
            break
        result = agent.run_episode(spec, env, initial_obs)
        result.rhae = _episode_rhae(spec, result)
        results.append(result)

    weight_total = sum(max(1, spec.level_index) for spec in load_task_specs()[: len(results)])
    weighted_score = 0.0
    for spec, result in zip(load_task_specs(), results):
        weighted_score += max(1, spec.level_index) * result.rhae

    avg_actions = sum(r.actions_taken for r in results) / max(1, len(results))
    return {
        "rhae_score": weighted_score / max(1, weight_total),
        "avg_actions": avg_actions,
        "goal_reached_pct": sum(r.goal_reached for r in results) / max(1, len(results)),
        "episodes_run": len(results),
    }


def _episode_rhae(spec: TaskSpec, result: EpisodeResult) -> float:
    if not result.goal_reached:
        return 0.0
    if result.actions_taken > 5 * spec.human_actions:
        return 0.0
    ratio = spec.human_actions / max(1, result.actions_taken)
    return min(1.0, math.pow(ratio, RHAE_EXPONENT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ARC-AGI-3 public environments")
    parser.add_argument("--offline", action="store_true", help="Do not query the ARC service")
    args = parser.parse_args()

    print(f"ARC cache directory: {CACHE_DIR}")
    game_ids = discover_public_games(allow_online=not args.offline)
    write_manifest(game_ids)

    count = len(load_task_specs())
    print(f"Prepared {count}/{NUM_PUBLIC_TASKS} public ARC task specs.")
    if count < NUM_PUBLIC_TASKS:
        raise SystemExit("Not enough public environments discovered. Set ARC_API_KEY and rerun.")
    print("Done! Ready to run `uv run agent.py`.")


if __name__ == "__main__":
    main()
