from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


@dataclass(frozen=True)
class FrozenLakeSpec:
    map_size: int = 4
    is_slippery: bool = False
    map_seed: int = 10
    render_mode: str = "rgb_array"
    desc: Optional[List[str]] = None  # If provided, overrides generation


def make_frozenlake_env(spec: FrozenLakeSpec) -> tuple[gym.Env, list[str]]:
    """
    Create FrozenLake environment and return (env, desc).
    If spec.desc is provided, uses it. Otherwise generates a random map using seed.
    """
    if spec.desc is None:
        desc = generate_random_map(size=spec.map_size, seed=spec.map_seed)
    else:
        desc = spec.desc

    env = gym.make(
        "FrozenLake-v1",
        render_mode=spec.render_mode,
        desc=desc,
        is_slippery=spec.is_slippery,
    )
    return env, desc
