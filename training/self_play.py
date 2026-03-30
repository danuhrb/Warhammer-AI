"""
Self-play callback for SB3 MaskablePPO training.

Periodically snapshots the current policy as a new opponent, then samples
from the opponent pool (biased toward recent checkpoints) to keep the
training agent challenged.
"""
from __future__ import annotations
import os
from typing import List, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SelfPlayCallback(BaseCallback):
    """
    Every `save_freq` steps:
      1. Save the current model as a new generation in `opponent_dir`.
      2. Rebuild the opponent pool from all saved checkpoints.
      3. Update the environment's opponent function to sample from the pool.

    The opponent model is cached in memory and only reloaded when the
    pool is updated, avoiding disk I/O on every action.
    """

    def __init__(self, save_freq: int = 50_000,
                 opponent_dir: str = "opponents",
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.opponent_dir = opponent_dir
        self.generation = 0
        self._cached_model = None
        os.makedirs(opponent_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.opponent_dir, f"gen_{self.generation}")
            self.model.save(path)
            self.generation += 1

            pool = self._get_all_opponents()
            if self.verbose:
                print(f"[SelfPlay] gen {self.generation}, pool size {len(pool)}")

            self._update_env_opponent(pool)
        return True

    def _get_all_opponents(self) -> List[str]:
        files = sorted([
            os.path.join(self.opponent_dir, f)
            for f in os.listdir(self.opponent_dir)
            if f.startswith("gen_") and f.endswith(".zip")
        ])
        return files

    def _update_env_opponent(self, pool: List[str]):
        """Pick one opponent from the pool, cache it, and set it on the env."""
        if not pool:
            return

        from sb3_contrib import MaskablePPO

        weights = np.arange(1, len(pool) + 1, dtype=float)
        weights /= weights.sum()
        chosen_path = np.random.choice(pool, p=weights)

        try:
            self._cached_model = MaskablePPO.load(chosen_path)
            if self.verbose:
                print(f"[SelfPlay] loaded opponent: {os.path.basename(chosen_path)}")
        except Exception as e:
            if self.verbose:
                print(f"[SelfPlay] failed to load {chosen_path}: {e}")
            self._cached_model = None

        cached = self._cached_model

        def opponent_fn(obs, mask):
            if cached is not None:
                try:
                    action, _ = cached.predict(obs, action_masks=mask)
                    return action
                except Exception:
                    pass
            return _random_fallback(obs, mask)

        try:
            self.training_env.env_method("set_opponent_fn", opponent_fn)
        except Exception:
            env = self.training_env.envs[0]
            if hasattr(env, "set_opponent_fn"):
                env.set_opponent_fn(opponent_fn)


def _random_fallback(obs, mask):
    """Emergency fallback if model loading fails."""
    from env.action_codec import MAX_TARGETS, NUM_ACTION_TYPES
    from game_engine import MAX_UNITS_PER_PLAYER
    nvec = [MAX_UNITS_PER_PLAYER, NUM_ACTION_TYPES, MAX_TARGETS]

    def _sample(m):
        v = np.where(m > 0)[0]
        return int(np.random.choice(v)) if len(v) else 0

    s1, s2 = nvec[0], nvec[1]
    return np.array([_sample(mask[:s1]), _sample(mask[s1:s1 + s2]),
                     _sample(mask[s1 + s2:])])
