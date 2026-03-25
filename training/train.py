#!/usr/bin/env python3
"""
Training entry point for Warhammer 40K RL agent.

Usage:
  # Phase 1: Train against random opponent
  python -m training.train --opponent random --timesteps 200000

  # Phase 2: Train against heuristic opponent
  python -m training.train --opponent heuristic --timesteps 500000

  # Phase 3: Self-play training
  python -m training.train --opponent self-play --timesteps 1000000

  # Resume from checkpoint
  python -m training.train --opponent self-play --timesteps 500000 --resume models/wh40k_agent
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback,
)

from env.wh40k_env import Warhammer40kEnv
from bots.random_bot import random_policy
from training.self_play import SelfPlayCallback


def _mask_fn(env: Warhammer40kEnv) -> np.ndarray:
    return env.action_masks()


def make_env(opponent: str = "random", seed: int = 42):
    """Create and wrap the training environment."""
    if opponent == "heuristic":
        from bots.heuristic_bot import heuristic_policy
        opp_fn = lambda obs, mask: heuristic_policy(obs, mask)
    else:
        opp_fn = random_policy

    env = Warhammer40kEnv(opponent_fn=opp_fn, rng_seed=seed)
    env = ActionMasker(env, _mask_fn)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train WH40K RL Agent")
    parser.add_argument("--opponent", choices=["random", "heuristic", "self-play"],
                        default="random")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model checkpoint to resume from")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-play-freq", type=int, default=20_000,
                        help="Steps between self-play snapshots")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = make_env(opponent=args.opponent, seed=args.seed)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=args.log_dir,
            seed=args.seed,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=50_000,
            save_path=args.save_dir,
            name_prefix="wh40k_agent",
        ),
    ]

    if args.opponent == "self-play":
        callbacks.append(
            SelfPlayCallback(
                save_freq=args.self_play_freq,
                opponent_dir=os.path.join(args.save_dir, "opponents"),
                verbose=1,
            )
        )

    print(f"Training: opponent={args.opponent}, timesteps={args.timesteps}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=f"wh40k_{args.opponent}",
    )

    final_path = os.path.join(args.save_dir, "wh40k_agent_final")
    model.save(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
