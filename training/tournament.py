#!/usr/bin/env python3
"""
Round-robin tournament with ELO ratings.

Usage:
  python -m training.tournament --model-dir models/opponents --num-games 20

Pits every saved model checkpoint against every other one in round-robin
matches, computes ELO ratings, and prints a leaderboard.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sb3_contrib import MaskablePPO
from env.wh40k_env import Warhammer40kEnv
from env.action_codec import build_action_mask, decode_action
from env.obs_builder import encode_observation
from game_engine import MAX_UNITS_PER_PLAYER
from bots.random_bot import random_policy


def load_agent(path: str):
    """Load a trained MaskablePPO agent, or return None for 'random'."""
    if path == "__random__":
        return None
    return MaskablePPO.load(path)


def agent_action(agent, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if agent is None:
        return random_policy(obs, mask)
    action, _ = agent.predict(obs, action_masks=mask)
    return action


def play_game(agent_a, agent_b, seed: int = 0) -> Optional[int]:
    """
    Play one full game. agent_a = player 0, agent_b = player 1.
    Returns 0 if agent_a wins, 1 if agent_b wins, None for draw.
    """
    def opp_fn(obs, mask):
        return agent_action(agent_b, obs, mask)

    env = Warhammer40kEnv(opponent_fn=opp_fn, rng_seed=seed)
    obs, info = env.reset(seed=seed)

    done = False
    while not done:
        mask = info["action_mask"]
        action = agent_action(agent_a, obs, mask)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return env.engine.winner


def run_tournament(agent_paths: List[str], num_games: int = 20,
                   base_seed: int = 0) -> Dict[str, float]:
    """
    Round-robin tournament. Returns ELO dict sorted by rating.
    """
    elo: Dict[str, float] = {p: 1500.0 for p in agent_paths}
    K = 32.0

    agents = {}
    for path in agent_paths:
        agents[path] = load_agent(path)

    total_matches = len(agent_paths) * (len(agent_paths) - 1) // 2
    match_num = 0

    for i, path_a in enumerate(agent_paths):
        for path_b in agent_paths[i + 1:]:
            match_num += 1
            wins_a, wins_b, draws = 0, 0, 0

            for g in range(num_games):
                seed = base_seed + match_num * 1000 + g
                winner = play_game(agents[path_a], agents[path_b], seed=seed)
                if winner == 0:
                    wins_a += 1
                elif winner == 1:
                    wins_b += 1
                else:
                    draws += 1

            score_a = (wins_a + 0.5 * draws) / num_games
            expected_a = 1.0 / (1.0 + 10.0 ** ((elo[path_b] - elo[path_a]) / 400.0))
            elo[path_a] += K * (score_a - expected_a)
            elo[path_b] += K * ((1.0 - score_a) - (1.0 - expected_a))

            name_a = os.path.basename(path_a)
            name_b = os.path.basename(path_b)
            print(f"  [{match_num}/{total_matches}] {name_a} vs {name_b}: "
                  f"{wins_a}W/{draws}D/{wins_b}L")

    return dict(sorted(elo.items(), key=lambda x: x[1], reverse=True))


def main():
    parser = argparse.ArgumentParser(description="WH40K Tournament")
    parser.add_argument("--model-dir", type=str, default="models/opponents")
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--include-random", action="store_true",
                        help="Include a random bot in the tournament")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    agent_paths = []
    if os.path.isdir(args.model_dir):
        for f in sorted(os.listdir(args.model_dir)):
            if f.endswith(".zip"):
                agent_paths.append(os.path.join(args.model_dir, f))

    if args.include_random:
        agent_paths.append("__random__")

    if len(agent_paths) < 2:
        print(f"Need at least 2 agents. Found {len(agent_paths)} in {args.model_dir}")
        sys.exit(1)

    print(f"Tournament: {len(agent_paths)} agents, {args.num_games} games each pair")
    print("Agents:", [os.path.basename(p) for p in agent_paths])
    print()

    elo = run_tournament(agent_paths, num_games=args.num_games,
                         base_seed=args.seed)

    print("\n=== LEADERBOARD ===")
    for rank, (path, rating) in enumerate(elo.items(), 1):
        name = os.path.basename(path)
        print(f"  #{rank}  {name:30s}  ELO: {rating:.1f}")


if __name__ == "__main__":
    main()
