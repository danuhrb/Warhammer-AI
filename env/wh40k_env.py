"""
Warhammer 40K RL Environment.

Two variants:
  1. Warhammer40kAEC  - PettingZoo AEC for multi-agent / generic usage
  2. Warhammer40kEnv  - Gymnasium single-agent wrapper for SB3 MaskablePPO training
     (opponent actions are handled internally by an opponent policy callback)
"""
from __future__ import annotations
import functools
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wh_types import GameConfig, TerrainPiece, Unit, Squad
from game_engine import GameEngine, Action, Phase, MAX_UNITS_PER_PLAYER, MAX_TURNS
from env.action_codec import (
    build_action_space, decode_action, build_action_mask,
    MAX_TARGETS, NUM_ACTION_TYPES,
)
from env.obs_builder import build_observation_space, encode_observation, OBS_SIZE
from env.reward import RewardCalculator
from unit_data import (
    create_tau_army, create_ultramarine_army, create_blood_angels_army,
    army_points,
)
from terrain import generate_terrain

MAX_STEPS_PER_GAME = MAX_TURNS * 2 * 4 * MAX_UNITS_PER_PLAYER * 2


def _default_armies(points_limit: int = 1000,
                    table_size: tuple = (44.0, 60.0),
                    rng: np.random.RandomState = None):
    p0_units, p0_squads = create_tau_army(owner=0, points_limit=points_limit,
                                          table_size=table_size)
    p1_units, p1_squads = create_blood_angels_army(owner=1, points_limit=points_limit,
                                                    table_size=table_size)
    terrain = generate_terrain(table_size=table_size, num_pieces=6, rng=rng)
    return p0_units, p0_squads, p1_units, p1_squads, terrain


# ---------------------------------------------------------------------------
#  PettingZoo AEC Environment
# ---------------------------------------------------------------------------

class Warhammer40kAEC(AECEnv):
    """PettingZoo AEC environment for Warhammer 40K."""

    metadata = {"render_modes": ["human"], "name": "wh40k_v0", "is_parallelizable": False}

    def __init__(self, render_mode=None, cfg: GameConfig = None, rng_seed: int = 42):
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = ["player_0", "player_1"]
        self._agent_to_pid = {"player_0": 0, "player_1": 1}

        self.engine = GameEngine(cfg=cfg, rng_seed=rng_seed)
        self._reward_calc = {0: RewardCalculator(), 1: RewardCalculator()}

        self._action_space = build_action_space()
        self._obs_space = build_observation_space()

        self.action_spaces = {a: self._action_space for a in self.possible_agents}
        self.observation_spaces = {a: self._obs_space for a in self.possible_agents}

        self._step_count = 0

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        pid = self._agent_to_pid[agent]
        return encode_observation(self.engine, pid)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.engine.rng = np.random.RandomState(seed)

        pts = self.engine.cfg.points_limit
        p0_units, p0_squads, p1_units, p1_squads, terrain = _default_armies(
            pts, rng=self.engine.rng)
        self.engine.reset(p0_units, p1_units, p0_squads, p1_squads,
                          terrain=terrain)

        self.agents = list(self.possible_agents)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {"action_mask": self._get_mask_for(a)} for a in self.agents}

        for pid in [0, 1]:
            self._reward_calc[pid].reset(self.engine)

        self._step_count = 0

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        pid = self._agent_to_pid[agent]

        if action is not None:
            decoded = decode_action(np.asarray(action))
        else:
            decoded = Action(unit_idx=0, action_type=Action.PASS, target_idx=0)

        result = self.engine.execute_action(decoded)

        for a in self.agents:
            p = self._agent_to_pid[a]
            if a == agent:
                r = self._reward_calc[p].compute(
                    self.engine, p,
                    damage_dealt=result.damage_dealt,
                    damage_taken=result.damage_taken,
                )
            else:
                r = 0.0
            self.rewards[a] = r

        if self.engine.all_units_acted() or not result.valid:
            self.engine.advance_phase()

        self._step_count += 1

        game_done = self.engine.game_over
        truncated = self._step_count >= MAX_STEPS_PER_GAME

        if game_done or truncated:
            for a in self.agents:
                self.terminations[a] = game_done
                self.truncations[a] = truncated and not game_done

        current_pid = self.engine.current_player
        self.agent_selection = self.possible_agents[current_pid]

        for a in self.agents:
            self.infos[a] = {"action_mask": self._get_mask_for(a)}

        self._accumulate_rewards()

    def _get_mask_for(self, agent: str) -> np.ndarray:
        pid = self._agent_to_pid[agent]
        saved_player = self.engine.current_player
        self.engine.current_player = pid
        mask = build_action_mask(self.engine)
        self.engine.current_player = saved_player
        return mask


# ---------------------------------------------------------------------------
#  Gymnasium single-agent wrapper (for SB3 MaskablePPO self-play)
# ---------------------------------------------------------------------------

class Warhammer40kEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper.
    The learning agent always plays as player 0.
    The opponent is controlled by `opponent_fn(obs, mask) -> action`.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent_fn: Callable = None,
                 cfg: GameConfig = None, rng_seed: int = 42,
                 render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.engine = GameEngine(cfg=cfg, rng_seed=rng_seed)
        self._reward_calc = RewardCalculator()
        self._opponent_fn = opponent_fn or self._random_opponent

        self.action_space = build_action_space()
        self.observation_space = build_observation_space()

        self._step_count = 0
        self._opponent_pool: List[str] = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.engine.rng = np.random.RandomState(seed)

        pts = self.engine.cfg.points_limit
        p0_units, p0_squads, p1_units, p1_squads, terrain = _default_armies(
            pts, rng=self.engine.rng)
        self.engine.reset(p0_units, p1_units, p0_squads, p1_squads,
                          terrain=terrain)
        self._reward_calc.reset(self.engine)
        self._step_count = 0

        self._play_opponent_turns()

        obs = encode_observation(self.engine, player_id=0)
        info = {"action_mask": build_action_mask(self.engine)}
        return obs, info

    def step(self, action):
        decoded = decode_action(np.asarray(action))
        result = self.engine.execute_action(decoded)

        reward = self._reward_calc.compute(
            self.engine, player_id=0,
            damage_dealt=result.damage_dealt,
            damage_taken=result.damage_taken,
        )

        if self.engine.all_units_acted() or not result.valid:
            self.engine.advance_phase()

        self._play_opponent_turns()

        self._step_count += 1
        terminated = self.engine.game_over
        truncated = self._step_count >= MAX_STEPS_PER_GAME

        obs = encode_observation(self.engine, player_id=0)
        info = {"action_mask": build_action_mask(self.engine)}

        return obs, reward, terminated, truncated, info

    def _play_opponent_turns(self):
        safety = 0
        while self.engine.current_player == 1 and not self.engine.game_over:
            safety += 1
            if safety > 500:
                break

            obs = encode_observation(self.engine, player_id=1)
            mask = build_action_mask(self.engine)
            opp_action = self._opponent_fn(obs, mask)

            decoded = decode_action(np.asarray(opp_action))
            self.engine.execute_action(decoded)

            if self.engine.all_units_acted():
                self.engine.advance_phase()

    def _random_opponent(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        nvec = self.action_space.nvec
        seg1 = nvec[0]
        seg2 = nvec[1]

        unit_mask = mask[:seg1]
        type_mask = mask[seg1:seg1 + seg2]
        target_mask = mask[seg1 + seg2:]

        def _sample(m):
            valid = np.where(m > 0)[0]
            if len(valid) == 0:
                return 0
            return int(np.random.choice(valid))

        return np.array([_sample(unit_mask), _sample(type_mask), _sample(target_mask)])

    def action_masks(self) -> np.ndarray:
        return build_action_mask(self.engine)

    def set_opponent_fn(self, fn: Callable):
        self._opponent_fn = fn

    def set_opponent_pool(self, model_paths: List[str]):
        self._opponent_pool = model_paths
