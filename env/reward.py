from __future__ import annotations
import numpy as np
from game_engine import GameEngine


class RewardCalculator:
    """
    Layered reward shaping for WH40K RL training.

    Components:
      - Terminal: win/loss at game end
      - VP delta: change in victory point differential per step
      - Damage: damage dealt minus damage taken
      - Objective control: bonus for controlling objectives
      - Step penalty: small negative to encourage decisive play
    """

    WIN_REWARD = 10.0
    LOSS_REWARD = -10.0
    DRAW_REWARD = 0.0
    VP_WEIGHT = 1.0
    DAMAGE_WEIGHT = 0.05
    OBJ_CONTROL_WEIGHT = 0.1
    STEP_PENALTY = -0.001

    def __init__(self):
        self._prev_vp = {0: 0, 1: 0}
        self._prev_wounds = {0: 0, 1: 0}

    def reset(self, engine: GameEngine) -> None:
        self._prev_vp = dict(engine.state.vp)
        self._prev_wounds = dict(engine.get_total_wounds())

    def compute(self, engine: GameEngine, player_id: int,
                damage_dealt: int = 0, damage_taken: int = 0) -> float:
        reward = 0.0

        if engine.game_over:
            if engine.winner == player_id:
                reward += self.WIN_REWARD
            elif engine.winner is None:
                reward += self.DRAW_REWARD
            else:
                reward += self.LOSS_REWARD
            return reward

        vp_delta_mine = engine.state.vp[player_id] - self._prev_vp[player_id]
        vp_delta_opp = engine.state.vp[1 - player_id] - self._prev_vp[1 - player_id]
        reward += (vp_delta_mine - vp_delta_opp) * self.VP_WEIGHT

        reward += (damage_dealt - damage_taken) * self.DAMAGE_WEIGHT

        for obj in engine.state.objectives:
            friendly_oc = 0
            enemy_oc = 0
            for uid, u in engine.state.units.items():
                if not u.alive:
                    continue
                dist = np.linalg.norm(u.pos - obj.pos)
                if dist <= obj.r + u.base_r:
                    if u.owner == player_id:
                        friendly_oc += u.OC
                    else:
                        enemy_oc += u.OC
            if friendly_oc > enemy_oc:
                reward += self.OBJ_CONTROL_WEIGHT

        reward += self.STEP_PENALTY

        self._prev_vp = dict(engine.state.vp)
        return reward
