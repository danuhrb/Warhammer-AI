from __future__ import annotations
import numpy as np
from gymnasium import spaces
from game_engine import (
    GameEngine, Phase, MAX_UNITS_PER_PLAYER, MAX_WEAPONS, MAX_TURNS,
)
from movement import check_coherency

# pos_x, pos_y, alive, wounds, max_wounds, toughness, save, M, OC,
# base_r, height, weapon_range_max, weapon_damage_max,
# squad_id_norm, is_leader, coherent
UNIT_FEATURES = 16
GLOBAL_FEATURES = 7  # phase(4 one-hot) + turn_norm + vp_0 + vp_1
MAX_OBJECTIVES = 3
OBJ_FEATURES = 3     # x, y, radius

OBS_SIZE = (
    MAX_UNITS_PER_PLAYER * UNIT_FEATURES * 2
    + GLOBAL_FEATURES
    + MAX_OBJECTIVES * OBJ_FEATURES
    + MAX_UNITS_PER_PLAYER  # distances to nearest enemy per own unit
)


def build_observation_space() -> spaces.Box:
    return spaces.Box(
        low=-1.0, high=1.0,
        shape=(OBS_SIZE,),
        dtype=np.float32,
    )


def encode_observation(engine: GameEngine, player_id: int) -> np.ndarray:
    state = engine.state
    tw, th = state.cfg.table_size
    scale_pos = max(tw, th)

    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    own_ids = engine.get_player_unit_ids(player_id)
    enemy_ids = engine.get_player_unit_ids(1 - player_id)

    def _encode_units(unit_ids, offset):
        for i in range(MAX_UNITS_PER_PLAYER):
            base = offset + i * UNIT_FEATURES
            if i < len(unit_ids):
                u = state.units[unit_ids[i]]
                obs[base + 0] = u.pos[0] / scale_pos
                obs[base + 1] = u.pos[1] / scale_pos
                obs[base + 2] = 1.0 if u.alive else 0.0
                obs[base + 3] = u.W / 10.0
                obs[base + 4] = u.max_W / 10.0
                obs[base + 5] = u.T / 10.0
                obs[base + 6] = u.Sv / 6.0
                obs[base + 7] = u.M / 12.0
                obs[base + 8] = u.OC / 4.0
                obs[base + 9] = u.base_r / 2.0
                obs[base + 10] = u.height / 3.0
                max_range = max((w.range for w in u.weapons), default=0)
                max_dmg = max((w.damage for w in u.weapons), default=0)
                obs[base + 11] = max_range / 36.0
                obs[base + 12] = max_dmg / 6.0
                obs[base + 13] = (u.squad_id % 10) / 10.0 if u.squad_id >= 0 else 0.0
                obs[base + 14] = 1.0 if u.is_leader else 0.0
                obs[base + 15] = 1.0 if (u.alive and check_coherency(u, state)) else 0.0

    _encode_units(own_ids, 0)
    _encode_units(enemy_ids, MAX_UNITS_PER_PLAYER * UNIT_FEATURES)
    idx = MAX_UNITS_PER_PLAYER * UNIT_FEATURES * 2

    phase_vec = np.zeros(4, dtype=np.float32)
    phase_vec[int(engine.current_phase)] = 1.0
    obs[idx:idx + 4] = phase_vec
    idx += 4

    obs[idx] = engine.turn / MAX_TURNS
    idx += 1
    obs[idx] = state.vp[player_id] / 20.0
    idx += 1
    obs[idx] = state.vp[1 - player_id] / 20.0
    idx += 1

    for oi in range(MAX_OBJECTIVES):
        base = idx + oi * OBJ_FEATURES
        if oi < len(state.objectives):
            obj = state.objectives[oi]
            obs[base + 0] = obj.pos[0] / scale_pos
            obs[base + 1] = obj.pos[1] / scale_pos
            obs[base + 2] = obj.r / 5.0
    idx += MAX_OBJECTIVES * OBJ_FEATURES

    for i in range(MAX_UNITS_PER_PLAYER):
        if i < len(own_ids):
            u = state.units[own_ids[i]]
            if u.alive:
                min_dist = float("inf")
                for eid in enemy_ids:
                    eu = state.units[eid]
                    if eu.alive:
                        d = np.linalg.norm(u.pos - eu.pos)
                        min_dist = min(min_dist, d)
                obs[idx + i] = min(min_dist / scale_pos, 1.0)
            else:
                obs[idx + i] = 1.0
        else:
            obs[idx + i] = 1.0

    return np.clip(obs, -1.0, 1.0)
