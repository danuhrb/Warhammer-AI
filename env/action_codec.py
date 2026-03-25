from __future__ import annotations
import numpy as np
from gymnasium import spaces
from game_engine import (
    GameEngine, Action, Phase, MAX_UNITS_PER_PLAYER,
    NUM_MOVE_CANDIDATES,
)

NUM_ACTION_TYPES = 5   # PASS=0, MOVE=1, SHOOT=2, CHARGE=3, FIGHT=4
MAX_TARGETS = max(MAX_UNITS_PER_PLAYER, NUM_MOVE_CANDIDATES)


def build_action_space() -> spaces.MultiDiscrete:
    """
    Factored action: [unit_index, action_type, target_or_direction].
    - unit_index:  0 .. MAX_UNITS_PER_PLAYER - 1
    - action_type: 0..4 (PASS, MOVE, SHOOT, CHARGE, FIGHT)
    - target:      0 .. MAX_TARGETS - 1
    """
    return spaces.MultiDiscrete([
        MAX_UNITS_PER_PLAYER,
        NUM_ACTION_TYPES,
        MAX_TARGETS,
    ])


def decode_action(raw: np.ndarray) -> Action:
    return Action(
        unit_idx=int(raw[0]),
        action_type=int(raw[1]),
        target_idx=int(raw[2]),
    )


def build_action_mask(engine: GameEngine) -> np.ndarray:
    """
    Build a flat boolean mask over the factored action space.
    PettingZoo / sb3-contrib MaskablePPO expects a flat mask of length
    sum(nvec) for MultiDiscrete, where each segment corresponds to
    one dimension of the MultiDiscrete space.

    Segment layout:
      [0 .. MAX_UNITS-1]  unit mask
      [MAX_UNITS .. MAX_UNITS+4]  action_type mask
      [MAX_UNITS+5 .. MAX_UNITS+5+MAX_TARGETS-1]  target mask
    """
    unit_mask = np.zeros(MAX_UNITS_PER_PLAYER, dtype=np.int8)
    type_mask = np.zeros(NUM_ACTION_TYPES, dtype=np.int8)
    target_mask = np.zeros(MAX_TARGETS, dtype=np.int8)

    type_mask[Action.PASS] = 1

    pid = engine.current_player
    my_ids = engine.get_player_unit_ids(pid)

    for ui in range(min(len(my_ids), MAX_UNITS_PER_PLAYER)):
        legal = engine.get_legal_unit_actions(ui)
        has_real_action = False
        for atype, targets in legal.items():
            if atype == Action.PASS:
                continue
            if targets:
                has_real_action = True
                type_mask[atype] = 1
                for t in targets:
                    if t < MAX_TARGETS:
                        target_mask[t] = 1

        if has_real_action or engine.state.units[my_ids[ui]].alive:
            unit_mask[ui] = 1

    target_mask[0] = 1

    if unit_mask.sum() == 0:
        unit_mask[0] = 1

    return np.concatenate([unit_mask, type_mask, target_mask])
