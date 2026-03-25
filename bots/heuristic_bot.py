"""
Heuristic bot -- rule-based opponent that:
  Movement:  advance toward nearest objective or nearest enemy
  Shooting:  focus fire on the lowest-HP enemy in range
  Charge:    charge weakest nearby enemy
  Fight:     attack the weakest adjacent enemy
"""
from __future__ import annotations
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game_engine import GameEngine, Action, Phase, MAX_UNITS_PER_PLAYER
from env.action_codec import (
    build_action_mask, MAX_TARGETS, NUM_ACTION_TYPES,
)


def heuristic_policy(obs: np.ndarray, mask: np.ndarray,
                     engine: GameEngine = None) -> np.ndarray:
    """
    If engine is provided, use game state for smarter decisions.
    Otherwise, fall back to mask-only heuristic (pick first legal action).
    """
    if engine is None:
        return _mask_only_policy(mask)
    return _smart_policy(engine, mask)


def _mask_only_policy(mask: np.ndarray) -> np.ndarray:
    """When we don't have engine access, pick first legal non-PASS action."""
    seg1 = MAX_UNITS_PER_PLAYER
    seg2 = NUM_ACTION_TYPES

    unit_mask = mask[:seg1]
    type_mask = mask[seg1:seg1 + seg2]
    target_mask = mask[seg1 + seg2:]

    def _first_valid(m, prefer_nonzero_idx=None):
        valid = np.where(m > 0)[0]
        if len(valid) == 0:
            return 0
        if prefer_nonzero_idx is not None:
            for idx in prefer_nonzero_idx:
                if idx in valid:
                    return int(idx)
        return int(valid[0])

    unit = _first_valid(unit_mask)
    action_type = _first_valid(type_mask, prefer_nonzero_idx=[
        Action.SHOOT, Action.FIGHT, Action.CHARGE, Action.MOVE, Action.PASS
    ])
    target = _first_valid(target_mask)

    return np.array([unit, action_type, target])


def _smart_policy(engine: GameEngine, mask: np.ndarray) -> np.ndarray:
    pid = engine.current_player
    my_ids = engine.get_player_unit_ids(pid)
    enemy_ids = engine.get_player_unit_ids(1 - pid)
    state = engine.state

    best_unit_idx = 0
    best_action_type = Action.PASS
    best_target = 0
    best_priority = -1

    for ui in range(min(len(my_ids), MAX_UNITS_PER_PLAYER)):
        uid = my_ids[ui]
        unit = state.units[uid]
        if not unit.alive:
            continue

        legal = engine.get_legal_unit_actions(ui)

        if Action.FIGHT in legal and legal[Action.FIGHT]:
            target_ei = _pick_weakest_target(legal[Action.FIGHT], enemy_ids, state)
            priority = 100
            if priority > best_priority:
                best_unit_idx, best_action_type, best_target = ui, Action.FIGHT, target_ei
                best_priority = priority

        elif Action.SHOOT in legal and legal[Action.SHOOT]:
            target_ei = _pick_weakest_target(legal[Action.SHOOT], enemy_ids, state)
            priority = 80
            if priority > best_priority:
                best_unit_idx, best_action_type, best_target = ui, Action.SHOOT, target_ei
                best_priority = priority

        elif Action.CHARGE in legal and legal[Action.CHARGE]:
            target_ei = _pick_weakest_target(legal[Action.CHARGE], enemy_ids, state)
            priority = 60
            if priority > best_priority:
                best_unit_idx, best_action_type, best_target = ui, Action.CHARGE, target_ei
                best_priority = priority

        elif Action.MOVE in legal and legal[Action.MOVE]:
            target_ci = _pick_best_move(unit, legal[Action.MOVE], engine)
            priority = 40
            if priority > best_priority:
                best_unit_idx, best_action_type, best_target = ui, Action.MOVE, target_ci
                best_priority = priority

    return np.array([best_unit_idx, best_action_type, best_target])


def _pick_weakest_target(target_indices, enemy_ids, state) -> int:
    """Among legal target indices, pick the one with lowest remaining wounds."""
    best_ei = target_indices[0]
    best_w = float("inf")
    for ei in target_indices:
        if ei < len(enemy_ids):
            eu = state.units[enemy_ids[ei]]
            if eu.alive and eu.W < best_w:
                best_w = eu.W
                best_ei = ei
    return best_ei


def _pick_best_move(unit, move_indices, engine: GameEngine) -> int:
    """Pick the move candidate that gets closest to the nearest objective or enemy."""
    state = engine.state
    candidates = engine._move_cache.get(unit.id, [])

    best_idx = move_indices[0] if move_indices else 0
    best_dist = float("inf")

    targets = []
    for obj in state.objectives:
        targets.append(obj.pos)
    enemy_ids = engine.get_player_unit_ids(1 - unit.owner)
    for eid in enemy_ids:
        eu = state.units[eid]
        if eu.alive:
            targets.append(eu.pos)

    if not targets:
        return best_idx

    for ci in move_indices:
        if ci >= len(candidates):
            continue
        pos = candidates[ci]
        min_d = min(np.linalg.norm(pos - t) for t in targets)
        if min_d < best_dist:
            best_dist = min_d
            best_idx = ci

    return best_idx
