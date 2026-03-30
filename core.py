from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from wh_types import Unit, Weapon, GameState
from movement import is_in_engagement


def resolve_melee(state: GameState, attacker: Unit, target: Unit,
                  weapon: Weapon, rng: np.random.RandomState) -> Dict[str, Any]:
    """
    Resolve a melee attack. Wounds allocated across the target's squad
    using the same bodyguard/allocation rules as shooting.
    """
    from combat import (
        roll_d6, wound_threshold, save_needed,
        _get_allocation_order, _allocate_damage, _update_squad_alive,
    )

    result = {
        "attacks": weapon.attacks, "hits": 0, "wounds": 0,
        "failed_saves": 0, "damage": 0, "models_killed": 0, "note": "",
    }

    if not attacker.alive or not target.alive:
        result["note"] = "one model dead"
        return result

    if not is_in_engagement(attacker, target):
        result["note"] = "not in engagement range"
        return result

    hit_rolls = roll_d6(rng, weapon.attacks)
    hits = int((hit_rolls >= weapon.to_hit).sum())

    need_w = wound_threshold(weapon.strength, target.T)
    wound_rolls = roll_d6(rng, hits)
    wounds = int((wound_rolls >= need_w).sum())

    need_sv = save_needed(target.Sv, weapon.ap)
    if need_sv <= 6:
        save_rolls = roll_d6(rng, wounds)
        failed = int((save_rolls < need_sv).sum())
    else:
        failed = wounds

    damage_pool = failed * weapon.damage

    squad = state.squads.get(target.squad_id) if target.squad_id >= 0 else None

    if squad is not None and squad.alive:
        alloc_order = _get_allocation_order(squad, state, attacker)
        alloc_result = _allocate_damage(damage_pool, alloc_order, state,
                                        weapon.damage)
        _update_squad_alive(squad, state)
        result["damage"] = alloc_result["damage_applied"]
        result["models_killed"] = alloc_result["models_killed"]
    else:
        target.W = max(0, target.W - damage_pool)
        if target.W == 0:
            target.alive = False
            result["models_killed"] = 1
        result["damage"] = damage_pool

    result.update({"hits": hits, "wounds": wounds, "failed_saves": failed})
    return result


def resolve_melee_manual(state: GameState, attacker: Unit, target: Unit,
                         weapon: Weapon,
                         hits: int, wounds: int, failed_saves: int) -> Dict[str, Any]:
    """Resolve melee with player-provided dice results."""
    from combat import (
        _get_allocation_order, _allocate_damage, _update_squad_alive,
    )

    result = {
        "attacks": weapon.attacks, "hits": hits, "wounds": wounds,
        "failed_saves": failed_saves, "damage": 0, "models_killed": 0, "note": "",
    }

    if not attacker.alive or not target.alive:
        result["note"] = "one model dead"
        return result
    if not is_in_engagement(attacker, target):
        result["note"] = "not in engagement range"
        return result

    damage_pool = failed_saves * weapon.damage
    squad = state.squads.get(target.squad_id) if target.squad_id >= 0 else None

    if squad is not None and squad.alive:
        alloc_order = _get_allocation_order(squad, state, attacker)
        alloc_result = _allocate_damage(damage_pool, alloc_order, state,
                                        weapon.damage)
        _update_squad_alive(squad, state)
        result["damage"] = alloc_result["damage_applied"]
        result["models_killed"] = alloc_result["models_killed"]
    else:
        target.W = max(0, target.W - damage_pool)
        if target.W == 0:
            target.alive = False
            result["models_killed"] = 1
        result["damage"] = damage_pool

    return result


def get_targets_in_range(unit: Unit, all_units: Dict[int, Unit],
                         weapon: Weapon) -> List[int]:
    """Return IDs of enemy units within weapon range."""
    targets = []
    for uid, other in all_units.items():
        if other.owner == unit.owner or not other.alive:
            continue
        dist = float(np.linalg.norm(unit.pos - other.pos))
        if dist - unit.base_r <= weapon.range + 1e-9:
            targets.append(uid)
    return targets


def get_melee_targets(unit: Unit, all_units: Dict[int, Unit]) -> List[int]:
    """Return IDs of enemy units in engagement range."""
    targets = []
    for uid, other in all_units.items():
        if other.owner == unit.owner or not other.alive:
            continue
        if is_in_engagement(unit, other):
            targets.append(uid)
    return targets
