from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional
from wh_types import GameState, Unit, Weapon, Squad
from los import has_los


def roll_d6(rng: np.random.RandomState, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    return rng.randint(1, 7, size=int(n))


def wound_threshold(S: int, T: int) -> int:
    if S >= 2 * T: return 2
    if S > T:      return 3
    if S == T:     return 4
    if S * 2 <= T: return 6
    return 5


def save_needed(Sv: int, ap: int) -> int:
    return max(2, Sv - ap)


def center_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def within_range(shooter: Unit, target: Unit, weapon: Weapon) -> bool:
    return (center_distance(shooter.pos, target.pos) - shooter.base_r) <= weapon.range + 1e-9


# ------------------------------------------------------------------
# Wound allocation across a squad
# ------------------------------------------------------------------

def _get_allocation_order(squad: Squad, state: GameState,
                          shooter: Unit) -> List[int]:
    """
    Return unit IDs in wound-allocation order:
      1. Non-leader models already damaged (W < max_W), closest first
      2. Non-leader models at full health, closest first
      3. Leader model last (bodyguard rule)
    """
    leader_id = squad.leader_id
    members = []
    for uid in squad.all_member_ids():
        u = state.units[uid]
        if not u.alive:
            continue
        members.append(u)

    damaged_nonleader = sorted(
        [u for u in members if not u.is_leader and u.W < u.max_W],
        key=lambda u: center_distance(u.pos, shooter.pos),
    )
    healthy_nonleader = sorted(
        [u for u in members if not u.is_leader and u.W >= u.max_W],
        key=lambda u: center_distance(u.pos, shooter.pos),
    )
    leaders = [u for u in members if u.is_leader]

    order = [u.id for u in damaged_nonleader + healthy_nonleader + leaders]
    return order


def _allocate_damage(damage_pool: int, allocation_order: List[int],
                     state: GameState, weapon_damage: int) -> Dict[str, Any]:
    """
    Distribute damage_pool across models in allocation_order.
    Each failed save inflicts weapon_damage wounds on the current model.
    If it kills the model, remaining damage spills to the next.
    Returns summary of total damage, kills, etc.
    """
    total_applied = 0
    kills = 0
    remaining = damage_pool

    for uid in allocation_order:
        if remaining <= 0:
            break
        u = state.units[uid]
        if not u.alive:
            continue

        while remaining > 0 and u.alive:
            dmg = min(weapon_damage, remaining)
            u.W = max(0, u.W - dmg)
            total_applied += dmg
            remaining -= dmg
            if u.W == 0:
                u.alive = False
                kills += 1

    return {"damage_applied": total_applied, "models_killed": kills}


def _update_squad_alive(squad: Squad, state: GameState) -> None:
    """Mark squad as dead if all its members are dead."""
    alive_any = any(state.units[uid].alive for uid in squad.all_member_ids())
    squad.alive = alive_any


# ------------------------------------------------------------------
# Resolve shooting (squad-aware)
# ------------------------------------------------------------------

def resolve_shooting(state: GameState, shooter: Unit, target: Unit,
                     weapon: Weapon,
                     rng: np.random.RandomState) -> Dict[str, Any]:
    """
    Resolve a ranged attack. Wounds are allocated across the target's squad
    using the bodyguard rule (leader last, closest model first).
    """
    result = {
        "attacks": weapon.attacks, "hits": 0, "wounds": 0,
        "failed_saves": 0, "damage": 0,
        "models_killed": 0, "note": "",
    }
    if not shooter.alive or not target.alive:
        result["note"] = "one model dead"
        return result
    if not within_range(shooter, target, weapon):
        result["note"] = "out of range"
        return result
    if not has_los(state, shooter, target):
        result["note"] = "no LOS"
        return result

    # 1) To-Hit
    hit_rolls = roll_d6(rng, weapon.attacks)
    hits = int((hit_rolls >= weapon.to_hit).sum())

    # 2) To-Wound
    need_w = wound_threshold(weapon.strength, target.T)
    wound_rolls = roll_d6(rng, hits)
    wounds = int((wound_rolls >= need_w).sum())

    # 3) Saves (use target's save -- in a squad, all models share the same
    #    save profile for simplicity; real 40K uses the allocated model's save)
    need_sv = save_needed(target.Sv, weapon.ap)
    if need_sv <= 6:
        save_rolls = roll_d6(rng, wounds)
        failed = int((save_rolls < need_sv).sum())
    else:
        failed = wounds

    # 4) Allocate damage across the squad
    damage_pool = failed * weapon.damage

    squad = state.squads.get(target.squad_id) if target.squad_id >= 0 else None

    if squad is not None and squad.alive:
        alloc_order = _get_allocation_order(squad, state, shooter)
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
