from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from shapely.geometry import Point
from wh_types import Unit, GameState, TerrainPiece, Squad, COHERENCY_RANGE

NUM_DIRECTIONS = 16
NUM_DISTANCES = 4
NUM_MOVE_SLOTS = NUM_DIRECTIONS * NUM_DISTANCES
MOVE_ANGLES = [i * (2 * np.pi / NUM_DIRECTIONS) for i in range(NUM_DIRECTIONS)]


def _collides_terrain(pos: np.ndarray, base_r: float,
                      terrain: List[TerrainPiece]) -> bool:
    circle = Point(pos[0], pos[1]).buffer(base_r)
    for tp in terrain:
        if tp.blocks_movement and circle.intersects(tp.polygon):
            return True
    return False


def _collides_units(pos: np.ndarray, base_r: float, unit_id: int,
                    all_units: dict, margin: float = 0.1) -> bool:
    for uid, other in all_units.items():
        if uid == unit_id or not other.alive:
            continue
        dist = np.linalg.norm(pos - other.pos)
        if dist < base_r + other.base_r + margin:
            return True
    return False


def _in_bounds(pos: np.ndarray, base_r: float,
               table_w: float, table_h: float) -> bool:
    return (base_r <= pos[0] <= table_w - base_r and
            base_r <= pos[1] <= table_h - base_r)


# ------------------------------------------------------------------
# Coherency
# ------------------------------------------------------------------

def check_coherency(unit: Unit, state: GameState,
                    proposed_pos: Optional[np.ndarray] = None) -> bool:
    """
    Check if a model satisfies squad coherency.
    A model is coherent if it is within COHERENCY_RANGE of at least one
    other alive model in its squad. Single-model squads always pass.
    If proposed_pos is given, check coherency as if the unit were at that
    position instead of its current one.
    """
    if unit.squad_id < 0:
        return True

    squad = state.squads.get(unit.squad_id)
    if squad is None:
        return True

    squad_members = squad.all_member_ids()
    alive_others = [
        uid for uid in squad_members
        if uid != unit.id and state.units[uid].alive
    ]
    if len(alive_others) == 0:
        return True

    pos = proposed_pos if proposed_pos is not None else unit.pos
    coh = state.cfg.coherency_range

    for other_id in alive_others:
        other = state.units[other_id]
        dist = np.linalg.norm(pos - other.pos)
        if dist <= coh + unit.base_r + other.base_r:
            return True

    return False


def squad_is_coherent(squad: Squad, state: GameState) -> bool:
    """Check if every alive model in the squad is coherent."""
    alive_ids = [uid for uid in squad.all_member_ids()
                 if state.units[uid].alive]
    if len(alive_ids) <= 1:
        return True
    for uid in alive_ids:
        if not check_coherency(state.units[uid], state):
            return False
    return True


# ------------------------------------------------------------------
# Movement
# ------------------------------------------------------------------

def get_move_candidates(unit: Unit, state: GameState) -> List[np.ndarray]:
    """
    Generate candidate move destinations sampled from the full movement
    circle (radius = unit.M). Returns up to NUM_DIRECTIONS * NUM_DISTANCES
    valid positions. Positions are ordered [angle_0_dist_0, angle_0_dist_1,
    ..., angle_N_dist_M] so the RL agent has a stable index mapping.

    Coherency is NOT checked here -- in 40K you move the whole unit at once,
    so per-model coherency checks during movement are too restrictive.
    Coherency is checked at the end of the movement phase instead.
    """
    candidates: List[np.ndarray] = []
    tw, th = state.cfg.table_size
    steps = np.linspace(unit.M / NUM_DISTANCES, unit.M, NUM_DISTANCES)

    for angle in MOVE_ANGLES:
        direction = np.array([np.cos(angle), np.sin(angle)])
        for dist in steps:
            new_pos = unit.pos + direction * dist
            if (not _in_bounds(new_pos, unit.base_r, tw, th)
                    or _collides_terrain(new_pos, unit.base_r, state.terrain)
                    or _collides_units(new_pos, unit.base_r, unit.id,
                                       state.units, state.cfg.move_margin)):
                candidates.append(None)
            else:
                candidates.append(new_pos)

    return candidates


def move_unit(unit: Unit, new_pos: np.ndarray, state: GameState) -> bool:
    """
    Attempt to move a unit to new_pos. Validates distance, bounds,
    and collisions. Coherency is not enforced per-move (checked at
    end of movement phase instead, matching 40K rules for unit-level movement).
    """
    dist = np.linalg.norm(new_pos - unit.pos)
    if dist > unit.M + 1e-9:
        return False

    tw, th = state.cfg.table_size
    if not _in_bounds(new_pos, unit.base_r, tw, th):
        return False
    if _collides_terrain(new_pos, unit.base_r, state.terrain):
        return False
    if _collides_units(new_pos, unit.base_r, unit.id,
                       state.units, state.cfg.move_margin):
        return False

    unit.pos = new_pos.copy()
    return True


def get_charge_targets(unit: Unit, state: GameState,
                       charge_range: float = 12.0) -> List[int]:
    """Return IDs of enemy units within charge range."""
    targets = []
    for uid, other in state.units.items():
        if other.owner == unit.owner or not other.alive:
            continue
        dist = np.linalg.norm(unit.pos - other.pos)
        if dist <= charge_range:
            targets.append(uid)
    return targets


def execute_charge(unit: Unit, target: Unit, state: GameState,
                   rng: np.random.RandomState) -> Tuple[bool, int, float]:
    """
    Roll 2d6 for charge distance. If enough to reach engagement range (1"),
    move into base contact. Coherency is relaxed during charges per 40K rules
    (you must attempt to maintain it but the charge itself isn't blocked by it).

    Returns (succeeded, charge_roll, distance_needed).
    """
    charge_roll = int(rng.randint(1, 7)) + int(rng.randint(1, 7))
    dist_to_target = np.linalg.norm(unit.pos - target.pos)
    engagement_range = unit.base_r + target.base_r + 1.0
    needed = dist_to_target - engagement_range

    if charge_roll >= needed:
        direction = target.pos - unit.pos
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        contact_pos = target.pos - direction * (unit.base_r + target.base_r + 0.1)
        unit.pos = contact_pos.copy()
        return True, charge_roll, needed
    return False, charge_roll, needed


def execute_charge_manual(unit: Unit, target: Unit, state: GameState,
                          charge_roll: int) -> Tuple[bool, float]:
    """Execute a charge with a player-provided 2d6 roll. Returns (succeeded, needed)."""
    dist_to_target = np.linalg.norm(unit.pos - target.pos)
    engagement_range = unit.base_r + target.base_r + 1.0
    needed = dist_to_target - engagement_range

    if charge_roll >= needed:
        direction = target.pos - unit.pos
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        contact_pos = target.pos - direction * (unit.base_r + target.base_r + 0.1)
        unit.pos = contact_pos.copy()
        return True, needed
    return False, needed


def is_in_engagement(unit: Unit, target: Unit) -> bool:
    """True if the two units are within 1\" engagement range."""
    dist = np.linalg.norm(unit.pos - target.pos)
    return dist <= unit.base_r + target.base_r + 1.0
