from __future__ import annotations
import numpy as np
from math import atan2, cos, sin, pi
from typing import List, Tuple
from shapely.geometry import Point
from .types import GameState, Unit, Objective, TerrainPiece
from .geometry import legal_destination, within_move

def _unit_enemy_list(state: GameState, unit: Unit) -> list[Unit]:
    return [u for u in state.units.values() if u.alive and u.owner != unit.owner]

def _objective_ring_points(obj: Objective, unit: Unit, samples: int = 6) -> np.ndarray:
    """Points on the ring where the unit's base would touch the objective radius."""
    angles = np.linspace(0, 2*pi, samples, endpoint=False)
    ring_r = obj.r + unit.base_r
    pts = np.stack([obj.pos[0] + ring_r*np.cos(angles), obj.pos[1] + ring_r*np.sin(angles)], axis=1)
    return pts

def _radial_spokes(unit: Unit, num_dirs: int = 8, dist: float = None) -> np.ndarray:
    """Spoke endpoints around the unit at a given distance (default=unit.M)."""
    d = unit.M if dist is None else float(dist)
    angles = np.linspace(0, 2*pi, num_dirs, endpoint=False)
    pts = np.stack([unit.pos[0] + d*np.cos(angles), unit.pos[1] + d*np.sin(angles)], axis=1)
    return pts

def _behind_cover_points(state: GameState, unit: Unit, enemies: list[Unit], per_terrain: int = 4, offset: float = 0.75) -> np.ndarray:
    """
    Sample points just behind cover relative to enemy centroid.
    We pick several points along each blocking polygon's exterior and nudge away from enemies.
    """
    if not enemies:
        return np.empty((0,2))
    enemy_centroid = np.mean(np.stack([e.pos for e in enemies], axis=0), axis=0)
    pts = []
    for t in state.terrain:
        if not t.blocks_los:
            continue
        coords = np.array(t.polygon.exterior.coords)
        # pick evenly spaced vertices
        idxs = np.round(np.linspace(0, len(coords)-1, per_terrain, endpoint=False)).astype(int)
        for i in idxs:
            p = coords[i]
            v = np.array(p) - enemy_centroid
            n = v / (np.linalg.norm(v) + 1e-9)
            candidate = np.array(p) + n * (unit.base_r + offset)
            pts.append(candidate)
    if not pts:
        return np.empty((0,2))
    return np.stack(pts, axis=0)

def movement_candidates(state: GameState, unit: Unit, k: int | None = None) -> np.ndarray:
    """
    Generate destination candidates within unit.M, filtered for collisions.
    Mixes: objective ring entries, behind-cover samples, and radial spokes.
    Returns up to k candidates (default state.cfg.max_candidates), sorted by a simple heuristic:
      (more cover vs enemies) -> (greater min distance to enemies) -> (closer to nearest objective)
    """
    max_k = state.cfg.max_candidates if k is None else int(k)
    enemies = _unit_enemy_list(state, unit)

    # raw candidates
    pts = []
    # objective ring entries
    for obj in state.objectives:
        pts.append(_objective_ring_points(obj, unit, samples=6))
    # behind cover relative to enemies
    pts.append(_behind_cover_points(state, unit, enemies, per_terrain=4))
    # radial spokes at full and 60% move
    pts.append(_radial_spokes(unit, num_dirs=8, dist=unit.M))
    pts.append(_radial_spokes(unit, num_dirs=8, dist=0.6*unit.M))

    if not pts:
        return np.empty((0,2))
    cand = np.concatenate([p for p in pts if p.size > 0], axis=0)

    # keep only within move and legal destination
    legal = []
    for p in cand:
        if within_move(unit, p) and legal_destination(state, unit, p):
            legal.append(p)
    if not legal:
        return np.empty((0,2))
    legal = np.unique(np.round(np.stack(legal, axis=0), 3), axis=0)  # dedupe to ~1mm

    # scoring heuristic
    # 1) cover: how many enemies lose LOS to this position
    from .los import cover_score
    cover = np.array([cover_score(state, p, enemies) for p in legal], dtype=float)
    # 2) safety: min edge distance to any enemy
    if enemies:
        enemy_xy = np.stack([e.pos for e in enemies], axis=0)
        dists = np.linalg.norm(legal[:,None,:] - enemy_xy[None,:,:], axis=2)  # [P,E]
        min_center = dists.min(axis=1)
    else:
        min_center = np.full((len(legal),), 1e6)
    # 3) objective closeness (smaller is better)
    if state.objectives:
        obj_xy = np.stack([o.pos for o in state.objectives], axis=0)
        d_obj = np.linalg.norm(legal[:,None,:] - obj_xy[None,:,:], axis=2).min(axis=1)
    else:
        d_obj = np.zeros((len(legal),))

    # combine: sort by (-cover, -min_center, +d_obj)
    order = np.lexsort((d_obj, -min_center, -cover))
    out = legal[order][:max_k]
    return out
