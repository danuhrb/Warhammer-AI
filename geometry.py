from __future__ import annotations
import numpy as np
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from .types import Unit, TerrainPiece, GameState

def circle_geom(center: np.ndarray, r: float, resolution: int = 16) -> BaseGeometry:
    """Approximate a round base as a buffered point polygon."""
    return Point(float(center[0]), float(center[1])).buffer(r, resolution=resolution)

def edge_distance(a_pos: np.ndarray, a_r: float, b_pos: np.ndarray, b_r: float) -> float:
    """Distance between the circumferences of two circular bases (>= 0)."""
    d = np.linalg.norm(a_pos - b_pos) - a_r - b_r
    return float(max(0.0, d))

def within_move(unit: Unit, dest: np.ndarray) -> bool:
    """Check if the center-to-center move length is <= unit.M."""
    return float(np.linalg.norm(dest - unit.pos)) <= unit.M + 1e-9

def legal_destination(state: GameState, unit: Unit, dest: np.ndarray) -> bool:
    """Destination must be within the table, not intersect blocking terrain, and not overlap other units."""
    u_geom = circle_geom(dest, unit.base_r + state.cfg.move_margin)
    # Table bounds
    if not state.table_poly.buffer(-state.cfg.move_margin).contains(u_geom):
        return False
    # Terrain collisions
    for t in state.terrain:
        if t.blocks_movement and u_geom.intersects(t.polygon):
            return False
    # Other units
    for uid, other in state.units.items():
        if uid == unit.id or not other.alive:
            continue
        if u_geom.intersects(circle_geom(other.pos, other.base_r + state.cfg.move_margin)):
            return False
    return True
