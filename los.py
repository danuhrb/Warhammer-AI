from __future__ import annotations
import numpy as np
from shapely.geometry import LineString, Point
from wh_types import GameState, Unit, TerrainPiece


def _ray_between(a: Unit, b: Unit) -> LineString:
    """Center-to-center ray between two units."""
    return LineString([a.pos.tolist(), b.pos.tolist()])


def _terrain_blocks(ray: LineString, terrain: TerrainPiece,
                    shooter_h: int, target_h: int) -> bool:
    """
    A terrain piece blocks LOS if:
      1. The ray intersects its polygon footprint, AND
      2. The terrain is tall enough to block (height >= both shooter and target height),
         OR the terrain blocks_los flag is set and height >= min(shooter, target).
    For simplicity we use the 40k convention: terrain of height >= min(shooter, target)
    blocks if blocks_los is True.
    """
    if not terrain.blocks_los:
        return False
    if not ray.intersects(terrain.polygon):
        return False
    if terrain.height >= min(shooter_h, target_h):
        return True
    return False


def has_los(state: GameState, shooter: Unit, target: Unit) -> bool:
    """
    Returns True if shooter has line of sight to target.
    Checks all terrain pieces for blocking geometry.
    """
    if not shooter.alive or not target.alive:
        return False

    ray = _ray_between(shooter, target)

    for tp in state.terrain:
        if _terrain_blocks(ray, tp, shooter.height, target.height):
            return False

    return True
