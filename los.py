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
      2. The terrain is tall enough (height >= min(shooter, target))
         and blocks_los is True.
    """
    if not terrain.blocks_los:
        return False
    if not ray.intersects(terrain.polygon):
        return False
    if terrain.height >= min(shooter_h, target_h):
        return True
    return False


def _terrain_gives_cover(ray: LineString, terrain: TerrainPiece,
                         target_h: int) -> bool:
    """
    A terrain piece provides cover if the LOS ray intersects or is
    adjacent to it and the piece is at least as tall as the target.
    Cover doesn't require full blocking -- even partial intersection counts.
    """
    if not terrain.provides_cover:
        return False
    if terrain.height < target_h:
        return False
    near_dist = ray.distance(terrain.polygon)
    return near_dist < 2.0


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


def has_cover(state: GameState, shooter: Unit, target: Unit) -> bool:
    """
    Returns True if the target benefits from cover against the shooter.
    Cover grants +1 to the target's save roll (i.e. reduces effective AP by 1).
    A target gets cover if the LOS ray passes near/through a terrain piece
    that provides cover and is tall enough relative to the target.
    """
    if not shooter.alive or not target.alive:
        return False
    if not state.terrain:
        return False

    ray = _ray_between(shooter, target)

    for tp in state.terrain:
        if _terrain_gives_cover(ray, tp, target.height):
            return True

    return False
