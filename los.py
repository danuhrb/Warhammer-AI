from __future__ import annotations
import numpy as np
from shapely.geometry import LineString
from .types import Unit, TerrainPiece, GameState
from .geometry import circle_geom

def has_los(state: GameState, shooter: Unit, target: Unit) -> bool:
    """
    Height-aware LOS: returns True if there is a clear line segment between shooter and target
    that does NOT intersect any blocking terrain whose height >= min(shooter.height, target.height).
    We model shooter/target as discs; LOS is checked between disc centers, but we ignore intersections
    with those discs by subtracting a tiny epsilon if needed.
    """
    if not shooter.alive or not target.alive:
        return False
    seg = LineString([(float(shooter.pos[0]), float(shooter.pos[1])),
                      (float(target.pos[0]), float(target.pos[1]))])
    block_level = min(shooter.height, target.height)
    for t in state.terrain:
        if not t.blocks_los or t.height < block_level:
            continue
        if seg.crosses(t.polygon) or seg.within(t.polygon) or seg.touches(t.polygon) and seg.length > 0:
            # touches() can be finicky; treat any boundary touch as blocked for tabletop intent
            return False
    return True

def cover_score(state: GameState, pos: np.ndarray, enemies: list[Unit]) -> int:
    """How many enemies would lose LOS to this position (coarse heuristic)."""
    dummy = Unit(id=-1, name="dummy", owner=0, pos=pos, theta=0.0, base_r=0.001, height=1, M=0, W=1, OC=0, weapons=[], alive=True)
    count = 0
    for e in enemies:
        if not has_los(state, e, dummy):
            count += 1
    return count
