"""
Random terrain generation for Warhammer 40K.

Generates L-shapes, squares, and rectangles representing floor-to-sky
buildings that block LOS, block movement, and provide cover.
Terrain is placed only in the mid-field (outside both deployment zones)
and pieces cannot overlap each other.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import rotate, translate
from wh_types import TerrainPiece
from unit_data import DEPLOYMENT_DEPTH

MIN_PIECE_GAP = 1.0  # minimum gap between terrain pieces (inches)
MAX_PLACEMENT_ATTEMPTS = 80  # retries per piece before giving up


def _make_L_polygon(arm_len: float, arm_width: float) -> Polygon:
    """Create an L-shaped polygon at the origin."""
    return Polygon([
        (0, 0),
        (arm_len, 0),
        (arm_len, arm_width),
        (arm_width, arm_width),
        (arm_width, arm_len),
        (0, arm_len),
    ])


def _make_rect_polygon(w: float, h: float) -> Polygon:
    return box(0, 0, w, h)


def _make_square_polygon(size: float) -> Polygon:
    return box(0, 0, size, size)


def _random_piece(rng: np.random.RandomState) -> Tuple[Polygon, str]:
    """Return a random terrain polygon at the origin and its type name."""
    kind = rng.choice(["L", "square", "rect"])

    if kind == "L":
        arm_len = rng.uniform(4.0, 8.0)
        arm_width = rng.uniform(2.0, min(4.0, arm_len - 1.0))
        poly = _make_L_polygon(arm_len, arm_width)
    elif kind == "square":
        size = rng.uniform(3.0, 6.0)
        poly = _make_square_polygon(size)
    else:
        w = rng.uniform(3.0, 8.0)
        h = rng.uniform(2.0, 5.0)
        poly = _make_rect_polygon(w, h)

    angle = rng.choice([0, 90, 180, 270])
    if angle:
        poly = rotate(poly, angle, origin=(0, 0), use_radians=False)

    return poly, kind


def generate_terrain(
    table_size: Tuple[float, float] = (44.0, 60.0),
    num_pieces: int = 6,
    rng: Optional[np.random.RandomState] = None,
) -> List[TerrainPiece]:
    """
    Generate random terrain pieces for the table.

    Constraints:
      - Pieces stay within the mid-field band (DEPLOYMENT_DEPTH to table_h - DEPLOYMENT_DEPTH)
      - Pieces don't overlap each other (buffered by MIN_PIECE_GAP)
      - All pieces are floor-to-sky buildings (height=3)
    """
    if rng is None:
        rng = np.random.RandomState()

    tw, th = table_size
    deploy = DEPLOYMENT_DEPTH
    mid_y_min = deploy + 1.0
    mid_y_max = th - deploy - 1.0
    margin = 1.0

    placed: List[TerrainPiece] = []
    placed_polys: List[Polygon] = []

    for _ in range(num_pieces):
        for _attempt in range(MAX_PLACEMENT_ATTEMPTS):
            poly, kind = _random_piece(rng)

            bounds = poly.bounds
            poly_w = bounds[2] - bounds[0]
            poly_h = bounds[3] - bounds[1]

            x = rng.uniform(margin, tw - poly_w - margin)
            y = rng.uniform(mid_y_min, mid_y_max - poly_h)

            shifted = translate(poly, xoff=x - bounds[0], yoff=y - bounds[1])

            sb = shifted.bounds
            if sb[0] < margin or sb[2] > tw - margin:
                continue
            if sb[1] < mid_y_min or sb[3] > mid_y_max:
                continue

            buffered = shifted.buffer(MIN_PIECE_GAP)
            overlap = False
            for existing in placed_polys:
                if buffered.intersects(existing):
                    overlap = True
                    break
            if overlap:
                continue

            tp = TerrainPiece(
                polygon=shifted,
                height=3,
                blocks_los=True,
                blocks_movement=True,
                provides_cover=True,
                terrain_type=kind,
            )
            placed.append(tp)
            placed_polys.append(shifted)
            break

    return placed


def terrain_to_json(pieces: List[TerrainPiece]) -> List[dict]:
    """Serialize terrain pieces for the web UI."""
    result = []
    for tp in pieces:
        coords = list(tp.polygon.exterior.coords)
        result.append({
            "vertices": [[float(x), float(y)] for x, y in coords],
            "height": tp.height,
            "blocks_los": tp.blocks_los,
            "blocks_movement": tp.blocks_movement,
            "provides_cover": tp.provides_cover,
            "terrain_type": tp.terrain_type,
        })
    return result
