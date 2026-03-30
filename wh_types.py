from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from shapely.geometry import Polygon

COHERENCY_RANGE = 2.0  # inches -- models in a squad must be within 2" of another model


@dataclass
class Weapon:
    name: str
    range: float      # inches (0 = melee only)
    attacks: int
    to_hit: int       # 2-6 (d6 threshold)
    strength: int
    ap: int
    damage: int


@dataclass
class Unit:
    id: int
    name: str
    owner: int                 # 0 or 1
    pos: np.ndarray            # shape (2,) in inches
    theta: float               # radians (0 along +x)
    base_r: float              # base radius in inches
    height: int                # 1=infantry, 2=tall, 3=vehicle/monster
    M: float                   # move characteristic (inches)
    W: int                     # wounds remaining
    max_W: int                 # wounds at full health (for resetting / tracking)
    T: int                     # toughness
    Sv: int                    # armor save (2-6)
    OC: int                    # objective control
    weapons: List[Weapon]
    squad_id: int = -1         # which Squad this model belongs to (-1 = none)
    is_leader: bool = False    # True if this is a leader/character model
    alive: bool = True
    advanced: bool = False     # True if this model advanced this turn (cannot shoot/charge)


@dataclass
class Squad:
    id: int
    name: str
    owner: int
    unit_ids: List[int]              # IDs of member Units (not including leader)
    leader_id: Optional[int] = None  # attached leader Unit ID (bodyguarded)
    points: int = 0                  # matched-play points cost for this squad
    alive: bool = True

    def all_member_ids(self) -> List[int]:
        """All unit IDs in this squad including the leader."""
        ids = list(self.unit_ids)
        if self.leader_id is not None:
            ids.append(self.leader_id)
        return ids


@dataclass
class TerrainPiece:
    polygon: Polygon           # Shapely polygon in table coords (inches)
    height: int                # height class (3 = floor-to-sky building)
    blocks_los: bool = True
    blocks_movement: bool = True
    provides_cover: bool = True  # +1 to save for targets behind this
    terrain_type: str = "rect"   # "rect", "square", "L"


@dataclass
class Objective:
    pos: np.ndarray            # (x,y)
    r: float                   # radius (inches)


@dataclass
class GameConfig:
    table_size: Tuple[float, float] = (44.0, 60.0)  # width x height in inches
    los_epsilon: float = 1e-6
    move_margin: float = 0.0
    max_candidates: int = 20
    coherency_range: float = COHERENCY_RANGE
    points_limit: int = 1000           # matched-play army points cap


@dataclass
class GameState:
    turn: int
    phase: str
    cp: Dict[int, int]
    vp: Dict[int, int]
    units: Dict[int, Unit]
    squads: Dict[int, Squad]
    terrain: List[TerrainPiece]
    objectives: List[Objective]
    cfg: GameConfig
    rng_seed: int = 0

    @property
    def table_poly(self):
        w, h = self.cfg.table_size
        return Polygon([(0, 0), (w, 0), (w, h), (0, h)])
