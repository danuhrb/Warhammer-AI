from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon, Point

@dataclass
class Weapon:
    name: str
    range: float      # inches
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
    theta: float               # radians (0 along +x); not all rulesets use it
    base_r: float              # base radius in inches
    height: int                # 1=infantry, 2=tall, 3=vehicle/monster (used vs terrain height)
    Move: float                   # move characteristic (inches)
    Wounds: int                     # wounds remaining
    Obj_Ctrl: int                    # objective control
    weapons: List[Weapon]
    alive: bool = True

@dataclass
class TerrainPiece:
    polygon: Polygon           # Shapely polygon in table coords (inches)
    height: int                # height class; blocks LOS if >= min(shooter, target) and blocks_los=True
    blocks_los: bool = True
    blocks_movement: bool = True

@dataclass
class Objective:
    pos: np.ndarray            # (x,y)
    r: float                   # radius (inches)

@dataclass
class GameConfig:
    table_size: Tuple[float, float] = (44.0, 60.0)  # width x height in inches
    los_epsilon: float = 1e-6
    move_margin: float = 0.1     # small gap when checking collisions
    max_candidates: int = 20

@dataclass
class GameState:
    turn: int
    phase: str
    cp: Dict[int,int]
    vp: Dict[int,int]
    units: Dict[int, Unit]
    terrain: List[TerrainPiece]
    objectives: List[Objective]
    cfg: GameConfig
    rng_seed: int = 0

    @property
    def table_poly(self):
        w, h = self.cfg.table_size
        return Polygon([(0,0),(w,0),(w,h),(0,h)])
