from __future__ import annotations
import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any
from shapely.geometry import Polygon, Point
from .types import GameState, GameConfig, Unit, TerrainPiece, Objective, Weapon
from .geometry import circle_geom
from .los import has_los
from .movement import movement_candidates

class WH40KEnv(gym.Env):
    """
    Minimal skeleton. Observation/action spaces are left as placeholders—focus is on world state,
    LOS, and movement legality. You can extend this into a PettingZoo multi-agent later.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, state: GameState):
        super().__init__()
        self.state = state

        # placeholders—replace with your encoding/masks later
        self.observation_space = gym.spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        # You'd create a fresh GameState here; we reuse the provided one for demo purposes.
        obs = np.zeros((1,), dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        # No-op: implement your apply_action, rewards, and phase progression.
        obs = np.zeros((1,), dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

# --- tiny helpers for the demo ---

def make_demo_state(seed: int = 0) -> GameState:
    rng = np.random.RandomState(seed)
    cfg = GameConfig()

    # Terrain: two L-shape ruins that block LOS + movement, height=3
    ruin1 = Polygon([(10,10),(18,10),(18,18),(14,18),(14,22),(10,22)])
    ruin2 = Polygon([(30,35),(40,35),(40,45),(36,45),(36,50),(30,50)])
    terrain = [
        TerrainPiece(polygon=ruin1, height=3, blocks_los=True, blocks_movement=True),
        TerrainPiece(polygon=ruin2, height=3, blocks_los=True, blocks_movement=True),
    ]

    # Objectives
    obj1 = Objective(pos=np.array([15.0, 40.0]), r=3.0)
    obj2 = Objective(pos=np.array([38.0, 20.0]), r=3.0)

    # Units
    bolter = Weapon(name="Bolter", range=24.0, attacks=2, to_hit=3, strength=4, ap=0, damage=1)
    rail = Weapon(name="Rail", range=36.0, attacks=1, to_hit=4, strength=8, ap=-4, damage=3)

    u0 = Unit(id=0, name="Marine", owner=0, pos=np.array([8.0, 15.0]), theta=0.0, base_r=1.0, height=1,
              M=6.0, W=2, OC=1, weapons=[bolter])
    u1 = Unit(id=1, name="Marine", owner=0, pos=np.array([12.0, 16.0]), theta=0.0, base_r=1.0, height=1,
              M=6.0, W=2, OC=1, weapons=[bolter])
    e0 = Unit(id=2, name="Battlesuit", owner=1, pos=np.array([35.0, 42.0]), theta=0.0, base_r=1.25, height=2,
              M=10.0, W=7, OC=2, weapons=[rail])
    e1 = Unit(id=3, name="Drone", owner=1, pos=np.array([28.0, 38.0]), theta=0.0, base_r=1.0, height=1,
              M=10.0, W=1, OC=1, weapons=[])

    units = {u.id: u for u in [u0, u1, e0, e1]}

    state = GameState(
        turn=1,
        phase="p0_movement",
        cp={0:1, 1:1},
        vp={0:0, 1:0},
        units=units,
        terrain=terrain,
        objectives=[obj1, obj2],
        cfg=cfg,
        rng_seed=seed
    )
    return state
