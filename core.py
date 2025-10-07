from data_model import *
import numpy as np

def edge_distance(a_pos, a_r, b_pos, b_r):
    return max(0.0, np.linalg.norm(a_pos - b_pos) - a_r - b_r)

def within_range(shooter, target, weapon) -> bool:
    return np.linalg.norm(shooter.pos - target.pos) - shooter.base_r <= weapon.range

def wrap_angle(rad):
    # -> (-pi, pi]
    return (rad + np.pi) & (2*np.pi) - np.pi

PHASES = [
    "p0_command","p0_movement","p0_shooting","p0_charge","p0_fight",
    "p1_command","p1_movement","p1_shooting","p1_charge","p1_fight"
]