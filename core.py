import numpy as np
from movement import MOVE_DIRS
from wh_types import Unit
from typing import List
import random

def roll_d6() -> int:
    return random.randint(1, 6)
def get_distance(pos1: tuple, pos2: tuple) -> float:
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def get_targets_in_range(unit1: Unit, squads: List[Unit]) -> list[Unit]:
    targets = []
    for other in squads:
        if other.owner != unit1.owner and other.alive:
            distance = get_distance(unit1.pos, other.pos)
            if distance <= unit1.weapons[0].range:
                targets.append(other)
    return targets

def resolve_shooting(attacker: Unit, defender: Unit) -> int:
    """ Handles the Shooting Action From one Model to Another. """
    hits = 0
    rolls = [roll_d6() for _ in range(attacker.weapons[0].shots)]
    for roll in rolls:
        if roll >= attacker.weapons[0].to_hit:
            hits += 1
            break
    dmg = hits * attacker.weapons[0].damage
    return dmg

def resolve_melee(attacker: Unit, defender: Unit) -> int:
    """ Handles the Melee Action From one Model to Another. """
    hits = 0
    rolls = [roll_d6() for _ in range(attacker.weapons[0].shots)]
    for roll in rolls:
        if roll >= attacker.weapons[0].to_hit:
            hits += 1
            break
    dmg = hits * attacker.weapons[0].damage
    return dmg