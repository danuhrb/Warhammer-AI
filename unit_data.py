from __future__ import annotations
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Callable, Optional
sys.path.append(str(Path(__file__).parent.parent))

from wh_types import Unit, Weapon, Squad

# ============================================================================
# WEAPON DEFINITIONS (Wahapedia 10th Ed datasheets)
# ============================================================================

# --- Tau Empire Weapons ---

PULSE_RIFLE = Weapon(
    name="Pulse Rifle",
    range=30.0, attacks=1, to_hit=4, strength=5, ap=-1, damage=1,
)

PULSE_CARBINE = Weapon(
    name="Pulse Carbine",
    range=20.0, attacks=2, to_hit=4, strength=5, ap=0, damage=1,
)

BURST_CANNON = Weapon(
    name="Burst Cannon",
    range=18.0, attacks=4, to_hit=4, strength=5, ap=0, damage=1,
)

MISSILE_POD = Weapon(
    name="Missile Pod",
    range=30.0, attacks=2, to_hit=4, strength=7, ap=-1, damage=2,
)

CYCLIC_ION_BLASTER = Weapon(
    name="Cyclic Ion Blaster",
    range=18.0, attacks=3, to_hit=4, strength=7, ap=-2, damage=1,
)

PLASMA_RIFLE = Weapon(
    name="Plasma Rifle",
    range=30.0, attacks=1, to_hit=4, strength=8, ap=-3, damage=3,
)

RAIL_RIFLE = Weapon(
    name="Rail Rifle",
    range=30.0, attacks=1, to_hit=5, strength=10, ap=-4, damage=3,
)

ION_RIFLE = Weapon(
    name="Ion Rifle",
    range=30.0, attacks=3, to_hit=5, strength=7, ap=-1, damage=1,
)

FUSION_BLASTER = Weapon(
    name="Fusion Blaster",
    range=12.0, attacks=1, to_hit=4, strength=9, ap=-4, damage=4,
)

TWIN_PLASMA_RIFLE = Weapon(
    name="Twin Plasma Rifle",
    range=18.0, attacks=1, to_hit=4, strength=8, ap=-3, damage=3,
)

HIGH_YIELD_MISSILE_PODS = Weapon(
    name="High-yield Missile Pods",
    range=30.0, attacks=6, to_hit=4, strength=7, ap=-1, damage=2,
)

HEAVY_RAIL_RIFLE = Weapon(
    name="Heavy Rail Rifle",
    range=60.0, attacks=2, to_hit=4, strength=12, ap=-4, damage=4,
)

CYCLIC_ION_RAKER = Weapon(
    name="Cyclic Ion Raker",
    range=36.0, attacks=6, to_hit=4, strength=7, ap=-1, damage=2,
)

FUSION_COLLIDER = Weapon(
    name="Fusion Collider",
    range=18.0, attacks=2, to_hit=4, strength=12, ap=-4, damage=4,
)

HEAVY_BURST_CANNON = Weapon(
    name="Heavy Burst Cannon",
    range=36.0, attacks=12, to_hit=4, strength=6, ap=-1, damage=2,
)

ION_ACCELERATOR = Weapon(
    name="Ion Accelerator",
    range=72.0, attacks=6, to_hit=4, strength=9, ap=-2, damage=3,
)

RIPTIDE_FISTS = Weapon(
    name="Riptide Fists",
    range=0.0, attacks=6, to_hit=5, strength=6, ap=0, damage=2,
)

GHOSTKEEL_FISTS = Weapon(
    name="Ghostkeel Fists",
    range=0.0, attacks=3, to_hit=5, strength=6, ap=0, damage=2,
)

BATTLESUIT_FISTS = Weapon(
    name="Battlesuit Fists",
    range=0.0, attacks=2, to_hit=5, strength=4, ap=0, damage=1,
)

# --- Space Marine (shared) Weapons ---

BOLT_RIFLE = Weapon(
    name="Bolt Rifle",
    range=30.0, attacks=2, to_hit=3, strength=4, ap=-1, damage=1,
)

STORM_BOLTER = Weapon(
    name="Storm Bolter",
    range=24.0, attacks=2, to_hit=3, strength=4, ap=0, damage=1,
)

POWER_FIST = Weapon(
    name="Power Fist",
    range=0.0, attacks=3, to_hit=4, strength=8, ap=-2, damage=2,
)

MASTER_CRAFTED_BOLTER = Weapon(
    name="Master-crafted Boltgun",
    range=24.0, attacks=2, to_hit=2, strength=4, ap=0, damage=2,
)

RELIC_BLADE = Weapon(
    name="Relic Blade",
    range=0.0, attacks=5, to_hit=3, strength=6, ap=-2, damage=2,
)

# --- Blood Angels Weapons ---

HEAVY_BOLT_PISTOL = Weapon(
    name="Heavy Bolt Pistol",
    range=18.0, attacks=1, to_hit=3, strength=4, ap=-1, damage=1,
)

ASTARTES_CHAINSWORD = Weapon(
    name="Astartes Chainsword",
    range=0.0, attacks=4, to_hit=3, strength=4, ap=-1, damage=1,
)

POWER_WEAPON = Weapon(
    name="Power Weapon",
    range=0.0, attacks=4, to_hit=3, strength=5, ap=-2, damage=1,
)

THUNDER_HAMMER = Weapon(
    name="Thunder Hammer",
    range=0.0, attacks=3, to_hit=4, strength=8, ap=-2, damage=2,
)

ANGELUS_BOLTGUN = Weapon(
    name="Angelus Boltgun",
    range=12.0, attacks=2, to_hit=3, strength=4, ap=0, damage=1,
)

ENCARMINE_BLADE = Weapon(
    name="Encarmine Blade",
    range=0.0, attacks=4, to_hit=3, strength=6, ap=-3, damage=2,
)

PERDITION_PISTOL = Weapon(
    name="Perdition Pistol",
    range=6.0, attacks=1, to_hit=2, strength=9, ap=-4, damage=4,
)

AXE_MORTALIS = Weapon(
    name="The Axe Mortalis",
    range=0.0, attacks=8, to_hit=2, strength=8, ap=-3, damage=2,
)

ABSOLVOR_BOLT_PISTOL = Weapon(
    name="Absolvor Bolt Pistol",
    range=18.0, attacks=1, to_hit=2, strength=5, ap=-1, damage=2,
)

BLOOD_CROZIUS = Weapon(
    name="The Blood Crozius",
    range=0.0, attacks=5, to_hit=2, strength=6, ap=-2, damage=2,
)

ENCARMINE_BROADSWORD = Weapon(
    name="Encarmine Broadsword",
    range=0.0, attacks=8, to_hit=2, strength=6, ap=-3, damage=2,
)

# ============================================================================
# TAU EMPIRE UNITS
# ============================================================================

def create_fire_warrior(uid: int, owner: int, pos: np.ndarray,
                        squad_id: int = -1) -> Unit:
    # M6" T3 Sv4+ W1 OC2
    return Unit(
        id=uid, name="Fire Warrior", owner=owner, pos=pos,
        theta=0.0, base_r=0.75, height=1,
        M=6.0, W=1, max_W=1, T=3, Sv=4, OC=2,
        weapons=[PULSE_RIFLE], squad_id=squad_id, alive=True,
    )

def create_pathfinder(uid: int, owner: int, pos: np.ndarray,
                      squad_id: int = -1) -> Unit:
    # M7" T3 Sv4+ W1 OC1  -- BS5+ normally but markerlights buff allies
    return Unit(
        id=uid, name="Pathfinder", owner=owner, pos=pos,
        theta=0.0, base_r=0.75, height=1,
        M=7.0, W=1, max_W=1, T=3, Sv=4, OC=1,
        weapons=[PULSE_CARBINE, RAIL_RIFLE], squad_id=squad_id, alive=True,
    )

def create_stealth_battlesuit(uid: int, owner: int, pos: np.ndarray,
                               squad_id: int = -1) -> Unit:
    # M8" T4 Sv3+ W2 OC1
    return Unit(
        id=uid, name="Stealth Battlesuit", owner=owner, pos=pos,
        theta=0.0, base_r=1.0, height=1,
        M=8.0, W=2, max_W=2, T=4, Sv=3, OC=1,
        weapons=[BURST_CANNON, BATTLESUIT_FISTS], squad_id=squad_id, alive=True,
    )

def create_crisis_battlesuit(uid: int, owner: int, pos: np.ndarray,
                              squad_id: int = -1) -> Unit:
    # M8" T5 Sv3+ W4 OC2
    return Unit(
        id=uid, name="Crisis Battlesuit", owner=owner, pos=pos,
        theta=0.0, base_r=1.5, height=2,
        M=8.0, W=4, max_W=4, T=5, Sv=3, OC=2,
        weapons=[BURST_CANNON, MISSILE_POD], squad_id=squad_id, alive=True,
    )

def create_crisis_commander(uid: int, owner: int, pos: np.ndarray,
                             squad_id: int = -1) -> Unit:
    # M8" T5 Sv3+ W5 OC1 -- leader
    return Unit(
        id=uid, name="Crisis Commander", owner=owner, pos=pos,
        theta=0.0, base_r=1.5, height=2,
        M=8.0, W=5, max_W=5, T=5, Sv=3, OC=1,
        weapons=[PLASMA_RIFLE, CYCLIC_ION_BLASTER],
        squad_id=squad_id, is_leader=True, alive=True,
    )

def create_broadside(uid: int, owner: int, pos: np.ndarray,
                     squad_id: int = -1) -> Unit:
    # M5" T6 Sv2+ W8 OC2  -- heavy rail rifle variant
    return Unit(
        id=uid, name="Broadside Battlesuit", owner=owner, pos=pos,
        theta=0.0, base_r=1.75, height=2,
        M=5.0, W=8, max_W=8, T=6, Sv=2, OC=2,
        weapons=[HEAVY_RAIL_RIFLE, TWIN_PLASMA_RIFLE], squad_id=squad_id, alive=True,
    )

def create_ghostkeel(uid: int, owner: int, pos: np.ndarray,
                     squad_id: int = -1) -> Unit:
    # M10" T8 Sv2+ W12 OC3
    return Unit(
        id=uid, name="Ghostkeel Battlesuit", owner=owner, pos=pos,
        theta=0.0, base_r=2.0, height=3,
        M=10.0, W=12, max_W=12, T=8, Sv=2, OC=3,
        weapons=[CYCLIC_ION_RAKER, FUSION_BLASTER, GHOSTKEEL_FISTS],
        squad_id=squad_id, alive=True,
    )

def create_riptide(uid: int, owner: int, pos: np.ndarray,
                   squad_id: int = -1) -> Unit:
    # M10" T9 Sv2+ W14 OC4
    return Unit(
        id=uid, name="Riptide Battlesuit", owner=owner, pos=pos,
        theta=0.0, base_r=2.5, height=3,
        M=10.0, W=14, max_W=14, T=9, Sv=2, OC=4,
        weapons=[HEAVY_BURST_CANNON, ION_ACCELERATOR, RIPTIDE_FISTS],
        squad_id=squad_id, alive=True,
    )

# ============================================================================
# SPACE MARINES (ULTRAMARINES) UNITS
# ============================================================================

def create_intercessor(uid: int, owner: int, pos: np.ndarray,
                       squad_id: int = -1) -> Unit:
    # M6" T4 Sv3+ W2 OC2
    return Unit(
        id=uid, name="Intercessor", owner=owner, pos=pos,
        theta=0.0, base_r=1.0, height=1,
        M=6.0, W=2, max_W=2, T=4, Sv=3, OC=2,
        weapons=[BOLT_RIFLE], squad_id=squad_id, alive=True,
    )

def create_terminator(uid: int, owner: int, pos: np.ndarray,
                      squad_id: int = -1) -> Unit:
    # M5" T5 Sv2+ W3 OC1
    return Unit(
        id=uid, name="Terminator", owner=owner, pos=pos,
        theta=0.0, base_r=1.25, height=1,
        M=5.0, W=3, max_W=3, T=5, Sv=2, OC=1,
        weapons=[STORM_BOLTER, POWER_FIST], squad_id=squad_id, alive=True,
    )

def create_captain_terminator(uid: int, owner: int, pos: np.ndarray,
                               squad_id: int = -1) -> Unit:
    # M5" T5 Sv2+ W6 OC1 -- leader
    return Unit(
        id=uid, name="Captain (Terminator)", owner=owner, pos=pos,
        theta=0.0, base_r=1.25, height=1,
        M=5.0, W=6, max_W=6, T=5, Sv=2, OC=1,
        weapons=[MASTER_CRAFTED_BOLTER, RELIC_BLADE],
        squad_id=squad_id, is_leader=True, alive=True,
    )

# ============================================================================
# BLOOD ANGELS UNITS
# ============================================================================

def create_assault_intercessor(uid: int, owner: int, pos: np.ndarray,
                                squad_id: int = -1) -> Unit:
    # M6" T4 Sv3+ W2 OC2
    return Unit(
        id=uid, name="Assault Intercessor", owner=owner, pos=pos,
        theta=0.0, base_r=1.0, height=1,
        M=6.0, W=2, max_W=2, T=4, Sv=3, OC=2,
        weapons=[HEAVY_BOLT_PISTOL, ASTARTES_CHAINSWORD],
        squad_id=squad_id, alive=True,
    )

def create_death_company_marine(uid: int, owner: int, pos: np.ndarray,
                                 squad_id: int = -1) -> Unit:
    # M12" T4 Sv3+ W2 OC1 (Jump Pack variant)
    return Unit(
        id=uid, name="Death Company Marine", owner=owner, pos=pos,
        theta=0.0, base_r=1.0, height=1,
        M=12.0, W=2, max_W=2, T=4, Sv=3, OC=1,
        weapons=[HEAVY_BOLT_PISTOL, ASTARTES_CHAINSWORD],
        squad_id=squad_id, alive=True,
    )

def create_sanguinary_guard(uid: int, owner: int, pos: np.ndarray,
                             squad_id: int = -1) -> Unit:
    # M12" T4 Sv2+ W3 OC1 (4+ invuln not modelled yet)
    return Unit(
        id=uid, name="Sanguinary Guard", owner=owner, pos=pos,
        theta=0.0, base_r=1.0, height=1,
        M=12.0, W=3, max_W=3, T=4, Sv=2, OC=1,
        weapons=[ANGELUS_BOLTGUN, ENCARMINE_BLADE],
        squad_id=squad_id, alive=True,
    )

def create_commander_dante(uid: int, owner: int, pos: np.ndarray,
                            squad_id: int = -1) -> Unit:
    # M12" T4 Sv2+ W6 OC1 (4+ invuln, leader)
    return Unit(
        id=uid, name="Commander Dante", owner=owner, pos=pos,
        theta=0.0, base_r=1.25, height=1,
        M=12.0, W=6, max_W=6, T=4, Sv=2, OC=1,
        weapons=[PERDITION_PISTOL, AXE_MORTALIS],
        squad_id=squad_id, is_leader=True, alive=True,
    )

def create_lemartes(uid: int, owner: int, pos: np.ndarray,
                    squad_id: int = -1) -> Unit:
    # M12" T4 Sv3+ W4 OC1 (4+ invuln, leader for Death Company)
    return Unit(
        id=uid, name="Lemartes", owner=owner, pos=pos,
        theta=0.0, base_r=1.25, height=1,
        M=12.0, W=4, max_W=4, T=4, Sv=3, OC=1,
        weapons=[ABSOLVOR_BOLT_PISTOL, BLOOD_CROZIUS],
        squad_id=squad_id, is_leader=True, alive=True,
    )

def create_sanguinor(uid: int, owner: int, pos: np.ndarray,
                     squad_id: int = -1) -> Unit:
    # M12" T4 Sv2+ W7 OC1 (4+ invuln, lone operative)
    return Unit(
        id=uid, name="The Sanguinor", owner=owner, pos=pos,
        theta=0.0, base_r=1.25, height=1,
        M=12.0, W=7, max_W=7, T=4, Sv=2, OC=1,
        weapons=[ENCARMINE_BROADSWORD],
        squad_id=squad_id, alive=True,
    )

# ============================================================================
# SQUAD TEMPLATES  (Wahapedia 10th Ed / MFM points)
#
# Each template is a dict with:
#   name        - display name
#   model_count - number of regular (non-leader) models
#   points      - matched-play cost (includes leader if present)
#   create_fn   - factory for a single regular model
#   leader_fn   - factory for the attached leader (or None)
#   spacing     - how far apart to place models (inches)
# ============================================================================

TAU_TEMPLATES = [
    dict(name="Fire Warrior Strike Team (10)", model_count=10, points=70,
         create_fn=create_fire_warrior, leader_fn=None, spacing=2.0),
    dict(name="Fire Warrior Strike Team (5)", model_count=5, points=35,
         create_fn=create_fire_warrior, leader_fn=None, spacing=2.0),
    dict(name="Fire Warriors + Commander (5+1)", model_count=5, points=115,
         create_fn=create_fire_warrior, leader_fn=create_crisis_commander, spacing=2.0),
    dict(name="Fire Warriors + Commander (10+1)", model_count=10, points=150,
         create_fn=create_fire_warrior, leader_fn=create_crisis_commander, spacing=2.0),
    dict(name="Pathfinder Team (10)", model_count=10, points=90,
         create_fn=create_pathfinder, leader_fn=None, spacing=2.0),
    dict(name="Pathfinder Team (5)", model_count=5, points=45,
         create_fn=create_pathfinder, leader_fn=None, spacing=2.0),
    dict(name="Stealth Battlesuits (5)", model_count=5, points=100,
         create_fn=create_stealth_battlesuit, leader_fn=None, spacing=2.0),
    dict(name="Stealth Battlesuits (3)", model_count=3, points=60,
         create_fn=create_stealth_battlesuit, leader_fn=None, spacing=2.0),
    dict(name="Crisis Battlesuits (3)", model_count=3, points=200,
         create_fn=create_crisis_battlesuit, leader_fn=None, spacing=2.5),
    dict(name="Crisis Battlesuits + Commander (3+1)", model_count=3, points=280,
         create_fn=create_crisis_battlesuit, leader_fn=create_crisis_commander, spacing=2.5),
    dict(name="Broadside Team (2)", model_count=2, points=170,
         create_fn=create_broadside, leader_fn=None, spacing=4.0),
    dict(name="Broadside (1)", model_count=1, points=80,
         create_fn=create_broadside, leader_fn=None, spacing=4.0),
    dict(name="Ghostkeel Battlesuit", model_count=1, points=160,
         create_fn=create_ghostkeel, leader_fn=None, spacing=0.0),
    dict(name="Riptide Battlesuit", model_count=1, points=200,
         create_fn=create_riptide, leader_fn=None, spacing=0.0),
]

BLOOD_ANGELS_TEMPLATES = [
    dict(name="Assault Intercessor Squad (5)", model_count=5, points=75,
         create_fn=create_assault_intercessor, leader_fn=None, spacing=2.0),
    dict(name="Assault Intercessor Squad (10)", model_count=10, points=150,
         create_fn=create_assault_intercessor, leader_fn=None, spacing=2.0),
    dict(name="Death Company + Lemartes (5+1)", model_count=5, points=220,
         create_fn=create_death_company_marine, leader_fn=create_lemartes, spacing=2.0),
    dict(name="Death Company (5)", model_count=5, points=120,
         create_fn=create_death_company_marine, leader_fn=None, spacing=2.0),
    dict(name="Sanguinary Guard + Dante (5+1)", model_count=5, points=245,
         create_fn=create_sanguinary_guard, leader_fn=create_commander_dante, spacing=2.0),
    dict(name="Sanguinary Guard (5)", model_count=5, points=125,
         create_fn=create_sanguinary_guard, leader_fn=None, spacing=2.0),
    dict(name="Sanguinary Guard (3)", model_count=3, points=110,
         create_fn=create_sanguinary_guard, leader_fn=None, spacing=2.0),
    dict(name="Terminator Squad (5)", model_count=5, points=170,
         create_fn=create_terminator, leader_fn=None, spacing=2.5),
    dict(name="Terminators + Captain (5+1)", model_count=5, points=265,
         create_fn=create_terminator, leader_fn=create_captain_terminator, spacing=2.5),
    dict(name="The Sanguinor", model_count=1, points=130,
         create_fn=create_sanguinor, leader_fn=None, spacing=0.0),
]

ULTRAMARINE_TEMPLATES = [
    dict(name="Intercessor Squad (5)", model_count=5, points=80,
         create_fn=create_intercessor, leader_fn=None, spacing=2.5),
    dict(name="Intercessor Squad (10)", model_count=10, points=160,
         create_fn=create_intercessor, leader_fn=None, spacing=2.5),
    dict(name="Intercessors + Captain (10+1)", model_count=10, points=255,
         create_fn=create_intercessor, leader_fn=create_captain_terminator, spacing=2.5),
    dict(name="Terminator Squad (5)", model_count=5, points=170,
         create_fn=create_terminator, leader_fn=None, spacing=2.5),
    dict(name="Terminators + Captain (5+1)", model_count=5, points=265,
         create_fn=create_terminator, leader_fn=create_captain_terminator, spacing=2.5),
]


# ============================================================================
# GENERIC ARMY BUILDER
# ============================================================================

DEPLOYMENT_DEPTH = 9.0  # inches from table edge


def _place_squad(template: dict, owner: int, uid: int, sid: int,
                 start_x: float, start_y: float, y_direction: float = 1.0,
                 ) -> Tuple[List[Unit], Squad, int]:
    """Instantiate one squad from a template. Returns (units, squad, next_uid).
    y_direction: +1 means rows go upward from start_y, -1 means downward."""
    units: List[Unit] = []
    member_ids: List[int] = []
    spacing = template["spacing"] or 2.0
    count = template["model_count"]
    cols = min(count, 5)

    for i in range(count):
        col = i % cols
        row = i // cols
        pos = np.array([start_x + col * spacing,
                        start_y + row * spacing * y_direction])
        u = template["create_fn"](uid, owner, pos, squad_id=sid)
        units.append(u)
        member_ids.append(uid)
        uid += 1

    leader_id: Optional[int] = None
    if template["leader_fn"] is not None:
        ldr_row = (count - 1) // cols + 1
        ldr_pos = np.array([start_x + (cols - 1) * spacing * 0.5,
                            start_y + ldr_row * spacing * y_direction])
        ldr = template["leader_fn"](uid, owner, ldr_pos, squad_id=sid)
        units.append(ldr)
        leader_id = uid
        uid += 1

    squad = Squad(
        id=sid, name=template["name"], owner=owner,
        unit_ids=member_ids, leader_id=leader_id,
        points=template["points"],
    )
    return units, squad, uid


def build_army(templates: List[dict],
               priority_order: List[int],
               owner: int,
               points_limit: int = 1000,
               max_models: int = 20,
               table_size: Tuple[float, float] = (44.0, 60.0),
               ) -> Tuple[List[Unit], List[Squad]]:
    """
    Build an army within a proper deployment zone (within 9" of your table edge).

    Player 0 deploys near y=0 (bottom), player 1 deploys near y=table_h (top).
    Squads are laid out left-to-right, wrapping into new rows within the zone.
    """
    tw, th = table_size
    all_units: List[Unit] = []
    all_squads: List[Squad] = []
    uid = owner * 100
    sid = owner * 10
    spent = 0
    model_count = 0

    margin = 1.5  # base radius buffer from edge
    if owner == 0:
        zone_y_start = margin
        zone_y_end = DEPLOYMENT_DEPTH
        y_dir = 1.0
    else:
        zone_y_start = th - margin
        zone_y_end = th - DEPLOYMENT_DEPTH
        y_dir = -1.0

    cursor_x = margin + 1.0
    cursor_y = zone_y_start
    row_height = 0.0

    for idx in priority_order:
        t = templates[idx]
        squad_models = t["model_count"] + (1 if t["leader_fn"] else 0)
        if spent + t["points"] > points_limit:
            continue
        if model_count + squad_models > max_models:
            continue

        spacing = t["spacing"] or 2.0
        cols = min(t["model_count"], 5)
        squad_width = (cols - 1) * spacing + 2.0
        rows = (t["model_count"] - 1) // cols + 1
        if t["leader_fn"]:
            rows += 1
        squad_height = (rows - 1) * spacing

        if cursor_x + squad_width > tw - margin:
            cursor_x = margin + 1.0
            cursor_y += (row_height + 1.5) * y_dir
            row_height = 0.0

        farthest_y = cursor_y + squad_height * y_dir
        if owner == 0 and farthest_y > zone_y_end:
            continue
        if owner == 1 and farthest_y < zone_y_end:
            continue

        units, squad, uid = _place_squad(
            t, owner, uid, sid, cursor_x, cursor_y, y_dir)
        all_units.extend(units)
        all_squads.append(squad)
        sid += 1
        spent += t["points"]
        model_count += squad_models
        cursor_x += squad_width + 1.5
        row_height = max(row_height, squad_height)

    return all_units, all_squads


def army_points(squads: List[Squad]) -> int:
    """Total points spent on an army."""
    return sum(sq.points for sq in squads)


# ============================================================================
# PRESET ARMY BUILDERS (convenience wrappers)
# ============================================================================

def create_tau_army(owner: int, points_limit: int = 1000,
                    table_size: Tuple[float, float] = (44.0, 60.0),
                    ) -> Tuple[List[Unit], List[Squad]]:
    priority = [2, 5, 7, 8, 10, 13, 4, 0, 12, 6, 11, 3, 9, 1]
    return build_army(TAU_TEMPLATES, priority, owner,
                      points_limit=points_limit, table_size=table_size)


def create_blood_angels_army(owner: int, points_limit: int = 1000,
                              table_size: Tuple[float, float] = (44.0, 60.0),
                              ) -> Tuple[List[Unit], List[Squad]]:
    priority = [0, 2, 4, 7, 9, 3, 6, 5, 1, 8]
    return build_army(BLOOD_ANGELS_TEMPLATES, priority, owner,
                      points_limit=points_limit, table_size=table_size)


def create_ultramarine_army(owner: int, points_limit: int = 1000,
                             table_size: Tuple[float, float] = (44.0, 60.0),
                             ) -> Tuple[List[Unit], List[Squad]]:
    priority = [2, 4, 1, 0, 3]
    return build_army(ULTRAMARINE_TEMPLATES, priority, owner,
                      points_limit=points_limit, table_size=table_size)
