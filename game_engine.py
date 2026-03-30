from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from wh_types import GameState, GameConfig, Unit, Weapon, TerrainPiece, Objective, Squad
from combat import resolve_shooting, within_range, _allocate_damage, _get_allocation_order, _update_squad_alive
from core import resolve_melee, get_targets_in_range, get_melee_targets
from los import has_los
from movement import (
    get_move_candidates, move_unit, get_charge_targets,
    execute_charge, is_in_engagement, NUM_DIRECTIONS, NUM_DISTANCES,
    NUM_MOVE_SLOTS, MOVE_ANGLES, squad_is_coherent,
)


class Phase(IntEnum):
    MOVEMENT = 0
    SHOOTING = 1
    CHARGE = 2
    FIGHT = 3


PHASE_ORDER = [Phase.MOVEMENT, Phase.SHOOTING, Phase.CHARGE, Phase.FIGHT]
PHASE_NAMES = {Phase.MOVEMENT: "movement", Phase.SHOOTING: "shooting",
               Phase.CHARGE: "charge", Phase.FIGHT: "fight"}

MAX_UNITS_PER_PLAYER = 20
MAX_TURNS = 5
MAX_WEAPONS = 3
NUM_MOVE_CANDIDATES = NUM_MOVE_SLOTS


@dataclass
class Action:
    """Decoded action from the RL agent."""
    unit_idx: int
    action_type: int
    target_idx: int

    PASS = 0
    MOVE = 1
    SHOOT = 2
    CHARGE = 3
    FIGHT = 4


@dataclass
class StepResult:
    """Result of executing a single action in the engine."""
    valid: bool = True
    damage_dealt: int = 0
    damage_taken: int = 0
    units_killed: int = 0
    charge_succeeded: bool = False
    charge_roll: int = 0
    charge_needed: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)


class GameEngine:
    """
    Core 40K game simulation engine with squad coherency,
    bodyguard wound allocation, and leader mechanics.
    """

    def __init__(self, cfg: GameConfig = None, rng_seed: int = 42):
        self.cfg = cfg or GameConfig()
        self.rng = np.random.RandomState(rng_seed)
        self.state: Optional[GameState] = None
        self.current_player: int = 0
        self.current_phase: Phase = Phase.MOVEMENT
        self.turn: int = 1
        self.game_over: bool = False
        self.winner: Optional[int] = None

        self._units_acted: set = set()
        self._player_units: Dict[int, List[int]] = {0: [], 1: []}
        self._player_squads: Dict[int, List[int]] = {0: [], 1: []}
        self._move_cache: Dict[int, List[np.ndarray]] = {}
        self._damage_this_step: Dict[int, int] = {0: 0, 1: 0}

    def reset(self, player0_units: List[Unit], player1_units: List[Unit],
              player0_squads: List[Squad] = None,
              player1_squads: List[Squad] = None,
              terrain: List[TerrainPiece] = None,
              objectives: List[Objective] = None) -> GameState:
        """Initialize a new game with the given armies and squads."""
        units_dict: Dict[int, Unit] = {}
        squads_dict: Dict[int, Squad] = {}
        self._player_units = {0: [], 1: []}
        self._player_squads = {0: [], 1: []}

        for u in player0_units:
            units_dict[u.id] = u
            self._player_units[0].append(u.id)
        for u in player1_units:
            units_dict[u.id] = u
            self._player_units[1].append(u.id)

        for sq in (player0_squads or []):
            squads_dict[sq.id] = sq
            self._player_squads[0].append(sq.id)
        for sq in (player1_squads or []):
            squads_dict[sq.id] = sq
            self._player_squads[1].append(sq.id)

        if objectives is None:
            tw, th = self.cfg.table_size
            objectives = [
                Objective(pos=np.array([tw / 2, th / 2]), r=3.0),
                Objective(pos=np.array([tw / 4, th / 4]), r=3.0),
                Objective(pos=np.array([3 * tw / 4, 3 * th / 4]), r=3.0),
            ]

        self.state = GameState(
            turn=1,
            phase="movement",
            cp={0: 0, 1: 0},
            vp={0: 0, 1: 0},
            units=units_dict,
            squads=squads_dict,
            terrain=terrain or [],
            objectives=objectives,
            cfg=self.cfg,
            rng_seed=int(self.rng.randint(0, 2**31)),
        )

        self.current_player = 0
        self.current_phase = Phase.MOVEMENT
        self.turn = 1
        self.game_over = False
        self.winner = None
        self._units_acted.clear()
        self._move_cache.clear()
        self._damage_this_step = {0: 0, 1: 0}
        self._precompute_moves()

        return self.state

    # ------------------------------------------------------------------
    # Phase / Turn management
    # ------------------------------------------------------------------

    def advance_phase(self) -> None:
        self._units_acted.clear()
        self._move_cache.clear()

        current_idx = PHASE_ORDER.index(self.current_phase)

        if current_idx < len(PHASE_ORDER) - 1:
            self.current_phase = PHASE_ORDER[current_idx + 1]
        else:
            if self.current_player == 0:
                self.current_player = 1
                self.current_phase = Phase.MOVEMENT
            else:
                self._score_objectives()
                self.current_player = 0
                self.current_phase = Phase.MOVEMENT
                self.turn += 1
                self.state.turn = self.turn

                if self.turn > MAX_TURNS:
                    self._end_game()

        if self.current_phase == Phase.MOVEMENT:
            for uid in self._player_units[self.current_player]:
                self.state.units[uid].advanced = False

        self.state.phase = PHASE_NAMES[self.current_phase]
        self._precompute_moves()

    def _score_objectives(self) -> None:
        for obj in self.state.objectives:
            oc = {0: 0, 1: 0}
            for uid, u in self.state.units.items():
                if not u.alive:
                    continue
                dist = np.linalg.norm(u.pos - obj.pos)
                if dist <= obj.r + u.base_r:
                    oc[u.owner] += u.OC
            if oc[0] > oc[1]:
                self.state.vp[0] += 1
            elif oc[1] > oc[0]:
                self.state.vp[1] += 1

    def _end_game(self) -> None:
        self.game_over = True
        if self.state.vp[0] > self.state.vp[1]:
            self.winner = 0
        elif self.state.vp[1] > self.state.vp[0]:
            self.winner = 1
        else:
            self.winner = None

    def check_tabled(self) -> None:
        for pid in [0, 1]:
            alive = any(self.state.units[uid].alive
                        for uid in self._player_units[pid])
            if not alive:
                self.game_over = True
                self.winner = 1 - pid

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def execute_action(self, action: Action) -> StepResult:
        result = StepResult()
        self._damage_this_step = {0: 0, 1: 0}

        pid = self.current_player
        my_unit_ids = self._player_units[pid]

        if action.unit_idx >= len(my_unit_ids):
            result.valid = False
            return result

        unit_id = my_unit_ids[action.unit_idx]
        unit = self.state.units[unit_id]

        if not unit.alive:
            result.valid = False
            return result

        if action.action_type == Action.PASS:
            self._units_acted.add(unit_id)
            return result

        if action.action_type == Action.MOVE:
            result = self._execute_move(unit, action.target_idx)
        elif action.action_type == Action.SHOOT:
            result = self._execute_shoot(unit, action.target_idx)
        elif action.action_type == Action.CHARGE:
            result = self._execute_charge(unit, action.target_idx)
        elif action.action_type == Action.FIGHT:
            result = self._execute_fight(unit, action.target_idx)
        else:
            result.valid = False
            return result

        self._units_acted.add(unit_id)
        self.check_tabled()
        result.damage_dealt = self._damage_this_step.get(1 - pid, 0)
        result.damage_taken = self._damage_this_step.get(pid, 0)
        return result

    def _execute_move(self, unit: Unit, candidate_idx: int) -> StepResult:
        result = StepResult()
        if self.current_phase != Phase.MOVEMENT:
            result.valid = False
            return result

        candidates = self._move_cache.get(unit.id, [])
        if candidate_idx >= len(candidates) or candidates[candidate_idx] is None:
            result.valid = False
            return result

        success = move_unit(unit, candidates[candidate_idx], self.state)
        result.valid = success
        return result

    def execute_move_xy(self, unit_id: int, x: float, y: float) -> StepResult:
        """Move a unit to an arbitrary (x, y) position (for human drag-drop).
        Validates distance, bounds, collisions, and coherency."""
        result = StepResult()
        if self.current_phase != Phase.MOVEMENT:
            result.valid = False
            return result

        unit = self.state.units.get(unit_id)
        if unit is None or not unit.alive or unit.owner != self.current_player:
            result.valid = False
            return result
        if unit_id in self._units_acted:
            result.valid = False
            return result

        new_pos = np.array([x, y])
        success = move_unit(unit, new_pos, self.state)
        if success:
            self._units_acted.add(unit_id)
            self._move_cache.clear()
        result.valid = success
        return result

    def execute_shoot_manual(self, unit_id: int, target_unit_idx: int,
                             hits: int, wounds: int, failed_saves: int) -> StepResult:
        """Human manual shooting: player provides combined dice results for all weapons."""
        result = StepResult()
        if self.current_phase != Phase.SHOOTING:
            result.valid = False
            return result
        unit = self.state.units.get(unit_id)
        if unit is None or not unit.alive or unit.owner != self.current_player:
            result.valid = False
            return result
        if unit_id in self._units_acted or unit.advanced:
            result.valid = False
            return result

        enemy_pid = 1 - self.current_player
        enemy_ids = self._player_units[enemy_pid]
        if target_unit_idx >= len(enemy_ids):
            result.valid = False
            return result
        target = self.state.units[enemy_ids[target_unit_idx]]
        if not target.alive:
            result.valid = False
            return result

        max_dmg = max((w.damage for w in unit.weapons if w.range > 0), default=1)
        damage_pool = failed_saves * max_dmg

        squad = self.state.squads.get(target.squad_id) if target.squad_id >= 0 else None
        if squad is not None and squad.alive:
            alloc_order = _get_allocation_order(squad, self.state, unit)
            alloc_result = _allocate_damage(damage_pool, alloc_order, self.state, max_dmg)
            _update_squad_alive(squad, self.state)
            result.damage_dealt = alloc_result["damage_applied"]
            result.units_killed = alloc_result["models_killed"]
        else:
            target.W = max(0, target.W - damage_pool)
            if target.W == 0:
                target.alive = False
                result.units_killed = 1
            result.damage_dealt = damage_pool

        self._damage_this_step = {0: 0, 1: 0}
        self._damage_this_step[enemy_pid] += result.damage_dealt
        self._units_acted.add(unit_id)
        self.check_tabled()
        return result

    def execute_charge_manual(self, unit_id: int, target_unit_idx: int,
                              charge_roll: int) -> StepResult:
        """Human manual charge: player provides 2d6 result."""
        result = StepResult()
        if self.current_phase != Phase.CHARGE:
            result.valid = False
            return result
        unit = self.state.units.get(unit_id)
        if unit is None or not unit.alive or unit.owner != self.current_player:
            result.valid = False
            return result
        if unit_id in self._units_acted or unit.advanced:
            result.valid = False
            return result

        enemy_pid = 1 - self.current_player
        enemy_ids = self._player_units[enemy_pid]
        if target_unit_idx >= len(enemy_ids):
            result.valid = False
            return result
        target = self.state.units[enemy_ids[target_unit_idx]]
        if not target.alive:
            result.valid = False
            return result

        from movement import execute_charge_manual as _ecm
        succeeded, needed = _ecm(unit, target, self.state, charge_roll)
        result.charge_succeeded = succeeded
        result.charge_roll = charge_roll
        result.charge_needed = needed
        self._units_acted.add(unit_id)
        self.check_tabled()
        return result

    def execute_fight_manual(self, unit_id: int, target_unit_idx: int,
                             hits: int, wounds: int, failed_saves: int) -> StepResult:
        """Human manual fight: player provides combined dice results for all melee weapons."""
        result = StepResult()
        if self.current_phase != Phase.FIGHT:
            result.valid = False
            return result
        unit = self.state.units.get(unit_id)
        if unit is None or not unit.alive or unit.owner != self.current_player:
            result.valid = False
            return result
        if unit_id in self._units_acted:
            result.valid = False
            return result

        enemy_pid = 1 - self.current_player
        enemy_ids = self._player_units[enemy_pid]
        if target_unit_idx >= len(enemy_ids):
            result.valid = False
            return result
        target = self.state.units[enemy_ids[target_unit_idx]]
        if not target.alive:
            result.valid = False
            return result

        if not is_in_engagement(unit, target):
            result.valid = False
            return result

        max_dmg = max((w.damage for w in unit.weapons if w.range <= 0), default=1)
        damage_pool = failed_saves * max_dmg

        squad = self.state.squads.get(target.squad_id) if target.squad_id >= 0 else None
        if squad is not None and squad.alive:
            alloc_order = _get_allocation_order(squad, self.state, unit)
            alloc_result = _allocate_damage(damage_pool, alloc_order, self.state, max_dmg)
            _update_squad_alive(squad, self.state)
            result.damage_dealt = alloc_result["damage_applied"]
            result.units_killed = alloc_result["models_killed"]
        else:
            target.W = max(0, target.W - damage_pool)
            if target.W == 0:
                target.alive = False
                result.units_killed = 1
            result.damage_dealt = damage_pool

        self._damage_this_step = {0: 0, 1: 0}
        self._damage_this_step[enemy_pid] += result.damage_dealt
        self._units_acted.add(unit_id)
        self.check_tabled()
        return result

    def execute_advance(self, unit_id: int) -> StepResult:
        """Advance a squad: roll D6, extend M for all squad members, mark as advanced.
        Models are NOT marked as acted so they can still be dragged to move."""
        result = StepResult()
        if self.current_phase != Phase.MOVEMENT:
            result.valid = False
            return result
        unit = self.state.units.get(unit_id)
        if unit is None or not unit.alive or unit.owner != self.current_player:
            result.valid = False
            return result
        if unit_id in self._units_acted or unit.advanced:
            result.valid = False
            return result

        advance_roll = int(self.rng.randint(1, 7))
        result.detail["advance_roll"] = advance_roll

        if unit.squad_id >= 0 and unit.squad_id in self.state.squads:
            squad = self.state.squads[unit.squad_id]
            for mid in squad.all_member_ids():
                m = self.state.units[mid]
                if m.alive:
                    m.M += advance_roll
                    m.advanced = True
        else:
            unit.M += advance_roll
            unit.advanced = True

        self._move_cache.clear()
        self._precompute_moves()
        return result

    def _execute_shoot(self, unit: Unit, target_unit_idx: int) -> StepResult:
        result = StepResult()
        if self.current_phase != Phase.SHOOTING:
            result.valid = False
            return result
        if unit.advanced:
            result.valid = False
            return result

        enemy_pid = 1 - self.current_player
        enemy_ids = self._player_units[enemy_pid]
        if target_unit_idx >= len(enemy_ids):
            result.valid = False
            return result

        target = self.state.units[enemy_ids[target_unit_idx]]
        if not target.alive:
            result.valid = False
            return result

        total_damage = 0
        total_kills = 0
        for weapon in unit.weapons:
            if weapon.range <= 0:
                continue
            detail = resolve_shooting(self.state, unit, target, weapon, self.rng)
            total_damage += detail["damage"]
            total_kills += detail.get("models_killed", 0)
            result.detail[weapon.name] = detail

        self._damage_this_step[1 - self.current_player] += total_damage
        result.units_killed = total_kills
        result.damage_dealt = total_damage
        return result

    def _execute_charge(self, unit: Unit, target_unit_idx: int) -> StepResult:
        result = StepResult()
        if self.current_phase != Phase.CHARGE:
            result.valid = False
            return result
        if unit.advanced:
            result.valid = False
            return result

        enemy_pid = 1 - self.current_player
        enemy_ids = self._player_units[enemy_pid]
        if target_unit_idx >= len(enemy_ids):
            result.valid = False
            return result

        target = self.state.units[enemy_ids[target_unit_idx]]
        if not target.alive:
            result.valid = False
            return result

        succeeded, roll, needed = execute_charge(unit, target, self.state, self.rng)
        result.charge_succeeded = succeeded
        result.charge_roll = roll
        result.charge_needed = needed
        return result

    def _execute_fight(self, unit: Unit, target_unit_idx: int) -> StepResult:
        result = StepResult()
        if self.current_phase != Phase.FIGHT:
            result.valid = False
            return result

        enemy_pid = 1 - self.current_player
        enemy_ids = self._player_units[enemy_pid]
        if target_unit_idx >= len(enemy_ids):
            result.valid = False
            return result

        target = self.state.units[enemy_ids[target_unit_idx]]
        if not target.alive:
            result.valid = False
            return result

        if not is_in_engagement(unit, target):
            result.valid = False
            return result

        total_damage = 0
        total_kills = 0
        for weapon in unit.weapons:
            if weapon.range > 0:
                continue
            detail = resolve_melee(self.state, unit, target, weapon, self.rng)
            total_damage += detail["damage"]
            total_kills += detail.get("models_killed", 0)
            result.detail[weapon.name] = detail

        self._damage_this_step[1 - self.current_player] += total_damage
        result.units_killed = total_kills
        result.damage_dealt = total_damage
        return result

    # ------------------------------------------------------------------
    # Precomputation & queries
    # ------------------------------------------------------------------

    def _precompute_moves(self) -> None:
        self._move_cache.clear()

        pid = self.current_player
        my_ids = self._player_units[pid]
        for i, uid in enumerate(my_ids):
            if i >= MAX_UNITS_PER_PLAYER:
                self._units_acted.add(uid)

        if self.current_phase != Phase.MOVEMENT:
            return

        for uid in my_ids[:MAX_UNITS_PER_PLAYER]:
            unit = self.state.units[uid]
            if unit.alive:
                self._move_cache[uid] = get_move_candidates(unit, self.state)

    def get_legal_unit_actions(self, unit_idx: int) -> Dict[int, List[int]]:
        pid = self.current_player
        my_ids = self._player_units[pid]
        enemy_ids = self._player_units[1 - pid]

        if unit_idx >= len(my_ids):
            return {Action.PASS: [0]}

        uid = my_ids[unit_idx]
        unit = self.state.units[uid]

        if not unit.alive or uid in self._units_acted:
            return {Action.PASS: [0]}

        legal: Dict[int, List[int]] = {Action.PASS: [0]}

        if self.current_phase == Phase.MOVEMENT:
            candidates = self._move_cache.get(uid, [])
            valid_idxs = [i for i, c in enumerate(candidates) if c is not None]
            if valid_idxs:
                legal[Action.MOVE] = valid_idxs

        elif self.current_phase == Phase.SHOOTING:
            if not unit.advanced:
                ranged_weapons = [w for w in unit.weapons if w.range > 0]
                if ranged_weapons:
                    for ei, eid in enumerate(enemy_ids):
                        enemy = self.state.units[eid]
                        if not enemy.alive:
                            continue
                        can_shoot = any(
                            within_range(unit, enemy, w) and has_los(self.state, unit, enemy)
                            for w in ranged_weapons
                        )
                        if can_shoot:
                            legal.setdefault(Action.SHOOT, []).append(ei)

        elif self.current_phase == Phase.CHARGE:
            if not unit.advanced:
                charge_ids = get_charge_targets(unit, self.state)
                for eid in charge_ids:
                    if eid in enemy_ids:
                        ei = enemy_ids.index(eid)
                        legal.setdefault(Action.CHARGE, []).append(ei)

        elif self.current_phase == Phase.FIGHT:
            melee_weapons = [w for w in unit.weapons if w.range <= 0]
            if melee_weapons:
                melee_ids = get_melee_targets(unit, self.state.units)
                for eid in melee_ids:
                    if eid in enemy_ids:
                        ei = enemy_ids.index(eid)
                        legal.setdefault(Action.FIGHT, []).append(ei)

        return legal

    def all_units_acted(self) -> bool:
        for uid in self._player_units[self.current_player]:
            if self.state.units[uid].alive and uid not in self._units_acted:
                return False
        return True

    def get_player_unit_ids(self, pid: int) -> List[int]:
        return list(self._player_units[pid])

    def get_player_squad_ids(self, pid: int) -> List[int]:
        return list(self._player_squads.get(pid, []))

    def get_alive_counts(self) -> Dict[int, int]:
        counts = {0: 0, 1: 0}
        for uid, u in self.state.units.items():
            if u.alive:
                counts[u.owner] += 1
        return counts

    def get_total_wounds(self) -> Dict[int, int]:
        wounds = {0: 0, 1: 0}
        for uid, u in self.state.units.items():
            if u.alive:
                wounds[u.owner] += u.W
        return wounds
