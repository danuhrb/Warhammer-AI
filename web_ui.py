#!/usr/bin/env python3
"""
Human vs AI Web UI for Warhammer 40K.

Usage:
  python web_ui.py                        # play vs random bot
  python web_ui.py --ai models/wh40k_agent_final  # play vs trained model
  python web_ui.py --points 500           # 500pt game

Opens http://localhost:5000 in your browser.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import webbrowser
from threading import Timer
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_from_directory

from wh_types import GameConfig
from game_engine import GameEngine, Action, Phase, PHASE_NAMES, MAX_UNITS_PER_PLAYER, PHASE_ORDER
from env.action_codec import build_action_mask, MAX_TARGETS, NUM_ACTION_TYPES
from env.obs_builder import encode_observation
from unit_data import (
    create_tau_army, create_blood_angels_army,
    create_ultramarine_army, army_points,
)
from bots.random_bot import random_policy

app = Flask(__name__, static_folder="web_static")

engine: GameEngine = None
ai_model = None
ai_type: str = "random"
game_log: list = []

HUMAN_PLAYER = 1
AI_PLAYER = 0

ACTION_TYPE_NAMES = {0: "Pass", 1: "Move", 2: "Shoot", 3: "Charge", 4: "Fight"}

def init_game(points_limit: int = 1000, seed: int = 42,
              human_goes_first: bool = False):
    global engine, game_log, HUMAN_PLAYER, AI_PLAYER

    if human_goes_first:
        HUMAN_PLAYER = 0
        AI_PLAYER = 1
    else:
        HUMAN_PLAYER = 1
        AI_PLAYER = 0

    cfg = GameConfig(points_limit=points_limit)
    engine = GameEngine(cfg=cfg, rng_seed=seed)

    table_size = cfg.table_size
    ai_units, ai_squads = create_tau_army(owner=AI_PLAYER, points_limit=points_limit,
                                           table_size=table_size)
    human_units, human_squads = create_blood_angels_army(owner=HUMAN_PLAYER, points_limit=points_limit,
                                                         table_size=table_size)

    if human_goes_first:
        engine.reset(human_units, ai_units, human_squads, ai_squads)
    else:
        engine.reset(ai_units, human_units, ai_squads, human_squads)

    first = "Your" if human_goes_first else "AI"
    game_log = [f"Game started! {points_limit}pt match: Tau (AI) vs Blood Angels (You)"]
    game_log.append(f"Turn {engine.turn} - {first} {PHASE_NAMES[engine.current_phase]} phase")


def serialize_state():
    state = engine.state
    tw, th = state.cfg.table_size

    is_human_move = (engine.current_player == HUMAN_PLAYER
                     and engine.current_phase == Phase.MOVEMENT)

    units = []
    for uid, u in state.units.items():
        weapons = [{"name": w.name, "range": w.range, "attacks": w.attacks,
                     "strength": w.strength, "ap": w.ap, "damage": w.damage}
                    for w in u.weapons]
        squad_name = ""
        if u.squad_id >= 0 and u.squad_id in state.squads:
            squad_name = state.squads[u.squad_id].name

        can_move = False
        if is_human_move and u.alive and u.owner == HUMAN_PLAYER:
            can_move = uid not in engine._units_acted

        units.append({
            "id": uid, "name": u.name, "owner": u.owner,
            "x": float(u.pos[0]), "y": float(u.pos[1]),
            "base_r": u.base_r, "height": u.height, "alive": u.alive,
            "M": u.M, "W": u.W, "max_W": u.max_W, "T": u.T,
            "Sv": u.Sv, "OC": u.OC, "is_leader": u.is_leader,
            "squad_id": u.squad_id, "squad_name": squad_name,
            "weapons": weapons,
            "can_move": can_move,
        })

    objectives = [{"x": float(o.pos[0]), "y": float(o.pos[1]), "r": o.r}
                  for o in state.objectives]

    squads = []
    for sid, sq in state.squads.items():
        squads.append({
            "id": sid, "name": sq.name, "owner": sq.owner,
            "points": sq.points, "alive": sq.alive,
            "unit_ids": sq.all_member_ids(),
        })

    is_ai_turn = engine.current_player == AI_PLAYER and not engine.game_over

    ai_remaining = 0
    if is_ai_turn:
        ai_ids = engine.get_player_unit_ids(AI_PLAYER)
        ai_remaining = sum(1 for uid in ai_ids
                           if engine.state.units[uid].alive
                           and uid not in engine._units_acted)

    return {
        "table_w": tw, "table_h": th,
        "turn": engine.turn, "phase": PHASE_NAMES[engine.current_phase],
        "current_player": engine.current_player,
        "vp": {str(k): v for k, v in state.vp.items()},
        "game_over": engine.game_over,
        "winner": engine.winner,
        "units": units,
        "objectives": objectives,
        "squads": squads,
        "is_human_turn": engine.current_player == HUMAN_PLAYER,
        "is_ai_turn": is_ai_turn,
        "ai_actions_remaining": ai_remaining,
        "human_player": HUMAN_PLAYER,
        "ai_player": AI_PLAYER,
        "log": game_log[-40:],
    }


def get_human_actions():
    """Build structured legal actions for the human player."""
    if engine.current_player != HUMAN_PLAYER:
        return []

    pid = HUMAN_PLAYER
    my_ids = engine.get_player_unit_ids(pid)
    enemy_ids = engine.get_player_unit_ids(1 - pid)
    actions = []

    for ui in range(min(len(my_ids), MAX_UNITS_PER_PLAYER)):
        uid = my_ids[ui]
        unit = engine.state.units[uid]
        if not unit.alive:
            continue

        legal = engine.get_legal_unit_actions(ui)

        for atype, targets in legal.items():
            for tidx in targets:
                label = _describe_action(unit, atype, tidx, enemy_ids)
                actions.append({
                    "unit_idx": ui,
                    "action_type": atype,
                    "target_idx": tidx,
                    "unit_id": uid,
                    "unit_name": unit.name,
                    "label": label,
                })

    return actions


def _describe_action(unit, atype, tidx, enemy_ids):
    if atype == Action.PASS:
        return f"{unit.name} - Pass"
    elif atype == Action.MOVE:
        candidates = engine._move_cache.get(unit.id, [])
        if tidx < len(candidates):
            pos = candidates[tidx]
            return f"{unit.name} - Move to ({pos[0]:.1f}, {pos[1]:.1f})"
        return f"{unit.name} - Move #{tidx}"
    elif atype == Action.SHOOT:
        if tidx < len(enemy_ids):
            target = engine.state.units[enemy_ids[tidx]]
            return f"{unit.name} - Shoot {target.name} (W{target.W}/{target.max_W})"
        return f"{unit.name} - Shoot #{tidx}"
    elif atype == Action.CHARGE:
        if tidx < len(enemy_ids):
            target = engine.state.units[enemy_ids[tidx]]
            return f"{unit.name} - Charge {target.name}"
        return f"{unit.name} - Charge #{tidx}"
    elif atype == Action.FIGHT:
        if tidx < len(enemy_ids):
            target = engine.state.units[enemy_ids[tidx]]
            return f"{unit.name} - Fight {target.name} (W{target.W}/{target.max_W})"
        return f"{unit.name} - Fight #{tidx}"
    return f"{unit.name} - ??? {atype}/{tidx}"


def _do_one_ai_action():
    """Compute and execute one AI action. Returns log message or None."""
    if engine.current_player != AI_PLAYER or engine.game_over:
        return None

    if ai_model is not None:
        obs = encode_observation(engine, AI_PLAYER)
        mask = build_action_mask(engine)
        action_raw, _ = ai_model.predict(obs, action_masks=mask)
        action_raw = np.asarray(action_raw)
    else:
        mask = build_action_mask(engine)
        action_raw = random_policy(None, mask)

    action = Action(
        unit_idx=int(action_raw[0]),
        action_type=int(action_raw[1]),
        target_idx=int(action_raw[2]),
    )

    my_ids = engine.get_player_unit_ids(AI_PLAYER)
    enemy_ids = engine.get_player_unit_ids(HUMAN_PLAYER)
    unit_name = "?"
    if action.unit_idx < len(my_ids):
        unit_name = engine.state.units[my_ids[action.unit_idx]].name
    target_name = ""
    if action.action_type in (Action.SHOOT, Action.CHARGE, Action.FIGHT):
        if action.target_idx < len(enemy_ids):
            target_name = engine.state.units[enemy_ids[action.target_idx]].name

    result = engine.execute_action(action)

    msg = None
    atype_name = ACTION_TYPE_NAMES.get(action.action_type, "?")
    if action.action_type != Action.PASS and result.valid:
        target_str = f" → {target_name}" if target_name else ""
        detail = ""
        if result.damage_dealt > 0:
            detail = f" ({result.damage_dealt} dmg"
            if result.units_killed > 0:
                detail += f", {result.units_killed} killed"
            detail += ")"
        elif result.charge_succeeded:
            detail = " (succeeded!)"
        msg = f"[AI] {unit_name}: {atype_name}{target_str}{detail}"
        game_log.append(msg)

    if engine.all_units_acted():
        engine.advance_phase()
        if not engine.game_over:
            game_log.append(
                f"Turn {engine.turn} - "
                f"{'AI' if engine.current_player == AI_PLAYER else 'Your'} "
                f"{PHASE_NAMES[engine.current_phase]} phase"
            )

    return msg


def _skip_ai_phase():
    """Execute all remaining AI actions in the current phase, then stop."""
    if engine.current_player != AI_PLAYER or engine.game_over:
        return

    start_phase = engine.current_phase
    start_player = engine.current_player
    safety = 0
    while (engine.current_player == start_player
           and engine.current_phase == start_phase
           and not engine.game_over
           and safety < 300):
        safety += 1
        _do_one_ai_action()


# ---- Routes ----

@app.route("/")
def index():
    return send_from_directory("web_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("web_static", path)

@app.route("/api/state")
def api_state():
    if engine.state is None:
        return jsonify({"waiting": True})
    return jsonify(serialize_state())

@app.route("/api/actions")
def api_actions():
    if engine.state is None:
        return jsonify([])
    return jsonify(get_human_actions())

@app.route("/api/act", methods=["POST"])
def api_act():
    if engine.state is None:
        return jsonify({"valid": False, "state": {"waiting": True}})
    data = request.json
    action = Action(
        unit_idx=data["unit_idx"],
        action_type=data["action_type"],
        target_idx=data["target_idx"],
    )

    my_ids = engine.get_player_unit_ids(HUMAN_PLAYER)
    enemy_ids = engine.get_player_unit_ids(AI_PLAYER)
    unit_name = "?"
    if action.unit_idx < len(my_ids):
        unit_name = engine.state.units[my_ids[action.unit_idx]].name
    target_name = ""
    if action.action_type in (Action.SHOOT, Action.CHARGE, Action.FIGHT):
        if action.target_idx < len(enemy_ids):
            target_name = engine.state.units[enemy_ids[action.target_idx]].name

    result = engine.execute_action(action)

    atype_name = ACTION_TYPE_NAMES.get(action.action_type, "?")
    if action.action_type != Action.PASS and result.valid:
        target_str = f" → {target_name}" if target_name else ""
        detail = ""
        if result.damage_dealt > 0:
            detail = f" ({result.damage_dealt} dmg"
            if result.units_killed > 0:
                detail += f", {result.units_killed} killed"
            detail += ")"
        elif result.charge_succeeded:
            detail = " (succeeded!)"
        game_log.append(f"[You] {unit_name}: {atype_name}{target_str}{detail}")

    if engine.all_units_acted():
        engine.advance_phase()
        if not engine.game_over:
            game_log.append(
                f"Turn {engine.turn} - "
                f"{'AI' if engine.current_player == AI_PLAYER else 'Your'} "
                f"{PHASE_NAMES[engine.current_phase]} phase"
            )

    return jsonify({
        "valid": result.valid,
        "damage_dealt": result.damage_dealt,
        "units_killed": result.units_killed,
        "state": serialize_state(),
    })


@app.route("/api/ai_step", methods=["POST"])
def api_ai_step():
    """Execute one AI action."""
    if engine.state is None:
        return jsonify({"msg": None, "state": {"waiting": True}})
    if engine.current_player != AI_PLAYER or engine.game_over:
        return jsonify({"msg": None, "state": serialize_state()})

    msg = _do_one_ai_action()
    return jsonify({"msg": msg, "state": serialize_state()})

@app.route("/api/move", methods=["POST"])
def api_move():
    """Human drag-and-drop move: accepts {unit_id, x, y}."""
    if engine.state is None:
        return jsonify({"valid": False, "state": {"waiting": True}})
    if engine.current_player != HUMAN_PLAYER or engine.game_over:
        return jsonify({"valid": False, "reason": "not your turn", "state": serialize_state()})
    if engine.current_phase != Phase.MOVEMENT:
        return jsonify({"valid": False, "reason": "not movement phase", "state": serialize_state()})

    data = request.json
    unit_id = int(data["unit_id"])
    x = float(data["x"])
    y = float(data["y"])

    result = engine.execute_move_xy(unit_id, x, y)

    if result.valid:
        unit = engine.state.units[unit_id]
        game_log.append(f"[You] {unit.name}: Move to ({x:.1f}, {y:.1f})")

    if engine.all_units_acted():
        engine.advance_phase()
        if not engine.game_over:
            game_log.append(
                f"Turn {engine.turn} - "
                f"{'AI' if engine.current_player == AI_PLAYER else 'Your'} "
                f"{PHASE_NAMES[engine.current_phase]} phase"
            )

    return jsonify({
        "valid": result.valid,
        "state": serialize_state(),
    })


@app.route("/api/pass_phase", methods=["POST"])
def api_pass_phase():
    """Pass/skip the current phase (works for either player)."""
    if engine.state is None:
        return jsonify({"waiting": True})
    if engine.game_over:
        return jsonify(serialize_state())

    if engine.current_player == AI_PLAYER:
        _skip_ai_phase()
        return jsonify(serialize_state())

    my_ids = engine.get_player_unit_ids(HUMAN_PLAYER)
    for ui in range(len(my_ids)):
        uid = my_ids[ui]
        if engine.state.units[uid].alive and uid not in engine._units_acted:
            engine.execute_action(Action(unit_idx=ui, action_type=Action.PASS, target_idx=0))

    engine.advance_phase()
    if not engine.game_over:
        game_log.append(
            f"Turn {engine.turn} - "
            f"{'AI' if engine.current_player == AI_PLAYER else 'Your'} "
            f"{PHASE_NAMES[engine.current_phase]} phase"
        )

    return jsonify(serialize_state())

@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.json or {}
    human_first = data.get("human_first", False)
    pts = engine.cfg.points_limit if engine else 1000
    init_game(points_limit=pts,
              seed=int(np.random.randint(0, 2**31)),
              human_goes_first=human_first)
    return jsonify(serialize_state())


def open_browser():
    webbrowser.open("http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WH40K Human vs AI Web UI")
    parser.add_argument("--ai", type=str, default=None,
                        help="Path to trained MaskablePPO model (.zip)")
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.ai:
        from sb3_contrib import MaskablePPO
        print(f"Loading AI model from {args.ai}...")
        ai_model = MaskablePPO.load(args.ai)
        ai_type = "trained"
        print("Model loaded!")
    else:
        print("No model specified, AI will play randomly.")
        ai_type = "random"

    _cfg = GameConfig(points_limit=args.points)
    engine = GameEngine(cfg=_cfg, rng_seed=42)

    if not args.no_browser:
        Timer(1.0, open_browser).start()

    print(f"\nStarting server at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)
