"""
Battlefield visualization utility.
Renders the game state using matplotlib for debugging and observation.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon as MplPolygon

from wh_types import GameState


def render_game_state(state: GameState, title: str = "",
                      ax: Optional[plt.Axes] = None,
                      show: bool = True) -> plt.Axes:
    """
    Render the current GameState as a 2D top-down view.
    Shows unit positions (colored by owner), terrain, and objectives.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    tw, th = state.cfg.table_size
    ax.set_xlim(0, tw)
    ax.set_ylim(0, th)
    ax.set_aspect("equal")
    ax.set_facecolor("#e8e8e0")

    for tp in state.terrain:
        xs, ys = tp.polygon.exterior.xy
        poly = MplPolygon(list(zip(xs, ys)),
                          facecolor="gray" if tp.blocks_los else "tan",
                          edgecolor="black", alpha=0.6, linewidth=1)
        ax.add_patch(poly)

    for obj in state.objectives:
        c = Circle(obj.pos, obj.r, facecolor="gold", edgecolor="black",
                   alpha=0.3, linewidth=1.5, linestyle="--")
        ax.add_patch(c)

    colors = {0: "#3366cc", 1: "#cc3333"}
    for uid, unit in state.units.items():
        if not unit.alive:
            continue
        color = colors[unit.owner]
        c = Circle(unit.pos, unit.base_r, facecolor=color,
                   edgecolor="white", alpha=0.8, linewidth=0.8)
        ax.add_patch(c)
        ax.text(unit.pos[0], unit.pos[1], str(unit.W),
                ha="center", va="center", fontsize=6,
                color="white", fontweight="bold")

    p0_patch = mpatches.Patch(color=colors[0], label="Player 0")
    p1_patch = mpatches.Patch(color=colors[1], label="Player 1")
    obj_patch = mpatches.Patch(color="gold", alpha=0.3, label="Objective")
    ax.legend(handles=[p0_patch, p1_patch, obj_patch], loc="upper right", fontsize=8)

    phase_str = f"Turn {state.turn} | Phase: {state.phase}"
    vp_str = f"VP: P0={state.vp[0]} P1={state.vp[1]}"
    ax.set_title(f"{title}  {phase_str}  {vp_str}" if title else f"{phase_str}  {vp_str}")

    ax.grid(True, alpha=0.2)

    if show:
        plt.tight_layout()
        plt.show()

    return ax
