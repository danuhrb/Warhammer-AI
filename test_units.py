"""
Test script to make sure unit_data.py works correctly
Run this first to verify everything is set up properly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from unit_data import (
    create_tau_army, create_ultramarine_army, create_blood_angels_army,
    army_points,
)


def _print_army(label, units, squads, pts_cap):
    total_pts = army_points(squads)
    print(f"\n{'='*60}")
    print(f" {label}: {len(units)} models, {len(squads)} squads, "
          f"{total_pts}/{pts_cap} pts")
    print(f"{'='*60}")
    for sq in squads:
        leader_name = "none"
        if sq.leader_id is not None:
            leader_name = next(u.name for u in units if u.id == sq.leader_id)
        member_count = len(sq.unit_ids)
        total = member_count + (1 if sq.leader_id is not None else 0)
        print(f"  {sq.name}: {total} models, {sq.points} pts"
              f" (leader={leader_name})")

    print("  Sample units:")
    for unit in units[:5]:
        wpns = ", ".join(w.name for w in unit.weapons)
        print(f"    {unit.name}: M{unit.M}\" W{unit.W}/{unit.max_W} T{unit.T} "
              f"Sv{unit.Sv}+ OC{unit.OC} [{wpns}]")


def test_unit_data():
    for pts_cap in [500, 1000, 2000]:
        print(f"\n{'#'*60}")
        print(f"  POINTS CAP: {pts_cap}")
        print(f"{'#'*60}")

        tau_u, tau_s = create_tau_army(owner=0, points_limit=pts_cap)
        _print_army("Tau Empire", tau_u, tau_s, pts_cap)
        assert army_points(tau_s) <= pts_cap, "Tau exceeded points cap!"

        ba_u, ba_s = create_blood_angels_army(owner=1, points_limit=pts_cap)
        _print_army("Blood Angels", ba_u, ba_s, pts_cap)
        assert army_points(ba_s) <= pts_cap, "BA exceeded points cap!"

        um_u, um_s = create_ultramarine_army(owner=1, points_limit=pts_cap)
        _print_army("Ultramarines", um_u, um_s, pts_cap)
        assert army_points(um_s) <= pts_cap, "UM exceeded points cap!"

    print(f"\nAll armies respect points caps!")


if __name__ == "__main__":
    test_unit_data()
