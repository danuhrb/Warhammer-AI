"""Random legal action bot -- baseline opponent for initial training."""
from __future__ import annotations
import numpy as np


def random_policy(obs: np.ndarray, mask: np.ndarray,
                  nvec: np.ndarray | None = None) -> np.ndarray:
    """
    Sample a uniformly random legal action from the mask.
    Works with the MultiDiscrete [units, action_types, targets] layout.

    Parameters
    ----------
    obs   : observation vector (unused, kept for interface consistency)
    mask  : flat action mask (length = sum(nvec))
    nvec  : the MultiDiscrete sizes; defaults to [14, 5, 24]
    """
    if nvec is None:
        from env.action_codec import MAX_TARGETS, NUM_ACTION_TYPES
        from game_engine import MAX_UNITS_PER_PLAYER
        nvec = np.array([MAX_UNITS_PER_PLAYER, NUM_ACTION_TYPES, MAX_TARGETS])

    def _sample_segment(m):
        valid = np.where(m > 0)[0]
        return int(np.random.choice(valid)) if len(valid) > 0 else 0

    seg1, seg2 = int(nvec[0]), int(nvec[1])
    unit_mask = mask[:seg1]
    type_mask = mask[seg1:seg1 + seg2]
    target_mask = mask[seg1 + seg2:]

    return np.array([
        _sample_segment(unit_mask),
        _sample_segment(type_mask),
        _sample_segment(target_mask),
    ])
