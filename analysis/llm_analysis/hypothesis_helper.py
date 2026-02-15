"""
Hypothesis helper for computing number of valid hypotheses from state trajectory.
Used by reasoning_models.py. Implements full (subset, rule) hypothesis space.
"""

import numpy as np


def all_subsets(n):
    """All subsets of {0, ..., n-1} including empty set."""
    out = []
    for b in range(0, 1 << n):
        out.append(frozenset(i for i in range(n) if (b >> i) & 1))
    return out


def all_full_hypotheses(n):
    """All (subset, rule) pairs: subset of objects, conjunctive or disjunctive."""
    subsets = all_subsets(n)
    out = []
    for s in subsets:
        out.append((s, "conjunctive"))
        out.append((s, "disjunctive"))
    return out


def consistent_with_test(hyp, objects_on, machine_lit):
    """Check if hypothesis (subset, rule) is consistent with (objects_on_machine, machine_lit)."""
    s, rule = hyp
    o = set(objects_on) if isinstance(objects_on, (list, tuple)) else set()
    if rule == "conjunctive":
        pred_on = s <= o  # all blickets on machine
    else:
        pred_on = bool(s & o)  # at least one blicket on machine
    return pred_on == bool(machine_lit)


def compute_num_valid_hypothesis(state_traj):
    """
    Compute number of hypotheses (subset, rule) consistent with state trajectory.
    state_traj: array of shape [T, num_objects+1]. Each row: [obj0_on, ..., objN_on, machine_lit].
    Returns: int, number of hypotheses remaining after last observation.
    """
    state_traj = np.asarray(state_traj)
    if state_traj.size == 0:
        return np.nan
    n_objects = state_traj.shape[1] - 1  # last column is machine state
    hyps_full = all_full_hypotheses(n_objects)
    remaining = set(hyps_full)
    for i in range(state_traj.shape[0]):
        row = state_traj[i]
        objects_on = set(j for j in range(n_objects) if row[j])
        machine_lit = bool(row[n_objects])
        remaining = {h for h in remaining if consistent_with_test(h, objects_on, machine_lit)}
    return len(remaining)
