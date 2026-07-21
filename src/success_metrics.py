"""
Generic success-criteria evaluation logic, shared across every environment
in this project. Per-environment specifics (the distance-to-goal function,
success threshold, timing fraction, default episode length) live in
env_registry.py -- this file only implements the pass/fail logic once
those numbers are given to it, so Reacher and Pusher (and anything added
later) can't quietly disagree about what "success" means to compute.
"""

import numpy as np


def evaluate_episode(distances_per_step, max_episode_steps: int,
                      threshold: float, time_fraction: float) -> dict:
    """
    Given the per-step goal-distances for one episode, return a dict
    describing whether/when the target was reached.

    distances_per_step: list/array of distances, one per environment step.
    max_episode_steps, threshold, time_fraction: from env_registry.get_env_spec().
    """
    distances_per_step = np.asarray(distances_per_step)
    reached_steps = np.where(distances_per_step <= threshold)[0]

    reached_at_all = reached_steps.size > 0
    first_reach_step = int(reached_steps[0]) if reached_at_all else None

    time_limit = time_fraction * max_episode_steps
    reached_in_time = reached_at_all and first_reach_step <= time_limit

    return {
        "reached_at_all": reached_at_all,
        "first_reach_step": first_reach_step,
        "reached_within_fraction": reached_in_time,
        "min_distance": float(distances_per_step.min()),
        "final_distance": float(distances_per_step[-1]),
    }
