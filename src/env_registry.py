"""
Per-environment registry, analogous to algo_registry.py. Each environment
plugs in its own goal-distance function (read straight out of that
environment's observation vector) and its own success threshold / timing
fraction / default episode length. The generic pass/fail logic in
success_metrics.evaluate_episode() is identical across environments --
only these specifics differ.

Reacher: fingertip-to-target distance, obs[8:10].
Pusher:  object-to-goal distance, obs[17:20] (object xyz) minus
         obs[20:23] (goal xyz). Indices taken directly from the Gymnasium
         docs (https://gymnasium.farama.org/environments/mujoco/pusher/),
         not guessed.

Short names ("reacher", "pusher") are used throughout the project for
directory naming (models/<short>/<algo>/seed<N>/, configs/<short>/<algo>.yml,
etc.) so paths don't have to embed the full "Reacher-v5" / "Pusher-v5" string.
"""

import numpy as np


def _reacher_distance(obs: np.ndarray) -> float:
    return float(np.linalg.norm(obs[8:10]))


def _pusher_distance(obs: np.ndarray) -> float:
    # xy only, deliberately -- obs[17:20] (object) and obs[20:23] (goal)
    # differ by a fixed ~0.048m in z even when perfectly centered, since
    # the object's body origin sits at its resting height above the table
    # (roughly its own radius) while the goal marker is flush with the
    # table surface. That z-gap is constant regardless of placement
    # quality, so a full 3D distance asymptotes at ~0.05m no matter how
    # well the policy centers the object -- confirmed empirically by
    # checking a real trained model's obs directly (xy-distance 0.027 vs.
    # full 3D distance 0.055, for a placement that looked well-centered on
    # video). Comparing xy only measures what actually matters here: is
    # the object over the target, not how tall the object happens to be.
    return float(np.linalg.norm(obs[17:19] - obs[20:22]))


ENVS = {
    "reacher": {
        "env_id": "Reacher-v5",
        "distance_fn": _reacher_distance,
        "success_threshold": 0.02,
        "success_time_fraction": 0.4,
        "default_max_episode_steps": 50,
    },
    "pusher": {
        "env_id": "Pusher-v5",
        "distance_fn": _pusher_distance,
        # Tighter than the cylinder's own size on purpose -- this is meant
        # to require real placement precision, not just "in the vicinity".
        # Object starts >0.17m from goal (enforced by the env itself), so
        # 0.02m is a meaningfully high bar. May be tightened further later.
        "success_threshold": 0.02,
        # Set to the full episode (1.0), not a tighter fraction like
        # Reacher's -- empirically, even the best-performing runs took
        # ~85-90 of Pusher's 100 steps to reach the threshold (TD3 seed 2,
        # 86% success, averaged step 83.8), so a tighter timing bar was
        # rejecting genuinely successful placements for being "too slow"
        # on a task whose physics can't realistically go faster. Pusher's
        # "main" and "preliminary" success criteria are therefore
        # numerically identical now -- positioning accuracy (did it reach
        # the threshold at all) is the criterion that matters here, not
        # timing.
        "success_time_fraction": 1.0,
        "default_max_episode_steps": 100,
    },
}

ENV_ID_TO_SHORT = {spec["env_id"]: name for name, spec in ENVS.items()}


def get_env_short_name(env_id: str) -> str:
    if env_id not in ENV_ID_TO_SHORT:
        raise ValueError(f"Unknown env_id '{env_id}'. Known: {list(ENV_ID_TO_SHORT)}")
    return ENV_ID_TO_SHORT[env_id]


def get_env_spec(env_short_name: str) -> dict:
    return ENVS[env_short_name]


def get_env_spec_by_id(env_id: str) -> dict:
    return ENVS[get_env_short_name(env_id)]


def env_choices():
    return list(ENVS.keys())
