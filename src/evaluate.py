"""
Evaluate a trained model against its environment's success criteria (see
env_registry.py for the per-environment threshold/timing/distance-function
definitions).

Usage:
    python src/evaluate.py --algo sac --model-path models/reacher/sac/seed0/sac_final.zip
    python src/evaluate.py --algo td3 --env-id Pusher-v5 --model-path models/pusher/td3/seed0/td3_final.zip --n-episodes 50

Runs the model for --n-episodes episodes on the raw (unnormalized) env so
that goal-distance can be read directly out of the observation vector, and
reports:
    - reached-at-all rate     (preliminary success criterion)
    - reached-within-time-fraction rate  (main success criterion)
    - average steps-to-first-reach (for episodes that succeeded)
    - average minimum distance achieved

Runs on CPU by default (see train.py's docstring for why).
"""

import argparse
import csv
import os

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from algo_registry import get_algo_class, variant_choices
from env_registry import env_choices, get_env_spec_by_id
from success_metrics import evaluate_episode


def load_vecnormalize_stats(vecnorm_path, env_id, seed):
    """Rebuild a VecNormalize wrapper purely to reuse its obs-normalization
    stats for feeding the policy; the raw env stays separate for metrics."""
    dummy_env = make_vec_env(env_id, n_envs=1, seed=seed)
    vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return vec_normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=variant_choices(), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vecnormalize-path", default=None,
                         help="Path to a saved VecNormalize .pkl, if the model was trained with normalize: true")
    parser.add_argument("--env-id", default="Reacher-v5",
                         help=f"Must be a registered environment (see env_registry.py). "
                              f"Registered short names: {env_choices()}")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=None,
                         help="Overrides the environment's default_max_episode_steps if set")
    parser.add_argument("--seed", type=int, default=42,
                         help="Episode-generation seed -- deliberately independent of the training seed, "
                              "so every model is evaluated against the same set of target/goal positions.")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--csv-out", default=None,
                         help="Optional path to write per-episode results as CSV")
    parser.add_argument("--summary-csv-out", default=None,
                         help="Optional path to APPEND a one-row summary (algo, seed, rates, etc.) -- "
                              "for aggregating many runs, e.g. from run_all.sh, into one comparison table.")
    parser.add_argument("--train-seed", type=int, default=None,
                         help="Which training seed this model came from, recorded in --summary-csv-out only.")
    args = parser.parse_args()

    env_spec = get_env_spec_by_id(args.env_id)
    distance_fn = env_spec["distance_fn"]
    threshold = env_spec["success_threshold"]
    time_fraction = env_spec["success_time_fraction"]
    max_episode_steps = args.max_episode_steps or env_spec["default_max_episode_steps"]

    algo_cls = get_algo_class(args.algo)
    model = algo_cls.load(args.model_path, device=args.device)

    obs_normalizer = None
    if args.vecnormalize_path:
        obs_normalizer = load_vecnormalize_stats(args.vecnormalize_path, args.env_id, args.seed)

    env = gym.make(args.env_id)

    episode_results = []
    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        distances = [distance_fn(obs)]
        done = False
        truncated = False
        while not (done or truncated):
            predict_obs = obs
            if obs_normalizer is not None:
                predict_obs = obs_normalizer.normalize_obs(obs)
            action, _ = model.predict(predict_obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            distances.append(distance_fn(obs))

        result = evaluate_episode(
            distances,
            max_episode_steps=max_episode_steps,
            threshold=threshold,
            time_fraction=time_fraction,
        )
        result["episode"] = ep
        episode_results.append(result)

    env.close()

    n = len(episode_results)
    reached_at_all = sum(r["reached_at_all"] for r in episode_results)
    reached_in_time = sum(r["reached_within_fraction"] for r in episode_results)
    reach_steps = [r["first_reach_step"] for r in episode_results if r["first_reach_step"] is not None]
    min_distances = [r["min_distance"] for r in episode_results]
    final_distances = [r["final_distance"] for r in episode_results]

    reached_at_all_rate = reached_at_all / n
    reached_in_time_rate = reached_in_time / n
    avg_reach_step = float(np.mean(reach_steps)) if reach_steps else None
    avg_min_distance = float(np.mean(min_distances))
    avg_final_distance = float(np.mean(final_distances))

    print(f"\n=== {args.algo.upper()} on {args.env_id} ({n} episodes) ===")
    print(f"Reached target at all:                {reached_at_all}/{n} ({100*reached_at_all_rate:.1f}%)")
    print(f"Reached within {int(time_fraction*100)}% of episode:      {reached_in_time}/{n} ({100*reached_in_time_rate:.1f}%)")
    if reach_steps:
        print(f"Avg. steps to first reach (successes only): {avg_reach_step:.1f}")
    print(f"Avg. minimum distance achieved:       {avg_min_distance:.4f}")
    print(f"Avg. final distance (at episode end): {avg_final_distance:.4f}")

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_results[0].keys()))
            writer.writeheader()
            writer.writerows(episode_results)
        print(f"Per-episode results written to {args.csv_out}")

    if args.summary_csv_out:
        write_header = not os.path.exists(args.summary_csv_out)
        os.makedirs(os.path.dirname(args.summary_csv_out) or ".", exist_ok=True)
        with open(args.summary_csv_out, "a", newline="") as f:
            fieldnames = ["algo", "env_id", "train_seed", "n_episodes", "reached_at_all_rate",
                          "reached_within_time_fraction_rate", "avg_steps_to_reach",
                          "avg_min_distance", "avg_final_distance"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "algo": args.algo,
                "env_id": args.env_id,
                "train_seed": args.train_seed if args.train_seed is not None else "",
                "n_episodes": n,
                "reached_at_all_rate": reached_at_all_rate,
                "reached_within_time_fraction_rate": reached_in_time_rate,
                "avg_steps_to_reach": avg_reach_step if avg_reach_step is not None else "",
                "avg_min_distance": avg_min_distance,
                "avg_final_distance": avg_final_distance,
            })
        print(f"Summary row appended to {args.summary_csv_out}")


if __name__ == "__main__":
    main()
