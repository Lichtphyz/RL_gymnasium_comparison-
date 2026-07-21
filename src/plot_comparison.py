"""
Plot comparisons of training progress across "conditions" -- each
condition is an algorithm, optionally with a --run-tag pointing at an
experimental variant (e.g. ddpg trained with more exploration noise). This
is what lets you compare stock DDPG against a noise-ablated DDPG, or the
whole 4-algorithm stock lineup against a reward-shaped lineup, in one call.

Usage:
    python src/plot_comparison.py
    python src/plot_comparison.py --env-id Pusher-v5 --algos ddpg,td3,sac,ppo --seeds 0,1,2

    # Compare stock DDPG against the noise-ablation experiment, same call:
    python src/plot_comparison.py --env-id Pusher-v5 --algos ddpg,ddpg:noise03 --seeds 0,1,2

    # Compare all four stock algorithms against all four reward-shaped ones:
    python src/plot_comparison.py --env-id Pusher-v5 \\
        --algos ddpg,td3,sac,ppo,ddpg:distweight2,td3:distweight2,sac:distweight2,ppo:distweight2 \\
        --seeds 0,1,2

Each --algos entry is "algo" (stock, uses --run-tag if given globally) or
"algo:tag" (that specific tag, overriding --run-tag for this entry only).
Same algorithm appearing twice gets the same base color but a different
linestyle (solid = no tag, dashed = tagged) so stock-vs-variant comparisons
stay visually grouped; different algorithms keep their own distinct colors
regardless of tag.

Looks for TensorBoard logs under logs/<env-short>/<algo>/seed<seed>[_<tag>]/,
captured stdout logs under logs/run_all_logs/<env-short>_<algo>_seed<seed>[_<tag>]_train.log,
and checkpoint files under models/<env-short>/<algo>/seed<seed>[_<tag>]/<algo>_*_steps.zip.
Missing combinations are skipped with a warning rather than failing the
whole script.

Produces four files under plots/<env-short>/[<run-tag>/]:

  training_comparison.png -- 2 stacked panels:
    1. mean episode reward vs. % of training budget (mean +/- std)
    2. mean episode reward vs. actual wall-clock training time in minutes

  goal_reaching_time.png -- 1 panel: avg. steps-to-first-reach vs. % of
  training budget (gaps = ~0% success across all seeds at that checkpoint).
  Split out from training_comparison.png so that figure doesn't grow a
  third panel every time this script changes.

  success_rate_vs_time.png -- 2 stacked panels, both vs. wall-clock time:
    1. success rate within the environment's timing fraction (main criterion)
    2. success rate of reaching the goal at all, by end of episode
       (preliminary criterion)

  training_comparison_data.csv -- one row per checkpoint per seed per
  condition (algo + tag), with step, wall-clock time, both success rates,
  reach time, and both distance metrics.

NOTE on wall-clock-time panels: only a fair comparison across conditions
if every run had the same CPU thread budget and none of them competed for
cores with something else running at the same time (run_all.sh enforces
this by running everything sequentially with a fixed thread cap). This
caveat used to be baked into every wall-clock panel's title; it's printed
once to the console instead now, to keep titles from running long.

NOTE on PPO normalization: PPO's checkpoints are evaluated using the FINAL
saved VecNormalize stats (train.py only saves them once, at the end), which
is an approximation for early PPO checkpoints -- see train.py/evaluate.py
docstrings for more.
"""

import argparse
import csv
import glob
import os
import re

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from algo_registry import get_algo_class, variant_choices
from env_registry import env_choices, get_env_short_name, get_env_spec_by_id
from success_metrics import evaluate_episode

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_EVAL_EPISODES_PER_CHECKPOINT = 10  # kept small since this runs many checkpoints x seeds

ALGO_LABELS = {
    "ddpg": "DDPG", "td3": "TD3", "sac": "SAC", "ppo": "PPO",
    "ddpg_1critic": "DDPG (1 critic)",
}
ALGO_COLORS = {
    "ddpg": "#d62728", "td3": "#1f77b4", "sac": "#2ca02c",
    "ppo": "#9467bd", "ddpg_1critic": "#ff7f0e",
}
DEFAULT_ALGO_ORDER = ["ddpg", "td3", "sac", "ppo"]


def seed_folder_name(seed, run_tag=None):
    """Matches train.py's convention: seed<N>, or seed<N>_<tag> for
    experimental runs launched with --run-tag."""
    return f"seed{seed}" + (f"_{run_tag}" if run_tag else "")


def parse_condition(entry, global_run_tag):
    """Parses one --algos entry: 'algo' or 'algo:tag'. An explicit tag
    overrides --run-tag for this entry only; no tag falls back to
    --run-tag (which may itself be None, i.e. stock). Returns
    (algo, effective_tag, condition_key, label, color, linestyle)."""
    parts = entry.split(":", 1)
    algo = parts[0].strip()
    explicit_tag = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    tag = explicit_tag if explicit_tag is not None else global_run_tag

    condition_key = f"{algo}:{tag}" if tag else algo
    base_label = ALGO_LABELS.get(algo, algo.upper())
    label = f"{base_label} ({tag})" if tag else base_label
    color = ALGO_COLORS.get(algo, None)
    linestyle = "--" if tag else "-"
    return algo, tag, condition_key, label, color, linestyle


# ---------------------------------------------------------------------------
# TensorBoard reward curves (panel 1)
# ---------------------------------------------------------------------------

def find_best_tb_run(env_short, algo, seed, run_tag=None):
    """logs/<env-short>/<algo>/seed<seed>[_<tag>]/ may contain multiple runs
    (e.g. reruns). Pick the one with the most logged steps, on the
    assumption that's the real full training run."""
    seed_dir = os.path.join(REPO_ROOT, "logs", env_short, algo, seed_folder_name(seed, run_tag))
    run_dirs = sorted(glob.glob(os.path.join(seed_dir, f"{algo}_*")))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    if not run_dirs:
        return None

    best_dir, best_max_step = None, -1
    for d in run_dirs:
        event_files = glob.glob(os.path.join(d, "events.out.tfevents.*"))
        if not event_files:
            continue
        try:
            ea = EventAccumulator(d)
            ea.Reload()
            if "rollout/ep_rew_mean" not in ea.Tags().get("scalars", []):
                continue
            events = ea.Scalars("rollout/ep_rew_mean")
            if not events:
                continue
            max_step = events[-1].step
            if max_step > best_max_step:
                best_max_step = max_step
                best_dir = d
        except Exception:
            continue
    return best_dir


def load_scalar(run_dir, tag):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def aggregate_reward_curves(env_short, algo, seeds, run_tag=None):
    """Returns (pct_grid, mean, std) interpolated onto a common 0-100%
    grid, or (None, None, None) if no seeds had usable data."""
    pct_grid = np.linspace(0, 100, 200)
    interpolated = []
    for seed in seeds:
        run_dir = find_best_tb_run(env_short, algo, seed, run_tag)
        if run_dir is None:
            sf = seed_folder_name(seed, run_tag)
            print(f"  [!] {algo} {sf}: no usable TensorBoard run under logs/{env_short}/{algo}/{sf}/")
            continue
        steps, rewards = load_scalar(run_dir, "rollout/ep_rew_mean")
        if steps is None or len(steps) == 0:
            continue
        pct = 100 * steps / steps[-1]
        interp_rewards = np.interp(pct_grid, pct, rewards)
        interpolated.append(interp_rewards)
        print(f"  {algo} {seed_folder_name(seed, run_tag)}: reward curve loaded "
              f"({len(steps)} points, run {os.path.basename(run_dir)})")

    if not interpolated:
        return None, None, None
    stacked = np.stack(interpolated, axis=0)
    return pct_grid, stacked.mean(axis=0), stacked.std(axis=0)


# ---------------------------------------------------------------------------
# Captured-stdout-log parsing (panel 2, and the step -> wall-clock-time
# mapping used by the checkpoint-based plots). Self-contained -- doesn't
# depend on which fields SB3 happens to write to TensorBoard.
# ---------------------------------------------------------------------------

def parse_log_table(env_short, algo, seed, run_tag=None):
    """Parse every SB3 progress-table block in a captured stdout log for
    (total_timesteps, time_elapsed, ep_rew_mean) triples. ep_rew_mean is
    NaN for blocks that don't have it -- those rows are still useful for
    the step->time mapping.

    Log filename convention: {env_short}_{algo}_seed{seed}_train.log for
    stock runs (matches run_all.sh), or {env_short}_{algo}_seed{seed}_{tag}_train.log
    for tagged experimental runs (matches manually-launched --run-tag runs)."""
    suffix = f"_{run_tag}" if run_tag else ""
    log_path = os.path.join(REPO_ROOT, "logs", "run_all_logs",
                             f"{env_short}_{algo}_seed{seed}{suffix}_train.log")
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", errors="ignore") as f:
        content = f.read()

    blocks = re.split(r"(?m)^-{5,}\s*$", content)
    steps, times, rewards = [], [], []
    for block in blocks:
        s_match = re.search(r"total_timesteps\s*\|\s*(\d+)", block)
        t_match = re.search(r"time_elapsed\s*\|\s*(\d+)", block)
        if s_match and t_match:
            r_match = re.search(r"ep_rew_mean\s*\|\s*(-?\d+\.?\d*)", block)
            steps.append(int(s_match.group(1)))
            times.append(float(t_match.group(1)))
            rewards.append(float(r_match.group(1)) if r_match else np.nan)

    if not steps:
        return None
    steps, times, rewards = np.array(steps), np.array(times), np.array(rewards)
    order = np.argsort(steps)
    return steps[order], times[order], rewards[order]


def aggregate_reward_vs_time(env_short, algo, seeds, run_tag=None):
    """Returns (time_grid_minutes, mean, std), grid extending only to the
    shortest seed's finish time."""
    curves = []
    max_common_time = None
    for seed in seeds:
        parsed = parse_log_table(env_short, algo, seed, run_tag)
        if parsed is None:
            print(f"  [!] {algo} {seed_folder_name(seed, run_tag)}: no reward-vs-time data found in logs/run_all_logs/")
            continue
        _, times, rewards = parsed
        valid = ~np.isnan(rewards)
        if not valid.any():
            continue
        t, r = times[valid], rewards[valid]
        curves.append((t, r))
        max_common_time = t[-1] if max_common_time is None else min(max_common_time, t[-1])

    if not curves:
        return None, None, None

    time_grid = np.linspace(0, max_common_time, 200)
    interpolated = [np.interp(time_grid, t, r) for t, r in curves]
    stacked = np.stack(interpolated, axis=0)
    return time_grid / 60.0, stacked.mean(axis=0), stacked.std(axis=0)  # minutes


# ---------------------------------------------------------------------------
# Checkpoint evaluation (shared by the reach-time panel and the
# success-rate-vs-time figure -- evaluated once per condition/seed and
# cached).
# ---------------------------------------------------------------------------

def find_checkpoints(env_short, algo, seed, run_tag=None):
    pattern = os.path.join(REPO_ROOT, "models", env_short, algo, seed_folder_name(seed, run_tag), f"{algo}_*_steps.zip")
    files = glob.glob(pattern)
    checkpoints = []
    for f in files:
        match = re.search(rf"{algo}_(\d+)_steps\.zip$", os.path.basename(f))
        if match:
            checkpoints.append((int(match.group(1)), f))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def get_vecnormalize_for_ppo(env_short, env_id, algo, seed, run_tag=None):
    vecnorm_path = os.path.join(REPO_ROOT, "models", env_short, algo, seed_folder_name(seed, run_tag), f"{algo}_vecnormalize.pkl")
    if not os.path.exists(vecnorm_path):
        return None
    dummy_env = make_vec_env(env_id, n_envs=1, seed=0)
    vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return vec_normalize


def evaluate_checkpoint(algo, model_path, obs_normalizer, env, distance_fn,
                         max_episode_steps, threshold, time_fraction, n_episodes, seed=42):
    """Runs n_episodes through one checkpoint and returns
    (success_rate_within_time_fraction, reached_at_all_rate, avg_steps_to_reach,
    avg_final_distance, avg_min_distance)."""
    algo_cls = get_algo_class(algo)
    model = algo_cls.load(model_path, device="cpu")

    within_time_count = 0
    reached_at_all_count = 0
    reach_steps = []
    final_distances = []
    min_distances = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        distances = [distance_fn(obs)]
        done, truncated = False, False
        while not (done or truncated):
            predict_obs = obs_normalizer.normalize_obs(obs) if obs_normalizer is not None else obs
            action, _ = model.predict(predict_obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            distances.append(distance_fn(obs))

        result = evaluate_episode(
            distances,
            max_episode_steps=max_episode_steps,
            threshold=threshold,
            time_fraction=time_fraction,
        )
        if result["reached_at_all"]:
            reached_at_all_count += 1
        if result["reached_within_fraction"]:
            within_time_count += 1
            reach_steps.append(result["first_reach_step"])
        final_distances.append(result["final_distance"])
        min_distances.append(result["min_distance"])

    success_rate = within_time_count / n_episodes
    reached_at_all_rate = reached_at_all_count / n_episodes
    avg_reach_step = float(np.mean(reach_steps)) if reach_steps else None
    avg_final_distance = float(np.mean(final_distances))
    avg_min_distance = float(np.mean(min_distances))
    return success_rate, reached_at_all_rate, avg_reach_step, avg_final_distance, avg_min_distance


def evaluate_checkpoints_for_seed(env_short, env_id, algo, seed, env_spec, run_tag=None):
    """Evaluates every saved checkpoint for one condition/seed and attaches
    an estimated wall-clock time (interpolated from the captured training
    log) to each. Returns a list of dicts sorted by step, or [] if nothing
    was found."""
    sf = seed_folder_name(seed, run_tag)
    checkpoints = find_checkpoints(env_short, algo, seed, run_tag)
    if not checkpoints:
        print(f"  [!] {algo} {sf}: no checkpoint files under models/{env_short}/{algo}/{sf}/")
        return []

    log_parsed = parse_log_table(env_short, algo, seed, run_tag)
    obs_normalizer = get_vecnormalize_for_ppo(env_short, env_id, algo, seed, run_tag) if algo == "ppo" else None
    env = gym.make(env_id)
    print(f"  {algo} {sf}: evaluating {len(checkpoints)} checkpoints "
          f"({N_EVAL_EPISODES_PER_CHECKPOINT} episodes each)...")

    results = []
    for step, path in checkpoints:
        success_rate, reached_at_all_rate, avg_reach, avg_final_distance, avg_min_distance = evaluate_checkpoint(
            algo, path, obs_normalizer, env, env_spec["distance_fn"],
            env_spec["default_max_episode_steps"], env_spec["success_threshold"],
            env_spec["success_time_fraction"], N_EVAL_EPISODES_PER_CHECKPOINT,
        )
        wall_clock_min = None
        if log_parsed is not None:
            log_steps, log_times, _ = log_parsed
            wall_clock_min = float(np.interp(step, log_steps, log_times)) / 60.0
        results.append({
            "step": step,
            "wall_clock_min": wall_clock_min,
            "success_rate_within_time_fraction": success_rate,
            "reached_at_all_rate": reached_at_all_rate,
            "avg_steps_to_reach": avg_reach,
            "avg_final_distance": avg_final_distance,
            "avg_min_distance": avg_min_distance,
        })
    env.close()
    return results


def aggregate_reach_time_curves(algo, tag, seeds, checkpoint_cache, csv_rows):
    """(pct_grid, mean_reach_time, std_reach_time) aligned across seeds by
    matching checkpoint step values. Also appends full rows to csv_rows."""
    condition_key = f"{algo}:{tag}" if tag else algo
    per_seed_data = {}
    per_seed_success = {}

    for seed in seeds:
        results = checkpoint_cache.get((condition_key, seed), [])
        if not results:
            continue
        total_steps = results[-1]["step"]
        for r in results:
            per_seed_success.setdefault(r["step"], []).append(r["success_rate_within_time_fraction"])
            if r["avg_steps_to_reach"] is not None:
                per_seed_data.setdefault(r["step"], []).append(r["avg_steps_to_reach"])
            csv_rows.append({
                "algo": algo, "run_tag": tag or "", "seed": seed, "timesteps": r["step"],
                "pct_of_training": 100 * r["step"] / total_steps,
                "wall_clock_min": r["wall_clock_min"] if r["wall_clock_min"] is not None else "",
                "success_rate_within_time_fraction": r["success_rate_within_time_fraction"],
                "reached_at_all_rate": r["reached_at_all_rate"],
                "avg_steps_to_reach": r["avg_steps_to_reach"] if r["avg_steps_to_reach"] is not None else "",
                "avg_final_distance": r["avg_final_distance"],
                "avg_min_distance": r["avg_min_distance"],
            })

    if not per_seed_success:
        return None, None, None

    steps_sorted = sorted(per_seed_success.keys())
    means, stds = [], []
    for step in steps_sorted:
        vals = per_seed_data.get(step, [])
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        else:
            means.append(np.nan)
            stds.append(0.0)

    total_steps = steps_sorted[-1]
    pct = 100 * np.array(steps_sorted) / total_steps
    return pct, np.array(means), np.array(stds)


def aggregate_metric_vs_pct(algo, tag, seeds, checkpoint_cache, metric_key):
    """Generic (pct_grid, mean, std) aligned across seeds by matching
    checkpoint step values, for any always-defined per-checkpoint metric
    (e.g. avg_final_distance, avg_min_distance)."""
    condition_key = f"{algo}:{tag}" if tag else algo
    per_seed_data = {}
    for seed in seeds:
        for r in checkpoint_cache.get((condition_key, seed), []):
            per_seed_data.setdefault(r["step"], []).append(r[metric_key])

    if not per_seed_data:
        return None, None, None

    steps_sorted = sorted(per_seed_data.keys())
    means = np.array([np.mean(per_seed_data[s]) for s in steps_sorted])
    stds = np.array([np.std(per_seed_data[s]) if len(per_seed_data[s]) > 1 else 0.0 for s in steps_sorted])
    total_steps = steps_sorted[-1]
    pct = 100 * np.array(steps_sorted) / total_steps
    return pct, means, stds


def aggregate_metric_vs_time(algo, tag, seeds, checkpoint_cache, metric_key):
    """(time_grid_minutes, mean, std) for any per-checkpoint metric."""
    condition_key = f"{algo}:{tag}" if tag else algo
    curves = []
    max_common_time = None
    for seed in seeds:
        results = [r for r in checkpoint_cache.get((condition_key, seed), [])
                   if r["wall_clock_min"] is not None]
        if not results:
            continue
        times = np.array([r["wall_clock_min"] for r in results])
        values = np.array([r[metric_key] for r in results])
        curves.append((times, values))
        max_common_time = times[-1] if max_common_time is None else min(max_common_time, times[-1])

    if not curves:
        return None, None, None

    time_grid = np.linspace(0, max_common_time, 100)
    interpolated = [np.interp(time_grid, t, v) for t, v in curves]
    stacked = np.stack(interpolated, axis=0)
    return time_grid, stacked.mean(axis=0), stacked.std(axis=0)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="Reacher-v5",
                         help=f"Registered short names: {env_choices()}")
    parser.add_argument("--algos", default=",".join(DEFAULT_ALGO_ORDER),
                         help="Comma-separated conditions: 'algo' or 'algo:tag'. "
                              "E.g. 'ddpg,ddpg:noise03' compares stock DDPG against the noise-ablation run.")
    parser.add_argument("--seeds", default="0,1,2",
                         help="Comma-separated list of training seeds to aggregate across")
    parser.add_argument("--run-tag", default=None,
                         help="Fallback tag applied to any --algos entry that doesn't specify its own "
                              "(e.g. 'distweight2'). Output goes to plots/<env>/<tag>/ if set and every "
                              "condition shares it; otherwise plots/<env>/.")
    args = parser.parse_args()

    env_short = get_env_short_name(args.env_id)
    env_spec = get_env_spec_by_id(args.env_id)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    conditions = []  # (algo, tag, condition_key, label, color, linestyle)
    for entry in [e.strip() for e in args.algos.split(",") if e.strip()]:
        algo, tag, condition_key, label, color, linestyle = parse_condition(entry, args.run_tag)
        if algo not in variant_choices():
            raise ValueError(f"Unknown algo '{algo}' (from entry '{entry}'). Choices: {variant_choices()}")
        conditions.append((algo, tag, condition_key, label, color, linestyle))

    all_tags = {tag for _, tag, *_ in conditions}
    common_tag = args.run_tag if (args.run_tag and all_tags == {args.run_tag}) else None
    plots_dir = (os.path.join(REPO_ROOT, "plots", env_short, common_tag) if common_tag
                 else os.path.join(REPO_ROOT, "plots", env_short))
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Environment: {args.env_id} ({env_short})")
    print(f"Conditions: {[label for *_, label, _, _ in conditions]}")
    print(f"Seeds: {seeds}")
    print("NOTE: wall-clock-time panels below are only a fair comparison if every "
          "condition ran with the same CPU thread budget, sequentially (see run_all.sh).\n")

    print("Evaluating checkpoints (this runs the policies -- may take a few minutes)...")
    checkpoint_cache = {}
    for algo, tag, condition_key, label, color, linestyle in conditions:
        for seed in seeds:
            checkpoint_cache[(condition_key, seed)] = evaluate_checkpoints_for_seed(
                env_short, args.env_id, algo, seed, env_spec, tag
            )

    csv_rows = []

    # --- Figure 1: training_comparison.png (reward vs %, reward vs time) ---
    fig1, (ax_reward, ax_time) = plt.subplots(2, 1, figsize=(9, 8))

    print("\nLoading reward curves from TensorBoard logs...")
    for algo, tag, condition_key, label, color, linestyle in conditions:
        pct_grid, mean_r, std_r = aggregate_reward_curves(env_short, algo, seeds, tag)
        if pct_grid is None:
            continue
        ax_reward.plot(pct_grid, mean_r, label=label, color=color, linestyle=linestyle)
        ax_reward.fill_between(pct_grid, mean_r - std_r, mean_r + std_r, color=color, alpha=0.15)

    ax_reward.set_ylabel("Mean episode reward")
    ax_reward.set_title(f"Training reward on {args.env_id} (mean\u00b1std, seeds {seeds})")
    ax_reward.set_xlabel("% of training budget completed")
    ax_reward.legend(fontsize=8)
    ax_reward.grid(alpha=0.3)

    print("\nLoading reward-vs-wall-clock-time curves from captured logs...")
    for algo, tag, condition_key, label, color, linestyle in conditions:
        time_grid, mean_r, std_r = aggregate_reward_vs_time(env_short, algo, seeds, tag)
        if time_grid is None:
            continue
        ax_time.plot(time_grid, mean_r, label=label, color=color, linestyle=linestyle)
        ax_time.fill_between(time_grid, mean_r - std_r, mean_r + std_r, color=color, alpha=0.15)

    ax_time.set_ylabel("Mean episode reward")
    ax_time.set_xlabel("Wall-clock training time (minutes)")
    ax_time.set_title(f"Training reward vs. wall-clock time (mean\u00b1std, seeds {seeds})")
    ax_time.legend(fontsize=8)
    ax_time.grid(alpha=0.3)

    fig1.tight_layout()
    fig1_path = os.path.join(plots_dir, "training_comparison.png")
    fig1.savefig(fig1_path, dpi=150)
    print(f"\nSaved plot to {fig1_path}")

    # --- Figure 2: goal_reaching_time.png (split out on its own) ---
    fig2, ax_reach = plt.subplots(1, 1, figsize=(9, 5))

    print("\nBuilding goal-reaching-time panel...")
    for algo, tag, condition_key, label, color, linestyle in conditions:
        pct, mean_rt, std_rt = aggregate_reach_time_curves(algo, tag, seeds, checkpoint_cache, csv_rows)
        if pct is None:
            continue
        ax_reach.plot(pct, mean_rt, marker="o", markersize=3, label=label, color=color, linestyle=linestyle)
        valid = ~np.isnan(mean_rt)
        ax_reach.fill_between(pct[valid], (mean_rt - std_rt)[valid], (mean_rt + std_rt)[valid],
                               color=color, alpha=0.15)

    ax_reach.set_ylabel("Avg. steps to first reach goal\n(successful episodes only)")
    ax_reach.set_xlabel("% of training budget completed")
    ax_reach.set_title(f"Goal-reaching time (mean\u00b1std, seeds {seeds})\n"
                        f"gaps = 0% success across all seeds at that checkpoint")
    ax_reach.legend(fontsize=8)
    ax_reach.grid(alpha=0.3)

    fig2.tight_layout()
    fig2_path = os.path.join(plots_dir, "goal_reaching_time.png")
    fig2.savefig(fig2_path, dpi=150)
    print(f"Saved plot to {fig2_path}")

    # --- Figure 3: success_rate_vs_time.png ---
    fig3, (ax_succ_time, ax_succ_all) = plt.subplots(2, 1, figsize=(9, 8))

    print("\nBuilding success-rate-vs-time panels...")
    for algo, tag, condition_key, label, color, linestyle in conditions:
        t, mean_s, std_s = aggregate_metric_vs_time(algo, tag, seeds, checkpoint_cache, "success_rate_within_time_fraction")
        if t is not None:
            ax_succ_time.plot(t, mean_s, label=label, color=color, linestyle=linestyle)
            ax_succ_time.fill_between(t, mean_s - std_s, mean_s + std_s, color=color, alpha=0.15)

        t, mean_a, std_a = aggregate_metric_vs_time(algo, tag, seeds, checkpoint_cache, "reached_at_all_rate")
        if t is not None:
            ax_succ_all.plot(t, mean_a, label=label, color=color, linestyle=linestyle)
            ax_succ_all.fill_between(t, mean_a - std_a, mean_a + std_a, color=color, alpha=0.15)

    pct_label = int(env_spec["success_time_fraction"] * 100)
    ax_succ_time.set_ylabel(f"Success rate\n(within {pct_label}% of episode)")
    ax_succ_time.set_title(f"Success rate vs. wall-clock time -- main criterion (mean\u00b1std, seeds {seeds})")
    ax_succ_time.set_ylim(-0.05, 1.05)
    ax_succ_time.legend(fontsize=8)
    ax_succ_time.grid(alpha=0.3)

    ax_succ_all.set_ylabel("Success rate\n(reached at all, by end of episode)")
    ax_succ_all.set_xlabel("Wall-clock training time (minutes)")
    ax_succ_all.set_title("Success rate vs. wall-clock time -- preliminary criterion")
    ax_succ_all.set_ylim(-0.05, 1.05)
    ax_succ_all.legend(fontsize=8)
    ax_succ_all.grid(alpha=0.3)

    fig3.tight_layout()
    fig3_path = os.path.join(plots_dir, "success_rate_vs_time.png")
    fig3.savefig(fig3_path, dpi=150)
    print(f"Saved plot to {fig3_path}")

    # --- Figure 4: distance_progress.png ---
    fig4, (ax_dist_pct, ax_dist_time) = plt.subplots(2, 1, figsize=(9, 8))
    threshold = env_spec["success_threshold"]

    print("\nBuilding distance-progress panels...")
    for algo, tag, condition_key, label, color, linestyle in conditions:
        pct, mean_f, std_f = aggregate_metric_vs_pct(algo, tag, seeds, checkpoint_cache, "avg_final_distance")
        if pct is not None:
            ax_dist_pct.plot(pct, mean_f, label=f"{label} (final)", color=color, linestyle=linestyle)
            ax_dist_pct.fill_between(pct, mean_f - std_f, mean_f + std_f, color=color, alpha=0.12)
        pct, mean_m, std_m = aggregate_metric_vs_pct(algo, tag, seeds, checkpoint_cache, "avg_min_distance")
        if pct is not None:
            ax_dist_pct.plot(pct, mean_m, label=f"{label} (min)", color=color,
                              linestyle=":" if linestyle == "-" else "-.", alpha=0.8)

        t, mean_f, std_f = aggregate_metric_vs_time(algo, tag, seeds, checkpoint_cache, "avg_final_distance")
        if t is not None:
            ax_dist_time.plot(t, mean_f, label=f"{label} (final)", color=color, linestyle=linestyle)
            ax_dist_time.fill_between(t, mean_f - std_f, mean_f + std_f, color=color, alpha=0.12)
        t, mean_m, std_m = aggregate_metric_vs_time(algo, tag, seeds, checkpoint_cache, "avg_min_distance")
        if t is not None:
            ax_dist_time.plot(t, mean_m, label=f"{label} (min)", color=color,
                               linestyle=":" if linestyle == "-" else "-.", alpha=0.8)

    for ax in (ax_dist_pct, ax_dist_time):
        ax.axhline(threshold, color="black", linestyle=":", linewidth=1, alpha=0.6,
                   label=f"success threshold ({threshold})")

    ax_dist_pct.set_ylabel("Mean distance to goal\n(solid/dashed=final, dotted=closest approach)")
    ax_dist_pct.set_xlabel("% of training budget completed")
    ax_dist_pct.set_title(f"Distance to goal vs. % of training on {args.env_id} (mean\u00b1std, seeds {seeds})")
    ax_dist_pct.legend(fontsize=7, ncol=2)
    ax_dist_pct.grid(alpha=0.3)

    ax_dist_time.set_ylabel("Mean distance to goal\n(solid/dashed=final, dotted=closest approach)")
    ax_dist_time.set_xlabel("Wall-clock training time (minutes)")
    ax_dist_time.set_title(f"Distance to goal vs. wall-clock time (mean\u00b1std, seeds {seeds})")
    ax_dist_time.legend(fontsize=7, ncol=2)
    ax_dist_time.grid(alpha=0.3)

    fig4.tight_layout()
    fig4_path = os.path.join(plots_dir, "distance_progress.png")
    fig4.savefig(fig4_path, dpi=150)
    print(f"Saved plot to {fig4_path}")

    if csv_rows:
        csv_path = os.path.join(plots_dir, "training_comparison_data.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved raw per-checkpoint, per-seed data to {csv_path}")


if __name__ == "__main__":
    main()
