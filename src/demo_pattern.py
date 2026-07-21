"""
Sends a trained Reacher agent's arm through a sequence of fixed target
coordinates, tracing a pattern (rectangle or star) one or more times, and
records the result as a video.

MECHANISM (worth understanding before trusting the output): Reacher's
target position isn't something you set through the normal step()/reset()
API -- it's randomized once at reset() and fixed for that episode. To
trace an arbitrary sequence of waypoints in one continuous run (arm motion
carrying over between waypoints, not resetting/teleporting), this script
writes directly into the underlying MuJoCo simulation state (qpos[2:4],
the target's x/y slide-joint positions) and calls mj_forward() to
propagate that change, then lets the policy continue stepping normally.
qpos[0:2] (the arm's own joint angles) are left untouched, so the arm's
motion is continuous across waypoint transitions -- it's actually tracking
a moving target, not resetting between segments. Empirically verified: set
qpos[2:4], step once, observation correctly reflects the new target while
the arm's own joint angles are unaffected.

This relies on Reacher's qpos layout being [joint0, joint1, target_x,
target_y] -- true for every version of this environment to date, but
worth a quick sanity check before a long run: the script prints qpos on
startup, and --diagnostic-only exits after one waypoint so you can eyeball
a short clip before committing to the full pattern.

Two modes:

1. Single model:
    python src/demo_pattern.py --algo sac --model-path models/reacher/sac/seed0/sac_final.zip --pattern rectangle --repeats 2

2. Checkpoint progression -- same model, several points across training,
   each tracing the full pattern in turn, concatenated into one video so
   you can watch tracking accuracy improve over training:
    python src/demo_pattern.py --algo sac --seed 0 --pattern rectangle --repeats 1
    python src/demo_pattern.py --algo sac --seed 0 --pattern star --fractions 0.02,0.1,0.5,1.0 --use-best-as-final

--repeats applies per checkpoint in progression mode (each checkpoint gets
an equal, fair number of loops through the pattern, not just the final one).
"""

import argparse
import os
import re

import cv2
import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from algo_registry import get_algo_class, variant_choices
from camera_utils import add_camera_args, build_camera_config
from make_progression_video import find_checkpoints, pick_checkpoints_for_fractions, seed_folder_name

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_SHORT = "reacher"  # this script is Reacher-specific (qpos[2:4] layout assumption)


def load_vecnormalize_stats(vecnorm_path, env_id, seed):
    """Rebuild a VecNormalize wrapper purely to reuse its obs-normalization
    stats for feeding the policy -- same pattern used in every other video/
    eval script in this project, previously missing here. Needed for PPO's
    Reacher config specifically (normalize: true); without this, PPO's
    policy gets raw unnormalized observations completely unlike what it was
    trained on, producing erratic actions unrelated to the pattern-tracing
    task itself."""
    dummy_env = make_vec_env(env_id, n_envs=1, seed=seed)
    vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return vec_normalize


def rectangle_pattern(size=0.1):
    """4 corners of a square, safely inside Reacher's ~0.2 max reach
    (corner radius = size*sqrt(2) ~= 0.141 at the default size)."""
    return [(size, size), (size, -size), (-size, -size), (-size, size)]


def star_pattern(outer=0.15, inner=0.06, n_points=5):
    """A 2*n_points-vertex star traced as one continuous path, alternating
    between outer and inner radius."""
    coords = []
    for i in range(n_points * 2):
        angle = np.pi / 2 + i * np.pi / n_points  # start pointing straight up
        r = outer if i % 2 == 0 else inner
        coords.append((r * np.cos(angle), r * np.sin(angle)))
    return coords


def label_frame(frame, text):
    frame = frame.copy()
    origin = (10, 30)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def banner_frame(frame, text):
    """Bigger, top-center banner, distinct from the small per-waypoint
    corner label -- used in progression mode to mark which checkpoint is
    currently playing. Placed well below the corner label's row (fixed
    70px gap, not a subtle percentage) so the two definitely don't overlap."""
    frame = frame.copy()
    h, w, _ = frame.shape
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    origin = (max((w - text_w) // 2, 0), 100)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
    return frame


def resolve_checkpoints(algo, seed, run_tag, fractions, use_best_as_final):
    """Returns list of (checkpoint_path, checkpoint_label) for progression
    mode, honoring --fractions and (optionally) substituting best_model.zip
    for whichever checkpoint the highest requested fraction would pick."""
    sf = seed_folder_name(seed, run_tag)
    checkpoints = find_checkpoints(ENV_SHORT, algo, seed, run_tag)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under models/{ENV_SHORT}/{algo}/{sf}/")
    picked = pick_checkpoints_for_fractions(checkpoints, fractions)
    if not picked:
        raise SystemExit(f"No checkpoints matched the requested fractions for {algo} {sf}.")

    best_path = os.path.join(REPO_ROOT, "models", ENV_SHORT, algo, sf, "best_model.zip")
    use_best = use_best_as_final and os.path.exists(best_path)
    max_frac_idx = max(range(len(picked)), key=lambda i: picked[i][2])

    results = []
    for i, (step, path, frac) in enumerate(picked):
        if use_best and i == max_frac_idx:
            results.append((best_path, f"step {step:,} (best)"))
        else:
            results.append((path, f"step {step:,} (~{frac*100:.0f}%)"))
    return results


def run_pattern_on_model(env, model, full_sequence, steps_per_waypoint, algo_label,
                          checkpoint_label=None, obs_normalizer=None):
    """Runs the given model through the full waypoint sequence on the
    already-created env, returning the collected frames."""
    obs, _ = env.reset(seed=0)
    unwrapped = env.unwrapped
    frames = []
    for i, (tx, ty) in enumerate(full_sequence):
        unwrapped.data.qpos[2] = tx
        unwrapped.data.qpos[3] = ty
        unwrapped.data.qvel[2:4] = 0.0
        mujoco.mj_forward(unwrapped.model, unwrapped.data)

        label = f"{algo_label}  waypoint {i+1}/{len(full_sequence)}  target=({tx:.2f}, {ty:.2f})"
        for _ in range(steps_per_waypoint):
            predict_obs = obs_normalizer.normalize_obs(obs) if obs_normalizer is not None else obs
            action, _ = model.predict(predict_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            frame = label_frame(frame, label)
            if checkpoint_label:
                frame = banner_frame(frame, checkpoint_label)
            frames.append(frame)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=variant_choices(), required=True)
    parser.add_argument("--pattern", choices=["rectangle", "star"], default="rectangle")
    parser.add_argument("--repeats", type=int, default=2,
                         help="How many loops through the pattern -- applies per checkpoint in progression mode.")
    parser.add_argument("--steps-per-waypoint", type=int, default=100)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--diagnostic-only", action="store_true",
                         help="Run just the first waypoint (of the first checkpoint, in progression mode) and "
                              "exit -- use this first to confirm things look right before a long run.")

    # Single-model mode
    parser.add_argument("--model-path", default=None,
                         help="Single-model mode: path to one trained model.")
    parser.add_argument("--vecnormalize-path", default=None,
                         help="Single-model mode: path to a saved VecNormalize .pkl, if the model was "
                              "trained with normalize: true (PPO's Reacher config uses this).")

    # Checkpoint-progression mode
    parser.add_argument("--seed", type=int, default=None,
                         help="Progression mode: training seed whose checkpoints to sample from.")
    parser.add_argument("--fractions", default="0.02,0.05,0.1,0.2,0.5,1.0",
                         help="Progression mode: comma-separated fractions of total training to sample at "
                              "(same convention as make_progression_video.py / make_grid_video.py).")
    parser.add_argument("--run-tag", default=None,
                         help="Progression mode: point at an experimental run launched with train.py --run-tag.")
    parser.add_argument("--use-best-as-final", action="store_true",
                         help="Progression mode: substitute best_model.zip for whichever checkpoint the "
                              "highest requested fraction would otherwise pick, if it exists.")

    add_camera_args(parser)
    args = parser.parse_args()

    if bool(args.model_path) == (args.seed is not None):
        raise SystemExit("Pass exactly one of --model-path (single-model mode) or --seed (progression mode).")

    waypoints = rectangle_pattern() if args.pattern == "rectangle" else star_pattern()
    full_sequence = waypoints if args.diagnostic_only else waypoints * args.repeats

    camera_config = build_camera_config(args)
    make_kwargs = {}
    if camera_config:
        make_kwargs["default_camera_config"] = camera_config
    # max_episode_steps is generously overridden since we're deliberately
    # running one continuous "episode" far longer than Reacher's normal 50
    # steps (many waypoints x steps-per-waypoint x repeats, x checkpoints
    # in progression mode).
    env = gym.make("Reacher-v5", render_mode="rgb_array", max_episode_steps=100_000, **make_kwargs)
    fps = env.metadata.get("render_fps", 30)

    unwrapped = env.unwrapped
    print(f"qpos shape: {unwrapped.data.qpos.shape} (expect (4,): [joint0, joint1, target_x, target_y])")
    print(f"Pattern: {args.pattern}, {len(waypoints)} waypoints"
          + (f", x{args.repeats} repeats" if not args.diagnostic_only else " (diagnostic: first waypoint only)"))

    algo_cls = get_algo_class(args.algo)
    all_frames = []

    if args.model_path:
        model = algo_cls.load(args.model_path, device=args.device)
        obs_normalizer = (load_vecnormalize_stats(args.vecnormalize_path, "Reacher-v5", 0)
                           if args.vecnormalize_path else None)
        all_frames = run_pattern_on_model(env, model, full_sequence, args.steps_per_waypoint, args.algo.upper(),
                                           obs_normalizer=obs_normalizer)
        tag = "diagnostic" if args.diagnostic_only else f"{args.pattern}_x{args.repeats}"
        default_out = os.path.join(REPO_ROOT, "videos", ENV_SHORT, args.algo, f"{args.algo}_pattern_{tag}.mp4")
    else:
        fractions = [float(f.strip()) for f in args.fractions.split(",") if f.strip()]
        if args.diagnostic_only:
            fractions = fractions[:1]  # just check the first checkpoint if diagnosing
        checkpoints = resolve_checkpoints(args.algo, args.seed, args.run_tag, fractions, args.use_best_as_final)
        print(f"Progression: {len(checkpoints)} checkpoints -> {[label for _, label in checkpoints]}")

        # PPO's VecNormalize stats are only saved once, at the very end of
        # training -- so the same file is used to approximate normalization
        # for every checkpoint in the progression, including early ones.
        # This is the same approximation used elsewhere in the project
        # (see evaluate.py/train.py docstrings) -- not exact for early
        # checkpoints, but far better than skipping normalization entirely.
        sf = seed_folder_name(args.seed, args.run_tag)
        vecnorm_path = os.path.join(REPO_ROOT, "models", ENV_SHORT, args.algo, sf, f"{args.algo}_vecnormalize.pkl")
        obs_normalizer = (load_vecnormalize_stats(vecnorm_path, "Reacher-v5", args.seed)
                           if os.path.exists(vecnorm_path) else None)
        if args.algo == "ppo" and obs_normalizer is None:
            print(f"  [!] No VecNormalize file found at {vecnorm_path} -- PPO's actions will likely be poor "
                  f"without it, since its Reacher config trains with normalize: true.")

        for path, checkpoint_label in checkpoints:
            model = algo_cls.load(path, device=args.device)
            segment = run_pattern_on_model(
                env, model, full_sequence, args.steps_per_waypoint, args.algo.upper(),
                checkpoint_label=f"{args.algo.upper()}  {checkpoint_label}",
                obs_normalizer=obs_normalizer,
            )
            all_frames.extend(segment)

        tag = "diagnostic" if args.diagnostic_only else f"{args.pattern}_x{args.repeats}"
        default_out = os.path.join(REPO_ROOT, "videos", ENV_SHORT, args.algo,
                                    f"{args.algo}_{sf}_progression_pattern_{tag}.mp4")

    env.close()

    if not all_frames:
        raise SystemExit("No frames captured -- nothing to write.")

    out_path = args.out or default_out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        from moviepy import ImageSequenceClip
    except ImportError:
        from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(all_frames, fps=fps)
    clip.write_videofile(out_path, codec="libx264", audio=False, logger=None)
    print(f"Saved pattern demo ({len(all_frames)} frames) to {out_path}")


if __name__ == "__main__":
    main()
