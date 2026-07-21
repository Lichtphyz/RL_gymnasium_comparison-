"""
Generate one combined video showing an agent's behavior at several points
across training -- e.g. what it looked like at 1%, 10%, 50%, 100% of the
way through. Useful for a presentation: one clip that visibly shows
learning progress, rather than a bunch of separate before/after clips.

This runs entirely AFTER training, against already-saved checkpoints.
CheckpointCallback already saves a full model snapshot every
--checkpoint-freq steps during train.py (25000 by default) -- that's all
the raw material this needs. Nothing has to be recorded live during
training.

Usage:
    python src/make_progression_video.py --algo sac --env-id Pusher-v5 --seed 0
    python src/make_progression_video.py --algo sac --env-id Pusher-v5 --seed 0 \\
        --fractions 0.01,0.05,0.1,0.25,0.5,0.75,1.0 --episodes-per-checkpoint 1

Each requested fraction is matched to whichever saved checkpoint is
actually closest to it (checkpoints only exist every --checkpoint-freq
steps, so an exact 1% checkpoint may not exist -- the nearest one is used,
and duplicates are dropped if two fractions round to the same checkpoint).
The step count and % are burned into each clip's frames (via OpenCV, not
moviepy's TextClip, to avoid an ImageMagick dependency) so the label is
visible while watching, not just in the filename.
"""

import argparse
import glob
import os
import re

import cv2
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from algo_registry import get_algo_class, variant_choices
from env_registry import env_choices, get_env_short_name
from camera_utils import add_camera_args, build_camera_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FRACTIONS = "0.02,0.05,0.1,0.2,0.5,1.0"


def seed_folder_name(seed, run_tag=None):
    """Matches train.py's convention: seed<N>, or seed<N>_<tag> for
    experimental runs launched with --run-tag."""
    return f"seed{seed}" + (f"_{run_tag}" if run_tag else "")


def find_checkpoints(env_short, algo, seed, run_tag=None):
    pattern = os.path.join(REPO_ROOT, "models", env_short, algo, seed_folder_name(seed, run_tag), f"{algo}_*_steps.zip")
    files = glob.glob(pattern)
    checkpoints = []
    for f in files:
        m = re.search(rf"{algo}_(\d+)_steps\.zip$", os.path.basename(f))
        if m:
            checkpoints.append((int(m.group(1)), f))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def pick_checkpoints_for_fractions(checkpoints, fractions):
    """checkpoints: sorted [(step, path), ...]. Picks whichever checkpoint's
    step is closest to each requested fraction of the final checkpoint's
    step count. Drops duplicates if two fractions round to the same
    nearest checkpoint, preserving the requested order."""
    if not checkpoints:
        return []
    total_steps = checkpoints[-1][0]
    picked = []
    seen_steps = set()
    for frac in fractions:
        target = frac * total_steps
        step, path = min(checkpoints, key=lambda sp: abs(sp[0] - target))
        if step not in seen_steps:
            picked.append((step, path, frac))
            seen_steps.add(step)
    return picked


def load_vecnormalize_stats(vecnorm_path, env_id, seed):
    dummy_env = make_vec_env(env_id, n_envs=1, seed=seed)
    vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return vec_normalize


def label_frame(frame, text):
    """Burns a readable label into the top-left of an RGB frame: black
    outline behind white text, so it stays legible on any background."""
    frame = frame.copy()
    origin = (10, 30)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=variant_choices(), required=True)
    parser.add_argument("--env-id", default="Reacher-v5",
                         help=f"Registered short names: {env_choices()}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fractions", default=DEFAULT_FRACTIONS,
                         help="Comma-separated fractions of total training to sample checkpoints at")
    parser.add_argument("--episodes-per-checkpoint", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default=None,
                         help="Output path; defaults to videos/<env-short>/<algo>/progression_seed<seed>[_<tag>].mp4")
    parser.add_argument("--run-tag", default=None,
                         help="Point at an experimental run launched with train.py --run-tag "
                              "(e.g. 'distweight2') instead of the stock seed<N> folder.")
    parser.add_argument("--use-best-as-final", action="store_true",
                         help="Substitute best_model.zip for whichever checkpoint the highest requested "
                              "fraction would otherwise pick, if best_model.zip exists -- useful when a "
                              "run's final checkpoint isn't actually its best (e.g. late-training regression).")
    add_camera_args(parser)
    args = parser.parse_args()

    env_short = get_env_short_name(args.env_id)
    fractions = [float(f.strip()) for f in args.fractions.split(",") if f.strip()]
    sf = seed_folder_name(args.seed, args.run_tag)

    checkpoints = find_checkpoints(env_short, args.algo, args.seed, args.run_tag)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under models/{env_short}/{args.algo}/{sf}/")

    picked = pick_checkpoints_for_fractions(checkpoints, fractions)
    if not picked:
        raise SystemExit(f"No checkpoints matched the requested fractions for {args.algo} {sf}.")

    best_path = os.path.join(REPO_ROOT, "models", env_short, args.algo, sf, "best_model.zip")
    use_best = args.use_best_as_final and os.path.exists(best_path)
    max_frac_idx = max(range(len(picked)), key=lambda i: picked[i][2])

    print(f"Selected {len(picked)} checkpoints (from {len(checkpoints)} available):")
    cell_specs = []  # (step, path, frac, label_override_or_None)
    for i, (step, path, frac) in enumerate(picked):
        if use_best and i == max_frac_idx:
            print(f"  target {frac*100:.0f}% -> using best_model.zip instead of step {step:,}")
            cell_specs.append((step, best_path, frac, "best"))
        else:
            print(f"  target {frac*100:.0f}% -> step {step:,} ({os.path.basename(path)})")
            cell_specs.append((step, path, frac, None))

    vecnorm_path = os.path.join(REPO_ROOT, "models", env_short, args.algo, sf, f"{args.algo}_vecnormalize.pkl")
    obs_normalizer = (load_vecnormalize_stats(vecnorm_path, args.env_id, args.seed)
                       if os.path.exists(vecnorm_path) else None)

    algo_cls = get_algo_class(args.algo)
    camera_config = build_camera_config(args)
    env_kwargs = {"render_mode": "rgb_array"}
    if camera_config:
        env_kwargs["default_camera_config"] = camera_config
    env = gym.make(args.env_id, **env_kwargs)
    fps = env.metadata.get("render_fps", 30)

    all_frames = []
    for step, path, frac, label_override in cell_specs:
        model = algo_cls.load(path, device=args.device)
        if label_override:
            label = f"{args.algo.upper()}  {label_override} (step {step:,})"
        else:
            label = f"{args.algo.upper()}  step {step:,}  (~{frac*100:.0f}% of training)"
        for ep in range(args.episodes_per_checkpoint):
            obs, _ = env.reset(seed=1000 + ep)
            done, truncated = False, False
            while not (done or truncated):
                predict_obs = obs_normalizer.normalize_obs(obs) if obs_normalizer is not None else obs
                action, _ = model.predict(predict_obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                frame = env.render()
                all_frames.append(label_frame(frame, label))

    env.close()

    if not all_frames:
        raise SystemExit("No frames captured -- nothing to write.")

    out_path = args.out or os.path.join(REPO_ROOT, "videos", env_short, args.algo,
                                         f"progression_{sf}.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # moviepy 2.x flattened its import structure; fall back for older installs.
    try:
        from moviepy import ImageSequenceClip
    except ImportError:
        from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(all_frames, fps=fps)
    clip.write_videofile(out_path, codec="libx264", audio=False, logger=None)
    print(f"Saved progression video ({len(all_frames)} frames, {len(picked)} checkpoints) to {out_path}")


if __name__ == "__main__":
    main()
