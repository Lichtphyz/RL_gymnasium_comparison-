"""
Record demo video clips of a trained agent -- useful for the live
demo/presentation, and as a sanity check that the success threshold in
env_registry.py looks right visually.

Usage:
    python src/record_demo.py --algo sac --model-path models/reacher/sac/seed0/sac_final.zip --n-episodes 3
    python src/record_demo.py --algo ppo --env-id Pusher-v5 --model-path models/pusher/ppo/seed0/ppo_final.zip --vecnormalize-path models/pusher/ppo/seed0/ppo_vecnormalize.pkl --n-episodes 3

Writes .mp4 files to videos/<env-short>/<algo>/. Requires `moviepy` (see
requirements.txt). Runs on CPU by default (see train.py's docstring for why).

Filenames are auto-tagged with the seed and checkpoint step (parsed from
--model-path) by default, e.g. sac_seed0_step475000-episode-0.mp4 -- this
is what stops successive calls from silently overwriting each other's clips,
which used to happen whenever you re-recorded from a different checkpoint of
the same algorithm. Pass --tag to override, or --tag "" to opt back into the
old untagged naming.
"""

import argparse
import os
import re

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from algo_registry import get_algo_class, variant_choices
from env_registry import env_choices, get_env_short_name
from camera_utils import add_camera_args, build_camera_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_vecnormalize_stats(vecnorm_path, env_id, seed):
    """Rebuild a VecNormalize wrapper purely to reuse its obs-normalization
    stats for feeding the policy; the recorded env stays raw for rendering."""
    dummy_env = make_vec_env(env_id, n_envs=1, seed=seed)
    vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return vec_normalize


def derive_tag_from_path(model_path: str) -> str:
    """Best-effort tag so repeated calls don't clobber each other: pulls the
    seed<N>[_<run-tag>] folder name (verbatim -- NOT reconstructed from a
    digits-only regex, since that would silently drop any --run-tag suffix
    like seed2_distweight2 and collide with the stock seed2 video) and
    either 'final' or the checkpoint step count out of the model path.
    Falls back to the bare filename stem if the path doesn't match the
    expected models/<env>/<algo>/seed<N>[_<tag>]/... layout."""
    parent_dir_name = os.path.basename(os.path.dirname(model_path))
    seed_part = parent_dir_name if parent_dir_name.startswith("seed") else None

    basename = os.path.splitext(os.path.basename(model_path))[0]
    step_match = re.search(r"_(\d+)_steps$", basename)
    if step_match:
        step_part = f"step{step_match.group(1)}"
    elif basename.endswith("_final"):
        step_part = "final"
    elif basename.endswith("_best_model") or basename == "best_model":
        step_part = "best"
    else:
        step_part = basename

    parts = [p for p in (seed_part, step_part) if p]
    return "_".join(parts) if parts else basename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=variant_choices(), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-id", default="Reacher-v5",
                         help=f"Must be a registered environment (see env_registry.py). "
                              f"Registered short names: {env_choices()}")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--vecnormalize-path", default=None,
                         help="Path to a saved VecNormalize .pkl, if the model was trained with normalize: true")
    parser.add_argument("--tag", default=None,
                         help="Overrides the auto-derived filename tag. Pass --tag \"\" for the old "
                              "untagged <algo>-episode-N.mp4 naming (will overwrite previous clips).")
    add_camera_args(parser)
    args = parser.parse_args()

    env_short = get_env_short_name(args.env_id)
    video_dir = os.path.join(REPO_ROOT, "videos", env_short, args.algo)
    os.makedirs(video_dir, exist_ok=True)

    tag = args.tag if args.tag is not None else derive_tag_from_path(args.model_path)
    name_prefix = f"{args.algo}_{tag}" if tag else args.algo

    camera_config = build_camera_config(args)
    env_kwargs = {"render_mode": "rgb_array"}
    if camera_config:
        env_kwargs["default_camera_config"] = camera_config
    env = gym.make(args.env_id, **env_kwargs)
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,  # record every episode
        name_prefix=name_prefix,
    )

    algo_cls = get_algo_class(args.algo)
    model = algo_cls.load(args.model_path, device=args.device)

    obs_normalizer = None
    if args.vecnormalize_path:
        obs_normalizer = load_vecnormalize_stats(args.vecnormalize_path, args.env_id, args.seed)

    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        while not (done or truncated):
            predict_obs = obs
            if obs_normalizer is not None:
                predict_obs = obs_normalizer.normalize_obs(obs)
            action, _ = model.predict(predict_obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

    env.close()
    print(f"Saved {args.n_episodes} clips (prefix '{name_prefix}') to {video_dir}")


if __name__ == "__main__":
    main()

