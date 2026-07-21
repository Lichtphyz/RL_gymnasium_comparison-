"""
Train one variant (ddpg, td3, sac, ppo, or ddpg_1critic) on any registered
environment (see env_registry.py -- currently Reacher-v5 and Pusher-v5).

Usage:
    python src/train.py --algo sac
    python src/train.py --algo td3 --env-id Pusher-v5 --n-timesteps 1000000 --seed 1
    python src/train.py --algo ddpg_1critic --seed 2

Loads hyperparameters from configs/<env-short>/<config-name>.yml (see
algo_registry.py for the variant -> config-name mapping, and
env_registry.py for the env-id -> short-name mapping), trains, checkpoints
periodically to models/<env-short>/<algo>/seed<seed>/, logs to
logs/<env-short>/<algo>/seed<seed>/ for TensorBoard, and saves the final
model as models/<env-short>/<algo>/seed<seed>/<algo>_final.zip.

Runs on CPU by default -- measured directly on this project (small MLP
policies, small batch sizes) that CPU matches or beats GPU here; pass
--device cuda to override if you want to re-check that on different
hardware.
"""

import argparse
import os

import yaml
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize

from algo_registry import NOISE_CAPABLE_VARIANTS, get_algo_class, get_config_name, variant_choices
from env_registry import get_env_short_name, env_choices

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(algo: str, env_short: str) -> dict:
    config_path = os.path.join(REPO_ROOT, "configs", env_short, f"{get_config_name(algo)}.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(algo: str, env, config: dict, seed: int, tensorboard_log: str, device: str = "cpu"):
    config = dict(config)  # don't mutate the loaded dict

    # policy_kwargs and noise are stored as strings in the yaml (matches the
    # rl-baselines3-zoo convention) so they can hold Python expressions like
    # "dict(net_arch=[400, 300], n_critics=1)" -- eval them here with a
    # restricted namespace.
    policy_kwargs = config.pop("policy_kwargs", None)
    if policy_kwargs is not None:
        policy_kwargs = eval(policy_kwargs, {"dict": dict})

    noise_type = config.pop("noise_type", None)
    noise_std = config.pop("noise_std", None)
    action_noise = None
    if noise_type == "normal":
        import numpy as np
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )

    # These keys control the training loop / env wrapping, not the SB3 constructor.
    config.pop("n_timesteps", None)
    config.pop("normalize", None)
    config.pop("n_envs", None)

    policy = config.pop("policy", "MlpPolicy")

    algo_cls = get_algo_class(algo)
    extra_kwargs = {}
    if algo in NOISE_CAPABLE_VARIANTS:
        extra_kwargs["action_noise"] = action_noise

    model = algo_cls(
        policy,
        env,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        device=device,
        **extra_kwargs,
        **config,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=variant_choices(), required=True)
    parser.add_argument("--env-id", default="Reacher-v5",
                         help=f"Must be one of the registered environments' env_ids (see env_registry.py). "
                              f"Currently registered short names: {env_choices()}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-timesteps", type=int, default=None,
                         help="Override n_timesteps from the config file")
    parser.add_argument("--n-envs", type=int, default=None,
                         help="Override n_envs (parallel envs) from the config file")
    parser.add_argument("--checkpoint-freq", type=int, default=25000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"],
                         help="Defaults to cpu -- measured faster than cuda for these network sizes on this project.")
    parser.add_argument("--resume-from", default=None,
                         help="Path to an existing checkpoint .zip to continue training from, instead of "
                              "building a fresh model. Useful for testing whether a plateaued reward curve "
                              "breaks out with more steps, without restarting from scratch. NOTE: this does "
                              "NOT restore the replay buffer (not saved by checkpoints currently), so the "
                              "off-policy buffer starts empty on resume -- policy/critic weights carry over, "
                              "but expect a brief dip in sample efficiency while it refills.")
    parser.add_argument("--additional-timesteps", type=int, default=None,
                         help="With --resume-from: how many MORE steps to train (added to whatever the "
                              "checkpoint already has). Defaults to the same n_timesteps as a fresh run "
                              "would use, i.e. doubles total training if resuming from a fully-finished run.")
    parser.add_argument("--noise-std", type=float, default=None,
                         help="Override the config's noise_std (DDPG/TD3/exploration noise magnitude). "
                              "Only meaningful for noise-capable variants (ddpg, td3, ddpg_1critic).")
    parser.add_argument("--reward-near-weight", type=float, default=None,
                         help="Pusher only: overrides reward_near_weight (default 0.5) via gym.make() kwargs.")
    parser.add_argument("--reward-dist-weight", type=float, default=None,
                         help="Pusher/Reacher: overrides reward_dist_weight (Pusher default 1.0) via gym.make() kwargs.")
    parser.add_argument("--reward-control-weight", type=float, default=None,
                         help="Pusher only: overrides reward_control_weight (default 0.1) via gym.make() kwargs.")
    parser.add_argument("--run-tag", default=None,
                         help="Appends to the seed folder name (e.g. seed0_explore instead of seed0) so "
                              "experimental runs never collide with or overwrite stock results, regardless "
                              "of reusing the same --algo/--seed/--env-id.")
    args = parser.parse_args()

    env_short = get_env_short_name(args.env_id)
    config = load_config(args.algo, env_short)
    n_timesteps = args.n_timesteps or int(config.get("n_timesteps", 300000))
    n_envs = args.n_envs or int(config.get("n_envs", 1))
    normalize = bool(config.get("normalize", False))

    if args.noise_std is not None:
        config["noise_std"] = args.noise_std
        print(f"Overriding noise_std -> {args.noise_std}")

    env_kwargs = {}
    if args.reward_near_weight is not None:
        env_kwargs["reward_near_weight"] = args.reward_near_weight
    if args.reward_dist_weight is not None:
        env_kwargs["reward_dist_weight"] = args.reward_dist_weight
    if args.reward_control_weight is not None:
        env_kwargs["reward_control_weight"] = args.reward_control_weight
    if env_kwargs:
        print(f"Overriding reward weights via env_kwargs: {env_kwargs}")

    seed_folder = f"seed{args.seed}" + (f"_{args.run_tag}" if args.run_tag else "")
    model_dir = os.path.join(REPO_ROOT, "models", env_short, args.algo, seed_folder)
    log_dir = os.path.join(REPO_ROOT, "logs", env_short, args.algo, seed_folder)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = make_vec_env(args.env_id, n_envs=n_envs, seed=args.seed,
                        monitor_dir=log_dir, env_kwargs=env_kwargs or None)
    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    eval_env = make_vec_env(args.env_id, n_envs=1, seed=args.seed + 1000,
                             env_kwargs=env_kwargs or None)
    if normalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                                 training=False)

    if args.resume_from:
        algo_cls = get_algo_class(args.algo)
        model = algo_cls.load(args.resume_from, env=env, device=args.device)
        additional_timesteps = args.additional_timesteps or n_timesteps
        print(f"Resumed from {args.resume_from} at {model.num_timesteps:,} steps; "
              f"training {additional_timesteps:,} more.")
        # SB3 treats total_timesteps as an ABSOLUTE target when
        # reset_num_timesteps=False, not "how many more steps" -- so this
        # has to be current + additional, not just additional, or the
        # training loop exits immediately (already >= target).
        n_timesteps_to_run = model.num_timesteps + additional_timesteps
        reset_num_timesteps = False
    else:
        model = build_model(args.algo, env, config, args.seed, log_dir, device=args.device)
        n_timesteps_to_run = n_timesteps
        reset_num_timesteps = True

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // n_envs, 1),
            save_path=model_dir,
            name_prefix=args.algo,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=max(args.eval_freq // n_envs, 1),
            deterministic=True,
            render=False,
        ),
    ]

    model.learn(total_timesteps=n_timesteps_to_run, callback=callbacks,
                tb_log_name=args.algo, progress_bar=True,
                reset_num_timesteps=reset_num_timesteps)

    final_path = os.path.join(model_dir, f"{args.algo}_final")
    model.save(final_path)
    if normalize:
        env.save(os.path.join(model_dir, f"{args.algo}_vecnormalize.pkl"))

    print(f"Saved final model to {final_path}.zip")


if __name__ == "__main__":
    main()
