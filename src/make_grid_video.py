"""
Combine several episodes (or several training-progress checkpoints) into
one grid video, all playing back simultaneously, each cell labeled --
useful for eyeballing "does this behavior generalize, or was that one
episode a fluke" at a glance, instead of watching clips one after another.

Both environments used in this project (Reacher, Pusher) truncate at a
fixed episode length and never terminate early, so every cell always has
the same number of frames -- no padding/sync logic needed to line them up.

Three modes:

1. Episodes mode -- N episodes from ONE model, side by side:
    python src/make_grid_video.py --algo sac --env-id Pusher-v5 \\
        --model-path models/pusher/sac/seed0/sac_final.zip --n-episodes 8

2. Checkpoints mode -- several points across training, side by side, all
   solving the SAME starting position (fixed seed across cells) so the
   only thing that differs between cells is how much training the model
   had -- a grid version of make_progression_video.py's clips:
    python src/make_grid_video.py --algo sac --env-id Pusher-v5 --seed 0 \\
        --fractions 0.02,0.05,0.1,0.2,0.5,1.0

   Pass --seeds (plural, comma-separated) instead of --seed to play
   multiple seeds' checkpoint-grids one after another in a single combined
   video, each segment banner-labeled "SEED N":
    python src/make_grid_video.py --algo ddpg --env-id Pusher-v5 --seeds 0,1,2

   Pass --use-best-as-final to substitute best_model.zip (EvalCallback's
   best-performing checkpoint during training) for whichever checkpoint
   the highest requested fraction would otherwise pick -- useful when a
   run's final checkpoint isn't actually its best (e.g. late-training
   regression, visible directly in distance_progress.png for some runs).

3. Compare-algorithms mode -- several algorithms' final models, side by
   side, all solving the SAME starting position -- so any visible
   difference is genuinely about the algorithms, not about different
   random starting conditions:
    python src/make_grid_video.py --env-id Pusher-v5 \\
        --compare-algos ddpg,td3,sac,ppo --compare-seed 0
"""

import argparse
import os

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from algo_registry import get_algo_class, variant_choices
from env_registry import env_choices, get_env_short_name
from make_progression_video import find_checkpoints, pick_checkpoints_for_fractions, seed_folder_name
from camera_utils import add_camera_args, build_camera_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_vecnormalize_stats(vecnorm_path, env_id, seed):
    dummy_env = make_vec_env(env_id, n_envs=1, seed=seed)
    vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return vec_normalize


def label_frame(frame, text):
    frame = frame.copy()
    origin = (8, 22)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def banner_frame(frame, text):
    """Bigger, top-center banner -- distinct from per-cell corner labels --
    used to mark which seed's segment is currently playing in a multi-seed
    combined video."""
    frame = frame.copy()
    h, w, _ = frame.shape
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    origin = ((w - text_w) // 2, 40)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
    return frame


def record_episode_frames(env, model, obs_normalizer, seed):
    """Runs one episode and returns the list of rendered RGB frames."""
    frames = []
    obs, _ = env.reset(seed=seed)
    done, truncated = False, False
    while not (done or truncated):
        predict_obs = obs_normalizer.normalize_obs(obs) if obs_normalizer is not None else obs
        action, _ = model.predict(predict_obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        frames.append(env.render())
    return frames


def choose_grid_shape(n):
    """Roughly-square rows x cols for n cells."""
    rows = int(np.floor(np.sqrt(n)))
    rows = max(rows, 1)
    cols = int(np.ceil(n / rows))
    return rows, cols


def compose_grid_frames(cells, pad=4, bg_color=(20, 20, 20), seed_banner=None):
    """cells: list of (frames_list, label). All frames_lists must be the
    same length (guaranteed for Reacher/Pusher -- fixed episode length, no
    early termination). Returns the composed grid frames as a list of
    numpy arrays (does NOT write a video file -- see write_video_from_frames).
    If seed_banner is given, burns it across the top of every frame in
    this segment (bigger/different placement than the per-cell labels)."""
    n = len(cells)
    n_frames = len(cells[0][0])
    h, w, _ = cells[0][0][0].shape
    rows, cols = choose_grid_shape(n)

    grid_h = rows * h + (rows + 1) * pad
    grid_w = cols * w + (cols + 1) * pad

    grid_frames = []
    for t in range(n_frames):
        canvas = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)
        for i, (frames, label) in enumerate(cells):
            r, c = divmod(i, cols)
            y = pad + r * (h + pad)
            x = pad + c * (w + pad)
            cell_frame = label_frame(frames[t], label)
            canvas[y:y + h, x:x + w] = cell_frame
        if seed_banner:
            canvas = banner_frame(canvas, seed_banner)
        grid_frames.append(canvas)
    return grid_frames


def write_video_from_frames(frames, fps, out_path):
    try:
        from moviepy import ImageSequenceClip
    except ImportError:
        from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(out_path, codec="libx264", audio=False, logger=None)


def resolve_checkpoint_cells(env_short, algo, seed, run_tag, fractions, use_best_as_final):
    """Returns list of (checkpoint_path, label) for one seed's checkpoints
    mode, honoring --fractions and (optionally) substituting best_model.zip
    for whichever checkpoint the highest requested fraction would pick --
    since the final checkpoint isn't always the best-performing one."""
    sf = seed_folder_name(seed, run_tag)
    checkpoints = find_checkpoints(env_short, algo, seed, run_tag)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under models/{env_short}/{algo}/{sf}/")
    picked = pick_checkpoints_for_fractions(checkpoints, fractions)
    if not picked:
        raise SystemExit(f"No checkpoints matched the requested fractions for {algo} {sf}.")

    best_path = os.path.join(REPO_ROOT, "models", env_short, algo, sf, "best_model.zip")
    use_best = use_best_as_final and os.path.exists(best_path)
    max_frac_idx = max(range(len(picked)), key=lambda i: picked[i][2])

    cells = []
    for i, (step, path, frac) in enumerate(picked):
        if use_best and i == max_frac_idx:
            cells.append((best_path, "best"))
            print(f"  ~{frac*100:.0f}% target -> using best_model.zip instead of step {step:,}")
        else:
            cells.append((path, f"step {step:,} (~{frac*100:.0f}%)"))
            print(f"  ~{frac*100:.0f}% target -> step {step:,} ({os.path.basename(path)})")
    return cells


def record_checkpoint_segment(env, algo_cls, cell_specs, obs_normalizer, episode_seed, device, seed_banner=None):
    """Records one seed's set of checkpoint cells and composes them into a
    grid-frame segment (list of numpy frames)."""
    cells = []
    for path, label in cell_specs:
        model = algo_cls.load(path, device=device)
        frames = record_episode_frames(env, model, obs_normalizer, episode_seed)
        cells.append((frames, label))
    return compose_grid_frames(cells, seed_banner=seed_banner)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=variant_choices(), default=None,
                         help="Required for episodes/checkpoints mode; unused/ignored for compare-algorithms mode.")
    parser.add_argument("--env-id", default="Reacher-v5",
                         help=f"Registered short names: {env_choices()}")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default=None)

    # Episodes mode
    parser.add_argument("--model-path", default=None,
                         help="Episodes mode: one model, --n-episodes cells, different starting positions.")
    parser.add_argument("--n-episodes", type=int, default=6)
    parser.add_argument("--episode-seed", type=int, default=123,
                         help="Base reset seed for episodes mode; cell i uses episode-seed + i.")
    parser.add_argument("--vecnormalize-path", default=None,
                         help="Path to a saved VecNormalize .pkl, if the model was trained with normalize: true "
                              "(episodes mode only -- checkpoints/compare mode look this up automatically).")

    # Checkpoints mode
    parser.add_argument("--seed", type=int, default=None,
                         help="Checkpoints mode: single training seed whose checkpoints to sample from.")
    parser.add_argument("--seeds", default=None,
                         help="Checkpoints mode: comma-separated seeds to play SEQUENTIALLY, one seed's "
                              "checkpoint-grid after another, in a single combined video (banner-labeled "
                              "'SEED N' per segment). Use this instead of --seed, not alongside it.")
    parser.add_argument("--fractions", default="0.02,0.05,0.1,0.2,0.5,1.0",
                         help="Checkpoints mode: comma-separated fractions of total training to sample at.")
    parser.add_argument("--checkpoint-episode-seed", type=int, default=1000,
                         help="Checkpoints mode: SAME reset seed used for every cell, so all checkpoints "
                              "solve the identical starting position -- training progress is the only "
                              "thing that differs between cells.")
    parser.add_argument("--use-best-as-final", action="store_true",
                         help="Checkpoints mode: substitute best_model.zip for whichever checkpoint the "
                              "highest requested fraction would otherwise pick, if best_model.zip exists.")

    # Compare-algorithms mode
    parser.add_argument("--compare-algos", default=None,
                         help="Compare-algorithms mode: comma-separated variant names, e.g. ddpg,td3,sac,ppo. "
                              "Each cell loads models/<env>/<algo>/seed<compare-seed>/<algo>_final.zip.")
    parser.add_argument("--compare-seed", type=int, default=0,
                         help="Compare-algorithms mode: which training seed's final model to use for each algo.")
    parser.add_argument("--compare-episode-seed", type=int, default=1000,
                         help="Compare-algorithms mode: SAME reset seed used for every cell, so every "
                              "algorithm solves the identical starting position -- a fair side-by-side.")
    parser.add_argument("--run-tag", default=None,
                         help="Checkpoints/compare-algorithms mode: point at experimental runs launched "
                              "with train.py --run-tag instead of the stock seed<N> folders.")
    add_camera_args(parser)

    args = parser.parse_args()

    checkpoints_mode = (args.seed is not None) or (args.seeds is not None)
    if args.seed is not None and args.seeds is not None:
        raise SystemExit("Pass --seed OR --seeds, not both.")
    modes_set = [bool(args.model_path), checkpoints_mode, bool(args.compare_algos)]
    if sum(modes_set) != 1:
        raise SystemExit("Pass exactly one of --model-path (episodes mode), --seed/--seeds (checkpoints mode), "
                          "or --compare-algos (compare-algorithms mode).")
    if (args.model_path or checkpoints_mode) and args.algo is None:
        raise SystemExit("--algo is required for episodes mode and checkpoints mode.")

    env_short = get_env_short_name(args.env_id)
    camera_config = build_camera_config(args)
    env_kwargs = {"render_mode": "rgb_array"}
    if camera_config:
        env_kwargs["default_camera_config"] = camera_config
    env = gym.make(args.env_id, **env_kwargs)
    fps = env.metadata.get("render_fps", 30)

    combined_frames = None  # set directly by checkpoints mode; other modes build `cells` instead
    cells = []

    if args.model_path:
        # --- Episodes mode ---
        algo_cls = get_algo_class(args.algo)
        model = algo_cls.load(args.model_path, device=args.device)
        obs_normalizer = (load_vecnormalize_stats(args.vecnormalize_path, args.env_id, args.episode_seed)
                           if args.vecnormalize_path else None)
        print(f"Recording {args.n_episodes} episodes from {args.model_path} ...")
        for i in range(args.n_episodes):
            seed = args.episode_seed + i
            frames = record_episode_frames(env, model, obs_normalizer, seed)
            cells.append((frames, f"episode {i} (seed {seed})"))
        default_out = os.path.join(REPO_ROOT, "videos", env_short, args.algo,
                                    f"{args.algo}_episodes_grid.mp4")

    elif checkpoints_mode:
        # --- Checkpoints mode (single seed, or multiple seeds played sequentially) ---
        algo_cls = get_algo_class(args.algo)
        fractions = [float(f.strip()) for f in args.fractions.split(",") if f.strip()]
        seeds_to_use = ([args.seed] if args.seed is not None
                         else [int(s.strip()) for s in args.seeds.split(",") if s.strip()])
        multi = len(seeds_to_use) > 1

        segments = []
        for seed in seeds_to_use:
            sf = seed_folder_name(seed, args.run_tag)
            print(f"Seed {seed} ({sf}): resolving checkpoints for fractions {fractions}"
                  + (" [best_model.zip substitution enabled]" if args.use_best_as_final else "") + ":")
            cell_specs = resolve_checkpoint_cells(env_short, args.algo, seed, args.run_tag,
                                                   fractions, args.use_best_as_final)

            vecnorm_path = os.path.join(REPO_ROOT, "models", env_short, args.algo, sf, f"{args.algo}_vecnormalize.pkl")
            obs_normalizer = (load_vecnormalize_stats(vecnorm_path, args.env_id, args.checkpoint_episode_seed)
                               if os.path.exists(vecnorm_path) else None)

            segment_frames = record_checkpoint_segment(
                env, algo_cls, cell_specs, obs_normalizer, args.checkpoint_episode_seed, args.device,
                seed_banner=f"SEED {seed}" if multi else None,
            )
            segments.append(segment_frames)

        combined_frames = [f for seg in segments for f in seg]

        if multi:
            seeds_str = "-".join(str(s) for s in seeds_to_use)
            tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
            default_out = os.path.join(REPO_ROOT, "videos", env_short, args.algo,
                                        f"{args.algo}_seeds{seeds_str}{tag_suffix}_checkpoints_grid.mp4")
        else:
            sf = seed_folder_name(seeds_to_use[0], args.run_tag)
            default_out = os.path.join(REPO_ROOT, "videos", env_short, args.algo,
                                        f"{args.algo}_{sf}_checkpoints_grid.mp4")

    else:
        # --- Compare-algorithms mode ---
        algos = [a.strip() for a in args.compare_algos.split(",") if a.strip()]
        for a in algos:
            if a not in variant_choices():
                raise SystemExit(f"Unknown algo '{a}'. Choices: {variant_choices()}")
        print(f"Comparing {len(algos)} algorithms, all replaying the same starting position "
              f"(seed {args.compare_episode_seed}):")

        for a in algos:
            sf = seed_folder_name(args.compare_seed, args.run_tag)
            model_path = os.path.join(REPO_ROOT, "models", env_short, a, sf, f"{a}_final.zip")
            if not os.path.exists(model_path):
                raise SystemExit(f"No final model found at {model_path}")
            vecnorm_path = os.path.join(REPO_ROOT, "models", env_short, a, sf, f"{a}_vecnormalize.pkl")
            obs_normalizer = (load_vecnormalize_stats(vecnorm_path, args.env_id, args.compare_episode_seed)
                               if os.path.exists(vecnorm_path) else None)

            print(f"  {a.upper()} -> {model_path}")
            algo_cls = get_algo_class(a)
            model = algo_cls.load(model_path, device=args.device)
            frames = record_episode_frames(env, model, obs_normalizer, args.compare_episode_seed)
            cells.append((frames, a.upper()))
        tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
        default_out = os.path.join(REPO_ROOT, "videos", env_short,
                                    f"compare_seed{args.compare_seed}{tag_suffix}_algos_grid.mp4")

    env.close()

    if combined_frames is None:
        if not cells:
            raise SystemExit("No cells recorded -- nothing to write.")
        combined_frames = compose_grid_frames(cells)

    if not combined_frames:
        raise SystemExit("No frames composed -- nothing to write.")

    out_path = args.out or default_out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_video_from_frames(combined_frames, fps, out_path)
    print(f"Saved grid video ({len(combined_frames)} frames) to {out_path}")


if __name__ == "__main__":
    main()
