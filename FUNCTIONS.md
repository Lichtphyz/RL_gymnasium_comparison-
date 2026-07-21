# Function Reference

Brief description of every function in `src/`, grouped by file. For setup,
usage examples, and the overall project structure, see `README.md` — this
file is a quick-lookup reference, not a tutorial.

## `algo_registry.py`
Maps each algorithm "variant" name (`ddpg`, `td3`, `sac`, `ppo`) to its SB3
class and config file — the single source of truth every other script
imports from, so they can't disagree about what a variant name means.

- `get_algo_class(variant)` — returns the SB3 algorithm class for a variant name.
- `get_config_name(variant)` — returns the config file stem (under `configs/<env>/`) for a variant.
- `variant_choices()` — returns the list of valid `--algo` values, used for argparse `choices=`.

## `env_registry.py`
Maps each environment to its short name (used in all file paths), its
goal-distance function, and its success-criteria constants (threshold,
timing fraction, episode length). Adding a third environment means adding
one entry here, not touching the generic evaluation logic elsewhere.

- `_reacher_distance(obs)` — fingertip-to-target distance from Reacher's observation vector (`obs[8:10]`).
- `_pusher_distance(obs)` — object-to-goal xy distance from Pusher's observation vector (`obs[17:19] - obs[20:22]`); deliberately 2D, not 3D — see the module comments for why.
- `get_env_short_name(env_id)` — e.g. `"Reacher-v5"` → `"reacher"`.
- `get_env_spec(env_short_name)` / `get_env_spec_by_id(env_id)` — return the full spec dict (distance fn, threshold, timing fraction, episode length) for an environment.
- `env_choices()` — returns the list of valid `--env-id` short names.

## `success_metrics.py`
Environment-agnostic pass/fail logic — the same evaluation code runs for
every environment, given that environment's own threshold/timing/distance
function from `env_registry.py`.

- `evaluate_episode(distances_per_step, max_episode_steps, threshold, time_fraction)` — given one episode's per-step goal-distances, returns whether/when the target was reached, both "at all" and "within the timing fraction."

## `train.py`
Trains one algorithm/seed/environment combination. Supports resuming from
a checkpoint, overriding hyperparameters (noise, reward weights) from the
command line, and tagging experimental runs so they never collide with
stock results.

- `load_config(algo, env_short)` — loads the YAML hyperparameter config for a variant/environment.
- `build_model(algo, env, config, seed, tensorboard_log, device)` — constructs the SB3 model from a config dict, handling the noise/policy_kwargs string-eval convention.
- `main()` — CLI entry point: parses args, builds the env and model (or resumes one), trains, saves checkpoints and the final model.

## `evaluate.py`
Runs a trained model for N episodes against its environment's success
criteria and reports/logs the results.

- `load_vecnormalize_stats(vecnorm_path, env_id, seed)` — reloads saved `VecNormalize` observation-normalization stats (needed for PPO's `normalize: true` configs).
- `main()` — CLI entry point: loads the model, runs evaluation episodes, prints results, optionally writes per-episode and/or summary CSVs.

## `record_demo.py`
Records `.mp4` clips of a trained model, auto-tagged by seed and
checkpoint so repeated calls don't silently overwrite each other's output.

- `load_vecnormalize_stats(...)` — same as in `evaluate.py`.
- `derive_tag_from_path(model_path)` — builds a filename tag (e.g. `seed2_distweight2_final`) from a model's path, used to keep output filenames unique.
- `main()` — CLI entry point: records N episodes to video.

## `camera_utils.py`
Shared camera zoom/rotation CLI flags and config-building, used by every
video-generating script.

- `add_camera_args(parser)` — adds `--camera-distance`/`--camera-azimuth`/`--camera-elevation` to an argparse parser.
- `build_camera_config(args)` — turns those parsed args into the `default_camera_config` dict `gym.make()` expects.

## `plot_comparison.py`
Generates the multi-seed comparison figures (reward curves, goal-reaching
time, success rate vs. time, distance progress) and the raw CSV behind
them. Supports comparing algorithm variants against tagged experimental
runs (`--algos ddpg,ddpg:noise03`) in one call.

- `seed_folder_name(seed, run_tag)` — builds the `seed<N>` or `seed<N>_<tag>` folder name.
- `parse_condition(entry, global_run_tag)` — parses one `--algos` entry (`algo` or `algo:tag`) into its class, tag, display label, color, and linestyle.
- `find_best_tb_run(env_short, algo, seed, run_tag)` — finds the TensorBoard run directory with the most logged steps for a condition/seed.
- `load_scalar(run_dir, tag)` — reads one scalar's full history out of a TensorBoard run.
- `aggregate_reward_curves(env_short, algo, seeds, run_tag)` — mean±std reward curve vs. % of training, across seeds.
- `parse_log_table(env_short, algo, seed, run_tag)` — parses SB3's captured stdout progress tables for (step, wall-clock time, reward) triples.
- `aggregate_reward_vs_time(env_short, algo, seeds, run_tag)` — mean±std reward curve vs. wall-clock time.
- `find_checkpoints(env_short, algo, seed, run_tag)` — lists a seed's saved checkpoint files, sorted by step.
- `get_vecnormalize_for_ppo(env_short, env_id, algo, seed, run_tag)` — loads PPO's saved normalization stats, if present.
- `evaluate_checkpoint(...)` — runs N episodes through one checkpoint, returning success rates and distance metrics.
- `evaluate_checkpoints_for_seed(...)` — evaluates every checkpoint for one condition/seed, attaching estimated wall-clock time to each.
- `aggregate_reach_time_curves(algo, tag, seeds, checkpoint_cache, csv_rows)` — mean±std steps-to-reach vs. % of training; also appends full rows to the output CSV.
- `aggregate_metric_vs_pct(...)` / `aggregate_metric_vs_time(...)` — generic mean±std aggregation of any per-checkpoint metric, vs. training % or wall-clock time respectively.
- `main()` — CLI entry point: orchestrates all of the above into the four output files.

## `make_progression_video.py`
Stitches clips from several training checkpoints into one sequential
video, showing how behavior changes across training.

- `seed_folder_name(seed, run_tag)` — same as in `plot_comparison.py`.
- `find_checkpoints(env_short, algo, seed, run_tag)` — same as in `plot_comparison.py`.
- `pick_checkpoints_for_fractions(checkpoints, fractions)` — picks whichever saved checkpoint is closest to each requested training-progress fraction.
- `load_vecnormalize_stats(...)` — same pattern as `evaluate.py`.
- `label_frame(frame, text)` — burns a small corner label (step count, % of training) into a frame.
- `main()` — CLI entry point: records one episode per selected checkpoint, concatenates into one video.

## `make_grid_video.py`
Combines several episodes, checkpoints, or algorithms into one video where
all cells play *simultaneously* in a grid, instead of one after another.
Three modes: episodes (one model, many starting positions), checkpoints
(one model, several training-progress points, optionally multiple seeds
played sequentially), and compare-algorithms (several algorithms' final
models, same starting position).

- `load_vecnormalize_stats(...)` — same pattern as elsewhere.
- `label_frame(frame, text)` / `banner_frame(frame, text)` — small per-cell label vs. larger top-banner label (used for "SEED N" segment markers in multi-seed mode).
- `record_episode_frames(env, model, obs_normalizer, seed)` — runs one episode, returns the rendered frames.
- `choose_grid_shape(n)` — picks a roughly-square (rows, cols) layout for `n` cells.
- `compose_grid_frames(cells, pad, bg_color, seed_banner)` — composites several equal-length frame sequences into one synchronized grid video's frames.
- `write_video_from_frames(frames, fps, out_path)` — writes a frame list to an `.mp4` file.
- `resolve_checkpoint_cells(env_short, algo, seed, run_tag, fractions, use_best_as_final)` — picks checkpoints for the requested fractions, optionally substituting `best_model.zip` for the final one.
- `record_checkpoint_segment(...)` — records and composes one seed's full set of checkpoint cells into a grid segment.
- `main()` — CLI entry point: dispatches to whichever of the three modes was requested.

## `demo_pattern.py`
Sends a trained Reacher agent through a sequence of target coordinates
(rectangle or star), tracing the pattern with continuous arm motion
between waypoints — a qualitative demo of generalization beyond the
single-target task the model was actually trained on.

- `load_vecnormalize_stats(...)` — same pattern as elsewhere.
- `rectangle_pattern(size)` / `star_pattern(outer, inner, n_points)` — generate the waypoint coordinate lists.
- `label_frame(frame, text)` / `banner_frame(frame, text)` — per-waypoint corner label vs. per-checkpoint top banner (progression mode).
- `resolve_checkpoints(algo, seed, run_tag, fractions, use_best_as_final)` — same checkpoint-picking pattern as the other video scripts.
- `run_pattern_on_model(env, model, full_sequence, steps_per_waypoint, algo_label, checkpoint_label, obs_normalizer)` — runs one model through the full waypoint sequence by directly overwriting the target's position in MuJoCo's simulation state (`qpos[2:4]`) between segments, keeping the arm's own state continuous.
- `main()` — CLI entry point: single-model mode or checkpoint-progression mode.
