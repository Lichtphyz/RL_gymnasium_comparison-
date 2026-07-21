# RL Algorithm Comparison: Reacher & Pusher

MECH 262 course project: a rigorous comparison of continuous-control RL
algorithms on OpenAI Gymnasium's `Reacher-v5` and `Pusher-v5` (both MuJoCo).

## Algorithmic arc

| Variant | Role in the comparison |
|---|---|
| **DDPG** | Baseline off-policy actor-critic. |
| **TD3**  | DDPG + delayed policy updates + target policy smoothing (SB3's DDPG already uses a single critic by default, so this isolates TD3's other two fixes cleanly -- see the note below). |
| **SAC**  | Stochastic, entropy-regularized policy. Isolates the *entropy/exploration* axis relative to TD3/DDPG's deterministic policies. |
| **PPO**  | On-policy contrast to all of the above. |

Tabular Q-learning and DQN are excluded: both require discrete action
spaces, and Reacher/Pusher's action spaces are continuous.

### A resolved caveat, worth knowing about

We initially added a `ddpg_1critic` variant to force `n_critics=1`, based on
documentation suggesting SB3's `DDPG` defaults to `n_critics=2` (since its
policy class aliases `TD3Policy`). Empirically checking the installed SB3
version (2.9.0) showed regular `ddpg` already trains with a single critic
by default -- confirmed by both configs producing byte-identical trained
models. That variant was removed; the plain `ddpg` vs. `td3` comparison was
already the clean single-critic-vs-twin-critic ablation the whole time.

## Environments

| | Reacher-v5 | Pusher-v5 |
|---|---|---|
| Task | Move a 2-DOF arm's fingertip to a target point | Push an object to a goal position with a 7-DOF arm |
| Observation / action dims | 10 / 2 | 23 / 7 |
| Default episode length | 50 steps | 100 steps |
| Success metric | fingertip-to-target distance (`obs[8:10]`) | object-to-goal distance (`obs[17:20] - obs[20:23]`) |
| Success threshold | 0.02 | 0.02 (deliberately tighter than the object's own size, for a real precision bar) |
| "Main" success criterion | within 40% of episode (20 steps) | within 40% of episode (40 steps) |
| RL-Zoo tuned configs available? | Only for PPO | None for any algorithm |

Both environments' success-criteria definitions (distance function,
threshold, timing fraction, episode length) live in `src/env_registry.py`
-- adding a third environment later means adding one entry there, not
touching the generic evaluation logic in `success_metrics.py`.

## Repo structure

```
configs/<env-short>/         Per-variant hyperparameters (YAML), one folder per environment
scripts/
  run_all.sh                  Sequential batch: train + evaluate every variant x seed, unattended
src/
  algo_registry.py             Variant name -> (SB3 class, config file)
  env_registry.py               Env id -> (short name, distance fn, success criteria)
  train.py                       Train one variant/seed on one environment
  evaluate.py                     n=50-episode success-rate metrics for a trained model
  record_demo.py                  Record .mp4 clips for the live demo
  make_progression_video.py        Combine clips from several training-progress checkpoints into one video
  plot_comparison.py               Multi-seed comparison plots (reward, reach-time, training-time, success-vs-time)
  success_metrics.py               Generic (environment-agnostic) pass/fail logic
models/<env-short>/<algo>/seed<N>/  Checkpoints and final models (gitignored, .gitkeep only)
logs/<env-short>/<algo>/seed<N>/    TensorBoard logs + Monitor CSVs (gitignored)
logs/run_all_logs/                  Per-run stdout/stderr from run_all.sh (gitignored)
videos/<env-short>/<algo>/          Recorded demo clips + progression videos (gitignored)
plots/<env-short>/                  Comparison figures + CSVs (tracked -- small, worth keeping in the repo)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run everything for one environment (recommended)

```bash
bash scripts/run_all.sh                                    # Reacher-v5, all variants, seeds 0-2
bash scripts/run_all.sh Pusher-v5                            # Pusher-v5, all variants, seeds 0-2
bash scripts/run_all.sh Pusher-v5 "ddpg td3" "0 1"            # subset
THREADS=8 bash scripts/run_all.sh Pusher-v5                    # different thread cap
```

Trains and evaluates every variant/seed combination **sequentially**
(deliberately -- see the training-time note below), logs each run
separately, appends a summary row per run to
`plots/<env-short>/batch_summary.csv`, and finishes by generating both
comparison figures. One failed run doesn't stop the rest of the batch.

### Run pieces individually

```bash
python src/train.py --algo sac --env-id Pusher-v5 --seed 1
python src/evaluate.py --algo td3 --env-id Pusher-v5 --model-path models/pusher/td3/seed0/td3_final.zip --n-episodes 50
python src/record_demo.py --algo sac --env-id Pusher-v5 --model-path models/pusher/sac/seed0/sac_final.zip --n-episodes 3
python src/plot_comparison.py --env-id Pusher-v5 --algos ddpg,td3,sac,ppo --seeds 0,1,2
```

Watch training live in TensorBoard:
```bash
tensorboard --logdir logs
```

### Progression videos (across training, not just the final model)

```bash
python src/make_progression_video.py --algo sac --env-id Pusher-v5 --seed 0
python src/make_progression_video.py --algo sac --env-id Pusher-v5 --seed 0 \
    --fractions 0.01,0.05,0.1,0.25,0.5,0.75,1.0 --episodes-per-checkpoint 1
```

This runs entirely against already-saved checkpoints -- nothing needs to
be recorded live during training. `CheckpointCallback` (used by
`train.py`) already saves a full model snapshot every `--checkpoint-freq`
steps (25,000 by default), and each one is an independently loadable
policy. The script picks whichever saved checkpoint is closest to each
requested fraction of total training, records one (or more) episodes from
each, burns the step count into the frames, and stitches everything into
a single `.mp4` -- so you get one video that visibly shows learning
progress rather than a pile of separate clips to explain.

## A note on training-time and success-vs-time comparisons

`plot_comparison.py`'s wall-clock-time-based panels (reward vs. time,
success rate vs. time) are **only a fair comparison if every run had the
same CPU thread budget and none of them competed for cores with something
else running at the same time**. `run_all.sh` enforces this by running
everything sequentially with a fixed `OMP_NUM_THREADS`/`MKL_NUM_THREADS`
cap. If you train runs in parallel across multiple terminals instead, the
resulting wall-clock numbers reflect CPU contention as much as the
algorithms themselves.

Also worth knowing: CPU was measured faster than GPU for this project's
network sizes (small MLPs, small batch sizes) --
`train.py`/`evaluate.py`/`record_demo.py`/`make_progression_video.py` all
default to `--device cpu` for this reason.

## A note on hyperparameters

Reacher's `configs/reacher/ppo.yml` uses rl-baselines3-zoo's actual
Reacher-tuned values; the other three Reacher configs and **all four**
Pusher configs (`configs/pusher/*.yml`) do not have a tuned upstream
reference -- RL Zoo has no Pusher entry for any algorithm, and only a
PyBullet-variant entry (not directly comparable) for DDPG/TD3/SAC on
Reacher. Pusher's configs additionally scale `n_timesteps` well above
Reacher's, since it's a meaningfully harder task (7-DOF contact
manipulation vs. 2-DOF pure reaching) -- these numbers are starting points
to validate against a real learning curve, not settled values.

## References

- Reacher environment: https://gymnasium.farama.org/environments/mujoco/reacher/
- Pusher environment: https://gymnasium.farama.org/environments/mujoco/pusher/
- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/
- rl-baselines3-zoo (tuned hyperparameters): https://github.com/DLR-RM/rl-baselines3-zoo
- DDPG: https://arxiv.org/abs/1509.02971
- TD3: https://arxiv.org/abs/1802.09477
- SAC: https://arxiv.org/abs/1801.01290, https://arxiv.org/abs/1812.05905
- PPO: https://arxiv.org/abs/1707.06347
