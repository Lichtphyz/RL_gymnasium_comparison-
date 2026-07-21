#!/usr/bin/env bash
# run_all.sh -- train + evaluate every algorithm variant across multiple
# seeds, for one environment, unattended.
#
# Usage:
#   bash scripts/run_all.sh                                       # Reacher-v5, sequential, all variants, seeds 0-2
#   bash scripts/run_all.sh Pusher-v5 "ddpg td3 sac ppo" "0 1 2"    # Pusher-v5, sequential
#   PARALLEL_SEEDS=1 THREADS=4 bash scripts/run_all.sh Pusher-v5     # Pusher-v5, all seeds of each algo run together
#
# PARALLEL_SEEDS=0 (default) runs everything sequentially, one run at a
# time -- this is what makes the wall-clock-time panels in
# plot_comparison.py (reward vs. time, success rate vs. time) a rigorous,
# apples-to-apples comparison across algorithms.
#
# PARALLEL_SEEDS=1 launches all seeds of one algorithm together (still one
# algorithm at a time, not everything at once), each capped to $THREADS.
# Meaningfully faster wall-clock, and since every algorithm gets the same
# "N seeds in parallel" treatment, cross-algorithm wall-clock comparisons
# stay ROUGHLY informative -- but this is an approximation, not as
# rigorous as true dedicated-core sequential timing. Worth a one-line
# caveat if you use these numbers in a presentation.
#
# One failed run does not stop the batch (set -e is deliberately NOT used
# here) -- failures are logged and reported in the summary at the end.

set -uo pipefail

ENV_ID=${1:-"Reacher-v5"}
ALGOS=${2:-"ddpg td3 sac ppo"}
SEEDS=${3:-"0 1 2"}
THREADS=${THREADS:-4}
PARALLEL_SEEDS=${PARALLEL_SEEDS:-0}

case "$ENV_ID" in
  Reacher-v5) ENV_SHORT="reacher" ;;
  Pusher-v5)  ENV_SHORT="pusher" ;;
  *) echo "Unknown env-id '$ENV_ID'. Add it to this case statement AND to src/env_registry.py first."; exit 1 ;;
esac

export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="logs/run_all_logs"
mkdir -p "$LOG_DIR"
mkdir -p "plots/$ENV_SHORT"
SUMMARY_CSV="plots/$ENV_SHORT/batch_summary.csv"
rm -f "$SUMMARY_CSV"

FAILED_RUNS=()
START_TIME=$(date +%s)

echo "=== run_all.sh starting ==="
echo "Environment:     $ENV_ID ($ENV_SHORT)"
echo "Algos:           $ALGOS"
echo "Seeds:           $SEEDS"
echo "Threads per run: $THREADS"
echo "Parallel seeds:  $([ "$PARALLEL_SEEDS" = "1" ] && echo "yes -- wall-clock-time panels are approximate this run" || echo "no -- fully sequential, timing-rigorous")"
echo "Per-run logs:    $LOG_DIR/"
echo "Summary CSV:     $SUMMARY_CSV"
echo

train_one() {
  local algo=$1 seed=$2
  local train_log="$LOG_DIR/${ENV_SHORT}_${algo}_seed${seed}_train.log"
  python src/train.py --algo "$algo" --env-id "$ENV_ID" --seed "$seed" --device cpu > "$train_log" 2>&1
}

eval_one() {
  local algo=$1 seed=$2
  local run_name="${ENV_SHORT}_${algo}_seed${seed}"
  local model_path="models/$ENV_SHORT/$algo/seed${seed}/${algo}_final.zip"
  local vecnorm_path="models/$ENV_SHORT/$algo/seed${seed}/${algo}_vecnormalize.pkl"
  local eval_log="$LOG_DIR/${run_name}_eval.log"
  local eval_args=(--algo "$algo" --env-id "$ENV_ID" --model-path "$model_path" --n-episodes 50
                    --device cpu --train-seed "$seed" --summary-csv-out "$SUMMARY_CSV")
  [ -f "$vecnorm_path" ] && eval_args+=(--vecnormalize-path "$vecnorm_path")

  echo "[$(date '+%H:%M:%S')] Evaluating $run_name ..."
  if python src/evaluate.py "${eval_args[@]}" > "$eval_log" 2>&1; then
    echo "[$(date '+%H:%M:%S')] Evaluation $run_name done."
  else
    echo "[$(date '+%H:%M:%S')] !!! Evaluation $run_name FAILED -- see $eval_log"
    FAILED_RUNS+=("eval:$run_name")
  fi
}

for algo in $ALGOS; do
  if [ "$PARALLEL_SEEDS" = "1" ]; then
    declare -A SEED_PIDS
    for seed in $SEEDS; do
      run_name="${ENV_SHORT}_${algo}_seed${seed}"
      echo "[$(date '+%H:%M:%S')] Launching $run_name (parallel, $THREADS threads) ..."
      train_one "$algo" "$seed" &
      SEED_PIDS[$seed]=$!
    done
    for seed in $SEEDS; do
      run_name="${ENV_SHORT}_${algo}_seed${seed}"
      if wait "${SEED_PIDS[$seed]}"; then
        echo "[$(date '+%H:%M:%S')] Training $run_name done."
        eval_one "$algo" "$seed"
      else
        echo "[$(date '+%H:%M:%S')] !!! Training $run_name FAILED -- see $LOG_DIR/${run_name}_train.log"
        FAILED_RUNS+=("train:$run_name")
      fi
    done
    unset SEED_PIDS
  else
    for seed in $SEEDS; do
      run_name="${ENV_SHORT}_${algo}_seed${seed}"
      echo "[$(date '+%H:%M:%S')] Training $run_name ..."
      if train_one "$algo" "$seed"; then
        echo "[$(date '+%H:%M:%S')] Training $run_name done."
        eval_one "$algo" "$seed"
      else
        echo "[$(date '+%H:%M:%S')] !!! Training $run_name FAILED -- see $LOG_DIR/${run_name}_train.log"
        FAILED_RUNS+=("train:$run_name")
      fi
    done
  fi
  echo
done

END_TIME=$(date +%s)
ELAPSED_MIN=$(( (END_TIME - START_TIME) / 60 ))

echo "=== run_all.sh finished in ${ELAPSED_MIN} minutes ==="
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
  echo "Failed runs (check their log files above for details):"
  for r in "${FAILED_RUNS[@]}"; do echo "  - $r"; done
else
  echo "All runs completed successfully."
fi
echo

ALGOS_CSV=$(echo "$ALGOS" | tr ' ' ',')
SEEDS_CSV=$(echo "$SEEDS" | tr ' ' ',')
echo "Generating comparison plots across all completed runs..."
python src/plot_comparison.py --env-id "$ENV_ID" --algos "$ALGOS_CSV" --seeds "$SEEDS_CSV"

echo
echo "Done. See:"
echo "  plots/$ENV_SHORT/training_comparison.png       (3-panel comparison figure)"
echo "  plots/$ENV_SHORT/success_rate_vs_time.png       (2-panel success-rate figure)"
echo "  plots/$ENV_SHORT/training_comparison_data.csv  (raw per-checkpoint, per-seed numbers)"
echo "  $SUMMARY_CSV       (one row per algo/seed -- n=50 episode eval results)"
if [ "$PARALLEL_SEEDS" = "1" ]; then
  echo
  echo "NOTE: this batch ran with PARALLEL_SEEDS=1 -- treat the wall-clock-time"
  echo "panels (reward vs. time, success rate vs. time, training-time comparisons)"
  echo "as approximate, not as rigorous as a fully sequential run."
fi
