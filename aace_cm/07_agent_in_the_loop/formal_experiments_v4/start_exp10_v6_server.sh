#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="formal_experiments_v4/10_exp10_v6_preference_audit.py"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-formal_experiments_v4/outputs/exp10_v6_formal_${RUN_TAG}}"
LOG_DIR="$OUT_DIR/logs"
API_ENV="${API_ENV:-$HOME/.config/exp10_api.env}"
SEEDS="${SEEDS:-11 22 33}"
EPISODE_GRID="${EPISODE_GRID:-100,300,500,1000}"

cd "$ROOT"
mkdir -p "$LOG_DIR"
echo "$OUT_DIR" > formal_experiments_v4/outputs/exp10_v6_latest_formal_path.txt
exec > >(tee -a "$OUT_DIR/launcher.log") 2>&1
if [[ ! -f "$API_ENV" ]]; then
  echo "Missing private API environment file: $API_ENV" >&2
  exit 2
fi
set -a
source "$API_ENV"
set +a
printf '%s\n' "$$" > "$OUT_DIR/launcher.pid"
mkdir -p "$OUT_DIR/source_snapshot"
cp "$SCRIPT" "$OUT_DIR/source_snapshot/"
cp formal_experiments_v4/09_exp9_qrdqn_v6_model_comparison.py "$OUT_DIR/source_snapshot/"
cp formal_experiments_v4/start_exp10_v6_server.sh "$OUT_DIR/source_snapshot/"
sha256sum \
  "$SCRIPT" \
  formal_experiments_v4/09_exp9_qrdqn_v6_model_comparison.py \
  formal_experiments_v4/start_exp10_v6_server.sh \
  > "$OUT_DIR/source_hashes.sha256"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader > "$OUT_DIR/gpu_preflight.csv"

echo "============================================================"
echo "Exp10-v6 formal launcher"
echo "ROOT         : $ROOT"
echo "PYTHON_BIN   : $PYTHON_BIN"
echo "SCRIPT       : $SCRIPT"
echo "OUT_DIR      : $OUT_DIR"
echo "SEEDS        : $SEEDS"
echo "EPISODE_GRID : $EPISODE_GRID"
echo "============================================================"

run_seed_stage() {
  local gpu="$1"
  local seed="$2"
  local phase="$3"
  local log="$LOG_DIR/${phase}_seed${seed}.log"
  echo "[$(date '+%F %T')] GPU=$gpu phase=$phase seed=$seed"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$SCRIPT" \
    --phase "$phase" \
    --out-dir "$OUT_DIR" \
    --seed "$seed" \
    --seeds "11,22,33" \
    --episode-grid "$EPISODE_GRID" \
    > "$log" 2>&1
}

worker() {
  local gpu="$1"
  local seed="$2"
  run_seed_stage "$gpu" "$seed" reproduce
  run_seed_stage "$gpu" "$seed" validation
}

gpu=0
pids=()
for seed in $SEEDS; do
  if [[ -f "$OUT_DIR/completion_reproduce_seed${seed}.json" && -f "$OUT_DIR/completion_validation_seed${seed}.json" ]]; then
    echo "[$(date '+%F %T')] stage1 seed=$seed already complete; skipping"
  else
    worker "$gpu" "$seed" > "$LOG_DIR/worker_seed${seed}.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    echo "$pid" > "$OUT_DIR/worker_seed${seed}.pid"
  fi
  gpu=$((gpu + 1))
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
touch "$OUT_DIR/STAGE_1_REPRODUCTION_VALIDATION_COMPLETE"

if [[ -f "$OUT_DIR/completion_select_seed11.json" && -f "$OUT_DIR/a2_selection/selection.json" ]]; then
  echo "[$(date '+%F %T')] Agent-guided selection already complete; skipping"
else
  echo "[$(date '+%F %T')] Agent-guided guarded selection"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$SCRIPT" \
    --phase select \
    --out-dir "$OUT_DIR" \
    --seeds "11,22,33" \
    --episode-grid "$EPISODE_GRID" \
    > "$LOG_DIR/select.log" 2>&1
fi
touch "$OUT_DIR/STAGE_2_SELECTION_COMPLETE"

gpu=0
pids=()
for seed in $SEEDS; do
  if [[ -f "$OUT_DIR/completion_final_seed${seed}.json" ]]; then
    echo "[$(date '+%F %T')] final seed=$seed already complete; skipping"
  else
    run_seed_stage "$gpu" "$seed" final > "$LOG_DIR/final_worker_seed${seed}.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    echo "$pid" > "$OUT_DIR/final_seed${seed}.pid"
  fi
  gpu=$((gpu + 1))
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
if [[ -f "$OUT_DIR/completion_merge_seed11.json" ]]; then
  echo "[$(date '+%F %T')] final merge already complete; skipping"
else
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$SCRIPT" \
    --phase merge \
    --out-dir "$OUT_DIR" \
    --seeds "11,22,33" \
    --episode-grid "$EPISODE_GRID" \
    > "$LOG_DIR/merge.log" 2>&1
fi
touch "$OUT_DIR/STAGE_3_FINAL_TEST_COMPLETE"

gpu=0
pids=()
for seed in $SEEDS; do
  if [[ -f "$OUT_DIR/completion_audit_seed${seed}.json" ]]; then
    echo "[$(date '+%F %T')] audit seed=$seed already complete; skipping"
  else
    run_seed_stage "$gpu" "$seed" audit > "$LOG_DIR/audit_worker_seed${seed}.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    echo "$pid" > "$OUT_DIR/audit_seed${seed}.pid"
  fi
  gpu=$((gpu + 1))
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
if [[ -f "$OUT_DIR/completion_audit-merge_seed11.json" ]]; then
  echo "[$(date '+%F %T')] audit merge already complete; skipping"
else
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$SCRIPT" \
    --phase audit-merge \
    --out-dir "$OUT_DIR" \
    --seeds "11,22,33" \
    > "$LOG_DIR/audit_merge.log" 2>&1
fi
touch "$OUT_DIR/STAGE_4_AUDIT_COMPLETE"

cat > "$OUT_DIR/FORMAL_RUN_COMPLETE.txt" <<EOF
Exp10-v6 formal run completed at $(date '+%F %T %Z')
Output: $OUT_DIR
EOF
echo "[$(date '+%F %T')] Exp10-v6 formal run complete: $OUT_DIR"
