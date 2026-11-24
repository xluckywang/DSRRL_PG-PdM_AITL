#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="formal_experiments_v4/09_exp9_qrdqn_v6_model_comparison.py"
LOG_DIR="formal_experiments_v4/outputs/exp9_qrdqn_v6_model_comparison_parallel_logs"
SEEDS="${SEEDS:-11,22,33}"
EPISODE_GRID="${EPISODE_GRID:-100,300,500,1000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$ROOT"
export EXP7_DSHI_CACHE_TAG="__exp9_20260701_rms_protocol__"
export EXP7_DSHI_STRICT=0
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Exp9-v6 formal 4-GPU launcher"
echo "ROOT         : $ROOT"
echo "PYTHON_BIN   : $PYTHON_BIN"
echo "SCRIPT       : $SCRIPT"
echo "SEEDS        : $SEEDS"
echo "EPISODE_GRID : $EPISODE_GRID"
echo "EXTRA_ARGS   : $EXTRA_ARGS"
echo "LOG_DIR      : $LOG_DIR"
echo "============================================================"

launch_shard() {
  local gpu="$1"
  local suffix="$2"
  local methods="$3"
  local log_file="$LOG_DIR/${suffix}.log"
  echo "[GPU $gpu] launching $suffix"
  echo "         methods=$methods"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$SCRIPT" \
    --seeds "$SEEDS" \
    --episode-grid "$EPISODE_GRID" \
    --methods "$methods" \
    --output-suffix "$suffix" \
    $EXTRA_ARGS \
    > "$log_file" 2>&1 &
  echo "         pid=$! log=$log_file"
}

launch_shard 0 "shard_gpu0_core_dqn" "DQN,Double-DQN,Dueling-DQN,Double-Dueling-DQN"
launch_shard 1 "shard_gpu1_temporal_dqn" "SCORER-DQN,LSTM-DQN,HiLSTM-Attn-DQN"
launch_shard 2 "shard_gpu2_qrdqn" "QR-DQN,LSTM-QR-DQN,HiLSTM-Attn-QR-DQN"
launch_shard 3 "shard_gpu3_policy_gradient" "PPO,SAC"

echo "All Exp9-v6 shards launched. Waiting for completion..."
wait

echo "All shards finished. Merging tables..."
"$PYTHON_BIN" formal_experiments_v4/merge_exp9_v6_parallel_results.py
echo "Exp9-v6 4-GPU run complete."
