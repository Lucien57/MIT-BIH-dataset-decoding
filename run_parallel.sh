#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Parallel launcher for ECG heartbeat classification project
# ============================================================
# Usage:
#   bash run_parallel.sh
#
# Notes:
#   1. Put data files under ./data by default:
#        data/dataset_raw.npy
#        data/labelset_raw.npy
#   2. Edit GPUS / SLOTS_PER_GPU / MODELS / SEEDS / FOLDS below as needed.
#   3. train.py expects --device in the form cuda:0, cuda:1, ...
#   4. Each job runs one stratified k-fold split: one test fold, next fold as val.
# ============================================================

# -------------------------
# user settings
# -------------------------
GPUS=(0 1 2 3 4 5 6 7)
SLOTS_PER_GPU=1

CONFIG_DIR="./configs"
LOG_DIR="./logs_parallel"
DATA_DIR="./data"

# Model names must match configs/<MODEL>.yaml.
MODELS=(
  NM2019
  LightCNN
  LSTM
  Transformer
  TransformerCNN
)

# For a quick sanity check, use SEEDS=(0) and FOLDS=(0).
SEEDS=(0 1 2 3 4)
FOLDS=(5)
N_SPLITS=10

# Optional overrides. Leave empty to use values in each config.
EPOCHS="100"
BATCH_SIZE=""

# -------------------------
# internal scheduler
# -------------------------
declare -A GPU_PIDS
for g in "${GPUS[@]}"; do GPU_PIDS[$g]=""; done

ALL_PIDS=()

cleanup() {
  echo
  echo "[WARN] Caught signal, terminating running jobs..."
  for pid in "${ALL_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${ALL_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  exit 1
}
trap cleanup INT TERM

prune_pids() {
  local g="$1"
  local keep=()
  for pid in ${GPU_PIDS[$g]}; do
    if kill -0 "$pid" 2>/dev/null; then
      keep+=("$pid")
    fi
  done
  GPU_PIDS[$g]="${keep[*]:-}"
}

num_running_on_gpu() {
  local g="$1"
  prune_pids "$g"
  if [[ -z "${GPU_PIDS[$g]}" ]]; then
    echo 0
  else
    echo "${GPU_PIDS[$g]}" | wc -w
  fi
}

RR_IDX=0

acquire_gpu_slot() {
  while true; do
    local num_gpus=${#GPUS[@]}
    for ((k=0; k<num_gpus; k++)); do
      local idx=$(( (RR_IDX + k) % num_gpus ))
      local g="${GPUS[$idx]}"
      local n
      n=$(num_running_on_gpu "$g")
      if (( n < SLOTS_PER_GPU )); then
        RR_IDX=$(( (idx + 1) % num_gpus ))
        echo "$g"
        return 0
      fi
    done
    sleep 1
  done
}

launch_job() {
  local model="$1"
  local seed="$2"
  local fold="$3"
  local gpu="$4"

  local cfg="${CONFIG_DIR}/${model}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] Config not found: $cfg"
    exit 1
  fi

  local out_dir="${LOG_DIR}/${model}/seed_${seed}"
  mkdir -p "$out_dir"
  local log="${out_dir}/fold_${fold}_gpu_${gpu}.log"

  local cmd=(
    python train.py
    --config "$cfg"
    --device "cuda:${gpu}"
    --seed "$seed"
    --fold "$fold"
    --n-splits "$N_SPLITS"
    --data-dir "$DATA_DIR"
  )

  if [[ -n "$EPOCHS" ]]; then
    cmd+=(--epochs "$EPOCHS")
  fi
  if [[ -n "$BATCH_SIZE" ]]; then
    cmd+=(--batch-size "$BATCH_SIZE")
  fi

  echo "[RUN] model=${model} seed=${seed} fold=${fold}/${N_SPLITS} gpu=${gpu} log=${log}"
  "${cmd[@]}" >"$log" 2>&1 &

  local pid=$!
  GPU_PIDS[$gpu]="${GPU_PIDS[$gpu]} $pid"
  ALL_PIDS+=("$pid")
}

# -------------------------
# build & run all jobs
# -------------------------
mkdir -p "$LOG_DIR"

total_jobs=0
for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      ((total_jobs+=1))
    done
  done
done

echo "[INFO] Models: ${MODELS[*]}"
echo "[INFO] Seeds: ${SEEDS[*]}"
echo "[INFO] Folds: ${FOLDS[*]} (N_SPLITS=${N_SPLITS})"
echo "[INFO] Total jobs: $total_jobs"
echo "[INFO] Concurrency: ${#GPUS[@]} GPUs × ${SLOTS_PER_GPU} slot(s) = $(( ${#GPUS[@]} * SLOTS_PER_GPU )) jobs in parallel"
echo

job_id=0
for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      ((job_id+=1))
      gpu=$(acquire_gpu_slot)
      echo "[INFO] Dispatch ${job_id}/${total_jobs} -> model=${model} seed=${seed} fold=${fold} GPU=${gpu}"
      launch_job "$model" "$seed" "$fold" "$gpu"
      sleep 0.1
    done
  done
done

echo
echo "[INFO] All jobs dispatched. Waiting for completion..."
fail_count=0
for pid in "${ALL_PIDS[@]}"; do
  if ! wait "$pid"; then
    ((fail_count+=1))
  fi
done

echo "[INFO] Done. Logs saved to: ${LOG_DIR}"
if (( fail_count > 0 )); then
  echo "[WARN] ${fail_count} job(s) failed. Check logs under: ${LOG_DIR}"
  exit 1
fi
