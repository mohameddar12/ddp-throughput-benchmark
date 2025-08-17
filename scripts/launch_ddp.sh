#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/base_config.yaml}
SEED=${2:-42}
SEQ_LEN=${3:-512}
PREC=${4:-fp16}
MAX_STEPS=${5:-100}
MASTER_PORT=${MASTER_PORT:-29501}
LOGTAG=${LOGTAG:-"bench"}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8
ulimit -n 65535 || true

RUN_ID=$(date +"%Y%m%d_%H%M%S")_${LOGTAG}_sl${SEQ_LEN}_${PREC}_seed${SEED}
OUTDIR=experiments/logs/${RUN_ID}
mkdir -p "$OUTDIR"

# Many repos let you override from CLI; adapt flags to your train.py
torchrun \
  --nproc_per_node=4 \
  --master_port=${MASTER_PORT} \
  train.py \
  --config ${CFG} \
  --seed ${SEED} \
  --model.max_length ${SEQ_LEN} \
  --training.precision ${PREC} \
  --training.max_steps ${MAX_STEPS} \
  --logging.log_dir ${OUTDIR} \
  --metrics.measure_throughput true \
  --metrics.measure_gpu_util true

# Optional: emit one-line JSON summary if your train.py writes metrics.jsonl
python scripts/summarize_one.py "$OUTDIR" || true