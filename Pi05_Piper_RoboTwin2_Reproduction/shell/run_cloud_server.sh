#!/usr/bin/env bash
# Cloud: load the trained step-19999 model and bind only to loopback.
set -euo pipefail
ROOT=/root/rivermind-data
PI05="$ROOT/XPolicyLab/policy/Pi_05"
OPENPI="$PI05/openpi"
source "$ROOT/pi05_cloud_env.sh"
source "$OPENPI/.venv/bin/activate"
export OPENPI_DATA_HOME="$ROOT/openpi-cache"
export PYTHONPATH="$ROOT:$ROOT/XPolicyLab:$OPENPI/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export PYTHONUNBUFFERED=1
RUN=RoboTwin-piper_stack_red_green_b16_20k-piper_single-joint-42
REPO=RoboTwin-piper_stack_red_green-piper_single-joint-v2
CKPT="$PI05/checkpoints/$RUN/19999"
DATASET="$ROOT/hf-cache/lerobot/$REPO"
test -d "$CKPT/params"
test -f "$DATASET/meta/info.json"
mkdir -p "$ROOT/experiment_logs"
python -u "$ROOT/serve_piper_pi05_xpl.py" \
  --checkpoint "$CKPT" \
  --dataset-root "$DATASET" \
  --host 127.0.0.1 \
  --port 6000 \
  --chunk-size 8 \
  2>&1 | tee "$ROOT/experiment_logs/pi05_piper_eval_server_19999.log"
