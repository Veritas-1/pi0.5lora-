#!/usr/bin/env bash
# Source this file: source /root/rivermind-data/pi05_cloud_env.sh
export PI05_ROOT=/root/rivermind-data
export UV_CACHE_DIR="$PI05_ROOT/uv-cache"
export HF_HOME="$PI05_ROOT/hf-cache"
export OPENPI_DATA_HOME="$PI05_ROOT/openpi-cache"
export UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1
export UV_PYTHON=3.11
export WANDB_MODE=offline
export PATH="$HOME/.local/bin:$PATH"
