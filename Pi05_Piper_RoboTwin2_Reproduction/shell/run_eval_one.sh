#!/usr/bin/env bash
# Local RoboTwin: seed=64 -> effective episode seed=6500000; one episode.
set -euo pipefail
cd "$HOME/robotics/RoboTwin"
export EVAL_ENV_TYPE=sim
export PYTHONPATH="$PWD:$PWD/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
test -f env_cfg/task_config/piper_pi05_eval.yml
mkdir -p eval_logs
bash scripts/eval_policy.sh \
  --bench_name RoboTwin \
  --task_name piper_stack_red_green \
  --env_cfg_type piper_single \
  --policy_name Pi_05 \
  --host 127.0.0.1 \
  --port 16000 \
  --protocol ws \
  --eval_batch false \
  --root_dir "$PWD" \
  --device_id 0 \
  --additional_info "ckpt_name=19999,action_type=joint" \
  --seed 64 \
  --task_config piper_pi05_eval \
  --test_num 1 \
  --instruction_type seen \
  --expert_check false \
  --frequency 50 \
  2>&1 | tee eval_logs/pi05_piper_19999_seed64_test1.log
