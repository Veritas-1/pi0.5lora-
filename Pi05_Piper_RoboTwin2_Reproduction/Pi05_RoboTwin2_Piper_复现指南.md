# Pi0.5 × RoboTwin 2.0：Piper 单臂堆叠任务复现指南

>
> **适用环境**：本地 Ubuntu 22.04 + 已配置的 RoboTwin 2.0 / Piper 单臂（RTX 4060）；云端 Ubuntu 22.04 + NVIDIA L40S 48GB（OpenPI + JAX）。
>

## 0. 最终实验参数与目录

| 项目 | 最终配置 |
|---|---|
| 任务 | `piper_stack_red_green` |
| 仿真 | RoboTwin 2.0，`piper_single`，`joint` 动作 |
| 演示数据 | HDF5 60 集，LeRobot 20,127 帧 |
| 数据集 ID | `RoboTwin-piper_stack_red_green-piper_single-joint-v2` |
| 训练配置 | `pi05_piper_single_lora` |
| 基础模型 | `pi05_base`，PaliGemma + action expert，LoRA |
| LoRA | PaliGemma rank 16；action expert rank 32 |
| 模型动作宽度 / 时域 | 32D（Piper 有效 7D）/ 50 steps |
| 优化 | batch size 16；20,000 train steps；seed 42 |
| Checkpoint | `RoboTwin-piper_stack_red_green_b16_20k-piper_single-joint-42/19999` |
| 观测 | 头部 RGB + 左腕 RGB + 7D 关节/夹爪状态 |
| 推理 chunk | 每次预测 50 步，实际返回前 8 步 |
| 网络 | 云端 `127.0.0.1:6000` → SSH → 本地 `127.0.0.1:16000` |
| 正式评测 | `seed=64`、20 轮、GUI、单环境闭环 |

**关键路径：**

```bash
# 云端
ROOT=/root/rivermind-data
PI05=$ROOT/XPolicyLab/policy/Pi_05
OPENPI=$PI05/openpi
REPO=RoboTwin-piper_stack_red_green-piper_single-joint-v2
RUN=RoboTwin-piper_stack_red_green_b16_20k-piper_single-joint-42
CKPT=$PI05/checkpoints/$RUN/19999
DATASET=$ROOT/hf-cache/lerobot/$REPO

# 本地
LOCAL_ROBOTWIN=$HOME/robotics/RoboTwin
```

**注意**：以下代码以“新环境复现”为顺序；对于已经完成训练的现有机器，直接跳转到第 7 节，不要重新运行安装、转换或训练命令。尤其不能重新执行可能覆盖 checkpoint 的训练脚本。

## 1. 本地 RoboTwin 环境

沿用已安装成功的 `RoboTwin` Conda 环境和现有 Piper 任务代码，不重装、不清理 Git 仓库。

```bash
conda activate RoboTwin
cd ~/robotics/RoboTwin

python - <<'PY'
import torch
import sapien
import mplib
import curobo
import websockets
import msgpack
import msgpack_numpy
import yaml

assert torch.cuda.is_available(), 'Local simulation CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(0))
print('websockets:', websockets.__version__)
print('LOCAL ROBOTWIN ENVIRONMENT OK')
PY
```

可选：保存可复现的版本记录（仅首次部署时执行一次）。

```bash
mkdir -p ~/robotics/experiment_logs
cd ~/robotics/RoboTwin

git -C XPolicyLab rev-parse HEAD > ~/robotics/experiment_logs/pi05_xpolicylab_sha.txt
conda env export -n RoboTwin > ~/robotics/experiment_logs/pi05_local_robotwin_env.yml
python -m pip freeze > ~/robotics/experiment_logs/pi05_local_robotwin_freeze.txt
```

## 2. 云端 OpenPI 安装

### 2.1 工作目录、基础依赖与缓存环境

```bash
apt-get update
apt-get install -y \
  git git-lfs curl wget ca-certificates build-essential \
  cmake ninja-build pkg-config ffmpeg libgl1 libglib2.0-0 libegl1 unzip

git lfs install
mkdir -p /root/rivermind-data/{tmp,uv-cache,hf-cache,openpi-cache,data,experiment_logs}
```

将压缩包中的 `shell/pi05_cloud_env.sh` 放到 `/root/rivermind-data/pi05_cloud_env.sh`，再加载：

```bash
source /root/rivermind-data/pi05_cloud_env.sh

curl -LsSf https://astral.sh/uv/install.sh -o /tmp/install_uv.sh
sh /tmp/install_uv.sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
```

### 2.2 固定 XPolicyLab 版本并安装

```bash
source /root/rivermind-data/pi05_cloud_env.sh
cd /root/rivermind-data

git clone https://github.com/XPolicyLab/XPolicyLab.git
cd XPolicyLab
git checkout 34d6135d609d46a10ccb81da24d5ddad0b4c1f7a
git submodule update --init --recursive
```

固定版本为 `34d6135d609d46a10ccb81da24d5ddad0b4c1f7a`。在已有仓库中不要重新 clone、reset 或 clean。

在网络较慢的环境中，安装时使用可访问的 PyPI 镜像；这是最终命令的一部分，不保留原文中的多轮镜像排错过程。

```bash
source /root/rivermind-data/pi05_cloud_env.sh
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export UV_HTTP_TIMEOUT=120
export UV_HTTP_RETRIES=5
export UV_CONCURRENT_DOWNLOADS=8
unset UV_OFFLINE

cd /root/rivermind-data/XPolicyLab/policy/Pi_05
bash -e install.sh 2>&1 | tee /root/rivermind-data/experiment_logs/pi05_install.log

source openpi/.venv/bin/activate
cd openpi
uv run python -c 'import openpi; print("OPENPI READY")'
python - <<'PY'
import jax
print('JAX devices:', jax.devices())
assert any(x.platform == 'gpu' for x in jax.devices()), 'JAX GPU unavailable'
PY
```

**不要设置 `UV_OFFLINE=1`**：`train.sh` 内部会使用 `uv run`，离线构建模式可能缺少 editable 构建依赖。这里保留的是排除该问题后的有效设置。

## 3. 60 集 HDF5 转换为 LeRobot 数据集

### 3.1 本地打包、上传、云端解压

**本地 Ubuntu：**

```bash
cd ~/robotics/RoboTwin

tar -czf ~/piper_60episodes_hdf5.tar.gz \
  -C data/piper_record_one/piper_stack_red_green/piper \
  data seed.txt
```

将 `~/piper_60episodes_hdf5.tar.gz` 上传至云端 `/root/rivermind-data/`（可通过 JupyterLab 上传）。

**云端：**

```bash
ROOT=/root/rivermind-data
mkdir -p "$ROOT/data/piper_record_one/piper_stack_red_green/piper"

tar -xzf "$ROOT/piper_60episodes_hdf5.tar.gz" \
  -C "$ROOT/data/piper_record_one/piper_stack_red_green/piper"
```

### 3.2 安装专用转换器并生成最终数据集

将 `python/process_piper_data.py` 复制到云端：

`/root/rivermind-data/XPolicyLab/policy/Pi_05/openpi/scripts/process_piper_data.py`

该脚本读取 HDF5 的左右关节与动作字段以及两个真实 RGB 相机；**不要使用默认针对双臂/三相机的通用转换器**。

```bash
ROOT=/root/rivermind-data
source "$ROOT/pi05_cloud_env.sh"
source "$ROOT/XPolicyLab/policy/Pi_05/openpi/.venv/bin/activate"
cd "$ROOT/XPolicyLab/policy/Pi_05/openpi"

python scripts/process_piper_data.py \
  --repo-id RoboTwin-piper_stack_red_green-piper_single-joint-v2 \
  --max-episodes 60 \
  2>&1 | tee "$ROOT/experiment_logs/pi05_piper_convert.log"
```

**为什么命令中明确填写 `-v2`：**原始记录前期出现过不带 `-v2` 的调用，但最终训练、统计量、checkpoint 与推理实际使用的 ID 都是 `...-v2`。此处统一为最终真实路径，避免重新复现时产生两个不同 ID 的数据目录。转换器对已存在目标数据集会拒绝覆盖；已完成转换时不要重复执行。

预期有效数据：**60 episodes / 20,127 frames**，每帧 7D state、7D action，头部+左腕两张 `240×320` RGB 图像。

可以执行包内 `python/verify_piper_dataset.py` 检查最终数据集的集数、帧数、7D 状态和双相机图像（将其上传到云端后运行）：

```bash
source /root/rivermind-data/pi05_cloud_env.sh
source /root/rivermind-data/XPolicyLab/policy/Pi_05/openpi/.venv/bin/activate
python /root/rivermind-data/verify_piper_dataset.py
```

## 4. 注册 Piper 专用 Pi0.5 适配与训练配置

### 4.1 复制文件、注册配置（一次性）

从本 ZIP 使用的是**最终 20,000-step 配置**，不需要再执行原记录中修改学习率日程和 batch size 的临时 Python 补丁。

在云端放置文件：

| 包内文件 | 云端目标位置 |
|---|---|
| `python/piper_policy.py` | `$OPENPI/src/openpi/policies/piper_policy.py` |
| `python/piper_config.py` | `$OPENPI/src/openpi/training/piper_config.py` |
| `shell/register_piper_config.sh` | `$ROOT/register_piper_config.sh` |

执行：

```bash
ROOT=/root/rivermind-data
OPENPI=$ROOT/XPolicyLab/policy/Pi_05/openpi
source "$ROOT/pi05_cloud_env.sh"
source "$OPENPI/.venv/bin/activate"

bash "$ROOT/register_piper_config.sh"

python - <<'PY'
from openpi.training.config import get_config
cfg = get_config('pi05_piper_single_lora')
assert cfg.model.pi05
assert cfg.model.action_dim == 32
assert cfg.model.action_horizon == 50
assert cfg.batch_size == 16
assert cfg.num_train_steps == 20000
print(cfg.name, 'batch=', cfg.batch_size, 'steps=', cfg.num_train_steps)
PY
```

注册脚本只对指定 XPolicyLab commit 执行一次；如果 `config.py` 已包含 `_PIPER_CONFIGS`，不要重复运行注册。

### 4.2 最终适配规则

| 项目 | 实际映射 / 处理 |
|---|---|
| 头部相机 | RoboTwin `cam_head` → `observation.images.cam_high` → `base_0_rgb` |
| 左腕相机 | RoboTwin `cam_left_wrist` → `observation.images.cam_left_wrist` → `left_wrist_0_rgb` |
| 右腕相机 | **不存在**；`right_wrist_0_rgb` 使用黑图、mask=False |
| 7D state/action | `[joint_1, …, joint_6, gripper]` |
| 动作语义 | 原始 `action[t] = state[t+1]`；前 6 个关节变为 delta，夹爪仍为绝对值 |
| 归一化 | 7D 统计量；模型空间补齐至 32D；推理先裁回 7D 再逆归一化 |
| 动作时域 | 50 步 |

语言指令来自数据集的 `task` 字段，训练配置使用 `prompt_from_task=True`：

```text
Place the green block on the base, then stack the red block on the green block.
```

`piper_config.py` 中仍有原始 `default_prompt = "Stack the red block on the green block."`，但**实际训练样本的 prompt 覆盖默认值**。已检查首帧、中间帧和末帧，实际语言均为上面完整的两阶段英文指令；云端服务的 fallback prompt 与它相同。无需因默认值不同而重新训练。

如需重新核对训练输入的语言，可在此配置已注册后，将包内 `python/verify_training_prompt.py` 放到云端 `/root/rivermind-data/verify_training_prompt.py`，执行：

```bash
source /root/rivermind-data/pi05_cloud_env.sh
source /root/rivermind-data/XPolicyLab/policy/Pi_05/openpi/.venv/bin/activate
python /root/rivermind-data/verify_training_prompt.py
```

## 5. 计算归一化统计量

```bash
ROOT=/root/rivermind-data
OPENPI=$ROOT/XPolicyLab/policy/Pi_05/openpi
source "$ROOT/pi05_cloud_env.sh"
source "$OPENPI/.venv/bin/activate"
cd "$OPENPI"

export TMPDIR=$ROOT/tmp
mkdir -p "$TMPDIR"

python scripts/compute_norm_stats.py \
  --config-name pi05_piper_single_lora \
  2>&1 | tee "$ROOT/experiment_logs/pi05_piper_norm_stats.log"
```

输出位置：

```text
/root/rivermind-data/XPolicyLab/policy/Pi_05/openpi/assets/
└── pi05_piper_single_lora/
    └── RoboTwin-piper_stack_red_green-piper_single-joint-v2/
        └── norm_stats.json
```

`state` 与 `actions` 的均值、标准差、1% 与 99% 分位数维度应为 **7D**。

## 6. Pi0.5 基础权重及正式训练

### 6.1 基础权重缓存

本次实验实际使用：

```text
/root/rivermind-data/openpi-cache/openpi-assets/checkpoints/pi05_base/params
```

如果该目录已有官方 `pi05_base` 参数，直接跳过权重下载。原始流程曾因云端网络较慢而先在本地下载，再上传云端；可使用以下可重现路径。

**本地 Ubuntu：**

```bash
python3 -m venv ~/gsutil-venv
~/gsutil-venv/bin/pip install gsutil
mkdir -p ~/pi05_download

~/gsutil-venv/bin/gsutil -m \
  cp -r gs://openpi-assets/checkpoints/pi05_base/params ~/pi05_download/

tar -czf ~/pi05_base_params.tar.gz -C ~/pi05_download params
scp -P 30185 ~/pi05_base_params.tar.gz \
  root@sh03-ssh.gpuhome.cc:/root/rivermind-data/
```

**云端：**

```bash
ROOT=/root/rivermind-data
mkdir -p "$ROOT/openpi-cache/openpi-assets/checkpoints/pi05_base"
tar -xzf "$ROOT/pi05_base_params.tar.gz" \
  -C "$ROOT/openpi-cache/openpi-assets/checkpoints/pi05_base"
```

上述打包/传输命令用于复现最终权重缓存目录；原始附件没有完整记录当时的传输细节，因此不能把这一组补齐的传输命令当作历史执行日志。基础权重和 20k-step checkpoint 都不在本 ZIP 内。

### 6.2 正式训练：batch 16、20,000 steps、seed 42

```bash
ROOT=/root/rivermind-data
PI05=$ROOT/XPolicyLab/policy/Pi_05
OPENPI=$PI05/openpi

source "$ROOT/pi05_cloud_env.sh"
source "$OPENPI/.venv/bin/activate"

unset UV_OFFLINE
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export UV_HTTP_TIMEOUT=120
export UV_HTTP_RETRIES=5
export UV_CONCURRENT_DOWNLOADS=8

export OPENPI_TRAIN_CONFIG_NAME=pi05_piper_single_lora
export OPENPI_LEROBOT_REPO_ID=RoboTwin-piper_stack_red_green-piper_single-joint-v2
export OPENPI_FSDP_DEVICES=1
export OPENPI_LOCAL_CACHE_ROOT=$ROOT/openpi-cache
export OPENPI_DATA_HOME=$ROOT/openpi-cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

cd "$OPENPI"
uv run python -c 'import openpi; print("OPENPI BUILD READY")'
cd "$PI05"

CKPT_NAME=piper_stack_red_green_b16_20k
CKPT_DIR="$PI05/checkpoints/RoboTwin-${CKPT_NAME}-piper_single-joint-42"
mkdir -p "$ROOT/experiment_logs"

if [ -e "$CKPT_DIR" ]; then
  echo "Existing checkpoint: $CKPT_DIR — do NOT overwrite."
else
  set -o pipefail
  bash train.sh \
    RoboTwin \
    "$CKPT_NAME" \
    piper_single \
    joint \
    42 \
    0 \
    2>&1 | tee "$ROOT/experiment_logs/pi05_piper_b16_20k_seed42.log"
fi
```

最终 checkpoint 路径：

```text
/root/rivermind-data/XPolicyLab/policy/Pi_05/checkpoints/
└── RoboTwin-piper_stack_red_green_b16_20k-piper_single-joint-42/
    ├── 5000/
    ├── 10000/
    ├── 15000/
    └── 19999/      # 实际用于推理
```

20,000 个训练 step 的最终目录是 **`19999`**，不是 `20000`。训练权重与相应的归一化 assets 必须成套保留。

## 7. 云端推理服务：Pi0.5 + WebSocket

### 7.1 部署专用服务脚本

将 `python/serve_piper_pi05_xpl.py` 上传到：

```text
/root/rivermind-data/serve_piper_pi05_xpl.py
```

这个脚本直接用 `policy_config.create_trained_policy(...)` 加载最终 checkpoint，同时复用 XPolicyLab 的 `PolicyServer` / `PolicyServerConfig` WebSocket 协议，不需要修改 XPolicyLab 通用 `model.py`。

在**云端独立终端**执行（同包 `shell/run_cloud_server.sh` 等效）：

```bash
ROOT=/root/rivermind-data
PI05=$ROOT/XPolicyLab/policy/Pi_05
OPENPI=$PI05/openpi

source "$ROOT/pi05_cloud_env.sh"
source "$OPENPI/.venv/bin/activate"

export OPENPI_DATA_HOME=$ROOT/openpi-cache
export PYTHONPATH="$ROOT:$ROOT/XPolicyLab:$OPENPI/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export PYTHONUNBUFFERED=1

RUN=RoboTwin-piper_stack_red_green_b16_20k-piper_single-joint-42
REPO=RoboTwin-piper_stack_red_green-piper_single-joint-v2
CKPT="$PI05/checkpoints/$RUN/19999"
DATASET="$ROOT/hf-cache/lerobot/$REPO"

mkdir -p "$ROOT/experiment_logs"
set -o pipefail
python -u "$ROOT/serve_piper_pi05_xpl.py" \
  --checkpoint "$CKPT" \
  --dataset-root "$DATASET" \
  --host 127.0.0.1 \
  --port 6000 \
  --chunk-size 8 \
  2>&1 | tee "$ROOT/experiment_logs/pi05_piper_eval_server_19999.log"
```

服务常驻终端，监听**云端 `127.0.0.1:6000`**。原始数据集缺少 `meta/tasks.jsonl` 时，服务会回退到完整的正确语言指令；这与训练样本的实际 `prompt` 一致，不是语言丢失。

### 7.2 本地 SSH 转发（最终有效端口）

在**本地另一个终端**执行并保持运行：

```bash
ssh -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -N \
  -L 127.0.0.1:16000:127.0.0.1:6000 \
  -p 30185 \
  root@sh03-ssh.gpuhome.cc
```

实际链路：

```text
本地 RoboTwin Ubuntu (RTX 4060)            云端 OpenPI (L40S)
   cam_head + cam_left_wrist + 7D State
                   |
        ws://127.0.0.1:16000
                   | SSH 端口转发
                   +----------------------> 127.0.0.1:6000
                                           Pi0.5 checkpoint 19999
                   <---------------------- 7D Action Chunk (8 steps)
                   |
                 Piper 仿真执行
```

**最终选用本地 16000 端口**；原先本地 6000 曾出现握手超时，已由改用 16000 后成功连通。两个环境都使用 6000 时的排错过程不纳入最终操作步骤。

### 7.3 一次性连接验收

在本地 `RoboTwin` 环境执行：

```bash
conda activate RoboTwin
cd ~/robotics/RoboTwin

python - <<'PY'
from client_server.ws import WsModelClient

client = WsModelClient(
    url='ws://127.0.0.1:16000',
    evaluation_id='piper-pi05-connectivity',
    trial_id='connection-test',
    action_case_id='piper-stack',
)
client.call(func_name='reset')
print('PI05 CLOUD WEBSOCKET CONNECTED')
client.close()
PY
```

本次已验证出现 `[CONNECTED]` 和 `PI05 CLOUD WEBSOCKET CONNECTED`。本地仿真 GPU 波动、云端显存占用较高但利用率间歇为零，符合“本地仿真 + 云端按需动作推理”的工作方式。

## 8. RoboTwin 2.0 正式仿真与评测（保留唯一有效路径）

### 8.1 评测前提

- 本地已有并且此前**成功运行过**的 Piper 任务配置：`env_cfg/task_config/piper_pi05_eval.yml`。
- 本地处于 `RoboTwin` Conda 环境；云端推理服务和本地 SSH 隧道一直运行。
- 策略使用 `Pi_05`，`joint` 动作，WebSocket 地址 `127.0.0.1:16000`。
- 本次采用 `scripts/eval_policy.sh` 直接显式指定单环境、seed、任务配置和轮数；**不使用** `setup_eval_env_client.sh` 的未明确参数化调用作为最终命令。

**重要资料边界**：原始附件没有包含 `piper_pi05_eval.yml` 的完整实际内容，因此 ZIP 不伪造这个配置。请保留你本地已经成功跑通的原版 YAML；下节只修改其中 GUI 相关字段。

### 8.2 启用 GUI（20 轮前执行）

将 `python/set_eval_gui.py` 放到本地任意位置，例如 `~/robotics/RoboTwin/set_eval_gui.py`，然后运行：

```bash
conda activate RoboTwin
cd ~/robotics/RoboTwin
python set_eval_gui.py \
  --config env_cfg/task_config/piper_pi05_eval.yml \
  --render-freq 5
```

仅设置下列选项，**其余机器人物理与相机参数不变**：

```yaml
render_freq: 5
eval_video_log: true
collect_data: false
```

`render_freq: 5` 用于打开 GUI 并周期性刷新；`1` 则刷新更频繁，但仿真开销可能更高。窗口应显示于本地 Ubuntu 桌面环境。

### 8.3 已跑通的单轮测试（seed=64）

```bash
cd ~/robotics/RoboTwin
conda activate RoboTwin

export EVAL_ENV_TYPE=sim
export PYTHONPATH="$PWD:$PWD/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p eval_logs

set -o pipefail
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
  --additional_info 'ckpt_name=19999,action_type=joint' \
  --seed 64 \
  --task_config piper_pi05_eval \
  --test_num 1 \
  --instruction_type seen \
  --expert_check false \
  --frequency 50 \
  2>&1 | tee eval_logs/pi05_piper_19999_seed64_test1.log
```

这条单轮闭环链路已在此前实际运行成功。可以用包内 `shell/run_eval_one.sh` 代替粘贴完整命令。

### 8.4 正式 20 轮 + GUI（保留此命令）

```bash
cd ~/robotics/RoboTwin
conda activate RoboTwin

export EVAL_ENV_TYPE=sim
export PYTHONPATH="$PWD:$PWD/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p eval_logs

set -o pipefail
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
  --additional_info 'ckpt_name=19999,action_type=joint' \
  --seed 64 \
  --task_config piper_pi05_eval \
  --test_num 20 \
  --instruction_type seen \
  --expert_check false \
  --frequency 50 \
  2>&1 | tee eval_logs/pi05_piper_19999_seed64_eval20_gui.log
```

也可以使用包内 `shell/run_eval_20_gui.sh`，其参数与上面一致。

#### Seed 与 DP 的对齐

本次沿用 DP 的种子规则：

\[
\text{actual\_seed}_i=(\text{seed}+1)\times100000+i.
\]

传入 `--seed 64`，首轮实际 seed 为 **6,500,000**。如果 20 轮都正常初始化，依次为 `6500000`、`6500001`、…、`6500019`。**不要直接向 `--seed` 传 `6500000`**，否则会被重复映射。若 RoboTwin 跳过某个初始化失败的 seed，须以日志中实际执行的 seed 为准。

#### 日志与结果判定

```bash
cd ~/robotics/RoboTwin
grep -Ei 'success rate|success!|fail!|current seed|traceback|error' \
  eval_logs/pi05_piper_19999_seed64_eval20_gui.log | tail -60
```

区分**“模型完成一个动作序列”**与**“环境成功判定”**：例如错误地把蓝块放到底座、再把红块放到蓝块上，机械臂虽然完成叠放，但任务语义不满足，不应计为成功。本次记录已经确认出现过这类现象；这属于模型目标选择/任务恢复能力需要分析的问题，不能直接归因于 Language 没传入。**本指南不填写尚未提供的 20 轮最终成功率。**

## 9. 重要实现细节（复现必须保持一致）

1. **两路真实相机**：`cam_head` 和 `cam_left_wrist`；未使用第二个实体腕部相机；右腕位仅黑图 + `False` mask，并非两个相机合并成一个。
2. **动作处理顺序**：7D state/action → 前 6 维 delta、夹爪绝对值 → 7D norm → 32D padding → 模型输出 → 裁回 7D → 7D 逆归一化 → 前 6 维恢复绝对关节目标。
3. **语言**：训练 sample prompt 和推理 fallback 一致，均为先放绿色、再将红色叠在绿色上的完整指令；但收到语言不等于策略一定遵循颜色语义。
4. **推理模式**：云端 JAX 模型常驻 GPU，按需生成 action chunk；本地 RTX 4060 负责 SAPIEN/RoboTwin 物理仿真和 GUI，无需把云端模型搬到本地。
5. **资产保留**：保存最终 checkpoint `19999` 及它的 `assets`；别误认为仅一个 LoRA 小文件就能在空环境独立运行。
6. **不要覆盖旧成果**：转换器会拒绝覆盖 dataset；`train.sh` 可能具有 overwrite 行为，现有 checkpoint 应停止在第 7 节开始使用。

## 10. ZIP 内文件与部署位置

| ZIP 相对路径 | 用途 |
|---|---|
| `python/process_piper_data.py` | HDF5 60 集 → LeRobot `-v2` 数据集 |
| `python/piper_policy.py` | 两相机、7D ↔ 32D、Delta/Absolute 适配 |
| `python/piper_config.py` | **最终** batch 16 / 20,000 steps / LoRA 配置 |
| `python/serve_piper_pi05_xpl.py` | 云端 Pi0.5 WebSocket 服务（实测可连接、完成单轮） |
| `python/verify_piper_dataset.py` | 最终数据 60 集、20,127 帧、7D 与双相机核验 |
| `python/verify_training_prompt.py` | 用 OpenPI 训练 DataLoader 核对实际语言指令 |
| `python/set_eval_gui.py` | 在现有本地任务 YAML 上开启 GUI/视频 |
| `shell/register_piper_config.sh` | 注册 Piper 训练配置（仅一次） |
| `shell/pi05_cloud_env.sh` | 云端缓存环境变量 |
| `shell/run_cloud_server.sh` | 云端推理启动命令 |
| `shell/start_ssh_tunnel.sh` | 本地 SSH 端口 16000 → 云端 6000 |
| `shell/run_eval_one.sh` | seed 64，1 轮已跑通的验收 |
| `shell/run_eval_20_gui.sh` | seed 64，20 轮 GUI 正式评测 |

**不包含**：原始 HDF5 数据、官方 `pi05_base` 模型、训练 checkpoint、原始本地 `piper_pi05_eval.yml`、旧环境快照或排错日志。这些不属于文档中的 Python 代码文件，且最后一个 YAML 在附件中没有提供完整内容。

---

**最终使用路径速记**：已有 checkpoint 的用户，只需确保 **云端运行第 7.1 节** → **本地打开第 7.2 节隧道** → **本地第 8.2 节设置 GUI** → **第 8.4 节执行 seed 64 的 20 轮仿真**。
