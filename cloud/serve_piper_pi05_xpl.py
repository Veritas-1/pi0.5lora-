"""Pi0.5 Piper 7D policy served over XPolicyLab WebSocket protocol.

Run on cloud OpenPI virtualenv. Does not modify XPolicyLab's generic Pi_05 model.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from openpi.policies import policy_config
from openpi.training import config as train_config

CONFIG_NAME = "pi05_piper_single_lora"
DATASET_NAME = "RoboTwin-piper_stack_red_green-piper_single-joint-v2"
FALLBACK_PROMPT = "Place the green block on the base, then stack the red block on the green block."


def dataset_task_prompt(dataset_root: Path) -> str:
    task_file = dataset_root / "meta" / "tasks.jsonl"
    if not task_file.is_file():
        logging.warning("No task metadata at %s; using fallback prompt", task_file)
        return FALLBACK_PROMPT
    lines = [json.loads(s) for s in task_file.read_text().splitlines() if s.strip()]
    tasks = {str(item["task"]) for item in lines}
    if len(tasks) != 1:
        raise ValueError(f"Expected one training task, got {sorted(tasks)}")
    return tasks.pop()


def image_from_vision(vision: dict, key: str) -> np.ndarray:
    if key not in vision:
        raise KeyError(f"Missing camera {key}; available: {list(vision)}")
    item = vision[key]
    if isinstance(item, dict):
        item = item.get("color", item.get("rgb"))
    if item is None:
        raise ValueError(f"Missing RGB payload for camera {key}")
    image = np.asarray(item)
    if image.ndim != 3 or (image.shape[-1] != 3 and image.shape[0] != 3):
        raise ValueError(f"Invalid RGB image {key}: shape={image.shape}")
    return image


class PiperPi05Model:
    def __init__(self, checkpoint: Path, dataset_root: Path, chunk_size: int):
        if not (checkpoint / "params").is_dir():
            raise FileNotFoundError(f"Missing checkpoint params: {checkpoint / 'params'}")
        config = train_config.get_config(CONFIG_NAME)
        # create_trained_policy loads checkpoint/params and checkpoint/assets norm stats.
        self.policy = policy_config.create_trained_policy(config, str(checkpoint))
        self.prompt = dataset_task_prompt(dataset_root)
        self.chunk_size = chunk_size
        self.obs = None
        logging.info("Using checkpoint: %s", checkpoint)
        logging.info("Training prompt: %s", self.prompt)
        logging.info("Executed actions per inference: %d (action horizon = 50)", chunk_size)

    def reset(self):
        self.obs = None

    def update_obs(self, obs: dict):
        if "vision" not in obs or "state" not in obs:
            raise KeyError("Expected RoboTwin/XPolicyLab observation keys 'vision' and 'state'")
        src = obs["state"]
        joints = np.asarray(src["left_arm_joint_state"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(src["left_ee_joint_state"], dtype=np.float32).reshape(-1)
        if joints.size != 6 or gripper.size != 1:
            raise ValueError(f"Piper state must be [6 arm,1 gripper], got {joints.shape}, {gripper.shape}")
        state = np.concatenate((joints, gripper)).astype(np.float32)
        if not np.isfinite(state).all():
            raise ValueError("Piper state contains NaN/Inf")
        vision = obs["vision"]
        self.obs = {
            "state": state,
            "images": {
                "cam_high": image_from_vision(vision, "cam_head"),
                "cam_left_wrist": image_from_vision(vision, "cam_left_wrist"),
            },
            "prompt": self.prompt,
        }

    def get_action(self):
        if self.obs is None:
            raise RuntimeError("Call update_obs before get_action")
        actions = np.asarray(self.policy.infer(self.obs)["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected (50,7) actions, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Nonfinite action from Pi0.5")
        actions = actions[: self.chunk_size].copy()
        actions[:, 6] = np.clip(actions[:, 6], 0.0, 1.0)
        return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.chunk_size <= 50:
        parser.error("--chunk-size must be between 1 and 50")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model = PiperPi05Model(args.checkpoint, args.dataset_root, args.chunk_size)
    server = PolicyServer(model, PolicyServerConfig(host=args.host, port=args.port))
    logging.info("Starting XPolicyLab WebSocket server at ws://%s:%d", args.host, args.port)
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
