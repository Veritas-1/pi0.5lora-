#!/usr/bin/env python3
"""Minimal final LeRobot dataset integrity check (cloud OpenPI venv)."""
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = "RoboTwin-piper_stack_red_green-piper_single-joint-v2"
ROOT = Path("/root/rivermind-data/hf-cache/lerobot") / REPO
PROMPT = "Place the green block on the base, then stack the red block on the green block."


def main():
    if not (ROOT / "meta/info.json").is_file():
        raise FileNotFoundError(f"Dataset not found: {ROOT}")
    ds = LeRobotDataset(repo_id=REPO, root=ROOT)
    assert ds.num_episodes == 60, ds.num_episodes
    assert ds.num_frames == 20127, ds.num_frames
    for i in (0, len(ds) // 2, len(ds) - 1):
        sample = ds[i]
        for k in ("observation.state", "action"):
            v = np.asarray(sample[k])
            assert v.shape == (7,) and np.isfinite(v).all(), (i, k, v.shape)
        for k in ("observation.images.cam_high", "observation.images.cam_left_wrist"):
            v = np.asarray(sample[k])
            assert v.shape in ((3, 240, 320), (240, 320, 3)), (i, k, v.shape)
        if "task" in sample:
            assert sample["task"] == PROMPT, (i, sample["task"])
        print(f"frame {i}: 7D state/action and two RGB cameras OK")
    print(f"PASS: {ds.num_episodes} episodes, {ds.num_frames} frames")


if __name__ == "__main__":
    main()
