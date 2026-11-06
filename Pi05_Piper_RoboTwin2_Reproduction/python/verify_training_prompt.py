#!/usr/bin/env python3
"""Read exact training prompts through the OpenPI LeRobot data loader.

Run AFTER Piper config registration, with HF_HOME=/root/rivermind-data/hf-cache.
"""
from openpi.training.config import get_config
from openpi.training import data_loader

EXPECTED_PROMPT = "Place the green block on the base, then stack the red block on the green block."


def main():
    cfg = get_config("pi05_piper_single_lora")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert dc.prompt_from_task is True
    dataset = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    print("Training dataset frames:", len(dataset))
    assert len(dataset) == 20127
    for i in (0, len(dataset) // 2, len(dataset) - 1):
        prompt = dataset[i]["prompt"]
        print(f"Sample {i} prompt:", repr(prompt))
        assert prompt == EXPECTED_PROMPT, (i, prompt)
    print("TRAINING PROMPTS MATCH INFERENCE FALLBACK")


if __name__ == "__main__":
    main()
