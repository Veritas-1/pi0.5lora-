#!/usr/bin/env python3
"""Set GUI/video options in the EXISTING RoboTwin Piper evaluation YAML.

This intentionally does not invent or replace the task's other simulation settings.
"""
from pathlib import Path
import argparse
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("env_cfg/task_config/piper_pi05_eval.yml"))
    p.add_argument("--render-freq", type=int, default=5)
    args = p.parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Expected existing verified task config: {args.config}")
    if args.render_freq < 1:
        p.error("--render-freq must be >= 1 to enable GUI rendering")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"YAML root must be a mapping: {args.config}")
    config["render_freq"] = args.render_freq
    config["eval_video_log"] = True
    config["collect_data"] = False
    args.config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("GUI config updated:", args.config)
    print("render_freq:", config["render_freq"])
    print("eval_video_log:", config["eval_video_log"])
    print("collect_data:", config["collect_data"])


if __name__ == "__main__":
    main()
