from pathlib import Path
import yaml

from envs.piper_stack_red_green import piper_stack_red_green

config_name = "piper_record_one"
task_name = "piper_stack_red_green"
robot_dir = "./assets/embodiments/piper"

cfg = yaml.safe_load(
    Path(f"env_cfg/task_config/{config_name}.yml").read_text()
)
robot_cfg = yaml.safe_load(
    Path(robot_dir, "config.yml").read_text()
)

save_dir = (
    Path(cfg["save_path"])
    / config_name
    / task_name
    / "piper"
)

if (save_dir / "seed.txt").exists():
    raise SystemExit("seed.txt already exists; inspect it first")

cfg.update({
    "task_name": task_name,
    "task_config": config_name,
    "now_ep_num": 0,
    "seed": 0,
    "save_path": str(save_dir),
    "left_robot_file": robot_dir,
    "right_robot_file": robot_dir,
    "left_embodiment_config": robot_cfg,
    "right_embodiment_config": robot_cfg,
    "dual_arm_embodied": True,
    "embodiment_name": "piper",
    "need_plan": True,
    "save_data": False,
})

env = piper_stack_red_green()
initialized = False

try:
    env.setup_demo(**cfg)
    initialized = True

    env.play_once()

    legal, excess = env.planned_joints_legal()
    success = env.check_success() and env.plan_success and legal

    print("Expert success:", success)
    print("Joint excess:", excess)

    if not success:
        raise RuntimeError("Expert failed; trajectory not saved")

    env.save_traj_data(0)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "seed.txt").write_text("0 ", encoding="utf-8")

    print("SAVED EXPERT TRAJECTORY:", save_dir)

finally:
    if initialized:
        env.close_env()
