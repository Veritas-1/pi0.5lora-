from pathlib import Path
import traceback
import yaml

from envs.piper_stack_red_green import piper_stack_red_green

cfg = yaml.safe_load(
    Path("env_cfg/task_config/piper_smoke.yml").read_text()
)

robot_dir = "./assets/embodiments/piper"
robot_cfg = yaml.safe_load(
    Path(robot_dir, "config.yml").read_text()
)

cfg.update({
    "task_name": "piper_stack_red_green",
    "task_config": "piper_smoke",
    "now_ep_num": 0,
    "seed": 0,
    "save_path": "./data/piper_smoke/piper_stack_red_green/piper",
    "left_robot_file": robot_dir,
    "right_robot_file": robot_dir,
    "left_embodiment_config": robot_cfg,
    "right_embodiment_config": robot_cfg,
    "dual_arm_embodied": True,
    "embodiment_name": "piper",
    "need_plan": True,
    "save_data": False,
    "render_freq": 0,
})

output = Path("piper_test_output")
output.mkdir(exist_ok=True)

env = piper_stack_red_green()
initialized = False

try:
    print("Initializing single-arm task...")
    env.setup_demo(**cfg)
    initialized = True

    env.save_camera_rgb(
        str(output / "blocks_before.png"),
        "head_camera"
    )

    print("Initial block positions:")
    for name, block in env.blocks.items():
        print(name, block.get_pose().p)

    print("\nRunning scripted expert...")
    env.play_once()

    env.save_camera_rgb(
        str(output / "blocks_after.png"),
        "head_camera"
    )

    success = env.check_success()
    legal, excess = env.planned_joints_legal()

    print("\n=== PICK AND PLACE RESULTS ===")
    print("Planner success:", env.plan_success)
    print("Task success:", success)
    print("Joint path legal:", legal)
    print("Joint excess (rad):", excess)
    print("Red position:", env.red.get_pose().p)
    print("Green position:", env.green.get_pose().p)
    print("Gripper:", env.robot.get_left_gripper_val())

    if success and legal and env.plan_success:
        print("EXPERT TEST PASSED")
    else:
        print("EXPERT TEST FAILED")

except Exception:
    traceback.print_exc()

finally:
    if initialized:
        env.close_env()
