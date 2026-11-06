from pathlib import Path
import yaml
import numpy as np

from envs.stack_blocks_two import stack_blocks_two
from envs.utils import ArmTag

cfg = yaml.safe_load(
    Path("env_cfg/task_config/piper_smoke.yml").read_text()
)

robot_dir = "./assets/embodiments/piper"
robot_cfg = yaml.safe_load(
    Path(robot_dir, "config.yml").read_text()
)

cfg.update({
    "task_name": "stack_blocks_two",
    "task_config": "piper_smoke",
    "now_ep_num": 0,
    "seed": 0,
    "save_path": "./data/piper_smoke/stack_blocks_two/piper",
    "left_robot_file": robot_dir,
    "right_robot_file": robot_dir,
    "left_embodiment_config": robot_cfg,
    "right_embodiment_config": robot_cfg,
    "dual_arm_embodied": True,
    "embodiment_name": "piper",
    "need_plan": True,
    "save_data": False,
})

out_dir = Path("piper_test_output")
out_dir.mkdir(exist_ok=True)

env = stack_blocks_two()
left = ArmTag("left")
initialized = False

try:
    print("\n[1] Initializing Piper...")
    env.setup_demo(**cfg)
    initialized = True

    q_before = np.array(
        env.robot.get_left_arm_real_jointState(),
        dtype=float
    )
    pose_before = np.array(
        env.get_arm_pose(left),
        dtype=float
    )

    env.save_camera_rgb(
        str(out_dir / "before.png"),
        "head_camera"
    )

    print("Initial 7D state:", q_before)

    print("\n[2] Closing gripper...")
    close_ok = env.move(env.close_gripper(left))
    print("Close result:", close_ok)
    print("Gripper value:", env.robot.get_left_gripper_val())

    print("\n[3] Opening gripper...")
    open_ok = env.move(env.open_gripper(left))
    print("Open result:", open_ok)
    print("Gripper value:", env.robot.get_left_gripper_val())

    print("\n[4] Planning a 3 cm upward motion...")
    move_ok = env.move(
        env.move_by_displacement(
            arm_tag=left,
            z=0.03
        )
    )

    q_after = np.array(
        env.robot.get_left_arm_real_jointState(),
        dtype=float
    )
    pose_after = np.array(
        env.get_arm_pose(left),
        dtype=float
    )

    env.save_camera_rgb(
        str(out_dir / "after.png"),
        "head_camera"
    )

    print("\n=== RESULTS ===")
    print("Move result:", move_ok)
    print("Planner success:", env.plan_success)
    print("Before 7D state:", q_before)
    print("After 7D state:", q_after)
    print("Joint delta:", q_after[:6] - q_before[:6])
    print("TCP delta (m):", pose_after[:3] - pose_before[:3])
    print("Screenshots:", out_dir.resolve())

finally:
    if initialized:
        env.close_env()
