import numpy as np
import sapien

from .stack_blocks_two import stack_blocks_two
from .utils import ArmTag, create_box


class piper_stack_red_green(stack_blocks_two):

    def load_actors(self):
        half = 0.025

        # 四个固定位置，随机分配颜色
        positions = [
            (-0.18, -0.02),
            (-0.06,  0.05),
            ( 0.08, -0.02),
            ( 0.20,  0.05),
        ]

        colors = {
            "red": (1, 0, 0),
            "green": (0, 1, 0),
            "blue": (0, 0, 1),
            "yellow": (1, 1, 0),
        }

        names = list(colors)
        np.random.shuffle(names)

        self.blocks = {}

        for (x, y), name in zip(positions, names):
            pose = sapien.Pose(
                [x, y, 0.741 + half],
                [1, 0, 0, 0],
            )

            block = create_box(
                scene=self,
                pose=pose,
                half_size=(half, half, half),
                color=colors[name],
                name=name,
            )

            self.blocks[name] = block
            self.add_prohibit_area(block, padding=0.07)

        self.red = self.blocks["red"]
        self.green = self.blocks["green"]

        self.prohibited_area.append(
            [-0.04, -0.13, 0.04, -0.05]
        )

    def checked_move(self, stage, action):
        result = self.move(action)
        print(f"[{stage}] result={result}, plan_success={self.plan_success}")
        if not result or not self.plan_success:
            raise RuntimeError(f"Expert failed at: {stage}")

    def pick_and_place_single(self, block, target_pose):
        arm = ArmTag("left")

        self.checked_move(
            "grasp",
            self.grasp_actor(
                block,
                arm_tag=arm,
                pre_grasp_dis=0.09,
            )
        )

        self.checked_move(
            "lift",
            self.move_by_displacement(
                arm_tag=arm,
                z=0.07,
            )
        )

        self.checked_move(
            "place",
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
            )
        )

        self.checked_move(
            "retract",
            self.move_by_displacement(
                arm_tag=arm,
                z=0.07,
            )
        )

    def play_once(self):
        # 先把绿色方块放到目标区域
        base_target = [
            0, -0.13, 0.75 + self.table_z_bias,
            0, 1, 0, 0
        ]

        print("Stage 1: Place green block")
        self.pick_and_place_single(
            self.green, base_target
        )

        # 然后将红色方块叠到绿色方块上
        print("Stage 2: Stack red on green")
        stack_target = self.green.get_functional_point(1)

        self.pick_and_place_single(
            self.red, stack_target
        )

        self.info["info"] = {}
        return self.info

    def check_success(self):
        red_pos = self.red.get_pose().p
        green_pos = self.green.get_pose().p

        target = green_pos + np.array([0, 0, 0.05])
        error = np.abs(red_pos - target)

        return bool(
            np.all(error < [0.025, 0.025, 0.012])
            and self.is_left_gripper_open()
        )
