"""Pi0.5 LoRA training configuration for the RoboTwin Piper single-arm task."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
import pathlib

from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.policies.piper_policy import PiperInputs, UnpadPiperStateAndActions
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders


PIPER_REPO_ID = "RoboTwin-piper_stack_red_green-piper_single-joint-v2"
PIPER_CONFIG_NAME = "pi05_piper_single_lora"
PIPER_DEFAULT_PROMPT = "Stack the red block on the green block."


@dataclasses.dataclass(frozen=True)
class LeRobotPiperDataConfig(_config.DataConfigFactory):
    """LeRobot adapter for Piper's 7D state/action and two RGB cameras."""

    default_prompt: str | None = PIPER_DEFAULT_PROMPT
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
    ) -> _config.DataConfig:
        repack_transforms = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # HDF5 action[t] == state[t+1]. Convert only six arm joints to delta;
        # keep the gripper's 0/1 command absolute.
        delta_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = _transforms.Group(inputs=[PiperInputs()]).push(
            inputs=[_transforms.DeltaActions(delta_mask)],
            outputs=[_transforms.AbsoluteActions(delta_mask)],
        )

        model_transforms = _config.ModelTransformFactory(
            default_prompt=self.default_prompt
        )(model_config)
        # Policy output order is: model_transforms.outputs -> Unnormalize ->
        # data_transforms.outputs. Trim 32D model outputs to 7D before the
        # 7D statistics are applied, then AbsoluteActions restores joint targets.
        model_transforms = dataclasses.replace(
            model_transforms,
            outputs=(
                UnpadPiperStateAndActions(action_dim=7),
                *model_transforms.outputs,
            ),
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            prompt_from_task=True,
        )


_piper_model = pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    action_dim=32,
    action_horizon=50,
    max_token_len=200,
)

PIPER_CONFIGS = [
    _config.TrainConfig(
        name=PIPER_CONFIG_NAME,
        model=_piper_model,
        data=LeRobotPiperDataConfig(
            repo_id=PIPER_REPO_ID,
            assets=_config.AssetsConfig(asset_id=PIPER_REPO_ID),
            base_config=_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=_piper_model.get_freeze_filter(),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=20000,
            decay_lr=5e-6,
        ),
        num_train_steps=20000,
        batch_size=16,
        num_workers=4,
        fsdp_devices=1,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        wandb_enabled=False,
    )
]
