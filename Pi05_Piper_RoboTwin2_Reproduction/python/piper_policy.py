"""Piper single-arm transforms for OpenPI Pi0.5.

LeRobot dataset fields:
  observation.state: [7] = six joint positions + gripper
  action: [7] = next-step absolute joint targets + absolute gripper
  observation.images.cam_high: HWC RGB
  observation.images.cam_left_wrist: HWC RGB

The data pipeline converts the first six action dimensions to deltas. The
seventh (gripper) dimension remains absolute. Pi0.5's model-space action
width remains 32; this module trims the model output back to seven dimensions
before unnormalization and conversion back to absolute joint targets.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from openpi import transforms


PIPER_ACTION_DIM = 7


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy or CPU torch values to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_hwc_uint8(image: Any, *, name: str) -> np.ndarray:
    """Convert CHW/HWC (or batched BCHW/BHWC) RGB image to uint8 HWC/BHWC."""
    image = _as_numpy(image)

    if image.ndim == 3:
        # LeRobot 0.4.4 returns individual images as CHW float32.
        if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 4:
        # Batched LeRobot images may be BCHW.
        if image.shape[1] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.transpose(image, (0, 2, 3, 1))
    else:
        raise ValueError(f"{name} must be a 3D or 4D image array, got {image.shape}")

    if image.shape[-1] != 3:
        raise ValueError(f"{name} must have 3 RGB channels, got {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError(f"{name} contains NaN or Inf")
        if image.size and image.min() >= 0.0 and image.max() <= 1.0 + 1e-6:
            image = image * 255.0
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def _image_mask(image: np.ndarray, valid: bool) -> np.ndarray | np.bool_:
    if image.ndim == 4:
        return np.full((image.shape[0],), valid, dtype=bool)
    return np.bool_(valid)


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Map Piper's two RGB cameras and 7D state to Pi0.5 observation keys."""

    action_dim: int = PIPER_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        source_images = data["images"]
        if "cam_high" not in source_images:
            raise KeyError("Piper dataset is missing observation.images.cam_high")
        if "cam_left_wrist" not in source_images:
            raise KeyError("Piper dataset is missing observation.images.cam_left_wrist")

        base_image = _to_hwc_uint8(source_images["cam_high"], name="cam_high")
        left_wrist_image = _to_hwc_uint8(
            source_images["cam_left_wrist"], name="cam_left_wrist"
        )
        if base_image.shape != left_wrist_image.shape:
            raise ValueError(
                "Piper head and wrist images must have matching shapes, got "
                f"{base_image.shape} and {left_wrist_image.shape}"
            )

        state = _as_numpy(data["state"]).astype(np.float32, copy=False)
        if state.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected Piper state width {self.action_dim}, got {state.shape}"
            )

        result = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": _image_mask(base_image, True),
                "left_wrist_0_rgb": _image_mask(left_wrist_image, True),
                "right_wrist_0_rgb": _image_mask(base_image, False),
            },
        }

        if "actions" in data:
            actions = _as_numpy(data["actions"]).astype(np.float32, copy=True)
            if actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Expected Piper action width {self.action_dim}, got {actions.shape}"
                )
            result["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            elif isinstance(prompt, np.ndarray) and prompt.ndim == 0:
                prompt = prompt.item()
            result["prompt"] = prompt

        return result


@dataclasses.dataclass(frozen=True)
class UnpadPiperStateAndActions(transforms.DataTransformFn):
    """Trim Pi0.5's padded model outputs before applying 7D normalization stats."""

    action_dim: int = PIPER_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        for key in ("state", "actions"):
            if key not in result:
                continue
            value = result[key]
            if value.shape[-1] < self.action_dim:
                raise ValueError(
                    f"Cannot unpad {key}: expected at least {self.action_dim} dims, "
                    f"got {value.shape}"
                )
            result[key] = value[..., : self.action_dim]
        return result
