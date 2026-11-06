#!/usr/bin/env python3
"""Audit RoboTwin Piper demonstrations and convert them to two-camera, 7D DP Zarr.

Run from the RoboTwin repository root in the `dp` conda environment:
    python piper_m2_prepare.py inspect
    python piper_m2_prepare.py convert --max-episodes 1

This script creates a dataset. The stock XPolicyLab DP Dataset/Runner still require
three cameras and cannot train on this two-camera Zarr without modification.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

DEFAULT_SOURCE = "data/piper_record_one/piper_stack_red_green/piper/data"
DEFAULT_DEST = "XPolicyLab/policy/DP/data/RoboTwin-piper_stack_red_green-piper_single-joint.zarr"
DEFAULT_PREVIEW = "piper_test_output/piper_camera_audit.png"


def decode_rgb(bits):
    # This is the official decoder. Do NOT add BGR2RGB after decoding.
    from XPolicyLab.utils.process_data import decode_image_bit

    image = np.asarray(decode_image_bit(bits))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB, got {image.dtype}")
    return image


def resize_rgb(image):
    if image.shape[:2] == (240, 320):
        return image
    import cv2
    return cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)


def pack7(file, prefix):
    arm = np.asarray(file[f"{prefix}/left_arm_joint_states"], dtype=np.float32)
    grip = np.asarray(file[f"{prefix}/left_ee_joint_states"], dtype=np.float32)
    if arm.ndim != 2 or arm.shape[1] != 6:
        raise ValueError(f"Bad {prefix} arm shape: {arm.shape}")
    if grip.shape != (len(arm), 1):
        raise ValueError(f"Bad {prefix} gripper shape: {grip.shape}")
    result = np.concatenate([arm, grip], axis=-1)
    if not np.isfinite(result).all():
        raise ValueError(f"{prefix} contains NaN or Inf")
    return result


def get_paths(source, maximum):
    paths = sorted(source.glob("episode_*.hdf5"))
    if maximum:
        paths = paths[:maximum]
    if not paths:
        raise FileNotFoundError(f"No episodes found in {source.resolve()}")
    return paths


def inspect(paths, preview):
    from PIL import Image, ImageDraw

    path = paths[0]
    with h5py.File(path, "r") as f:
        states = pack7(f, "state")
        actions = pack7(f, "action")
        n = len(states)
        assert n == len(actions) and n >= 2
        camera_keys = ("cam_head", "cam_left_wrist", "cam_right_wrist")
        for key in camera_keys:
            assert len(f[f"vision/{key}/colors"]) == n, key

        print(f"Episode: {path}")
        print(f"Frames: {n}")
        print("Single-arm state:", states.shape, "action:", actions.shape)
        print("State gripper min/max:", float(states[:, 6].min()), float(states[:, 6].max()))
        print("Action gripper min/max:", float(actions[:, 6].min()), float(actions[:, 6].max()))
        print("Action vs same-time state MAE:", float(np.mean(np.abs(actions - states))))
        print("Action[t] vs state[t+1] MAE:", float(np.mean(np.abs(actions[:-1] - states[1:]))))

        if "state/right_arm_joint_states" in f:
            dr = np.max(np.abs(f["state/left_arm_joint_states"][:] -
                               f["state/right_arm_joint_states"][:]))
            dg = np.max(np.abs(f["state/left_ee_joint_states"][:] -
                               f["state/right_ee_joint_states"][:]))
            print("Max left/right physical-joint difference:", float(dr))
            print("Max left/right gripper difference:", float(dg))

        indices = sorted(set([0, n // 2, n - 1]))
        canvas = Image.new("RGB", (320 * 3, (240 + 27) * len(indices)), "white")
        draw = ImageDraw.Draw(canvas)
        differences = []
        for row, t in enumerate(indices):
            imgs = []
            for key in camera_keys:
                imgs.append(resize_rgb(decode_rgb(f[f"vision/{key}/colors"][t])))
            differences.append(float(np.mean(np.abs(
                imgs[1].astype(np.float32) - imgs[2].astype(np.float32)))))
            for col, (key, img) in enumerate(zip(camera_keys, imgs)):
                x, y = col * 320, row * (240 + 27)
                draw.text((x + 5, y + 6), f"frame {t} | {key}", fill="black")
                canvas.paste(Image.fromarray(img), (x, y + 27))
        preview.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(preview)
        print("Wrist left/right pixel MAE at sampled frames:", differences)
        el = f["vision/cam_left_wrist/extrinsics_matrix"][0]
        er = f["vision/cam_right_wrist/extrinsics_matrix"][0]
        print("Wrist left/right extrinsics max difference:",
              float(np.max(np.abs(el - er))))
        if all(v == 0 for v in differences):
            print("The wrist views are identical in sampled frames; one is redundant.")
        print("Camera comparison saved:", preview.resolve())
        print("AUDIT PASSED (inspect the montage visually before conversion).")


def convert(paths, dest):
    import zarr

    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing Zarr: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(dest), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    arrays = {
        "head_camera": data.create_dataset(
            "head_camera", shape=(0, 3, 240, 320), chunks=(16, 3, 240, 320),
            dtype="u1", compressor=compressor),
        "left_camera": data.create_dataset(
            "left_camera", shape=(0, 3, 240, 320), chunks=(16, 3, 240, 320),
            dtype="u1", compressor=compressor),
        "state": data.create_dataset(
            "state", shape=(0, 7), chunks=(256, 7), dtype="f4", compressor=compressor),
        "action": data.create_dataset(
            "action", shape=(0, 7), chunks=(256, 7), dtype="f4", compressor=compressor),
    }
    episode_ends = []
    offset = 0
    for path in paths:
        with h5py.File(path, "r") as f:
            state, action = pack7(f, "state"), pack7(f, "action")
            n = len(state)
            if n == 0 or len(action) != n:
                raise ValueError(f"Invalid episode: {path}")
            for src in ("cam_head", "cam_left_wrist"):
                if len(f[f"vision/{src}/colors"]) != n:
                    raise ValueError(f"Camera length mismatch: {path}: {src}")
            new_size = offset + n
            for key in ("state", "action"):
                arrays[key].resize((new_size, 7))
            arrays["state"][offset:new_size] = state
            arrays["action"][offset:new_size] = action
            for key, src in (("head_camera", "cam_head"), ("left_camera", "cam_left_wrist")):
                arr = arrays[key]
                arr.resize((new_size, 3, 240, 320))
                for start in range(0, n, 16):
                    stop = min(start + 16, n)
                    batch = np.stack([
                        np.moveaxis(resize_rgb(decode_rgb(f[f"vision/{src}/colors"][j])), -1, 0)
                        for j in range(start, stop)
                    ])
                    arr[offset + start:offset + stop] = batch
            offset = new_size
            episode_ends.append(offset)
        print(f"Converted {path.name}: {n} steps (total {offset})")
    meta.create_dataset("episode_ends", data=np.array(episode_ends, dtype=np.int64),
                        compressor=compressor)
    root.attrs["camera_keys"] = ["head_camera", "left_camera"]
    root.attrs["action_description"] = "[joint1..joint6, left_gripper], joint positions"
    root.attrs["note"] = "Two-camera dataset; stock XPolicyLab DP dataset/runner require adaptation."
    print("Zarr:", dest.resolve())
    print("Shapes:", {key: arr.shape for key, arr in arrays.items()})
    print("episode_ends:", episode_ends)
    assert arrays["state"].shape == arrays["action"].shape
    print("CONVERSION PASSED (DP model/runner integration still pending).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inspect", "convert"))
    parser.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--dest", type=Path, default=Path(DEFAULT_DEST))
    parser.add_argument("--preview", type=Path, default=Path(DEFAULT_PREVIEW))
    parser.add_argument("--max-episodes", type=int, default=1)
    args = parser.parse_args()
    paths = get_paths(args.source, args.max_episodes)
    if args.command == "inspect":
        inspect(paths, args.preview)
    else:
        convert(paths, args.dest)


if __name__ == "__main__":
    main()
