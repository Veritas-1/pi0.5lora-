#!/usr/bin/env python3
"""Convert RoboTwin Piper single-arm HDF5 episodes to LeRobot 0.4.4 format.

Keeps the recorded action (joint targets + gripper), two RGB cameras, and
7D joint state. Never removes/replaces a pre-existing dataset directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from XPolicyLab.utils.process_data import decode_image_bit

DEFAULT_RAW = Path('/root/rivermind-data/data/piper_record_one/piper_stack_red_green/piper/data')
DEFAULT_REPO = 'RoboTwin-piper_stack_red_green-piper_single-joint-v2'
DEFAULT_TASK = 'Place the green block on the base, then stack the red block on the green block.'
CAMERAS = {
    'cam_head': 'cam_high',
    'cam_left_wrist': 'cam_left_wrist',
}


def vector(group: h5py.File, name: str, width: int) -> np.ndarray:
    x = np.asarray(group[name], dtype=np.float32)
    if x.ndim == 0:
        raise ValueError(f'{name}: scalar array')
    x = x.reshape(x.shape[0], -1)
    if x.shape[1] != width or not np.isfinite(x).all():
        raise ValueError(f'{name}: expected (*,{width}) finite, got {x.shape}')
    return x


def read_episode(path: Path):
    with h5py.File(path, 'r') as f:
        state = np.concatenate([
            vector(f, 'state/left_arm_joint_states', 6),
            vector(f, 'state/left_ee_joint_states', 1),
        ], axis=-1)
        action = np.concatenate([
            vector(f, 'action/left_arm_joint_states', 6),
            vector(f, 'action/left_ee_joint_states', 1),
        ], axis=-1)
        n = len(state)
        if len(action) != n or n == 0:
            raise ValueError(f'{path}: invalid state/action lengths: {n}/{len(action)}')
        images = {}
        for src, dst in CAMERAS.items():
            key = f'vision/{src}/colors'
            if len(f[key]) != n:
                raise ValueError(f'{path}: {key} length {len(f[key])} != {n}')
            images[dst] = np.asarray(decode_image_bit(f[key]))
            if images[dst].shape != (n, 240, 320, 3):
                raise ValueError(
                    f'{path}: decoded {key} shape {images[dst].shape}; '
                    'expected (frames, 240, 320, 3) in RGB order'
                )
            if images[dst].dtype != np.uint8:
                raise ValueError(f'{path}: decoded {key} dtype {images[dst].dtype}, expected uint8')
    return state, action, images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', type=Path, default=DEFAULT_RAW)
    ap.add_argument('--repo-id', default=DEFAULT_REPO)
    ap.add_argument('--max-episodes', type=int, default=60)
    ap.add_argument('--fps', type=int, default=50)
    ap.add_argument('--task', default=DEFAULT_TASK)
    args = ap.parse_args()

    raw = args.raw_dir.expanduser()
    paths = sorted(raw.glob('episode_*.hdf5'))
    if len(paths) != 60:
        raise ValueError(f'Expected 60 source episodes in {raw}, got {len(paths)}')
    if not 1 <= args.max_episodes <= len(paths):
        raise ValueError(f'--max-episodes must be 1..{len(paths)}')
    paths = paths[:args.max_episodes]

    output = HF_LEROBOT_HOME / args.repo_id
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)

    motor_names = [f'joint_{i}' for i in range(1, 7)] + ['gripper']
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (7,), 'names': motor_names},
        'action': {'dtype': 'float32', 'shape': (7,), 'names': motor_names},
        'observation.images.cam_high': {
            'dtype': 'image', 'shape': (240, 320, 3), 'names': ['height', 'width', 'channels'],
        },
        'observation.images.cam_left_wrist': {
            'dtype': 'image', 'shape': (240, 320, 3), 'names': ['height', 'width', 'channels'],
        },
    }
    print('Output:', output, flush=True)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        fps=args.fps,
        robot_type='piper_single',
        features=features,
        use_videos=False,
        image_writer_threads=1,
    )
    frames = 0
    for ep_index, ep in enumerate(paths):
        state, action, images = read_episode(ep)
        for i in range(len(state)):
            ds.add_frame({
                'observation.state': state[i],
                'action': action[i],
                'observation.images.cam_high': images['cam_high'][i],
                'observation.images.cam_left_wrist': images['cam_left_wrist'][i],
                'task': args.task,
            })
        ds.save_episode()
        frames += len(state)
        print(f'[{ep_index+1}/{len(paths)}] {ep.name}: {len(state)} frames; total={frames}', flush=True)
    ds.finalize()  # required by LeRobot 0.4.4 to finalize parquet footers
    print(f'CONVERSION PASSED: episodes={len(paths)}, frames={frames}, dataset={output}', flush=True)


if __name__ == '__main__':
    main()
