"""MolmoSpaces (MuJoCo Franka) LeRobot datasets for SpatialVLA fine-tuning, without RLDS/TensorFlow.

Reads a LeRobot v2 dataset made by molmospaces' mlspaces_multiview_to_lerobot.py (17-dim state/action:
eef_9d (xyz + first two rotation columns, robot base frame) + gripper + 7 joints; one exterior view) and
yields what the RLDS pipeline + data/dataset.py would: one exterior RGB image, the lower-cased task,
and a chunk of `chunk` normalized 7-D actions in SpatialVLA's EEF_POS convention:

    a_t = [dxyz, drpy, gripper]  with  dxyz = p_{t+1} - p_t  (robot base frame, like Bridge's
    relabeled state deltas), drpy = euler_xyz(R_{t+1} R_t^T) (applied on the left), and the gripper
    absolute with 1 = open (from the commanded gripper at t).

With drop_idle, no-op frames are removed first (as OpenVLA does for LIBERO): no motion
(|dxyz| < 0.2 mm, |drpy| < 1e-3 rad) with the gripper open and unchanged. That is the robot
waiting before it starts, while the planner settles, and the ~35 idle steps that end every
demo; otherwise this all-zero, gripper-open chunk is the most common label and the policy
collapses to it from unfamiliar views. Idle steps with the gripper closed (grasp settling) stay.

Chunks past the end of an episode get zero motion and repeat the last gripper value, as
data/traj_transforms.chunk_act_obs does. dxyz/drpy are normalized to [-1, 1] with this dataset's
q01/q99 (BOUNDS_Q99), and the statistics are exposed in the format the processor stores, so
decode_actions(..., unnorm_key=name) undoes it at inference.

The camera intrinsics feed Ego3D; the exterior view's pose and FOV change per episode, so each
sample carries its own K (from scripts/molmo/episode_intrinsics.py), scaled to the 224x224 input.
"""

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms

EEF = slice(0, 9)
GRIPPER = 9  # in the 17-dim action: commanded gripper, 1 = closed


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def episode_actions(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """(T, 7) per-step actions for one episode; the last step has zero motion."""
    pos = state[:, 0:3]
    rot = rot6d_to_matrix(state[:, 3:9])
    t = len(state)
    out = np.zeros((t, 7), dtype=np.float32)
    out[:-1, 0:3] = pos[1:] - pos[:-1]
    out[:-1, 3:6] = Rotation.from_matrix(rot[1:] @ np.swapaxes(rot[:-1], -1, -2)).as_euler("xyz")
    out[:, 6] = (action[:, GRIPPER] <= 0.5).astype(np.float32)  # 1 = open
    return out


def idle_mask(acts: np.ndarray) -> np.ndarray:
    """True for no-op steps: no motion, gripper open and unchanged from the previous step."""
    still = (np.linalg.norm(acts[:, 0:3], axis=1) < 2e-4) & (np.linalg.norm(acts[:, 3:6], axis=1) < 1e-3)
    prev = np.concatenate([acts[:1, 6], acts[:-1, 6]])
    return still & (acts[:, 6] > 0.5) & (prev == acts[:, 6])


def decode_video(path: Path) -> np.ndarray:
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


class MolmoLeRobotDataset(torch.utils.data.Dataset):
    use_raw_dataloader = False  # map-style: the Trainer's sampler handles shuffling / DDP

    def __init__(
        self,
        root: str,
        name: str,
        intrinsics_json: str,
        chunk: int = 4,
        image_key: str = "observation.images.exterior_1_left",
        max_length: int = 2048,
        augment: bool = True,
        vla_processor=None,
        drop_idle: bool = False,
    ):
        self.root, self.name, self.chunk, self.max_length = Path(root), name, chunk, max_length
        self.vla_processor = vla_processor
        info = json.load(open(self.root / "meta" / "info.json"))
        tasks = {}
        for line in open(self.root / "meta" / "tasks.jsonl"):
            row = json.loads(line)
            tasks[row["task_index"]] = row["task"]
        intr = json.load(open(intrinsics_json))
        if len(intr["intrinsics"]) != info["total_episodes"]:
            raise ValueError(f"{intrinsics_json}: {len(intr['intrinsics'])} K for {info['total_episodes']} episodes")
        scale = np.diag([224.0 / intr["width"], 224.0 / intr["height"], 1.0])

        self.frames, self.actions, self.lang, self.K, self.index = [], [], [], [], []
        for ep in range(info["total_episodes"]):
            chunk_dir = f"chunk-{ep // info['chunks_size']:03d}"
            df = pd.read_parquet(self.root / "data" / chunk_dir / f"episode_{ep:06d}.parquet")
            state = np.stack(df["observation.state"].to_numpy())[:, EEF]
            action = np.stack(df["action"].to_numpy())
            frames = decode_video(self.root / "videos" / chunk_dir / image_key / f"episode_{ep:06d}.mp4")[: len(df)]
            acts = episode_actions(state, action)
            if drop_idle:
                # the removed steps have no motion, so the kept steps' deltas still chain
                keep = ~idle_mask(acts)
                frames, acts = frames[keep], acts[keep]
            self.frames.append(frames)
            self.actions.append(acts)
            self.lang.append(tasks[int(df["task_index"].iloc[0])].lower())
            self.K.append(torch.tensor(scale @ np.asarray(intr["intrinsics"][ep]), dtype=torch.float32))
            self.index += [(ep, t) for t in range(len(acts))]

        motion = np.concatenate([a[:-1, :6] for a in self.actions])  # exclude the zero last steps
        allact = np.concatenate(self.actions)
        self.q01 = np.quantile(motion, 0.01, axis=0).astype(np.float32)
        self.q99 = np.quantile(motion, 0.99, axis=0).astype(np.float32)
        mask = [True] * 6 + [False]
        self.ds_stats_pc = {
            name: {
                "action": {
                    "mean": allact.mean(0).tolist(),
                    "std": allact.std(0).tolist(),
                    "max": allact.max(0).tolist(),
                    "min": allact.min(0).tolist(),
                    "q01": self.q01.tolist() + [0.0],
                    "q99": self.q99.tolist() + [1.0],
                    "mask": mask,
                },
                "num_transitions": int(len(allact)),
                "num_trajectories": int(len(self.actions)),
            }
        }
        # like data/dataset.py: crop scale 0.9 + color jitter (applied to the 224 image)
        self.augment = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.9, 0.9), ratio=(1.0, 1.0)),
                    transforms.ColorJitter(brightness=0.2, contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.05),
                ]
            )
            if augment
            else None
        )
        print(f"[MolmoLeRobotDataset] {name}: {len(self.actions)} episodes, {len(self.index)} samples"
              f"{' (idle frames dropped)' if drop_idle else ''}, "
              f"q01 {np.round(self.q01, 4).tolist()} q99 {np.round(self.q99, 4).tolist()}")

    def __len__(self):
        return len(self.index)

    def normalize(self, a: np.ndarray) -> np.ndarray:
        out = a.copy()
        span = self.q99 - self.q01
        norm = np.clip(2 * (a[:, :6] - self.q01) / np.where(span > 0, span, 1) - 1, -1, 1)
        out[:, :6] = np.where(span > 0, norm, 0.0)
        return out

    def get_chunk(self, ep: int, t: int) -> np.ndarray:
        acts = self.actions[ep]
        idx = np.arange(t, t + self.chunk)
        chunk = acts[np.minimum(idx, len(acts) - 1)].copy()
        past = idx >= len(acts) - 1  # the last step and beyond: no motion
        chunk[past, :6] = 0.0
        return chunk

    def __getitem__(self, i):
        ep, t = self.index[i]
        image = Image.fromarray(self.frames[ep][t]).resize((224, 224), Image.BILINEAR)
        if self.augment is not None:
            image = self.augment(image)
        actions = torch.from_numpy(self.normalize(self.get_chunk(ep, t)))
        ret = self.vla_processor(
            text=self.lang[ep],
            images=[image],
            suffix_actions=actions,
            return_tensors="pt",
            padding=False,
            max_length=self.max_length,
            truncation=True,
            do_normalize=False,  # Zoe and SigLIP normalize inside the model
        )
        return dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            token_type_ids=ret["token_type_ids"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=ret["pixel_values"],
            intrinsic=self.K[ep],
            actions=actions,
        )
