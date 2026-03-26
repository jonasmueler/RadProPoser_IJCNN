import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory for config imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent / "MotionBERT"))

from config import ANNOT_DIR, MB_CHECKPOINT_PATH, MB_CONFIG_PATH, MB_ROOTREL, PREPROCESS_CONFIG
from lib.model.DSTformer import DSTformer
from lib.utils.tools import get_config
from misc.constants import HUPR_TO_H36M

IMG_WIDTH = PREPROCESS_CONFIG["img_width"]
IMG_HEIGHT = PREPROCESS_CONFIG["img_height"]
JUMP_THRESHOLD_PX = PREPROCESS_CONFIG["jump_threshold_px"]


def load_model(device: torch.device) -> Tuple[nn.Module, int]:
    """Load MotionBERT model with pretrained weights.

    Args:
        device: Torch device to load model on

    Returns:
        Tuple of (model in eval mode, clip length)
    """
    args = get_config(str(MB_CONFIG_PATH))

    model = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=args.dim_feat,
        dim_rep=args.dim_rep,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        maxlen=args.maxlen,
        num_joints=args.num_joints,
    )

    checkpoint = torch.load(MB_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_pos"].items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    return model, args.maxlen


def hupr_to_h36m(joints_14: np.ndarray) -> np.ndarray:
    """Convert HuPR 14-joint format to H3.6M 17-joint format.

    Args:
        joints_14: Keypoints array of shape (num_frames, 14, 2)

    Returns:
        Keypoints array of shape (num_frames, 17, 2)
    """
    num_frames = joints_14.shape[0]
    joints_17 = np.zeros((num_frames, 17, 2), dtype=np.float32)

    for hupr_idx, h36m_idx in HUPR_TO_H36M.items():
        joints_17[:, h36m_idx] = joints_14[:, hupr_idx]

    joints_17[:, 0] = (joints_14[:, 0] + joints_14[:, 3]) / 2.0  # Hip
    joints_17[:, 8] = (joints_14[:, 8] + joints_14[:, 11]) / 2.0  # Thorax
    joints_17[:, 7] = joints_17[:, 0] + (joints_17[:, 8] - joints_17[:, 0]) / 3.0  # Spine

    return joints_17


def interpolate_jumps(keypoints: np.ndarray, threshold: float = JUMP_THRESHOLD_PX) -> np.ndarray:
    """Detect and interpolate frames with unrealistic joint jumps.

    Args:
        keypoints: Keypoints array of shape (num_frames, num_joints, 2)
        threshold: Maximum allowed movement in pixels per frame

    Returns:
        Cleaned keypoints with jumps interpolated
    """
    keypoints = keypoints.copy()
    num_frames, num_joints, _ = keypoints.shape

    # Compute per-joint velocities
    velocities = np.diff(keypoints, axis=0)
    speeds = np.linalg.norm(velocities, axis=-1)  # (num_frames-1, num_joints)

    # Find frames where any joint exceeds threshold
    max_speeds = speeds.max(axis=1)
    jump_mask = max_speeds > threshold

    # Find contiguous jump regions and interpolate
    i = 0
    total_interpolated = 0
    while i < len(jump_mask):
        if jump_mask[i]:
            # Found start of jump region (frame i+1 has the jump)
            start = i + 1
            end = start

            # Extend to find end of contiguous jump region
            while end < len(jump_mask) and jump_mask[end]:
                end += 1
            end += 1  # end is now the first good frame after jumps

            # Interpolate between last good frame (start-1) and first good frame (end)
            if start > 0 and end < num_frames:
                for j in range(num_joints):
                    keypoints[start:end, j, 0] = np.linspace(
                        keypoints[start - 1, j, 0],
                        keypoints[end, j, 0],
                        end - start + 2,
                    )[1:-1]
                    keypoints[start:end, j, 1] = np.linspace(
                        keypoints[start - 1, j, 1],
                        keypoints[end, j, 1],
                        end - start + 2,
                    )[1:-1]
                total_interpolated += end - start

            i = end
        else:
            i += 1

    return keypoints


def normalize_with_bbox(coords: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """Normalize pixel coordinates to [-1, 1] range using bounding boxes.

    Args:
        coords: Keypoints array of shape (num_frames, 17, 2)
        bboxes: Bounding boxes of shape (num_frames, 4) as [x_min, y_min, x_max, y_max]

    Returns:
        Normalized coordinates in [-1, 1] range
    """
    num_frames = coords.shape[0]
    result = np.zeros_like(coords)

    for i in range(num_frames):
        x_min, y_min, x_max, y_max = bboxes[i]
        w = x_max - x_min
        h = y_max - y_min
        scale = max(w, h)

        if scale == 0:
            continue

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        result[i, :, 0] = (coords[i, :, 0] - cx) / (scale / 2)
        result[i, :, 1] = (coords[i, :, 1] - cy) / (scale / 2)

    return np.clip(result, -1, 1).astype(np.float32)


def pad_sequence(sequence: np.ndarray, clip_len: int) -> np.ndarray:
    """Pad sequence to clip_len using reflection padding.

    Args:
        sequence: Input sequence of shape (num_frames, 17, 2 or 3)
        clip_len: Target length (243 for MotionBERT)

    Returns:
        Padded sequence
    """
    num_frames = sequence.shape[0]

    if num_frames >= clip_len:
        return sequence

    # Reflect pad to reach clip_len
    pad_left = (clip_len - num_frames) // 2
    pad_right = clip_len - num_frames - pad_left

    return np.pad(sequence, ((pad_left, pad_right), (0, 0), (0, 0)), mode="reflect")


def lift_2d_to_3d(
    model: nn.Module,
    sequence_2d: np.ndarray,
    clip_len: int,
    device: torch.device,
) -> np.ndarray:
    """Lift 2D keypoints to 3D using MotionBERT with sliding window.

    Args:
        model: MotionBERT model
        sequence_2d: Normalized 2D keypoints of shape (num_frames, 17, 2)
        clip_len: Window size (243)
        device: Torch device

    Returns:
        3D keypoints of shape (num_frames, 17, 3)
    """
    num_frames = sequence_2d.shape[0]

    # Add confidence channel (1.0 for ground truth keypoints)
    confidence = np.ones((num_frames, 17, 1), dtype=np.float32)
    sequence_3ch = np.concatenate([sequence_2d, confidence], axis=-1)

    # For short sequences, pad and run once
    if num_frames <= clip_len:
        padded = pad_sequence(sequence_3ch, clip_len)
        input_tensor = torch.from_numpy(padded).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(input_tensor)

        # Crop to original length
        pad_left = (clip_len - num_frames) // 2
        output_3d = output[0, pad_left:pad_left + num_frames].cpu().numpy()
        return output_3d

    # For longer sequences, use sliding window with stride
    stride = clip_len // 2  # 50% overlap
    counts = np.zeros(num_frames, dtype=np.float32)
    accumulated = np.zeros((num_frames, 17, 3), dtype=np.float32)

    for start in range(0, num_frames - clip_len + 1, stride):
        end = start + clip_len
        clip = sequence_3ch[start:end]
        input_tensor = torch.from_numpy(clip).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(input_tensor)

        output_np = output[0].cpu().numpy()
        accumulated[start:end] += output_np
        counts[start:end] += 1

    # Handle remaining frames at the end
    if (num_frames - clip_len) % stride != 0:
        start = num_frames - clip_len
        clip = sequence_3ch[start:]
        input_tensor = torch.from_numpy(clip).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(input_tensor)

        output_np = output[0].cpu().numpy()
        accumulated[start:] += output_np
        counts[start:] += 1

    # Average overlapping predictions
    output_3d = accumulated / counts[:, None, None]

    return output_3d


def process_split(
    split: str,
    model: nn.Module,
    clip_len: int,
    device: torch.device,
) -> List[List[Dict[str, Any]]]:
    """Process a single split and generate 3D annotations.

    Args:
        split: Split name ('train', 'val', or 'test')
        model: MotionBERT model
        clip_len: Model window size
        device: Torch device

    Returns:
        List of sequences with 3D joint annotations
    """
    input_path = ANNOT_DIR / f"hrnet_annot_{split}.json"

    with open(input_path) as f:
        annot_2d = json.load(f)

    annot_3d: List[List[Dict[str, Any]]] = []

    for seq_2d in tqdm(annot_2d, desc=f"Processing {split}"):
        num_frames = len(seq_2d)
        keypoints_14 = np.zeros((num_frames, 14, 2), dtype=np.float32)
        bboxes = np.zeros((num_frames, 4), dtype=np.float32)

        for i, frame in enumerate(seq_2d):
            keypoints_14[i] = np.array(frame["joints"], dtype=np.float32)
            bboxes[i] = np.array(frame["bbox"], dtype=np.float32)

        # Clean noisy 2D detections by interpolating unrealistic jumps
        keypoints_14 = interpolate_jumps(keypoints_14)

        keypoints_17 = hupr_to_h36m(keypoints_14)
        keypoints_norm = normalize_with_bbox(keypoints_17, bboxes)
        poses_3d = lift_2d_to_3d(model, keypoints_norm, clip_len, device)

        # Make root-relative if model doesn't output root-relative poses
        if not MB_ROOTREL:
            poses_3d = poses_3d - poses_3d[:, 0:1, :]

        seq_3d = []
        for i, frame_2d in enumerate(seq_2d):
            seq_3d.append({
                "image": frame_2d["image"],
                "joints_3d": poses_3d[i].tolist(),
            })
        annot_3d.append(seq_3d)

    return annot_3d


def main() -> None:
    """Create 3D ground truth annotations using MotionBERT."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading MotionBERT from {MB_CHECKPOINT_PATH}...")
    model, clip_len = load_model(device)
    print(f"Clip length: {clip_len}")

    for split in ["train", "val", "test"]:
        input_path = ANNOT_DIR / f"hrnet_annot_{split}.json"
        output_path = ANNOT_DIR / f"motionbert_3d_{split}.json"

        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping...")
            continue

        print(f"\nProcessing {split} split...")
        annot_3d = process_split(split, model, clip_len, device)

        with open(output_path, "w") as f:
            json.dump(annot_3d, f)

        print(f"Saved {output_path} ({len(annot_3d)} sequences)")

    print("\nDone!")


if __name__ == "__main__":
    main()
