import json
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "MotionBERT"))

from lib.model.DSTformer import DSTformer
from lib.utils.tools import get_config

from config import ANNOT_DIR, MB_CONFIG_PATH, MB_CHECKPOINT_PATH, MB_ROOTREL, PREPROCESS_CONFIG
from misc.constants import HUPR_TO_H36M, TEST_SEQUENCES


def load_motionbert(device: torch.device) -> Tuple[nn.Module, int]:
    """Load MotionBERT model with pretrained weights.

    Args:
        device: Torch device.

    Returns:
        Tuple of (model in eval mode, clip length).
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
        joints_14: Keypoints of shape (num_frames, 14, 2).

    Returns:
        Keypoints of shape (num_frames, 17, 2).
    """
    num_frames = joints_14.shape[0]
    joints_17 = np.zeros((num_frames, 17, 2), dtype=np.float32)

    for hupr_idx, h36m_idx in HUPR_TO_H36M.items():
        joints_17[:, h36m_idx] = joints_14[:, hupr_idx]

    # Compute derived joints
    joints_17[:, 0] = (joints_14[:, 0] + joints_14[:, 3]) / 2.0  # Hip
    joints_17[:, 8] = (joints_14[:, 8] + joints_14[:, 11]) / 2.0  # Thorax
    joints_17[:, 7] = joints_17[:, 0] + (joints_17[:, 8] - joints_17[:, 0]) / 3.0  # Spine

    return joints_17


def interpolate_jumps(keypoints: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """Detect and interpolate frames with unrealistic joint jumps.

    Args:
        keypoints: Keypoints of shape (num_frames, num_joints, 2).
        threshold: Maximum allowed movement in pixels per frame.

    Returns:
        Cleaned keypoints with jumps interpolated.
    """
    if threshold is None:
        threshold = PREPROCESS_CONFIG["jump_threshold_px"]

    keypoints = keypoints.copy()
    num_frames, num_joints, _ = keypoints.shape

    velocities = np.diff(keypoints, axis=0)
    speeds = np.linalg.norm(velocities, axis=-1)
    max_speeds = speeds.max(axis=1)
    jump_mask = max_speeds > threshold

    i = 0
    while i < len(jump_mask):
        if jump_mask[i]:
            start = i + 1
            end = start

            while end < len(jump_mask) and jump_mask[end]:
                end += 1
            end += 1

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

            i = end
        else:
            i += 1

    return keypoints


def normalize_with_bbox(coords: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """Normalize pixel coordinates to [-1, 1] using bounding boxes.

    Args:
        coords: Keypoints of shape (num_frames, 17, 2).
        bboxes: Bounding boxes of shape (num_frames, 4) as [x_min, y_min, x_max, y_max].

    Returns:
        Normalized coordinates in [-1, 1] range.
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


def normalize_screen_coordinates(coords: np.ndarray, w: int = 256, h: int = 256) -> np.ndarray:
    """Normalize pixel coordinates to [-1, 1] using image dimensions.

    Args:
        coords: Keypoints of shape (num_frames, 17, 2).
        w: Image width.
        h: Image height.

    Returns:
        Normalized coordinates.
    """
    result = coords / w * 2 - np.array([1, h / w])
    return result.astype(np.float32)


def compute_bboxes_from_keypoints(coords: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """Compute bounding boxes from keypoints with margin.

    Args:
        coords: Keypoints of shape (num_frames, num_joints, 2).
        margin: Margin to add around bbox as fraction of bbox size.

    Returns:
        Bounding boxes of shape (num_frames, 4) as [x_min, y_min, x_max, y_max].
    """
    num_frames = coords.shape[0]
    bboxes = np.zeros((num_frames, 4), dtype=np.float32)

    for i in range(num_frames):
        x_min, y_min = coords[i].min(axis=0)
        x_max, y_max = coords[i].max(axis=0)
        w = x_max - x_min
        h = y_max - y_min
        m = margin * max(w, h)
        bboxes[i] = [x_min - m, y_min - m, x_max + m, y_max + m]

    return bboxes


def pad_sequence(sequence: np.ndarray, clip_len: int) -> np.ndarray:
    """Pad sequence to clip_len using reflection padding.

    Args:
        sequence: Input sequence of shape (num_frames, 17, 2 or 3).
        clip_len: Target length.

    Returns:
        Padded sequence.
    """
    num_frames = sequence.shape[0]

    if num_frames >= clip_len:
        return sequence

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
        model: MotionBERT model.
        sequence_2d: Normalized 2D keypoints of shape (num_frames, 17, 2).
        clip_len: Window size (243).
        device: Torch device.

    Returns:
        3D keypoints of shape (num_frames, 17, 3).
    """
    num_frames = sequence_2d.shape[0]

    # Add confidence channel
    confidence = np.ones((num_frames, 17, 1), dtype=np.float32)
    sequence_3ch = np.concatenate([sequence_2d, confidence], axis=-1)

    # Short sequences: pad and run once
    if num_frames <= clip_len:
        padded = pad_sequence(sequence_3ch, clip_len)
        input_tensor = torch.from_numpy(padded).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(input_tensor)

        pad_left = (clip_len - num_frames) // 2
        return output[0, pad_left:pad_left + num_frames].cpu().numpy()

    # Long sequences: sliding window with 50% overlap
    stride = clip_len // 2
    counts = np.zeros(num_frames, dtype=np.float32)
    accumulated = np.zeros((num_frames, 17, 3), dtype=np.float32)

    for start in range(0, num_frames - clip_len + 1, stride):
        end = start + clip_len
        clip = sequence_3ch[start:end]
        input_tensor = torch.from_numpy(clip).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(input_tensor)

        accumulated[start:end] += output[0].cpu().numpy()
        counts[start:end] += 1

    # Handle remaining frames
    if (num_frames - clip_len) % stride != 0:
        start = num_frames - clip_len
        clip = sequence_3ch[start:]
        input_tensor = torch.from_numpy(clip).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(input_tensor)

        accumulated[start:] += output[0].cpu().numpy()
        counts[start:] += 1

    return accumulated / counts[:, None, None]


def load_gt_bboxes() -> Dict[int, np.ndarray]:
    """Load ground truth bboxes from hrnet_annot_test.json.

    Returns:
        Dictionary mapping seq_id to bboxes of shape (600, 4).
    """
    annot_path = ANNOT_DIR / "hrnet_annot_test.json"

    with open(annot_path) as f:
        annot = json.load(f)

    bboxes = {}
    for seq_idx, seq in enumerate(annot):
        if seq_idx >= len(TEST_SEQUENCES):
            break

        seq_id = TEST_SEQUENCES[seq_idx]
        seq_bboxes = np.array([frame["bbox"] for frame in seq], dtype=np.float32)
        bboxes[seq_id] = seq_bboxes

    return bboxes


def lift_predictions_to_3d(
    predictions_2d: Dict[int, np.ndarray],
    device: torch.device = None,
    no_bbox: bool = False,
    pred_bbox: bool = False,
) -> Dict[int, np.ndarray]:
    """Lift HuPR 2D predictions to 3D using MotionBERT.

    Args:
        predictions_2d: Dictionary mapping seq_id to 2D predictions (600, 14, 2).
        device: Torch device.
        no_bbox: If True, use image-level normalization instead of bbox.
        pred_bbox: If True, compute bbox from predicted keypoints instead of GT.

    Returns:
        Dictionary mapping seq_id to 3D predictions (600, 17, 3).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading MotionBERT from {MB_CHECKPOINT_PATH}...")
    model, clip_len = load_motionbert(device)
    print(f"Clip length: {clip_len}")

    gt_bboxes = None
    if no_bbox:
        print("Using image-level normalization (no bbox)")
    elif pred_bbox:
        print("Using bboxes computed from predicted keypoints")
    else:
        print("Loading GT bboxes...")
        gt_bboxes = load_gt_bboxes()

    predictions_3d: Dict[int, np.ndarray] = {}

    for seq_id, pred_2d in tqdm(predictions_2d.items(), desc="Lifting to 3D"):
        if not no_bbox and not pred_bbox:
            if seq_id not in gt_bboxes:
                print(f"Warning: No bbox for seq {seq_id}, skipping")
                continue
            bboxes = gt_bboxes[seq_id]

        # 1. Interpolate jumps in 2D predictions
        pred_2d_clean = interpolate_jumps(pred_2d)

        # 2. Convert 14 HuPR joints to 17 H3.6M joints
        pred_17 = hupr_to_h36m(pred_2d_clean)

        # 3. Normalize
        if no_bbox:
            pred_norm = normalize_screen_coordinates(pred_17, w=256, h=256)
        elif pred_bbox:
            bboxes = compute_bboxes_from_keypoints(pred_17, margin=0.1)
            pred_norm = normalize_with_bbox(pred_17, bboxes)
        else:
            pred_norm = normalize_with_bbox(pred_17, bboxes)

        # 4. Lift to 3D
        pred_3d = lift_2d_to_3d(model, pred_norm, clip_len, device)

        # 5. Make root-relative if model doesn't output root-relative poses
        if not MB_ROOTREL:
            pred_3d = pred_3d - pred_3d[:, 0:1, :]

        predictions_3d[seq_id] = pred_3d

    return predictions_3d


def main() -> None:
    """Lift 2D predictions with saved 2D predictions."""
    pred_path = Path(__file__).parent / "hupr_2d_predictions.npz"

    if not pred_path.exists():
        print(f"Error: {pred_path} not found. Run run_hupr_inference.py first.")
        return

    data = np.load(pred_path)
    predictions_2d = {int(k): v for k, v in data.items()}

    # Configuration
    no_bbox = False
    pred_bbox = True  # Best results

    predictions_3d = lift_predictions_to_3d(
        predictions_2d,
        no_bbox=no_bbox,
        pred_bbox=pred_bbox,
    )

    # Save
    suffix = ""
    if no_bbox:
        suffix += "_nobbox"
    if pred_bbox:
        suffix += "_predbbox"
    output_path = Path(__file__).parent / f"hupr_3d_predictions{suffix}.npz"
    np.savez(output_path, **{str(k): v for k, v in predictions_3d.items()})
    print(f"Saved 3D predictions to {output_path}")


if __name__ == "__main__":
    main()
