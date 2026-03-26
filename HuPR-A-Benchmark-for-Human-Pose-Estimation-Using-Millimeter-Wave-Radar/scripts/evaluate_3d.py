import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import ANNOT_DIR
from misc.constants import JOINT_NAMES, TEST_SEQUENCES
from misc.metrics import compute_mpjpe, compute_pa_mpjpe, print_pose_table


def load_gt_3d() -> Dict[int, np.ndarray]:
    """Load ground truth 3D poses from MotionBERT annotations.

    Returns:
        Dictionary mapping seq_id to 3D poses of shape (600, 17, 3).
    """
    annot_path = ANNOT_DIR / "motionbert_3d_test.json"

    with open(annot_path) as f:
        annot = json.load(f)

    gt = {}
    for seq_idx, seq in enumerate(annot):
        if seq_idx >= len(TEST_SEQUENCES):
            break

        seq_id = TEST_SEQUENCES[seq_idx]
        poses = np.array([frame["joints_3d"] for frame in seq], dtype=np.float32)
        gt[seq_id] = poses

    return gt


def evaluate_predictions(predictions_3d: Dict[int, np.ndarray]) -> None:
    """Evaluate 3D predictions against ground truth and print results.

    Args:
        predictions_3d: Dictionary mapping seq_id to 3D predictions (600, 17, 3).
    """
    print("Loading ground truth 3D poses (MotionBERT)...")
    gt_3d = load_gt_3d()

    all_preds = []
    all_gts = []

    for seq_id in tqdm(TEST_SEQUENCES, desc="Collecting predictions"):
        if seq_id not in predictions_3d:
            print(f"Warning: No predictions for seq {seq_id}")
            continue
        if seq_id not in gt_3d:
            print(f"Warning: No GT for seq {seq_id}")
            continue

        pred = predictions_3d[seq_id]
        gt = gt_3d[seq_id]

        # Ensure same length
        min_len = min(len(pred), len(gt))
        all_preds.append(pred[:min_len])
        all_gts.append(gt[:min_len])

    # Concatenate all sequences
    preds = torch.from_numpy(np.concatenate(all_preds, axis=0))
    gts = torch.from_numpy(np.concatenate(all_gts, axis=0))

    # Convert to cm (GT is in meters, predictions should be in same scale)
    preds_cm = preds * 100
    gts_cm = gts * 100

    print(f"\nTotal test frames: {len(preds)}")

    # Compute metrics
    mpjpe_per_joint, mpjpe_total = compute_mpjpe(preds_cm, gts_cm)
    pa_mpjpe_per_joint, pa_mpjpe_total = compute_pa_mpjpe(preds_cm, gts_cm)

    # Print table
    print_pose_table(mpjpe_per_joint, pa_mpjpe_per_joint, mpjpe_total, pa_mpjpe_total)


def main() -> None:
    """Evaluate saved 3D predictions."""
    pred_path = Path(__file__).parent / "hupr_3d_predictions_predbbox.npz"

    if not pred_path.exists():
        print(f"Error: {pred_path} not found. Run lift_to_3d.py first.")
        return

    data = np.load(pred_path)
    predictions_3d = {int(k): v for k, v in data.items()}

    evaluate_predictions(predictions_3d)


if __name__ == "__main__":
    main()
