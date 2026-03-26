import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import PATHORIGIN, TRAINCONFIG
from datasets.dataset_full import get_dataloader
from models import RadProPoserVAECov
from recalibration.recalibrator import HuPRRecalibrator

DEVICE = "cuda"
BATCH_SIZE = 32
SEQ_LEN = 8
NUM_WORKERS = 8
N_SAMPLES = 128

PREDICTIONS_DIR = SCRIPT_DIR / "predictions"


def get_predictions_path(gt_source: str, phase: str) -> Path:
    """Get path for saved predictions.

    Args:
        gt_source: Ground truth source (e.g., "motionbert")
        phase: Dataset phase ("val" or "test")

    Returns:
        Path to the predictions npz file
    """
    return PREDICTIONS_DIR / gt_source / f"{phase}_predictions.npz"


def save_predictions(
    mu: np.ndarray, var: np.ndarray, gt: np.ndarray, gt_source: str, phase: str
) -> None:
    """Save predictions to disk.

    Args:
        mu: Mean predictions of shape (N, 51)
        var: Variances of shape (N, 51)
        gt: Ground truth of shape (N, 51)
        gt_source: Ground truth source (e.g., "motionbert")
        phase: Dataset phase ("val" or "test")
    """
    save_path = get_predictions_path(gt_source, phase)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, mu=mu, var=var, gt=gt)
    print(f"Saved predictions to {save_path}")


def load_predictions(gt_source: str, phase: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load predictions from disk if they exist.

    Args:
        gt_source: Ground truth source (e.g., "motionbert")
        phase: Dataset phase ("val" or "test")

    Returns:
        Tuple of (mu, var, gt) arrays if file exists, None otherwise
    """
    load_path = get_predictions_path(gt_source, phase)
    if not load_path.exists():
        return None

    data = np.load(load_path)
    print(f"Loaded predictions from {load_path}")
    return data["mu"], data["var"], data["gt"]


def get_checkpoint_path() -> Path:
    """Get checkpoint path from TESTCONFIG.

    Returns:
        Path to the best checkpoint file

    Raises:
        FileNotFoundError: If no checkpoint exists
    """
    from config import TESTCONFIG
    checkpoint_name = TESTCONFIG["checkpoint_name"]

    best_path = Path(PATHORIGIN) / "checkpoints" / checkpoint_name / "best.pth"
    latest_path = Path(PATHORIGIN) / "checkpoints" / checkpoint_name / "latest.pth"

    if best_path.exists():
        return best_path
    if latest_path.exists():
        print(f"Best checkpoint not found, using latest: {latest_path}")
        return latest_path

    raise FileNotFoundError(f"No checkpoint found for {checkpoint_name}")


def run_inference(
    model: RadProPoserVAECov, dataloader: torch.utils.data.DataLoader
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference to collect predictions, variances, and ground truth.

    Args:
        model: Trained RadProPoserVAECov model
        dataloader: DataLoader for the dataset split

    Returns:
        Tuple of (predictions, variances, ground_truth) arrays, each of shape (N, 51)
    """
    all_preds, all_vars, all_gts = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference", unit="batch"):
            radar = batch["radar"].to(DEVICE).to(torch.complex64)
            gt = batch["pose_3d"].to(DEVICE).float().reshape(-1, 17 * 3) * 100

            _, _, _, mu, cov = model(radar)
            var = cov.diagonal(dim1=-2, dim2=-1)

            all_preds.append(mu.cpu().numpy())
            all_vars.append(var.cpu().numpy())
            all_gts.append(gt.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    variances = np.concatenate(all_vars, axis=0)
    gts = np.concatenate(all_gts, axis=0)

    return preds, variances, gts


def get_or_run_inference(
    gt_source: str, phase: str, model: RadProPoserVAECov | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load predictions from cache or run inference if not cached.

    Args:
        gt_source: Ground truth source (e.g., "motionbert")
        phase: Dataset phase ("val" or "test")
        model: Model instance (only needed if predictions not cached)

    Returns:
        Tuple of (mu, var, gt) arrays
    """
    cached = load_predictions(gt_source, phase)
    if cached is not None:
        return cached

    if model is None:
        raise ValueError(f"No cached predictions and no model provided for {phase}")

    print(f"\n--- Running inference on {phase} set ---")
    dataloader = get_dataloader(
        phase=phase,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )
    print(f"{phase.capitalize()} samples: {len(dataloader.dataset)}")

    mu, var, gt = run_inference(model, dataloader)
    save_predictions(mu, var, gt, gt_source, phase)

    return mu, var, gt


def format_sharpness_table(results: dict, gt_source: str) -> None:
    """Print formatted sharpness values for table creation.

    Args:
        results: Results dictionary from recalibration
        gt_source: Ground truth source name
    """
    # Joint indices in 17-joint H3.6M format:
    # 0:Hip, 1:R_Hip, 2:R_Knee, 3:R_Ankle, 4:L_Hip, 5:L_Knee, 6:L_Ankle
    # 7:Spine, 8:Thorax, 9:Neck, 10:Head
    # 11:L_Shoulder, 12:L_Elbow, 13:L_Wrist
    # 14:R_Shoulder, 15:R_Elbow, 16:R_Wrist

    joint_groups = {
        "Elbow": [12, 15],  # L_Elbow, R_Elbow
        "Wrist": [13, 16],  # L_Wrist, R_Wrist
        "Knee": [2, 5],     # R_Knee, L_Knee
        "Ankle": [3, 6],    # R_Ankle, L_Ankle
    }

    # Extract per-joint arrays
    unc_per_joint = results["sharpness_uncalibrated_per_joint"]
    cal_per_joint = results["sharpness_isotonic_per_joint"]

    # Compute grouped averages
    def compute_grouped(per_joint_array: np.ndarray, total_key: str) -> dict:
        grouped = {}
        for group_name, indices in joint_groups.items():
            grouped[group_name] = np.mean([per_joint_array[i] for i in indices])
        grouped["Total"] = results[total_key]
        return grouped

    unc_grouped = compute_grouped(unc_per_joint, "sharpness_uncalibrated")
    cal_grouped = compute_grouped(cal_per_joint, "sharpness_isotonic")

    # Print formatted output
    print("\n" + "=" * 80)
    print(f"TABLE DATA: {gt_source}")
    print("=" * 80)

    # Uncalibrated row (ECE in first column, empty ECE(Cal))
    print(f"Uncalibrated | {results['ece_uncalibrated']:.3f} |     - | "
          f"{unc_grouped['Elbow']:.3f} | {unc_grouped['Wrist']:.3f} | "
          f"{unc_grouped['Knee']:.3f} | {unc_grouped['Ankle']:.3f} | "
          f"{unc_grouped['Total']:.3f}")

    # Calibrated row (empty ECE, ECE(Cal) in second column)
    print(f"Calibrated   |     - | {results['ece_isotonic']:.3f} | "
          f"{cal_grouped['Elbow']:.3f} | {cal_grouped['Wrist']:.3f} | "
          f"{cal_grouped['Knee']:.3f} | {cal_grouped['Ankle']:.3f} | "
          f"{cal_grouped['Total']:.3f}")

    print("=" * 80)
    print("Format: ECE | ECE(Cal) | Elbow | Wrist | Knee | Ankle | Total")
    print("All sharpness values in cm²")


def main() -> None:
    """Run recalibration for the model specified in config."""
    model_name = "motionbert"
    print(f"Running recalibration for model: {model_name}")

    val_cached = load_predictions(model_name, "val")
    test_cached = load_predictions(model_name, "test")

    model = None
    if val_cached is None or test_cached is None:
        checkpoint_path = get_checkpoint_path()
        print(f"Loading checkpoint: {checkpoint_path}")

        model = RadProPoserVAECov(n_samples=N_SAMPLES).to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        if "epoch" in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if "mpjpe" in checkpoint:
            print(f"Checkpoint validation MPJPE: {checkpoint['mpjpe']:.3f} cm")

    mu_val, var_val, gt_val = get_or_run_inference(model_name, "val", model)
    mu_test, var_test, gt_test = get_or_run_inference(model_name, "test", model)

    print("\n--- Running recalibration ---")
    recalibrator = HuPRRecalibrator()
    results = recalibrator.quantify_improvement_with_holdout(
        mu_train=mu_val,
        var_train=var_val,
        gt_train=gt_val,
        mu_test=mu_test,
        var_test=var_test,
        gt_test=gt_test,
        model_name=model_name,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"ECE (Uncalibrated): {results['ece_uncalibrated']:.6f}")
    print(f"ECE (Isotonic):     {results['ece_isotonic']:.6f}")
    print(f"ECE Improvement:    {results['ece_uncalibrated'] - results['ece_isotonic']:.6f}")

    # Format per-joint sharpness for table
    format_sharpness_table(results, model_name)


if __name__ == "__main__":
    main()
