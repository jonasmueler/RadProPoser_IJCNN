import os

import torch
from tqdm import tqdm

from config import PATHORIGIN, TESTCONFIG
from datasets.dataset_full import get_dataloader
from misc import (
    JOINT_NAMES,
    compute_ece,
    compute_mpjpe,
    compute_pa_mpjpe,
    compute_sharpness,
    print_pose_table,
    print_uncertainty_table,
)
from models import RadProPoserVAECov

CHECKPOINT_PATH = os.path.join(
    PATHORIGIN, "checkpoints", TESTCONFIG["checkpoint_name"], "best.pth"
)
DEVICE = TESTCONFIG["device"]
BATCH_SIZE = TESTCONFIG["batch_size"]
SEQ_LEN = TESTCONFIG["seq_len"]
NUM_WORKERS = TESTCONFIG["num_workers"]
N_SAMPLES = TESTCONFIG["n_samples"]


def test() -> None:
    """Run evaluation on test set and print results."""
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    if not os.path.exists(CHECKPOINT_PATH):
        latest_path = os.path.join(PATHORIGIN, "checkpoints", TESTCONFIG["checkpoint_name"], "latest.pth")
        if os.path.exists(latest_path):
            print(f"Best checkpoint not found, using latest: {latest_path}")
            checkpoint_path = latest_path
        else:
            raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")
    else:
        checkpoint_path = CHECKPOINT_PATH

    test_loader = get_dataloader(
        phase="test",
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    model = RadProPoserVAECov(n_samples=N_SAMPLES).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "mpjpe" in checkpoint:
        print(f"Checkpoint validation MPJPE: {checkpoint['mpjpe']:.3f} cm")

    all_preds, all_gts, all_vars = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            radar = batch["radar"].to(DEVICE).to(torch.complex64)
            gt = batch["pose_3d"].to(DEVICE).float().reshape(-1, 17 * 3) * 100

            _, _, _, mu, cov = model(radar)
            var = cov.diagonal(dim1=-2, dim2=-1)

            all_preds.append(mu.cpu())
            all_gts.append(gt.cpu())
            all_vars.append(var.cpu())

    preds = torch.cat(all_preds, dim=0)
    gts = torch.cat(all_gts, dim=0)
    variances = torch.cat(all_vars, dim=0)

    mpjpe_per_joint, mpjpe_total = compute_mpjpe(preds, gts)
    pa_mpjpe_per_joint, pa_mpjpe_total = compute_pa_mpjpe(preds, gts)
    sharpness_per_joint, sharpness_total = compute_sharpness(variances)
    ece = compute_ece(preds.numpy(), variances.numpy(), gts.numpy())

    print_pose_table(mpjpe_per_joint, pa_mpjpe_per_joint, mpjpe_total, pa_mpjpe_total)
    print_uncertainty_table(sharpness_per_joint, sharpness_total, ece)


if __name__ == "__main__":
    test()
