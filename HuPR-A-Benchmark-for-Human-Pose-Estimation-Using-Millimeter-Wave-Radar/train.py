import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PATH_ORIGIN, TRAIN_CONFIG
from datasets.dataset_full import get_dataloader
from misc import kl_loss, compute_mpjpe, nll_from_cov
from misc.constants import NUM_DIMS
from models import RadProPoserVAECov


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    path: str,
    mpjpe: Optional[float] = None,
) -> None:
    """Save model and optimizer checkpoint to disk.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        path: Path to save checkpoint
        mpjpe: Optional validation MPJPE to store
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if mpjpe is not None:
        ckpt["mpjpe"] = mpjpe
    torch.save(ckpt, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    model: nn.Module, optimizer: Optional[optim.Optimizer], path: str
) -> int:
    """Load model and optimizer from checkpoint.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into (can be None)
        path: Path to checkpoint file

    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Checkpoint loaded: {path}")
    return checkpoint.get("epoch", 0)


def validate(model: nn.Module, val_loader: DataLoader, device: str) -> float:
    """Run validation and compute MPJPE.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to run on

    Returns:
        Mean MPJPE across validation set in cm
    """
    model.eval()
    total_mpjpe = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", unit="batch", mininterval=1.0):
            radar = batch["radar"].to(device).to(torch.complex64)
            gt = batch["pose_3d"].to(device).float().reshape(-1, NUM_DIMS) * 100

            _, _, _, preds, _ = model(radar)
            _, mpjpe = compute_mpjpe(preds, gt)
            total_mpjpe += mpjpe.item()
            count += 1

    model.train()
    return total_mpjpe / count if count > 0 else 0


def train() -> None:
    """Train RadProPoserVAECov model on HuPR full radar data."""
    cfg = TRAIN_CONFIG
    device = cfg["device"]

    # Enable TF32 for faster matmul on Ampere GPUs (RTX 3090)
    torch.set_float32_matmul_precision('high')

    # Initialize W&B
    if cfg["use_wandb"]:
        wandb.init(
            project="RadProPoser-HuPR",
            config=cfg,
        )

    # Data loaders
    train_loader = get_dataloader(
        phase="train",
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        stride=cfg["stride"],
        num_workers=cfg["num_workers"],
    )
    val_loader = get_dataloader(
        phase="val",
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        stride=cfg["stride"],
        num_workers=cfg["num_workers"],
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Model
    model = RadProPoserVAECov(n_samples=cfg["n_samples"]).to(device)
    # Note: torch.compile disabled - incompatible with complex tensor operations in radar data

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=cfg["betas"],
        weight_decay=cfg["weight_decay"],
    )

    # LR scheduler
    scheduler = optim.lr_scheduler.MultiplicativeLR(
        optimizer, lambda epoch: cfg["lr_decay"]
    )

    # Load checkpoint if resuming
    start_epoch = 0
    best_mpjpe = float("inf")
    if cfg["resume"]:
        checkpoint_path = os.path.join(PATH_ORIGIN, "checkpoints", cfg["name"], "latest.pth")
        if os.path.exists(checkpoint_path):
            start_epoch = load_checkpoint(model, optimizer, checkpoint_path) + 1
        best_path = os.path.join(PATH_ORIGIN, "checkpoints", cfg["name"], "best.pth")
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, weights_only=True)
            best_mpjpe = best_ckpt.get("mpjpe", float("inf"))

    # Training loop
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss = 0
        epoch_nll = 0
        epoch_kl = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", mininterval=1.0)
        for batch_idx, batch in enumerate(pbar):
            radar = batch["radar"].to(device).to(torch.complex64)
            gt = batch["pose_3d"].to(device).float().reshape(-1, NUM_DIMS) * 100  # to cm

            optimizer.zero_grad()

            # Forward pass
            _, mu, latent_spread, mu_out, cov = model(radar)

            # Compute losses
            kl = kl_loss(mu, latent_spread)
            nll, pen = nll_from_cov(gt, mu_out, cov)
            loss = nll + cfg["beta"] * kl

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_nll += nll.item()
            epoch_kl += kl.item()

            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.3f}", nll=f"{nll.item():.3f}", kl=f"{kl.item():.3f}")

            # Log batch
            if cfg["use_wandb"]:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/nll": nll.item(),
                    "train/kl": kl.item(),
                    "train/var_penalty": pen.item(),
                })

        scheduler.step()

        # Validation
        val_mpjpe = validate(model, val_loader, device)

        avg_loss = epoch_loss / len(train_loader)
        avg_nll = epoch_nll / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)

        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f} Val MPJPE: {val_mpjpe:.2f} cm")

        if cfg["use_wandb"]:
            wandb.log({
                "epoch": epoch,
                "train/avg_loss": avg_loss,
                "train/avg_nll": avg_nll,
                "train/avg_kl": avg_kl,
                "val/mpjpe": val_mpjpe,
                "lr": optimizer.param_groups[0]["lr"],
            })

        # Save checkpoints
        checkpoint_dir = os.path.join(PATH_ORIGIN, "checkpoints", cfg["name"])
        save_checkpoint(model, optimizer, epoch, os.path.join(checkpoint_dir, "latest.pth"), mpjpe=val_mpjpe)
        save_checkpoint(model, optimizer, epoch, os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"), mpjpe=val_mpjpe)

        # Save best model
        if val_mpjpe < best_mpjpe:
            best_mpjpe = val_mpjpe
            save_checkpoint(model, optimizer, epoch, os.path.join(checkpoint_dir, "best.pth"), mpjpe=val_mpjpe)
            print(f"New best model! MPJPE: {val_mpjpe:.2f} cm")

    if cfg["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    train()
