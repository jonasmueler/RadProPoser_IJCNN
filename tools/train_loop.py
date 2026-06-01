import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from edl_pytorch import evidential_regression

from config import MODELNAME, NUM_COORDS, NUM_JOINTS, PATHORIGIN, TRAINCONFIG
from losses import kl_loss, laplace_nll, nll_from_cov, nll_loss


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, filename: str
) -> None:
    """Save model and optimizer states to a checkpoint file.

    Args:
        model: The model whose state needs to be saved.
        optimizer: The optimizer whose state needs to be saved.
        filename: File path relative to PATHORIGIN.
    """
    filepath = os.path.join(PATHORIGIN, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print("checkpoint saved")


def load_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, path: str
) -> nn.Module:
    """Load model and optionally optimizer state from a checkpoint file.

    Args:
        model: The model instance to load weights into.
        optimizer: The optimizer to load state into. If None, only model is loaded.
        path: Path to the checkpoint file.

    Returns:
        The model if optimizer is None, otherwise [model, optimizer].
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("checkpoint loaded")
        return [model, optimizer]
    print("checkpoint loaded")
    return model


def lr_lambda(epoch: int) -> float:
    """Return multiplicative learning rate decay factor for the scheduler.

    Args:
        epoch: Current epoch number (unused, decay is constant).

    Returns:
        Learning rate decay multiplier.
    """
    return TRAINCONFIG["lrDecay"]


def train_loop(
    train_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    load_model: bool,
    model_name: str,
    params: dict,
    use_wandb: bool,
    device: str,
    criterion: nn.Module,
) -> None:
    """Train a model with checkpoint loading, optimizer/scheduler setup, and WandB logging.

    Args:
        train_loader: DataLoader for the training dataset.
        model: The neural network model to be trained.
        load_model: Whether to load a previously saved model checkpoint.
        model_name: The name of the model used for saving checkpoints and logging.
        params: Training parameters (optimizer, learningRate, betas, weightDecay, epochs).
        use_wandb: Whether to use Weights & Biases for logging.
        device: Device to use for training (e.g., "cpu" or "cuda").
        criterion: Loss function to use when TRAINCONFIG["nll"] is False.
    """

    if use_wandb:
        wandb.init(project=model_name, config=params)

    # get optimizer
    if params["optimizer"] == "adam":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learningRate"],
            betas=params["betas"],
            weight_decay=params["weightDecay"],
        )
    if params["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=params["learningRate"],
            betas=params["betas"],
            weight_decay=params["weightDecay"],
        )

    # load model and optimizer from checkpoint
    if load_model:
        os.chdir(os.path.join(PATHORIGIN, "models"))
        checkpoint_path = os.path.join(
            PATHORIGIN, "trainedModels", model_name, model_name
        )
        last_state = load_checkpoint(model, optimizer, checkpoint_path)
        model = last_state[0]
        optimizer = last_state[1]

    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)

    for b in range(params["epochs"]):
        train_counter = 0
        for radar, gt in train_loader:
            model.train()

            radar = radar.to(device).to(torch.complex64)
            gt = gt.to(device).float() * 100  # convert to cm

            optimizer.zero_grad()

            if TRAINCONFIG["nll"]:
                preds, mu, latent_spread, mu_out, var_out = model.forward(radar)

                if MODELNAME == "RPPgaussianGaussian":
                    kl = kl_loss(mu, latent_spread)
                    nll, pen = nll_loss(gt, mu_out, var_out)
                elif MODELNAME == "RPPgaussianGaussianCov":
                    kl = kl_loss(mu, latent_spread)
                    nll, pen = nll_from_cov(gt, mu_out, var_out)
                elif MODELNAME == "RPPlaplaceGaussian":
                    nll, pen = nll_loss(gt, mu_out, var_out)
                    kl = torch.zeros_like(nll).to(TRAINCONFIG["device"])
                elif MODELNAME == "RPPgaussianLaplace":
                    nll, pen = laplace_nll(gt, mu_out, var_out)
                    kl = kl_loss(mu, latent_spread)
                elif MODELNAME == "RPPlaplaceLaplace":
                    nll, pen = laplace_nll(gt, mu_out, var_out)
                    kl = torch.zeros_like(nll).to(TRAINCONFIG["device"])

                loss = nll + TRAINCONFIG["beta"] * kl

            elif TRAINCONFIG["nf"]:
                mu, kld = model.forward(radar, inference=False)
                loss = criterion(mu, gt) + TRAINCONFIG["beta"] * kld

            elif TRAINCONFIG["evd"]:
                pred_nig, _ = model(radar)
                loss = evidential_regression(
                    pred_nig, gt, lamb=TRAINCONFIG["lambda"]
                )

            elif TRAINCONFIG.get("HR_SINGLE"):
                root, offsets = model.forward(radar)
                gt = gt.reshape(gt.shape[0], NUM_JOINTS, NUM_COORDS)
                loss_root = criterion(root, gt[:, 0])
                gt_offset = gt - gt[:, 0].unsqueeze(dim=1)
                loss_offset = criterion(gt_offset[:, 1:], offsets)
                loss = loss_root + loss_offset

            else:
                preds = model.forward(radar)
                loss = criterion(preds, gt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_counter += 1

            # logging
            if TRAINCONFIG["nll"]:
                if use_wandb:
                    wandb.log({
                        "nll": nll.detach().cpu().item(),
                        "KLLoss": kl.detach().cpu().item(),
                        "loss": loss.detach().cpu().item(),
                        "varPenalty": pen.detach().cpu().item(),
                    })
            elif TRAINCONFIG["nf"]:
                if use_wandb:
                    wandb.log({
                        "loss": loss.detach().cpu().item(),
                        "KL": kld.detach().cpu().item(),
                    })
            elif TRAINCONFIG["evd"]:
                if use_wandb:
                    wandb.log({
                        "evidential_loss": loss.detach().cpu().item(),
                        "v": pred_nig[1].mean(),
                        "alpha": pred_nig[2].mean(),
                        "beta": pred_nig[3].mean(),
                    })
            elif TRAINCONFIG.get("HR_SINGLE"):
                if use_wandb:
                    wandb.log({"MSE": loss.detach().cpu().item()})
            else:
                if use_wandb:
                    wandb.log({"MSE": loss.detach().cpu().item()})

        scheduler.step()

        checkpoint_path = os.path.join(
            "trainedModels", model_name, model_name + str(b)
        )
        save_checkpoint(model, optimizer, checkpoint_path)

    print("Training done!")
