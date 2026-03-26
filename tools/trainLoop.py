import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from edl_pytorch import evidential_regression
from torch.optim.lr_scheduler import LambdaLR, SequentialLR

from config import *


def MPJPE(preds: torch.Tensor, 
          targets: torch.Tensor)->torch.Tensor:
    """Calculate the Mean Per Joint Position Error (MPJPE).

    Args:
        preds: Predicted joint positions, shape (batch_size, num_joints, 3).
        targets: Ground truth joint positions, shape (batch_size, num_joints, 3).

    Returns:
        The MPJPE for the batch.
    """
    assert preds.shape == targets.shape, "Predictions and targets must have the same shape"
    
    # Calculate the Euclidean distance between predicted and target joint positions
    #diff = preds.reshape(preds.size(0), 14, 3) - targets.reshape(preds.size(0), 14, 3)
    diff = preds.reshape(preds.size(0), 26, 3) - targets.reshape(preds.size(0), 26, 3)
    dist = torch.norm(diff, dim=-1)
    
    # Calculate the mean per joint position error
    mpjpe = dist.mean()
    
    return mpjpe

def saveCheckpoint(model: nn.Module, 
                   optimizer: torch.optim.Adam, 
                   filename: str):
    """Saves the model and optimizer states to a checkpoint file.

    Args:
        model: The model whose state needs to be saved.
        optimizer: The optimizer whose state needs to be saved.
        filename: The name of the file to save the checkpoint to (relative to PATHORIGIN).
    """
    filepath = os.path.join(PATHORIGIN, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, filepath)
    print("checkpoint saved")
    return

def loadCheckpoint(model: nn.Module, 
                   optimizer: torch.optim.Adam, 
                   path: str)->torch.nn.Module:
    """Loads a trained model and/or optimizer state from a checkpoint file.

    Args:
        model: The model instance to load the saved state into.
        optimizer: The optimizer instance. If None, only the model state will be loaded.
        path: Path to the checkpoint file.

    Returns:
        The model with the loaded state, or [model, optimizer] if optimizer is provided.
    """

    if optimizer is not None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("checkpoint loaded")
        return [model, optimizer]
    elif optimizer is None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        print("checkpoint loaded")
        return model
    
def lrLambda(epoch: int):
    """Return multiplicative learning rate decay factor for the scheduler."""
    return TRAINCONFIG["lrDecay"]

def KLLoss(mu: torch.Tensor,
           logvar: torch.Tensor)->torch.Tensor:
    """Compute KL divergence between learned Gaussian posterior and standard normal prior."""
    klloss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return klloss

def nllLoss(gt: torch.Tensor, 
            mu: torch.Tensor, 
            var: torch.Tensor) -> torch.Tensor:
    """Simplified Negative Log-Likelihood (NLL) loss function.

    Args:
        gt: Ground truth tensor.
        mu: Predicted mean tensor.
        var: Predicted variance tensor.

    Returns:
        Tuple of (NLL loss, variance penalty).
    """
    # Ensure variance is positive to avoid numerical instability
    var = torch.clamp(var, min=1e-6)

    # Compute the NLL loss
    pen = TRAINCONFIG["gamma"] * var.mean()
    loss = ((gt - mu) ** 2 / var).mean() + pen

    return loss, pen

def nll_from_cov(gt: torch.Tensor,
                 mu: torch.Tensor,
                 cov: torch.Tensor):
    """Negative log-likelihood loss for full covariance predictions.

    Args:
        gt: Ground truth of shape (B, K).
        mu: Predictive mean of shape (B, K).
        cov: Predictive covariance (SPD) of shape (B, K, K).

    Returns:
        Tuple of (NLL loss, covariance penalty).
    """
    B, K = gt.shape
    eps  = 1e-6
    eye  = torch.eye(K, device=cov.device, dtype=cov.dtype)
    cov  = cov + eps * eye                       # SPD “jitter”

    L    = torch.linalg.cholesky(cov)            # (B, K, K)
    diff = (gt - mu).unsqueeze(-1)               # (B, K, 1)
    y    = torch.linalg.solve_triangular(L, diff, upper=False)
    maha = (y.square()).sum(dim=(-2, -1))        # Mahalanobis  (B,)

    logdet = 2.0 * torch.log(torch.diagonal(L, 0, -2, -1)).sum(-1)

    nll = 0.5 * (maha + logdet + K * np.log(2.0 * np.pi))  # (B,)

    pen = TRAINCONFIG["gamma"] * cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1)  # (B,)

    return nll.mean() + pen.mean(), pen.mean()



def nllLoss_precision(gt: torch.Tensor, 
                        mu: torch.Tensor, 
                        log_variance: torch.Tensor) -> torch.Tensor:
    """NLL loss using predicted log variance, converting to precision-based Gaussian NLL.

    Args:
        gt: Ground truth tensor.
        mu: Predicted mean tensor.
        log_variance: Predicted log variance (log(sigma^2)).

    Returns:
        Tuple of (NLL loss, precision penalty).
    """
    # Step 1: Convert to variance
    #variance = torch.exp(log_variance)
    variance = log_variance

    # Step 2: Compute precision
    precision = 1.0 / variance

    # Step 3: Compute log precision
    log_precision = torch.log(precision)

    # penalty 
    pen = -(TRAINCONFIG["gamma"] *log_precision.mean())

    # Step 4: Final NLL computation
    nll = (pen + precision * (gt - mu) ** 2).mean() 
    return nll, pen


def laplace_nll(
    y_true: torch.Tensor,
    mu_pred: torch.Tensor,
    var_pred: torch.Tensor
) -> torch.Tensor:
    """Laplace negative log-likelihood loss computed from empirical variance.

    Args:
        y_true: Ground truth tensor.
        mu_pred: Empirical mean of predicted samples.
        var_pred: Empirical variance of predicted samples.

    Returns:
        Tuple of (loss, penalty) averaged over batch and dimensions.
    """
    # Convert variance to Laplace scale: var = 2b² ⇒ b = sqrt(var / 2)
    b = torch.sqrt(torch.clamp(var_pred, min=1e-10) / 2.0)

    abs_error = torch.abs(y_true - mu_pred)
    pen = TRAINCONFIG["gamma"] * b
    loss = pen + abs_error / b

    return loss.mean(), pen.mean()
    


def trainLoop(trainLoader: torch.utils.data.DataLoader, 
              valLoader: torch.utils.data.DataLoader, 
              model: nn.Module,
              loadModel: bool, 
              modelName: str, 
              params: dict,  
              WandB: bool, 
              device: str, 
              criterion: nn.Module):
        
    """Train a model with checkpoint loading, optimizer/scheduler setup, and WandB logging.

    Args:
        trainLoader: DataLoader for the training dataset.
        valLoader: DataLoader for the validation dataset.
        model: The neural network model to be trained.
        loadModel: Whether to load a previously saved model checkpoint.
        modelName: The name of the model used for saving checkpoints and logging.
        params: Training parameters (optimizer, learningRate, betas, weightDecay, epochs).
        WandB: Whether to use Weights & Biases for logging.
        device: Device to use for training (e.g., "cpu" or "cuda").
        criterion: Loss function to use when TRAINCONFIG["nll"] is False.
    """

    # WandB
    if WandB:
        wandb.init(
            # set the wandb project where this run will be logged
            project= modelName,

            # track hyperparameters and run metadata
            config=params
        )

    # get optimizer
    if params["optimizer"] == "adam":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr = params["learningRate"],
                                      betas = params["betas"],
                                      weight_decay= params["weightDecay"])
    if params["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                      lr = params["learningRate"],
                                      betas = params["betas"],
                                      weight_decay= params["weightDecay"])

    # load model and optimizer from checkpoint
    if loadModel:
        # get into folder
        os.chdir(os.path.join(PATHORIGIN, "models"))
        checkpoint_path = os.path.join(PATHORIGIN, "trainedModels", modelName, modelName)
        lastState = loadCheckpoint(model, optimizer, checkpoint_path)
        model = lastState[0]
        optimizer = lastState[1]

    # initilaize LR sheduling
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lrLambda)

    ###################### start training #############################
    for b in range(params["epochs"]):
        trainCounter = 0
        for radar,  gt in trainLoader:
            model.train()

            # move data to device and preprocess
            radar = radar.to(device).to(torch.complex64)
            #radar2 = radar.to(device).float() # .to(torch.complex64)
            gt = gt.to(device).float() * 100 # convert to cm
                
            # zero the parameter gradients
            optimizer.zero_grad()

            # loss functions
            if TRAINCONFIG["nll"] == True:
                preds, mu, latentSpread, muOut, varOut = model.forward(radar) # varOut can also be covariance, laplacian latent has no kl term

                # check for different likelihoods 
                if MODELNAME in ("RPPgaussianGaussian"):
                    KLloss = KLLoss(mu, latentSpread) 
                    nLL, pen = nllLoss(gt, muOut, varOut)
                elif MODELNAME in ("RPPgaussianGaussianCov"):
                     KLloss = KLLoss(mu, latentSpread)
                     nLL, pen = nll_from_cov(gt, muOut, varOut)
                
                elif MODELNAME in ("RPPlaplaceGaussian"): #laplace latent has no KL div term
                     nLL, pen = nllLoss(gt, muOut, varOut)
                     KLloss = torch.zeros_like(nLL).to(TRAINCONFIG["device"])

                elif MODELNAME in ("RPPgaussianLaplace"): 
                     nLL, pen = laplace_nll(gt, muOut, varOut)
                     KLloss = KLLoss(mu, latentSpread)

                elif MODELNAME in ("RPPlaplaceLaplace"):
                    nLL, pen = laplace_nll(gt, muOut, varOut)
                    KLloss = torch.zeros_like(nLL).to(TRAINCONFIG["device"])

                loss = nLL + TRAINCONFIG["beta"] * KLloss

            elif TRAINCONFIG["nf"] == True:
                    mu, kld = model.forward(radar, inference = False)
                    loss = criterion(mu, gt) +  TRAINCONFIG["beta"] * kld
            
            elif TRAINCONFIG["evd"] == True:
                pred_nig, _ = model(radar)

                loss = evidential_regression(
                        pred_nig,      # predicted Normal Inverse Gamma parameters
                        gt,             
                        lamb=TRAINCONFIG["lambda"],    # regularization coefficient 
                    )
            
            elif TRAINCONFIG["HR_SINGLE"] == True:
                     root, offsets = model.forward(radar)
                     gt = gt.reshape(gt.shape[0], 26, 3)
                     loss_root = criterion(root, gt[:,0])
                     gt_offset = gt - gt[:, 0].unsqueeze(dim = 1)
                     loss_offset = criterion(gt_offset[:, 1:], offsets)
                     loss = loss_root + loss_offset

               
            else:
                preds = model.forward(radar)
                loss = criterion(preds, gt)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            trainCounter += 1

            # logging 
            if TRAINCONFIG["nll"] == True:
                if WandB:
                        wandb.log({"nll": nLL.detach().cpu().item(), 
                                   "KLLOss": KLloss.detach().cpu().item(), 
                                   "loss": loss.detach().cpu().item(), 
                                   "varPenalty": pen.detach().cpu().item()})
                        
            elif TRAINCONFIG["nf"] == True:
                if WandB:
                        wandb.log({
                                   "loss": loss.detach().cpu().item(), 
                                   "KL": kld.detach().cpu().item()})
                        
            elif TRAINCONFIG["evd"] == True:
                if WandB:
                        wandb.log({"evidential_loss": loss.detach().cpu().item(), 
                                   "v": pred_nig[1].mean(), 
                                   "alpha": pred_nig[2].mean(), 
                                   "beta": pred_nig[3].mean()})

            elif TRAINCONFIG["HR_SINGLE"] == True:
                 if WandB:
                        wandb.log({"MSE": loss.detach().cpu().item()})

            else:
                if WandB:
                        wandb.log({"MSE": loss.detach().cpu().item()})

        scheduler.step()
        ########################### Validation Loop ############################
        with torch.no_grad():
            print("start validating model")
            valLossMean = 0
            counter = 0
            for x,  y in valLoader:
                model.eval()
                x = x.to(device).to(device).to(torch.complex64)
                y = y.to(device).float() * 100

                if TRAINCONFIG["nll"] == True:
                    _, _, _, preds, _ = model.forward(x)
                
                elif TRAINCONFIG["nf"] == True:
                    preds,_,_ = model.forward(x, inference = True)

                elif TRAINCONFIG["evd"] == True:
                    preds, _ = model(x) # (mu, v, alpha, beta)
                    preds = preds[0]
                    
                elif TRAINCONFIG["HR_SINGLE"] == True:
                     root, offset = model(x)
                     preds = root.unsqueeze(dim = 1) + offset
                     preds = torch.cat([root.unsqueeze(dim = 1), preds], dim = 1)
                     y = y.reshape(y.size(0), 26, 3)
                else:
                    preds = model.forward(x)

                # calculate validation loss    
                valLoss = MPJPE(preds, y)
                valLossMean += valLoss

                counter += 1
                
                        
            valLosses = valLossMean/counter + 1 

            

            if WandB:
                    wandb.log({"valLoss": valLosses.detach().cpu().item(), 
                               })
            print("valLoss: ", valLosses.detach().cpu().item())

            # save model and optimizer checkpoint
            checkpoint_path = os.path.join("trainedModels", modelName, modelName + str(b))
            saveCheckpoint(model, optimizer, checkpoint_path)

    ## save model state
    final_checkpoint_path = os.path.join("trainedModels", modelName, modelName + str(b))
    saveCheckpoint(model, optimizer, final_checkpoint_path)
    print("Model saved!")
    print("Training done!")

