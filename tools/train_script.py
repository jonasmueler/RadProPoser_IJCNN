import importlib
import os
import random
import sys

# Ensure tools/ directory is on the path so local imports resolve
# regardless of which directory the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

import data_loaders
import train_loop
from config import (
    MODEL_REGISTRY, MODELNAME, MODELPATH, PATHRAW, SEQLEN, TRAINCONFIG,
)

# Load model via registry
sys.path.append(MODELPATH)
module_name, class_name = MODEL_REGISTRY[MODELNAME]
Encoder = getattr(importlib.import_module(module_name), class_name)

model = Encoder().to(TRAINCONFIG["device"])


def set_seed(seed: int) -> None:
    """Set the seed for reproducibility across CPU, GPU, NumPy, and Python random.

    Args:
        seed: The seed value to use.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to: {seed}")


# Calculate the number of trainable and non-trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
print(f"Non-trainable parameters: {non_trainable_params}")

# get data loaders
single_frame_mode = TRAINCONFIG.get("HR_SINGLE", False)
dataset_train = data_loaders.RadarData(
    "train", PATHRAW, PATHRAW, SEQLEN, single_frame=single_frame_mode
)
data_train = DataLoader(
    dataset_train, TRAINCONFIG["batchSize"], shuffle=True, num_workers=1
)

mse = torch.nn.MSELoss()


if __name__ == "__main__":
    set_seed(42)
    train_loop.train_loop(
        data_train, model, False, MODELNAME,
        TRAINCONFIG, True, TRAINCONFIG["device"], mse,
    )
