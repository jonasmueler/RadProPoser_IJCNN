import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import DATA_DIR, HUPR_CHECKPOINT_PATH, HUPR_INFERENCE_CONFIG
from misc.constants import TEST_SEQUENCES
from models import HuPRNet

PROCESSED_DIR = DATA_DIR / "test_processed"

# Config values from central config
NUM_GROUP_FRAMES = HUPR_INFERENCE_CONFIG["num_group_frames"]
NUM_FRAMES = HUPR_INFERENCE_CONFIG["num_frames"]
NUM_CHIRPS = HUPR_INFERENCE_CONFIG["num_chirps"]
RANGE_SIZE = HUPR_INFERENCE_CONFIG["range_size"]
AZIMUTH_SIZE = HUPR_INFERENCE_CONFIG["azimuth_size"]
ELEVATION_SIZE = HUPR_INFERENCE_CONFIG["elevation_size"]
HEATMAP_SIZE = HUPR_INFERENCE_CONFIG["heatmap_size"]
IMG_SIZE = HUPR_INFERENCE_CONFIG["img_size"]
DURATION = HUPR_INFERENCE_CONFIG["duration"]
NUM_KEYPOINTS = HUPR_INFERENCE_CONFIG["num_keypoints"]


class ConfigObj:
    """Simple config object to mimic YAML config structure."""

    def __init__(self) -> None:
        """Initialize config object with HuPR inference parameters."""
        self.DATASET = type("DATASET", (), {
            "numFrames": NUM_FRAMES,
            "numGroupFrames": NUM_GROUP_FRAMES,
            "rangeSize": RANGE_SIZE,
            "azimuthSize": AZIMUTH_SIZE,
            "elevationSize": ELEVATION_SIZE,
            "heatmapSize": HEATMAP_SIZE,
            "numKeypoints": NUM_KEYPOINTS,
        })()
        self.MODEL = type("MODEL", (), {
            "numFilters": 32,
        })()


class Normalize:
    """Normalize radar data per-channel."""

    def __call__(self, radar_data: torch.Tensor) -> torch.Tensor:
        """Normalize radar data per-channel to zero mean and unit variance.

        Args:
            radar_data: Radar tensor of shape (C, H, W).

        Returns:
            Normalized tensor of same shape.
        """
        c = radar_data.size(0)
        min_values = torch.min(radar_data.view(c, -1), 1)[0].view(c, 1, 1)
        radar_zero = radar_data - min_values
        max_values = torch.max(radar_zero.view(c, -1), 1)[0].view(c, 1, 1)
        radar_norm = radar_zero / (max_values + 1e-8)
        std, mean = torch.std_mean(radar_norm.view(c, -1), 1)
        return (radar_norm - mean.view(c, 1, 1)) / (std.view(c, 1, 1) + 1e-8)


def load_model(device: torch.device) -> HuPRNet:
    """Load pretrained HuPR model.

    Args:
        device: Torch device.

    Returns:
        HuPR model in eval mode.
    """
    cfg = ConfigObj()
    model = HuPRNet(cfg).to(device)

    checkpoint = torch.load(HUPR_CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def load_frame_data(
    seq_id: int,
    frame_indices: List[int],
    transform: transforms.Compose,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and preprocess VRDAEmap data for a batch of frames.

    Args:
        seq_id: Sequence ID.
        frame_indices: List of frame indices to load (length NUM_GROUP_FRAMES).
        transform: Transform to apply to each frame.

    Returns:
        Tuple of (VRDAEmap_hori, VRDAEmap_vert) tensors.
    """
    seq_dir = PROCESSED_DIR / f"single_{seq_id}"

    vrdae_hori = torch.zeros(NUM_GROUP_FRAMES, NUM_FRAMES, 2, RANGE_SIZE, AZIMUTH_SIZE, ELEVATION_SIZE)
    vrdae_vert = torch.zeros(NUM_GROUP_FRAMES, NUM_FRAMES, 2, RANGE_SIZE, AZIMUTH_SIZE, ELEVATION_SIZE)

    for j, frame_idx in enumerate(frame_indices):
        hori_path = seq_dir / "hori" / f"{frame_idx:09d}.npy"
        vert_path = seq_dir / "vert" / f"{frame_idx:09d}.npy"

        # Load complex data: (16, 64, 64, 8) = (Chirps, Range, Azimuth, Elevation)
        hori_data = np.load(hori_path)
        vert_data = np.load(vert_path)

        # Sample center chirps
        chirp_start = NUM_CHIRPS // 2 - NUM_FRAMES // 2
        chirp_end = NUM_CHIRPS // 2 + NUM_FRAMES // 2

        for k, chirp_idx in enumerate(range(chirp_start, chirp_end)):
            # Convert to real/imag and apply transform
            hori_real = transform(hori_data[chirp_idx].real).permute(1, 2, 0)
            hori_imag = transform(hori_data[chirp_idx].imag).permute(1, 2, 0)
            vert_real = transform(vert_data[chirp_idx].real).permute(1, 2, 0)
            vert_imag = transform(vert_data[chirp_idx].imag).permute(1, 2, 0)

            vrdae_hori[j, k, 0] = hori_real
            vrdae_hori[j, k, 1] = hori_imag
            vrdae_vert[j, k, 0] = vert_real
            vrdae_vert[j, k, 1] = vert_imag

    return vrdae_hori, vrdae_vert


def get_frame_indices(target_frame: int) -> List[int]:
    """Get temporal context frame indices for a target frame.

    Args:
        target_frame: Target frame index.

    Returns:
        List of NUM_GROUP_FRAMES frame indices centered on target.
    """
    indices = []
    pad_size = target_frame % DURATION

    idx = target_frame - NUM_GROUP_FRAMES // 2 - 1

    for j in range(NUM_GROUP_FRAMES):
        if (j + pad_size) <= NUM_GROUP_FRAMES // 2:
            idx = target_frame - pad_size
        elif j > (DURATION - 1 - pad_size) + NUM_GROUP_FRAMES // 2:
            idx = target_frame + (DURATION - 1 - pad_size)
        else:
            idx += 1
        indices.append(idx)

    return indices


def heatmap_to_keypoints(heatmaps: torch.Tensor) -> np.ndarray:
    """Convert heatmaps to 2D keypoint coordinates.

    Args:
        heatmaps: Heatmaps of shape (batch, num_keypoints, H, W).

    Returns:
        Keypoints of shape (batch, num_keypoints, 2) in pixel coordinates.
    """
    batch_size, num_kpts, h, w = heatmaps.shape

    # Flatten and find argmax
    flat = heatmaps.view(batch_size, num_kpts, -1)
    max_idx = flat.argmax(dim=-1)

    # Convert to x, y coordinates
    y = (max_idx // w).float()
    x = (max_idx % w).float()

    # Scale from heatmap to image size
    scale = IMG_SIZE / HEATMAP_SIZE
    keypoints = torch.stack([x * scale, y * scale], dim=-1)

    return keypoints.cpu().numpy()


def run_hupr_inference(device: Optional[torch.device] = None, batch_size: int = 32) -> Dict[int, np.ndarray]:
    """Run HuPR model on all test sequences with batching.

    Args:
        device: Torch device (defaults to CUDA if available).
        batch_size: Number of frames to process in parallel.

    Returns:
        Dictionary mapping seq_id to predictions of shape (600, 14, 2).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading model from {HUPR_CHECKPOINT_PATH}...")
    print(f"Batch size: {batch_size}")

    model = load_model(device)
    transform = transforms.Compose([transforms.ToTensor(), Normalize()])

    predictions: Dict[int, np.ndarray] = {}

    for seq_id in tqdm(TEST_SEQUENCES, desc="Processing sequences"):
        seq_dir = PROCESSED_DIR / f"single_{seq_id}"

        # Check if sequence data exists
        if not (seq_dir / "hori").exists():
            print(f"Warning: {seq_dir} not found, skipping")
            continue

        seq_predictions = np.zeros((DURATION, NUM_KEYPOINTS, 2), dtype=np.float32)

        # Process in batches
        for batch_start in tqdm(range(0, DURATION, batch_size), desc=f"Seq {seq_id}", leave=False):
            batch_end = min(batch_start + batch_size, DURATION)

            # Collect batch data
            batch_hori = []
            batch_vert = []

            for frame_idx in range(batch_start, batch_end):
                frame_indices = get_frame_indices(frame_idx)
                vrdae_hori, vrdae_vert = load_frame_data(seq_id, frame_indices, transform)
                batch_hori.append(vrdae_hori)
                batch_vert.append(vrdae_vert)

            # Stack into batch tensors
            batch_hori = torch.stack(batch_hori, dim=0).float().to(device)
            batch_vert = torch.stack(batch_vert, dim=0).float().to(device)

            # Run batched inference
            with torch.no_grad():
                _, gcn_heatmaps = model(batch_hori, batch_vert)
                # gcn_heatmaps shape: (batch, 1, 14, 64, 64)
                # Permute and reshape to (batch, 14, 64, 64) like original code
                gcn_heatmaps = gcn_heatmaps.permute(0, 2, 1, 3, 4).squeeze(2)

            # Convert heatmaps to keypoints
            keypoints = heatmap_to_keypoints(gcn_heatmaps)
            seq_predictions[batch_start:batch_end] = keypoints

        predictions[seq_id] = seq_predictions

    return predictions


def main() -> None:
    """Run HuPR inference on test set."""
    predictions = run_hupr_inference()

    # Save predictions
    output_path = Path(__file__).parent / "hupr_2d_predictions.npz"
    np.savez(output_path, **{str(k): v for k, v in predictions.items()})
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
