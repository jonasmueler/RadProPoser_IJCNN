import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from config import DATA_DIR, RADAR_FULL_DIR, ANNOT_DIR, TRAINCONFIG

CONFIG_PATH = Path(__file__).parent.parent / "config" / "mscsa_prgcn.yaml"

FRAMES_PER_SEQUENCE = 600


def load_config() -> Dict:
    """Load dataset configuration from YAML file.

    Returns:
        Configuration dictionary
    """
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_phase_sequence_ids(phase: str) -> List[int]:
    """Get sequence IDs for the specified phase from config.

    Args:
        phase: One of 'train', 'val', 'test'

    Returns:
        List of sequence IDs for this phase
    """
    cfg = load_config()
    phase_key = f"{phase}Name"
    return cfg["DATASET"].get(phase_key, [])


class HuPRFullRadar(Dataset):
    """Dataset for full radar data with temporal sequences and 3D pose targets."""

    def __init__(self, phase: str, seq_len: int = 16, stride: int = 1) -> None:
        """Initialize dataset.

        Args:
            phase: One of 'train', 'val', 'test'
            seq_len: Number of temporal frames in each sample
            stride: Stride between consecutive samples within a sequence
        """
        if phase not in ("train", "val", "test"):
            raise ValueError(f"Invalid phase: {phase}")

        self.phase = phase
        self.seq_len = seq_len
        self.stride = stride

        self.available_seq_ids = self._get_available_sequences()
        self.annot_3d = self._load_3d_annotations()
        self.samples = self._build_sample_index()

    def _get_available_sequences(self) -> List[int]:
        """Get sequence IDs that have both radar data and are in this phase."""
        radar_seqs = {
            int(p.name.split("_")[1]) for p in RADAR_FULL_DIR.glob("single_*")
        }
        phase_seqs = set(get_phase_sequence_ids(self.phase))
        return sorted(radar_seqs & phase_seqs)

    def _load_3d_annotations(self) -> Dict[int, List[np.ndarray]]:
        """Load 3D pose annotations indexed by sequence ID.

        Returns:
            Dictionary mapping seq_id to list of (17, 3) pose arrays
        """
        gt_source = TRAINCONFIG.get("gt_source", "motionbert")
        annot_file = ANNOT_DIR / f"{gt_source}_3d_{self.phase}.json"

        with open(annot_file) as f:
            annot_list = json.load(f)

        phase_seq_ids = get_phase_sequence_ids(self.phase)
        annot_3d = {}

        for seq_idx, seq in enumerate(annot_list):
            if seq_idx < len(phase_seq_ids):
                seq_id = phase_seq_ids[seq_idx]
                if seq_id in self.available_seq_ids:
                    poses = [np.array(frame["joints_3d"], dtype=np.float32) for frame in seq]
                    annot_3d[seq_id] = poses

        return annot_3d

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """Build index of (seq_id, start_frame) for all valid samples.

        Returns:
            List of (seq_id, start_frame) tuples
        """
        samples = []

        for seq_id in self.available_seq_ids:
            if seq_id not in self.annot_3d:
                continue

            num_frames = len(self.annot_3d[seq_id])
            max_start = num_frames - self.seq_len

            for start_frame in range(0, max_start + 1, self.stride):
                samples.append((seq_id, start_frame))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - radar: Complex tensor (seq_len, 2, 64, 4, 4, 256) = (T, Radar, Doppler, TX, RX, Range)
                - pose_3d: Float tensor (17, 3) - 3D pose of the last frame
        """
        seq_id, start_frame = self.samples[idx]
        seq_dir = RADAR_FULL_DIR / f"single_{seq_id}"

        radar_frames = []
        for t in range(self.seq_len):
            frame_path = seq_dir / f"{start_frame + t:09d}.npy"
            radar_frames.append(np.load(frame_path))

        radar = torch.from_numpy(np.stack(radar_frames, axis=0))

        target_frame = start_frame + self.seq_len - 1
        pose_3d = torch.from_numpy(self.annot_3d[seq_id][target_frame])

        return {"radar": radar, "pose_3d": pose_3d}


def get_dataloader(
    phase: str,
    batch_size: int,
    seq_len: int = 16,
    stride: int = 1,
    num_workers: int = 4,
    shuffle: bool = None,
) -> DataLoader:
    """Create dataloader for full radar dataset.

    Args:
        phase: One of 'train', 'val', 'test'
        batch_size: Batch size
        seq_len: Number of temporal frames per sample
        stride: Stride between samples within sequences
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (defaults to True for train)

    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = phase == "train"

    dataset = HuPRFullRadar(phase=phase, seq_len=seq_len, stride=stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
