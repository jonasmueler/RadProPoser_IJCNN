import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import NUM_COORDS, NUM_JOINTS


class RadarData(Dataset):
    """Dataset for loading preprocessed radar sequences and skeleton targets."""

    def __init__(
        self,
        mode: str,
        root_path: str,
        train_data_path: str,
        seq_len: int,
        single_frame: bool = False,
    ) -> None:
        """Initialize the RadarData dataset.

        Args:
            mode: One of "train", "val", or "test".
            root_path: Root path where the radar and skeleton data are stored.
            train_data_path: Path to the specific training data files.
            seq_len: Length of the sequence to be loaded.
            single_frame: If True, load single frames instead of sequences.
        """
        self.seq_len = seq_len
        self.single_frame = single_frame
        self.mode = mode
        self.root_path = root_path
        self.train_path = train_data_path

        if single_frame:
            self.data_mode = "single_frame"
        else:
            self.data_mode = f"seq_{seq_len}"

        # augmentation
        self.augment = False
        self.augment_prob = 0.3
        self.mask_ratio = 0.2
        self.augmentations = [
            self.random_zero_mask,
            self.add_gaussian_noise,
            self.random_time_masking,
        ]

        self.participants, self.angles, self.actions, self.recording = (
            self._get_split_config()
        )

        trials = []
        for participant in self.participants:
            for angle in self.angles:
                for action in self.actions:
                    for recording in self.recording:
                        trials.append([participant, angle, action, recording])

        random.shuffle(trials)
        self.trials = trials

        data_dir = os.path.join(self.train_path, self.mode, self.data_mode)
        if not os.path.exists(data_dir):
            self.counter = self._generate_sequences(data_dir)
        else:
            self.counter = len(os.listdir(os.path.join(data_dir, "radar")))

        if self.mode in ("train", "val"):
            self.gt = self._load_directory_into_ram(
                os.path.join(self.train_path, self.mode, self.data_mode, "target")
            )

    def _get_split_config(self) -> Tuple[list, list, list, list]:
        """Return participant/angle/action/recording lists for the current split.

        Returns:
            Tuple of (participants, angles, actions, recordings).
        """
        angles = ["an0", "an1", "an2"]
        actions = ["ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9"]
        recordings = ["r0", "r1"]

        if self.mode == "train":
            participants = ["p3", "p4", "p6", "p7", "p8", "p9", "p10", "p11"]
        elif self.mode == "val":
            participants = ["p5"]
        elif self.mode == "test":
            participants = ["p1", "p2", "p12"]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return participants, angles, actions, recordings

    def _generate_sequences(self, data_dir: str) -> int:
        """Generate and save training sequences or single frames to disk.

        Args:
            data_dir: Directory to save generated data.

        Returns:
            Total number of generated samples.
        """
        counter = 0
        for i in tqdm(range(len(self.trials))):
            combination = self.trials[i]

            path = os.path.join(
                self.root_path, "radar",
                "data_cube_parsed_" + "_".join(combination) + ".npz",
            )
            if not os.path.exists(path):
                continue

            print("processing path ", path)
            data = np.load(path)
            radar = data[data.files[0]]
            target_path = os.path.join(
                self.root_path, "skeletons",
                "skeleton_" + "_".join(combination) + ".npy",
            )
            data_target = np.load(target_path)
            data_target = data_target.reshape(
                data_target.shape[0], NUM_JOINTS, NUM_COORDS
            )

            os.makedirs(os.path.join(data_dir, "radar"), exist_ok=True)
            os.makedirs(os.path.join(data_dir, "target"), exist_ok=True)

            if self.single_frame:
                for b in range(radar.shape[0]):
                    radar_tensor = torch.from_numpy(radar[b]).to(torch.complex64)
                    torch.save(
                        radar_tensor,
                        os.path.join(data_dir, "radar", str(counter) + ".pth"),
                    )

                    gt = torch.from_numpy(data_target[b])
                    torch.save(
                        gt, os.path.join(data_dir, "target", str(counter) + ".pth")
                    )

                    counter += 1
                    if counter % 200 == 0:
                        print("created ", counter, " frames")
            else:
                for b in range(radar.shape[0] - self.seq_len):
                    radar_seq = radar[b : b + self.seq_len]
                    radar_tensor = torch.from_numpy(radar_seq).to(torch.complex64)
                    torch.save(
                        radar_tensor,
                        os.path.join(data_dir, "radar", str(counter) + ".pth"),
                    )

                    gt = torch.from_numpy(data_target[b + self.seq_len])
                    torch.save(
                        gt, os.path.join(data_dir, "target", str(counter) + ".pth")
                    )

                    counter += 1
                    if counter % 200 == 0:
                        print("created ", counter, " sequences")

        print("data generated!")
        return counter

    def _load_directory_into_ram(self, directory_path: str) -> Dict[int, torch.Tensor]:
        """Load all .pth files in a directory into RAM in order.

        Args:
            directory_path: Path to the directory containing .pth files.

        Returns:
            Dictionary mapping file indices to loaded tensors.
        """
        file_names = sorted(
            [f for f in os.listdir(directory_path) if f.endswith(".pth")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )

        tensor_dict = {}
        for file_name in file_names:
            file_path = os.path.join(directory_path, file_name)
            file_index = int(os.path.splitext(file_name)[0])
            tensor_dict[file_index] = torch.load(file_path, weights_only=True)

        print(f"Loaded {len(tensor_dict)} files into RAM.")
        return tensor_dict

    def __len__(self) -> int:
        return self.counter

    def random_zero_mask(self, radar: torch.Tensor) -> torch.Tensor:
        """Randomly mask a sub-volume of the radar tensor with zeros.

        Args:
            radar: Input radar tensor of shape (D, A, E, R, T).

        Returns:
            Masked radar tensor with the same shape.
        """
        D, A, E, R, T = radar.shape
        d_mask = int(D * self.mask_ratio)
        a_mask = int(A * self.mask_ratio)
        e_mask = int(E * self.mask_ratio)
        r_mask = int(R * self.mask_ratio)

        d_start = random.randint(0, max(0, D - d_mask)) if d_mask > 0 else 0
        a_start = random.randint(0, max(0, A - a_mask)) if a_mask > 0 else 0
        e_start = random.randint(0, max(0, E - e_mask)) if e_mask > 0 else 0
        r_start = random.randint(0, max(0, R - r_mask)) if r_mask > 0 else 0

        mask = torch.ones_like(radar)
        mask[
            d_start : d_start + d_mask,
            a_start : a_start + a_mask,
            e_start : e_start + e_mask,
            r_start : r_start + r_mask,
            :,
        ] = 0

        return radar * mask

    def add_gaussian_noise(
        self, radar: torch.Tensor, noise_std: float = 0.05
    ) -> torch.Tensor:
        """Add Gaussian noise to the radar tensor.

        Args:
            radar: Input radar tensor.
            noise_std: Standard deviation of the noise.

        Returns:
            Noisy radar tensor.
        """
        noise = torch.randn_like(radar) * noise_std
        return radar + noise

    def random_time_masking(
        self, radar: torch.Tensor, mask_prob: float = 0.2
    ) -> torch.Tensor:
        """Randomly mask time steps in the radar tensor.

        Args:
            radar: Input radar tensor.
            mask_prob: Probability of masking each time step.

        Returns:
            Time-masked radar tensor.
        """
        mask = (torch.rand(radar.shape[-1]) > mask_prob).float()
        return radar * mask.view(1, 1, 1, 1, -1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample by index.

        Args:
            idx: Index of the data sample.

        Returns:
            Tuple of (radar input tensor, ground truth target tensor).
        """
        radar = torch.load(
            os.path.join(
                self.root_path, self.mode, self.data_mode,
                "radar", str(idx) + ".pth",
            ),
            weights_only=True,
        )

        if self.augment and random.random() < self.augment_prob and self.mode == "train":
            augmentation = random.choice(self.augmentations)
            radar = augmentation(radar)

        gt = self.gt[idx]
        gt = gt.flatten()
        return radar, gt
