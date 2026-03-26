"""Preprocess raw ADC radar data to VRDAEmaps for HuPR model using GPU-accelerated FFT."""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

RAW_DIR = Path("/home/XXXX-1/data/HuPR/test_raw")
OUTPUT_DIR = Path("/home/XXXX-1/data/HuPR/test_processed")

# Radar hardware parameters
NUM_RX = 4
NUM_LANES = 2
NUM_ADC_SAMPLES = 256
NUM_CHIRPS_TOTAL = 192  # 64 chirps × 3 TX
NUM_CHIRPS_PER_TX = 64
NUM_FRAMES = 600  # 10 FPS × 60 seconds
NUM_ANGLE_BINS = 64  # After padding
NUM_ELE_BINS = 8
ADC_RATIO = 4
NUM_GROUP_CHIRPS = 4

TEST_SEQUENCES = [
    15, 16, 17, 38, 39, 40, 41, 42,
    244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256,
]

# Device for GPU acceleration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_adc_data(filepath: Path) -> np.ndarray:
    """Load raw ADC data from DCA1000 binary file.

    Args:
        filepath: Path to adc_data.bin file.

    Returns:
        Complex array of shape (NUM_RX, num_chirps, NUM_ADC_SAMPLES).
    """
    adc_data = np.fromfile(filepath, dtype=np.int16)
    file_size = adc_data.shape[0]

    # Deinterleave LVDS lanes
    adc_data = adc_data.reshape(-1, NUM_LANES * 2).T
    file_size = file_size // 2

    lvds = np.zeros((2, file_size), dtype=np.float32)

    temp = np.empty(adc_data[0].size + adc_data[1].size, dtype=np.float32)
    temp[0::2] = adc_data[0]
    temp[1::2] = adc_data[1]
    lvds[0] = temp

    temp = np.empty(adc_data[2].size + adc_data[3].size, dtype=np.float32)
    temp[0::2] = adc_data[2]
    temp[1::2] = adc_data[3]
    lvds[1] = temp

    # Convert to complex - vectorized
    num_samples_per_rx = file_size // NUM_RX
    complex_data = np.zeros((NUM_RX, num_samples_per_rx), dtype=np.complex64)

    for rx in range(NUM_RX):
        indices = np.arange(rx * NUM_ADC_SAMPLES, file_size, NUM_ADC_SAMPLES * 4)
        for i, idx in enumerate(indices):
            if idx + NUM_ADC_SAMPLES <= file_size:
                start = i * NUM_ADC_SAMPLES
                complex_data[rx, start:start + NUM_ADC_SAMPLES] = (
                    lvds[0, idx:idx + NUM_ADC_SAMPLES] +
                    1j * lvds[1, idx:idx + NUM_ADC_SAMPLES]
                )

    return complex_data.reshape(NUM_RX, -1, NUM_ADC_SAMPLES)


def process_all_frames_gpu(adc_data: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Process all frames using GPU-accelerated FFT.

    Args:
        adc_data: Raw ADC data of shape (NUM_RX, total_chirps, NUM_ADC_SAMPLES).
        batch_size: Number of frames to process in parallel on GPU.

    Returns:
        VRDAEmaps of shape (NUM_FRAMES, 16, 64, 64, 8) matching original HuPR format.
    """
    num_range_bins = NUM_ADC_SAMPLES // ADC_RATIO
    all_heatmaps = np.zeros(
        (NUM_FRAMES, 16, num_range_bins, NUM_ANGLE_BINS, NUM_ELE_BINS),
        dtype=np.complex64,
    )

    # Process in batches
    for batch_start in range(0, NUM_FRAMES, batch_size):
        batch_end = min(batch_start + batch_size, NUM_FRAMES)
        batch_frames = []

        for frame_idx in range(batch_start, batch_end):
            frame = adc_data[
                :,
                NUM_CHIRPS_TOTAL * frame_idx : NUM_CHIRPS_TOTAL * (frame_idx + 1),
                :NUM_ADC_SAMPLES,
            ]
            batch_frames.append(frame)

        # Stack batch and process on GPU
        batch_np = np.stack(batch_frames, axis=0)
        batch_results = generate_heatmap_batch_gpu(batch_np)

        all_heatmaps[batch_start:batch_end] = batch_results

    return all_heatmaps


def generate_heatmap_batch_gpu(frames: np.ndarray) -> np.ndarray:
    """Generate VRDAEmaps for a batch of frames using GPU.

    Args:
        frames: Batch of frames of shape (batch, NUM_RX, NUM_CHIRPS_TOTAL, NUM_ADC_SAMPLES).

    Returns:
        VRDAEmaps of shape (batch, 16, 64, 64, 8).
    """
    batch_size = frames.shape[0]

    # TDM-MIMO demultiplexing
    tx0_indices = np.arange(0, NUM_CHIRPS_TOTAL, 3)
    tx1_indices = np.arange(1, NUM_CHIRPS_TOTAL, 3)
    tx2_indices = np.arange(2, NUM_CHIRPS_TOTAL, 3)

    # Horizontal: TX0 (RX 0-3) and TX2 (RX 4-7) → 8 virtual antennas
    data_hori = np.zeros((batch_size, NUM_RX * 2, NUM_CHIRPS_PER_TX, NUM_ADC_SAMPLES), dtype=np.complex64)
    data_hori[:, :NUM_RX] = frames[:, :, tx0_indices]
    data_hori[:, NUM_RX:] = frames[:, :, tx2_indices]

    # Vertical: TX1 → 4 virtual antennas
    data_vert = frames[:, :, tx1_indices].copy()

    # Move to GPU
    data_hori_t = torch.from_numpy(data_hori).to(DEVICE)
    data_vert_t = torch.from_numpy(data_vert).to(DEVICE)

    # Clutter removal (matching original: transpose, mean on chirp axis, transpose back)
    # Original shape: (rx, chirp, adc) -> transpose to (chirp, rx, adc) -> mean on axis 0 -> transpose back
    # Our shape: (batch, rx, chirp, adc) -> transpose to (batch, chirp, rx, adc) -> mean on dim 1
    data_hori_t = data_hori_t.permute(0, 2, 1, 3)
    data_hori_t = data_hori_t - data_hori_t.mean(dim=1, keepdim=True)
    data_hori_t = data_hori_t.permute(0, 2, 1, 3)

    data_vert_t = data_vert_t.permute(0, 2, 1, 3)
    data_vert_t = data_vert_t - data_vert_t.mean(dim=1, keepdim=True)
    data_vert_t = data_vert_t.permute(0, 2, 1, 3)

    # Range-Doppler FFT
    data_hori_t = torch.fft.fft2(data_hori_t, dim=(2, 3))
    data_vert_t = torch.fft.fft2(data_vert_t, dim=(2, 3))

    # Angle FFT with padding
    data_hori_padded = torch.zeros(
        (batch_size, NUM_ANGLE_BINS, NUM_CHIRPS_PER_TX, NUM_ADC_SAMPLES),
        dtype=torch.complex64, device=DEVICE
    )
    data_hori_padded[:, :NUM_RX * 2] = data_hori_t

    data_vert_padded = torch.zeros(
        (batch_size, NUM_ANGLE_BINS, NUM_CHIRPS_PER_TX, NUM_ADC_SAMPLES),
        dtype=torch.complex64, device=DEVICE
    )
    data_vert_padded[:, 2:2 + NUM_RX] = data_vert_t

    # Stack and pad elevation
    data_merge = torch.zeros(
        (batch_size, NUM_ELE_BINS, NUM_ANGLE_BINS, NUM_CHIRPS_PER_TX, NUM_ADC_SAMPLES),
        dtype=torch.complex64, device=DEVICE
    )
    data_merge[:, 0] = data_hori_padded
    data_merge[:, 1] = data_vert_padded

    # Elevation FFT on angle bins 2,3,4,5 (FFT applied along elevation dimension=1)
    # Original does this per chirp and ADC, but vectorized is equivalent
    for angle_idx in [2, 3, 4, 5]:
        data_merge[:, :, angle_idx, :, :] = torch.fft.fft(data_merge[:, :, angle_idx, :, :], dim=1)

    # Angle FFT for all elevation bins (FFT applied along angle dimension=2)
    data_merge = torch.fft.fft(data_merge, dim=2)

    # Select ADC samples (range region of interest) - indices 94 down to 31
    adc_indices = list(range(94, 30, -1))[:NUM_ADC_SAMPLES // ADC_RATIO]

    # Reorder and select
    data_temp = data_merge[:, :, :, :, adc_indices]
    # (batch, ele, angle, chirps, range) -> (batch, chirps, range, angle, ele)
    data_temp = data_temp.permute(0, 3, 4, 2, 1)

    # Velocity shift
    data_temp = torch.fft.fftshift(data_temp, dim=1)

    # Select velocity region of interest
    chirp_pad = NUM_CHIRPS_PER_TX // NUM_GROUP_CHIRPS  # 16
    start_chirp = NUM_CHIRPS_PER_TX // 2 - chirp_pad // 2  # 24
    end_chirp = start_chirp + chirp_pad  # 40

    data_fft_group = data_temp[:, start_chirp:end_chirp]

    # Post-process (matching original postProcessFFT3D):
    # Original takes (range, angle, ele), transposes to (angle, ele, range),
    # applies fftshift on (0,1), transposes to (range, angle, ele), flips (1,2)
    # Current shape: (batch, 16, range, angle, ele)
    # Step 1: Transpose to (batch, 16, angle, ele, range)
    data_post = data_fft_group.permute(0, 1, 3, 4, 2)
    # Step 2: fftshift on (angle, ele) = dims (2, 3)
    data_post = torch.fft.fftshift(data_post, dim=(2, 3))
    # Step 3: Transpose to (batch, 16, range, angle, ele)
    data_post = data_post.permute(0, 1, 4, 2, 3)
    # Step 4: Flip on (angle, ele) = dims (3, 4)
    data_post = torch.flip(data_post, dims=(3, 4))

    return data_post.cpu().numpy()


def process_sequence_radar(seq_id: int, radar: str) -> bool:
    """Process one radar (hori or vert) for a sequence using GPU.

    Args:
        seq_id: Sequence ID.
        radar: Radar type ('hori' or 'vert').

    Returns:
        True if successful, False otherwise.
    """
    raw_dir = RAW_DIR / f"single_{seq_id}"
    out_dir = OUTPUT_DIR / f"single_{seq_id}" / radar

    adc_path = raw_dir / radar / "adc_data.bin"
    if not adc_path.exists():
        return False

    # Force reprocess by removing existing files
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw ADC data
    adc_data = load_adc_data(adc_path)

    # Process all frames using GPU
    all_heatmaps = process_all_frames_gpu(adc_data)

    # Save all frames
    for frame_idx in range(NUM_FRAMES):
        output_path = out_dir / f"{frame_idx:09d}.npy"
        np.save(output_path, all_heatmaps[frame_idx])

    return True


def preprocess_all_test_sequences() -> None:
    """Preprocess all test sequences using GPU acceleration."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")

    # Process sequences sequentially (GPU handles parallelism)
    total_tasks = len(TEST_SEQUENCES) * 2
    pbar = tqdm(total=total_tasks, desc="Preprocessing")

    for seq_id in TEST_SEQUENCES:
        for radar in ["hori", "vert"]:
            success = process_sequence_radar(seq_id, radar)
            status = "✓" if success else "✗"
            pbar.set_postfix_str(f"seq_{seq_id}/{radar} {status}")
            pbar.update(1)

    pbar.close()


def main() -> None:
    """Preprocess test set radar data."""
    preprocess_all_test_sequences()


if __name__ == "__main__":
    main()
