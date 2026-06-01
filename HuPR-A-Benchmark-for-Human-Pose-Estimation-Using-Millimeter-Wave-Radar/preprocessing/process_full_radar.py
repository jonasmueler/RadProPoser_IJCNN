import shutil
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

# Radar hardware constants
NUM_RX = 4
NUM_TX = 3
NUM_TX_PADDED = 4
NUM_ADC_SAMPLES = 256
NUM_CHIRPS_TOTAL = 192  # 64 chirps × 3 TX
NUM_CHIRPS_PER_TX = 64
NUM_LANES = 2
NUM_FRAMES = 600  # 10 FPS × 60 seconds

# Parallel processing config
NUM_WORKERS = 10  # 24 cores, 125GB RAM - use 10 workers (~12GB each)


def load_raw_adc(file_path: Path) -> np.ndarray:
    """Load raw ADC data from DCA1000 binary file.

    Args:
        file_path: Path to adc_data.bin

    Returns:
        Complex array of shape (NUM_RX, total_chirps, NUM_ADC_SAMPLES)
    """
    adc_data = np.fromfile(file_path, dtype=np.int16)
    file_size = adc_data.shape[0]

    adc_data = adc_data.reshape(-1, NUM_LANES * 2).T

    file_size = file_size // 2
    lvds = np.zeros((2, file_size), dtype=np.int16)

    temp = np.empty(adc_data[0].size + adc_data[1].size, dtype=adc_data[0].dtype)
    temp[0::2] = adc_data[0]
    temp[1::2] = adc_data[1]
    lvds[0] = temp

    temp = np.empty(adc_data[2].size + adc_data[3].size, dtype=adc_data[2].dtype)
    temp[0::2] = adc_data[2]
    temp[1::2] = adc_data[3]
    lvds[1] = temp

    adc_complex = np.zeros((NUM_RX, file_size // NUM_RX), dtype=np.complex64)

    for i in range(0, file_size, NUM_ADC_SAMPLES * 4):
        idx = i // (NUM_ADC_SAMPLES * 4) * NUM_ADC_SAMPLES
        for rx in range(NUM_RX):
            start = i + rx * NUM_ADC_SAMPLES
            end = start + NUM_ADC_SAMPLES
            adc_complex[rx, idx:idx + NUM_ADC_SAMPLES] = (
                lvds[0, start:end] + 1j * lvds[1, start:end]
            )

    return adc_complex.reshape(NUM_RX, -1, NUM_ADC_SAMPLES)


def extract_frame_tdm(adc_data: np.ndarray, frame_idx: int) -> np.ndarray:
    """Extract single frame and organize by TDM-MIMO structure.

    Args:
        adc_data: Full ADC array of shape (NUM_RX, total_chirps, NUM_ADC_SAMPLES)
        frame_idx: Frame index (0-599)

    Returns:
        Complex array of shape (NUM_CHIRPS_PER_TX, NUM_TX_PADDED, NUM_RX, NUM_ADC_SAMPLES)
    """
    start_chirp = frame_idx * NUM_CHIRPS_TOTAL
    end_chirp = start_chirp + NUM_CHIRPS_TOTAL
    frame_data = adc_data[:, start_chirp:end_chirp, :]

    output = np.zeros(
        (NUM_CHIRPS_PER_TX, NUM_TX_PADDED, NUM_RX, NUM_ADC_SAMPLES),
        dtype=np.complex64
    )

    for tx in range(NUM_TX):
        chirp_indices = np.arange(tx, NUM_CHIRPS_TOTAL, NUM_TX)
        tx_chirps = frame_data[:, chirp_indices, :]
        output[:, tx, :, :] = tx_chirps.transpose(1, 0, 2)

    return output


def clutter_removal(data: torch.Tensor) -> torch.Tensor:
    """Remove static clutter by subtracting mean across Doppler axis.

    Args:
        data: Complex tensor of shape (Doppler, TX, RX, Range)

    Returns:
        Clutter-removed tensor of same shape
    """
    return data - data.mean(dim=0, keepdim=True)


def apply_fft_4d(data: torch.Tensor) -> torch.Tensor:
    """Apply 4D FFT with forward normalization and fftshift across all dimensions.

    Args:
        data: Complex tensor of shape (Doppler, TX, RX, Range)

    Returns:
        FFT-processed and shifted tensor of same shape
    """
    x = torch.fft.fftshift(torch.fft.fft(data, dim=-1, norm="forward"), dim=-1)
    x = torch.fft.fftshift(torch.fft.fft(x, dim=-2, norm="forward"), dim=-2)
    x = torch.fft.fftshift(torch.fft.fft(x, dim=-3, norm="forward"), dim=-3)
    x = torch.fft.fftshift(torch.fft.fft(x, dim=-4, norm="forward"), dim=-4)
    return x


def process_frame(
    adc_hori: np.ndarray,
    adc_vert: np.ndarray,
    frame_idx: int
) -> np.ndarray:
    """Process single frame from both radars with clutter removal and 4D FFT.

    Args:
        adc_hori: Horizontal radar ADC data
        adc_vert: Vertical radar ADC data
        frame_idx: Frame index (0-599)

    Returns:
        Complex array of shape (2, NUM_CHIRPS_PER_TX, NUM_TX_PADDED, NUM_RX, NUM_ADC_SAMPLES)
        Axes: (Radar, Doppler, TX, RX, Range) = (2, 64, 4, 4, 256)
    """
    frame_hori = extract_frame_tdm(adc_hori, frame_idx)
    frame_vert = extract_frame_tdm(adc_vert, frame_idx)

    frame_hori_t = torch.from_numpy(frame_hori)
    frame_vert_t = torch.from_numpy(frame_vert)

    frame_hori_t = clutter_removal(frame_hori_t)
    frame_vert_t = clutter_removal(frame_vert_t)

    frame_hori_t = apply_fft_4d(frame_hori_t)
    frame_vert_t = apply_fft_4d(frame_vert_t)

    output = torch.stack([frame_hori_t, frame_vert_t], dim=0)

    return output.numpy()


def process_sequence_worker(args: Tuple[int, str]) -> Tuple[int, bool]:
    """Worker function for parallel processing.

    Args:
        args: Tuple of (seq_id, output_dir)

    Returns:
        Tuple of (seq_id, success)
    """
    seq_id, output_dir = args
    output_dir = Path(output_dir)

    seq_dir = DATA_DIR / f"single_{seq_id}"
    hori_path = seq_dir / "hori" / "adc_data.bin"
    vert_path = seq_dir / "vert" / "adc_data.bin"

    if not hori_path.exists() or not vert_path.exists():
        print(f"[{seq_id}] Skipping: missing ADC data")
        return (seq_id, False)

    out_seq_dir = output_dir / f"single_{seq_id}"

    # Skip if already processed
    if out_seq_dir.exists() and len(list(out_seq_dir.glob("*.npy"))) == NUM_FRAMES:
        print(f"[{seq_id}] Already processed, skipping")
        # Still delete raw data if it exists
        if seq_dir.exists():
            shutil.rmtree(seq_dir)
        return (seq_id, True)

    out_seq_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{seq_id}] Loading ADC data...")
    adc_hori = load_raw_adc(hori_path)
    adc_vert = load_raw_adc(vert_path)

    for frame_idx in range(NUM_FRAMES):
        frame_data = process_frame(adc_hori, adc_vert, frame_idx)
        out_path = out_seq_dir / f"{frame_idx:09d}.npy"
        np.save(out_path, frame_data)

    print(f"[{seq_id}] Completed {NUM_FRAMES} frames")

    # Delete raw data to save storage
    shutil.rmtree(seq_dir)
    print(f"[{seq_id}] Deleted raw data")

    return (seq_id, True)


def get_sequence_ids() -> List[int]:
    """Find all sequence IDs in the data directory.

    Returns:
        Sorted list of sequence IDs
    """
    seq_dirs = sorted(DATA_DIR.glob("single_*"))
    seq_ids = []

    for seq_dir in seq_dirs:
        try:
            seq_id = int(seq_dir.name.split("_")[1])
            seq_ids.append(seq_id)
        except (ValueError, IndexError):
            continue

    return seq_ids


def main() -> None:
    """Process all sequences with clutter removal and 4D FFT using parallel processing."""
    output_dir = DATA_DIR / "radar_full"
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Output format: (2, 64, 4, 4, 256) complex64 per frame")
    print(f"Processing: clutter removal -> 4D FFT (norm='forward') -> fftshift")
    print(f"Parallel workers: {NUM_WORKERS}")
    print()

    seq_ids = get_sequence_ids()
    print(f"Found {len(seq_ids)} sequences")

    # Prepare arguments for parallel processing
    args_list = [(seq_id, str(output_dir)) for seq_id in seq_ids]

    # Process in parallel
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(process_sequence_worker, args_list)

    success_count = sum(1 for _, success in results if success)

    print(f"\n{'=' * 50}")
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(seq_ids)} sequences")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
