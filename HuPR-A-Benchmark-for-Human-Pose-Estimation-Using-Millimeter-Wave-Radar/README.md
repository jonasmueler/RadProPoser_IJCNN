# HuPR: Human Pose Estimation Using Millimeter Wave Radar

This repository contains the implementation for 3D human pose estimation using mmWave radar, building on the [HuPR benchmark](https://arxiv.org/abs/2210.12564).

## Citation

If you use this code or dataset, please cite:

```bibtex
@InProceedings{Lee_2023_WACV,
    author    = {Lee, Shih-Po and Kini, Niraj Prakash and Peng, Wen-Hsiao and Ma, Ching-Wen and Hwang, Jenq-Neng},
    title     = {HuPR: A Benchmark for Human Pose Estimation Using Millimeter Wave Radar},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {5715-5724}
}
```

## Setup

### Environment

```bash
conda env create -f environment.yml
conda activate hupr
```

### Dataset

Download the HuPR dataset from [HuggingFace](https://huggingface.co/datasets/nirajpkini/HuPR).

Request access with your institutional email and provide:
- Full Name, Institution, Advisor Name
- Current Position, Email Address
- Intended Purpose

### MotionBERT

Download the pretrained MotionBERT checkpoint:

```bash
cd MotionBERT
mkdir -p checkpoint/pose3d/FT_MB_release_MB_ft_h36m
# Download best_epoch.bin from MotionBERT releases
```

## Directory Structure

```
HuPR/
├── config.py                 # Central configuration
├── train.py                  # RadProPoserVAECov training
├── test.py                   # Testing with uncertainty metrics
├── main.py                   # Original HuPR 2D training
├── models/                   # Neural network architectures
│   ├── vae_lstm_ho.py       # RadProPoserVAECov model
│   └── networks.py          # HuPRNet for 2D pose
├── datasets/
│   ├── dataset_full.py      # Full radar dataset (3D training)
│   └── dataset.py           # VRDAEmap dataset (2D training)
├── preprocessing/            # Data preparation scripts
│   ├── process_full_radar.py # GPU-optimized 4D FFT processing
│   ├── generate_3d_gt.py    # Generate 3D GT using MotionBERT
│   └── preprocess_test_radar.py
├── scripts/                  # Inference and evaluation
│   ├── run_hupr_inference.py # HuPR 2D inference
│   ├── lift_to_3d.py        # 2D→3D lifting with MotionBERT
│   ├── evaluate_3d.py       # 3D pose evaluation
│   └── recalibrate.py       # Uncertainty recalibration
├── misc/
│   ├── constants.py         # Shared constants (joints, skeleton)
│   ├── metrics.py           # MPJPE, PA-MPJPE, ECE, sharpness
│   └── losses.py            # KL loss, NLL with covariance
├── recalibration/
│   └── recalibrator.py      # Isotonic regression recalibration
└── MotionBERT/              # 2D-to-3D lifting model
```

## Workflow

### 1. Preprocess Radar Data

Process raw IWR1843 radar data into VRDAEmap format:

```bash
cd preprocessing
python process_iwr1843.py
```

Or use GPU-accelerated 4D FFT processing:

```bash
python process_full_radar.py
```

### 2. Generate 3D Ground Truth

Generate 3D pose annotations from 2D HRNet detections using MotionBERT:

```bash
python preprocessing/generate_3d_gt.py
```

This creates `motionbert_3d_{train,val,test}.json` in the annotations directory.

### 3. Train HuPR 2D Model

Train the 2D pose estimation model:

```bash
python main.py --config mscsa_prgcn.yaml --dir hupr_pretrained
```

### 4. Train RadProPoserVAECov

Train the 3D pose model with uncertainty estimation:

```bash
python train.py
```

Configuration is in `config.py` under `TRAIN_CONFIG`.

### 5. Evaluate

Test the model with uncertainty metrics:

```bash
python test.py
```

Outputs:
- MPJPE and PA-MPJPE per joint (cm)
- Sharpness (predictive variance)
- ECE (Expected Calibration Error)

### 6. Run Recalibration

Improve uncertainty calibration using isotonic regression:

```bash
python scripts/recalibrate.py
```

## Configuration

All settings are centralized in `config.py`:

```python
TRAIN_CONFIG = {
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 16,
    "beta": 20,      # KL loss weight
    "gamma": 5,      # Variance penalty
    "n_samples": 128, # Ensemble samples
    ...
}
```

## Models

### HuPRNet
2D pose estimation from radar using MSCSA (Multi-Scale Cross-modal Self-Attention) and PR-GCN (Pose Refinement Graph Convolutional Network).

### RadProPoserVAECov
3D pose estimation with full covariance uncertainty using:
- VAE encoder for radar features
- LSTM for temporal modeling
- Higher-order ensemble for covariance estimation

## License

MIT License. See LICENSE file.
