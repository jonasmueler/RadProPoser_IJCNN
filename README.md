
# RadProPoser: Probabilistic Radar Tensor Human Pose Estimation That Knows Its Limits
 
<div align="center">
  <img src="assets/overview.png" alt="Project Overview" width="1000">
</div>

## Folder Structure
The folder structure of this repository is organized as follows:

```
RadProPoser/
├── data/
│   └── raw_radar_data/          # Our dataset (radar + skeletons)
├── models/                       # Neural network architectures
├── tools/                        # Training, testing, calibration scripts
├── trainedModels/                # Pretrained model weights
├── HuPR-.../                     # HuPR dataset experiments
│   ├── config.py                 # Central configuration
│   ├── train.py                  # RadProPoserVAECov training
│   ├── test.py                   # Testing with uncertainty metrics
│   ├── models/                   # Model architectures
│   ├── datasets/                 # Data loaders
│   ├── preprocessing/            # Radar processing, 3D GT generation
│   ├── scripts/                  # Inference and evaluation
│   ├── misc/                     # Losses, metrics, utilities
│   ├── recalibration/            # Uncertainty recalibration
│   └── MotionBERT/               # 2D-to-3D lifting model
└── requirements.txt
```

---

## Installation
A working installation of CUDA should be provided if GPU should be used. We used python 3.12 for the analysis.
To get started with the project, follow the steps below:

1. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

2. **Install torch with cuda_12.4**:
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **Install additional dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download data**:
   Our data can be downloaded anonymously here: https://doi.org/10.5281/zenodo.14035103. Place our dataset in the `/data` folder. Refer to `radar_raw_data/README.md` for dataset details.

---

## Radar-based Human Pose Estimation

### 1. Configuration
Before training or testing, configure the model in `tools/config.py`:
- Set `MODELNAME` to the model you want to train. default is the best performing model as described in the paper
- Set `PATHORIGIN` to your data root path
- Set `PATHRAW` to your raw radar data path

The dataloader currently includes all training participants, including the original validation participant, because the hyperparameters are already optimized and we aim for maximum performance on the held-out test set. If you want to tune your own hyperparameters, remove the validation participant from the training set to restore a proper validation split and track generalization each epoch.

### 2. Training
The training data (unfolded sequences) is automatically generated based on specified parameters and directories in `config.py`. Around 1.7 to 2 TB of free disk space should be available. The model is trained using Weights and Biases (wandb). You need to create a wandb account.

Create data and train the model:
```bash
python tools/trainScript.py
```

### 3. Testing
Run evaluation on the test set. Results are saved to `tools/calibration_analysis/`:
```bash
python tools/testing.py
```

### 4. Calibration Analysis and Recalibration
After testing, run recalibration analysis. This fits recalibration models on validation data (p1) and evaluates on test data (p2, p12). Results and plots are saved to `tools/calibration_plots/` and `tools/calibrated_models/`:
```bash
python tools/calibration_test.py
```

---

## HuPR Dataset Experiments

This section describes how to reproduce our experiments on the HuPR dataset. All scripts are in the `HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar/` directory.

### 1. Configuration
Configure paths in `HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar/config.py`:
- Set `DATA_DIR` to your HuPR data directory (download from [HuPR GitHub](https://github.com/robert80203/HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar))
- Set `HUPR_CHECKPOINT_PATH` to the HuPR pretrained weights (download from [HuPR releases](https://github.com/robert80203/HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar/releases))

### 2. Data Preparation
Process the raw radar data and generate 3D ground truth annotations:
```bash
cd HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar

# Process raw radar ADC data to 4D tensors
python preprocessing/process_full_radar.py

# Generate MotionBERT 3D pose annotations from 2D keypoints
python preprocessing/generate_3d_gt.py
```

### 3. Training
Train the RadProPoserVAECov model on HuPR radar data:
```bash
python train.py
```
Checkpoints are saved to `checkpoints/`. Training uses Weights and Biases (wandb) for logging.

### 4. Testing
Evaluate the trained model with MPJPE, PA-MPJPE, Sharpness, and ECE metrics:
```bash
python test.py
```

### 5. Recalibration
Run uncertainty recalibration on the trained model:
```bash
python scripts/recalibrate.py
```

### 6. HuPR 2D-to-3D Lifting Baseline
To evaluate the HuPR model's 2D predictions lifted to 3D (baseline comparison):
```bash
# Run HuPR 2D inference
python scripts/run_hupr_inference.py

# Lift 2D predictions to 3D using MotionBERT
python scripts/lift_to_3d.py

# Evaluate 3D pose accuracy
python scripts/evaluate_3d.py
```




