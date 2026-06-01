from pathlib import Path

# Base paths
PATH_ORIGIN = Path(__file__).parent
DATA_DIR = None  # Set this to your HuPR data directory, e.g. Path("/path/to/HuPR")

if DATA_DIR is None:
    raise ValueError("DATA_DIR must be set in config.py to your HuPR data directory.")
ANNOT_DIR = DATA_DIR / "annotations"
RADAR_FULL_DIR = DATA_DIR / "radar_full"

# MotionBERT configuration
MOTIONBERT_DIR = PATH_ORIGIN / "MotionBERT"

# Select model: "full" or "lite"
# - "full": MB_ft_h36m (512 dim, rootrel=True, outputs root-relative poses directly)
# - "lite": MB_ft_h36m_global_lite (256 dim, rootrel=False, outputs global poses)
MOTIONBERT_MODEL = "full"

MOTIONBERT_CONFIGS = {
    "full": {
        "config": MOTIONBERT_DIR / "configs/pose3d/MB_ft_h36m.yaml",
        "checkpoint": MOTIONBERT_DIR / "checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin",
        "rootrel": True,  # Model already outputs root-relative poses
    },
    "lite": {
        "config": MOTIONBERT_DIR / "configs/pose3d/MB_ft_h36m_global_lite.yaml",
        "checkpoint": MOTIONBERT_DIR / "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        "rootrel": False,  # Model outputs global poses, need to subtract hip
    },
}

# Active MotionBERT paths (based on selection)
MB_CONFIG_PATH = MOTIONBERT_CONFIGS[MOTIONBERT_MODEL]["config"]
MB_CHECKPOINT_PATH = MOTIONBERT_CONFIGS[MOTIONBERT_MODEL]["checkpoint"]
MB_ROOTREL = MOTIONBERT_CONFIGS[MOTIONBERT_MODEL]["rootrel"]

# HuPR model paths
HUPR_CHECKPOINT_PATH = PATH_ORIGIN / "logs" / "hupr_pretrained" / "model_best.pth"

# Preprocessing configuration
PREPROCESS_CONFIG = {
    "img_width": 256,
    "img_height": 256,
    "clip_len_motionbert": 243,
    "jump_threshold_px": 20,
}

# Training configuration for RadProPoserVAECov model
TRAIN_CONFIG = {
    # Optimizer
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "betas": (0.9, 0.999),
    "lr_decay": 0.99,

    # Training
    "epochs": 50,
    "batch_size": 16,
    "device": "cuda",

    # Loss weights
    "beta": 20,     # KL loss weight
    "gamma": 5,     # Variance/covariance penalty weight

    # Model
    "n_samples": 128,  # Number of ensemble samples for covariance estimation

    # Data
    "seq_len": 8,
    "stride": 1,
    "num_workers": 8,

    # Experiment
    "name": "radproposer_cov_motionbert",
    "use_wandb": True,
    "resume": True,
}

# Testing configuration
TEST_CONFIG = {
    "checkpoint_name": "radproposer_cov_motionbert",
    "device": "cuda",
    "batch_size": 16,
    "seq_len": 8,
    "num_workers": 8,
    "n_samples": 128,
}

# HuPR inference configuration
HUPR_INFERENCE_CONFIG = {
    "num_group_frames": 8,
    "num_frames": 8,
    "num_chirps": 16,
    "range_size": 64,
    "azimuth_size": 64,
    "elevation_size": 8,
    "heatmap_size": 64,
    "img_size": 256,
    "duration": 600,
    "num_keypoints": 14,
    "batch_size": 32,
}
