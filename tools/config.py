import os

# Skeleton dimensions
NUM_JOINTS = 26
NUM_COORDS = 3
NUM_DIMS = NUM_JOINTS * NUM_COORDS  # 78

# Globals
SEQ_LEN = 8
MODEL_NAME = "RPPgaussianGaussian"

# Shared training defaults
_BASE_CONFIG = {
    "weight_decay": 0.0001,
    "optimizer": "adam",
    "device": "cuda",
    "betas": (0.9, 0.999),
    "lr_decay": 0.99,
    "nll": False,
    "evd": False,
    "nf": False,
    "testing_epoch": None,
}

# Per-model overrides (only keys that differ from _BASE_CONFIG)
_MODEL_CONFIGS = {
    "RPPgaussianGaussian": {
        "learning_rate": 0.001, "epochs": 24, "batch_size": 16,
        "beta": 20, "gamma": 5, "nll": True,
    },
    "RPPgaussianGaussianCov": {
        "learning_rate": 0.001, "epochs": 24, "batch_size": 16,
        "beta": 20, "gamma": 5, "nll": True,
    },
    "RPPlaplaceLaplace": {
        "learning_rate": 0.001, "epochs": 24, "batch_size": 16,
        "beta": 1, "gamma": 1, "nll": True,
    },
    "RPPlaplaceGaussian": {
        "learning_rate": 0.001, "epochs": 24, "batch_size": 16,
        "beta": 1, "gamma": 5, "nll": True,
    },
    "RPPgaussianLaplace": {
        "learning_rate": 0.001, "epochs": 24, "batch_size": 16,
        "beta": 20, "gamma": 5, "nll": True,
    },
    "RPPevidential": {
        "learning_rate": 0.0001, "epochs": 24, "batch_size": 16,
        "lambda": 0.001, "evd": True,
    },
    "RPPnormalizingFlow": {
        "learning_rate": 0.0001, "epochs": 24, "batch_size": 32,
        "gamma": 1, "beta": 1, "nf": True,
    },
    "HoEtAlBaseline": {
        "learning_rate": 0.0001, "epochs": 40, "batch_size": 16,
        "HR_SINGLE": True,
    },
}

if MODEL_NAME not in _MODEL_CONFIGS:
    raise ValueError(
        f"Invalid MODEL_NAME: {MODEL_NAME}. "
        f"Valid options: {list(_MODEL_CONFIGS.keys())}"
    )
TRAIN_CONFIG = {**_BASE_CONFIG, **_MODEL_CONFIGS[MODEL_NAME]}

# Model registry: maps MODEL_NAME to (module_name, class_name)
MODEL_REGISTRY = {
    "RPPgaussianGaussian": ("radproposer_vae", "RadProPoserVAE"),
    "RPPgaussianGaussianCov": ("radproposer_vae_cov", "RadProPoserVAECov"),
    "RPPlaplaceLaplace": ("radproposer_vae", "RadProPoserVAE"),
    "RPPlaplaceGaussian": ("radproposer_vae", "RadProPoserVAE"),
    "RPPgaussianLaplace": ("radproposer_vae", "RadProPoserVAE"),
    "RPPevidential": ("evidential_pose_regression", "RadProPoserEvidential"),
    "RPPnormalizingFlow": ("normalizing_flow", "RadProPoserVAENF"),
    "HoEtAlBaseline": ("ho_single_frame", "HRRadarPose"),
}

# Paths
PATH_ORIGIN = None  # Set this to your root path
PATH_RAW = None  # Set this to your raw radar data path

if PATH_ORIGIN is None:
    raise ValueError("PATH_ORIGIN must be set in config.py")
if PATH_RAW is None:
    raise ValueError("PATH_RAW must be set in config.py")

MODEL_PATH = os.path.join(PATH_ORIGIN, "models")
HP_ECKPT = os.path.join(PATH_ORIGIN, "trainedModels", "humanPoseEstimation")
