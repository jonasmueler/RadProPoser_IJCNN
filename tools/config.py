import os

# Skeleton dimensions
NUM_JOINTS = 26
NUM_COORDS = 3
NUM_DIMS = NUM_JOINTS * NUM_COORDS  # 78

# Globals
SEQLEN = 8
MODELNAME = "RPPgaussianGaussian"

# Shared training defaults
_BASE_CONFIG = {
    "weightDecay": 0.0001,
    "optimizer": "adam",
    "device": "cuda",
    "betas": (0.9, 0.999),
    "lrDecay": 0.99,
    "nll": False,
    "evd": False,
    "nf": False,
    "testing_epoch": None,
}

# Per-model overrides (only keys that differ from _BASE_CONFIG)
_MODEL_CONFIGS = {
    "RPPgaussianGaussian": {
        "learningRate": 0.001, "epochs": 24, "batchSize": 16,
        "beta": 20, "gamma": 5, "nll": True,
    },
    "RPPgaussianGaussianCov": {
        "learningRate": 0.001, "epochs": 24, "batchSize": 16,
        "beta": 20, "gamma": 5, "nll": True,
    },
    "RPPlaplaceLaplace": {
        "learningRate": 0.001, "epochs": 24, "batchSize": 16,
        "beta": 1, "gamma": 1, "nll": True,
    },
    "RPPlaplaceGaussian": {
        "learningRate": 0.001, "epochs": 24, "batchSize": 16,
        "beta": 1, "gamma": 5, "nll": True,
    },
    "RPPgaussianLaplace": {
        "learningRate": 0.001, "epochs": 24, "batchSize": 16,
        "beta": 20, "gamma": 5, "nll": True,
    },
    "RPPevidential": {
        "learningRate": 0.0001, "epochs": 24, "batchSize": 16,
        "lambda": 0.001, "evd": True,
    },
    "RPPnormalizingFlow": {
        "learningRate": 0.0001, "epochs": 24, "batchSize": 32,
        "gamma": 1, "beta": 1, "nf": True,
    },
    "HoEtAlBaseline": {
        "learningRate": 0.0001, "epochs": 40, "batchSize": 16,
        "HR_SINGLE": True,
    },
}

if MODELNAME not in _MODEL_CONFIGS:
    raise ValueError(
        f"Invalid MODELNAME: {MODELNAME}. "
        f"Valid options: {list(_MODEL_CONFIGS.keys())}"
    )
TRAINCONFIG = {**_BASE_CONFIG, **_MODEL_CONFIGS[MODELNAME]}

# Model registry: maps MODELNAME to (module_name, class_name)
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
PATHORIGIN = None  # Set this to your root path
PATHRAW = None  # Set this to your raw radar data path

if PATHORIGIN is None:
    raise ValueError("PATHORIGIN must be set in config.py")
if PATHRAW is None:
    raise ValueError("PATHRAW must be set in config.py")

MODELPATH = os.path.join(PATHORIGIN, "models")
HPECKPT = os.path.join(PATHORIGIN, "trainedModels", "humanPoseEstimation")
