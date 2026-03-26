import numpy as np
import os
import sys
from config import MODELNAME
from calibration_analysis import MultiModelRecalibrationVisualizer as ParametricVisualizer
from calibration_analysis_empirical import MultiModelRecalibrationVisualizer as EmpiricalVisualizer

MODELS = {
    "rppgaussiangaussian": {"type": "parametric", "laplace": False},
    "rppgaussiangaussiancov": {"type": "parametric", "laplace": False},
    "rpplaplacelaplace": {"type": "parametric", "laplace": True},
    "rpplaplacegaussian": {"type": "parametric", "laplace": False},
    "rppgaussianlaplace": {"type": "parametric", "laplace": True},
    "rppevidential": {"type": "empirical"},
    "rppnormalizingflow": {"type": "empirical"},
}

def load_data(model_suffix: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    calib_dir = os.path.join(base_dir, "calibration_analysis")
    
    mu_val = np.load(os.path.join(calib_dir, f"mu_validation_{model_suffix}.npy"))
    var_val = np.load(os.path.join(calib_dir, f"var_validation_{model_suffix}.npy"))
    gt_val = np.load(os.path.join(calib_dir, f"gt_validation_{model_suffix}.npy"))
    
    mu_test = np.load(os.path.join(calib_dir, f"mu_testing_{model_suffix}.npy"))
    var_test = np.load(os.path.join(calib_dir, f"var_testing_{model_suffix}.npy"))
    gt_test = np.load(os.path.join(calib_dir, f"gt_testing_{model_suffix}.npy"))
    
    ensemble_val = None
    ensemble_test = None
    
    val_path = os.path.join(calib_dir, f"ensemble_validation_{model_suffix}.npy")
    test_path = os.path.join(calib_dir, f"ensemble_testing_{model_suffix}.npy")
    
    if os.path.exists(val_path):
        ensemble_val = np.load(val_path)
    if os.path.exists(test_path):
        ensemble_test = np.load(test_path)
    
    return mu_val, var_val, gt_val, ensemble_val, mu_test, var_test, gt_test, ensemble_test

def main():
    if MODELNAME is None:
        print("ERROR: MODELNAME not set in config.py")
        return
    
    model_suffix = MODELNAME.lower()
    config = MODELS.get(model_suffix)
    
    if config is None:
        print(f"ERROR: Unknown model {MODELNAME}")
        return
    
    mu_val, var_val, gt_val, ensemble_val, mu_test, var_test, gt_test, ensemble_test = load_data(model_suffix)
    
    if config["type"] == "parametric":
        visualizer = ParametricVisualizer(num_dims=78)
        visualizer.quantify_improvement_with_holdout(
            mu_val, var_val, gt_val,
            mu_test, var_test, gt_test,
            laplace_=config["laplace"],
            model_name=model_suffix
        )
    else:
        if ensemble_val is None or ensemble_test is None:
            print(f"ERROR: Missing ensemble data for {MODELNAME}")
            return
        visualizer = EmpiricalVisualizer(num_dims=78)
        visualizer.quantify_improvement_with_holdout(
            mu_val, var_val, gt_val, ensemble_val,
            mu_test, var_test, gt_test, ensemble_test,
            model_name=model_suffix
        )

if __name__ == "__main__":
    main()

