import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Callable
from functools import partial

from scipy.stats import t, norm
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.isotonic import IsotonicRegression
from typing import List
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
import os
import joblib
# Disable scientific notation
np.set_printoptions(suppress=True)
# Optional: Only include these if you actually use them
# from sklearn.linear_model import Ridge, Lasso, ElasticNet

class MultiModelRecalibrationVisualizer:
    def __init__(self, num_dims: int):
        self.num_dims = num_dims
        self.models = {
            "isotonic": [IsotonicRegression(out_of_bounds="clip") for _ in range(num_dims)],
        }

    def save_models(self, directory: str, model_name: str):
        """
        Serialize each model-type's list of per-dimension regressors
        to disk under `directory/{model_name}/`, one .pkl per model-type.
        
        Args:
            directory: Base directory name (e.g., "recalibration_models")
            model_name: Name of the pose estimation model (e.g., "rppgaussiangaussiancov")
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, directory, model_name)
        os.makedirs(full_path, exist_ok=True)
        for model_type, mdl_list in self.models.items():
            path = os.path.join(full_path, f"{model_type}.pkl")
            joblib.dump(mdl_list, path)
        print(f"Saved {len(self.models)} model types to '{directory}/{model_name}'")

    def load_models(self, directory: str, model_name: str):
        """
        Load each model-type's .pkl file from `directory/{model_name}/` back into self.models.
        Expects files like `isotonic.pkl`, `rf100.pkl`, etc.
        
        Args:
            directory: Base directory name (e.g., "recalibration_models")
            model_name: Name of the pose estimation model (e.g., "rppgaussiangaussiancov")
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, directory, model_name)
        for model_type in list(self.models.keys()):
            path = os.path.join(full_path, f"{model_type}.pkl")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Could not find calibration file '{path}'")
            self.models[model_type] = joblib.load(path)
        print(f"Loaded {len(self.models)} model types from '{full_path}'")

    def _get_cdf_at_y(self, samples: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Empirical CDF at each ground-truth value y, given samples of shape (B, D, S).

        Args:
        samples: np.ndarray, shape (B, D, S)
                samples[i, d, :] are the S MC draws for batch‐item i, dimension d
        gt:      np.ndarray, shape (B, D)
                gt[i, d] is the true value for batch‐item i, dimension d

        Returns:
        cdf_y:   np.ndarray, shape (B, D)
                cdf_y[i, d] = fraction of samples[i, d, :] <= gt[i, d]
        """
        # Broadcast gt from (B, D) to (B, D, 1), compare to each sample draw:
        mask = samples <= gt[:, :, None]    # shape (B, D, S)

        # Mean over the sample‐axis gives empirical CDF at y:
        cdf_y = mask.mean(axis=2)           # shape (B, D)

        return cdf_y

    def _fit_models(self, 
                    gt: np.ndarray, 
                    cdf: np.ndarray,
                    clip_percentile: float = 0.01):
        cdf = self._get_cdf_at_y(cdf, gt)

        for model_type in self.models:
            print(f"Fitting model type: {model_type}")
            for d in range(self.num_dims):
                print(d)
                x = cdf[:, d].reshape(-1)
                y = np.array([(x <= v).mean() for v in x])

                # Exclude extreme CDF values
                q_low, q_high = np.quantile(x, [clip_percentile, 1 - clip_percentile])
                mask = (x >= q_low) & (x <= q_high)

                x_clip = x[mask].reshape(-1, 1)
                y_clip = y[mask]

                model = self.models[model_type][d]
                model.fit(x_clip, y_clip)

    def _apply_model(self, 
                     gt: np.ndarray,
                     cdf: np.ndarray, 
                     model_type: str) -> np.ndarray:
        cdf = self._get_cdf_at_y(cdf, gt)
        if model_type == "rf_joint":
            return self.models[model_type].predict(cdf).reshape(cdf.shape)
        else:
            recalibrated = np.zeros_like(cdf)
            for d in range(self.num_dims):
                recalibrated[:, d] = self.models[model_type][d].predict(cdf[:, d].reshape(-1, 1))
            return recalibrated

    def _plot_calibration_coverage(
        self,
        cdf_uncal: np.ndarray,
        cdf_cal_dict: dict,
        model_labels: List[str],
        num_bins: int = 50,
        font_scale: float = 4,
    ):
        expected_p = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
        base_fontsize = 10 * font_scale
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": base_fontsize * 1.2,
            "axes.labelsize": base_fontsize,
            "xtick.labelsize": base_fontsize * 0.9,
            "ytick.labelsize": base_fontsize * 0.9,
            "legend.fontsize": base_fontsize * 0.9,
        })

        fig, ax = plt.subplots(figsize=(18, 18))

        def plot_cdf(cdf_vals: np.ndarray, label: str):
            flat = cdf_vals.flatten()
            observed_p = [(flat <= q).mean() for q in expected_p]
            ax.plot(expected_p, observed_p, label=label, linewidth=2)

        plot_cdf(cdf_uncal, "Uncalibrated")
        for method, cdf_vals in cdf_cal_dict.items():
            plot_cdf(cdf_vals, f"Recalibrated ({method})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
        ax.set_xlabel("Expected Confidence Level")
        ax.set_ylabel("Observed Confidence Level")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":")
        ax.legend(frameon=False)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(base_dir, "calibration_plots")
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "calibration.pdf"), dpi=500)
        plt.show()

    def _compute_calibration_error(self, cdf: np.ndarray, num_bins: int = 100) -> float:
        expected_p = np.linspace(0.01, 0.99, num_bins)
        flat = cdf.flatten()
        observed_p = np.array([(flat <= p).mean() for p in expected_p])
        return np.mean(np.abs(observed_p - expected_p))

    def run(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray):
        cdf_uncal = self._get_cdf(mu, np.sqrt(var), gt)
        self._fit_models(mu, var, gt)

        cdf_calibrated = {
            model_type: self._apply_model(mu, var, gt, model_type)
            for model_type in self.models
        }

        self._plot_calibration_coverage(
            cdf_uncal=cdf_uncal,
            cdf_cal_dict=cdf_calibrated,
            model_labels=["Uncalibrated"] + [f"Recalibrated ({m})" for m in self.models.keys()],
        )

    def quantify_improvement(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray):
        cdf_uncal = self._get_cdf(mu, np.sqrt(var), gt)
        self._fit_models(mu, var, gt)

        print(f"Calibration Error (Uncalibrated): {self._compute_calibration_error(cdf_uncal):.6f}")
        for model_type in self.models:
            cdf_cal = self._apply_model(mu, var, gt, model_type)
            err = self._compute_calibration_error(cdf_cal)
            print(f"Calibration Error (Recalibrated - {model_type}): {err:.6f}")

    def quantify_improvement_Kfold(self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray, n_splits: int = 2):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        unc_errors = []
        cal_errors = {model_type: [] for model_type in self.models}

        for train_idx, val_idx in kf.split(mu):
            mu_train, var_train, gt_train = mu[train_idx], var[train_idx], gt[train_idx]
            mu_val, var_val, gt_val = mu[val_idx], var[val_idx], gt[val_idx]

            self._fit_models(mu_train, var_train, gt_train)

            cdf_uncal = self._get_cdf(mu_val, np.sqrt(var_val), gt_val)
            unc_errors.append(self._compute_calibration_error(cdf_uncal))

            for model_type in self.models:
                cdf_cal = self._apply_model(mu_val, var_val, gt_val, model_type)
                cal_errors[model_type].append(self._compute_calibration_error(cdf_cal))

        print(f"Cross-Validated Calibration Error (Uncalibrated): {np.mean(unc_errors):.6f} ± {np.std(unc_errors):.6f}")
        for model_type in self.models:
            mean_err = np.mean(cal_errors[model_type])
            std_err = np.std(cal_errors[model_type])
            print(f"Cross-Validated Calibration Error (Recalibrated - {model_type}): {mean_err:.6f} ± {std_err:.6f}")

    def run_with_holdout(
        self,
        mu_train: np.ndarray, var_train: np.ndarray, gt_train: np.ndarray, cdf_train: np.ndarray,
        mu_test: np.ndarray, var_test: np.ndarray, gt_test: np.ndarray, cdf_test: np.ndarray
    ):
        """Fit on train set, visualize calibration on test set."""
        self._fit_models(gt_train, cdf_train)
        cdf_uncal = self._get_cdf_at_y(cdf_test, gt_test)

        cdf_calibrated = {
            model_type: self._apply_model(gt_test, cdf_test, model_type)
            for model_type in self.models
        }

        self._plot_calibration_coverage(
            cdf_uncal=cdf_uncal,
            cdf_cal_dict=cdf_calibrated,
            model_labels=["Uncalibrated"] + [f"Recalibrated ({m})" for m in self.models.keys()],
        )

    def _estimate_variance_from_isotonic(self,
        model, y_support: np.ndarray
    ) -> float:
        """
        Estimate the variance of a random variable whose CDF is defined
        by a fitted isotonic regression model.

        Args:
            model: sklearn IsotonicRegression model (monotonic CDF estimator)
            y_support: 1D array of evenly spaced y values over the support of the variable

        Returns:
            Estimated variance (scalar)
        """
        # Get CDF values across support
        cdf_vals = model.predict(y_support)

        # Compute PDF as numerical derivative (finite difference)
        pdf_vals = np.diff(cdf_vals) / np.diff(y_support)
        y_midpoints = 0.5 * (y_support[:-1] + y_support[1:])  # match PDF shape

        # Normalize in case model is not strictly [0,1]
        pdf_vals = np.clip(pdf_vals, 1e-12, None)
        pdf_vals /= np.sum(pdf_vals)  # ensure integrates to 1

        # Compute mean
        mean = np.sum(pdf_vals * y_midpoints)

        # Compute variance
        var = np.sum(pdf_vals * (y_midpoints - mean) ** 2)

        return var
    
    def fast_nongaussian_variance(self, mu, sigma, recalibrator,
                              L=4, G=200, P=200):

        # 1. y-grid
        y = np.linspace(mu - L*sigma, mu + L*sigma, G)
        # 2. calibrated CDF
        u = norm.cdf((y - mu) / sigma)
        F_cal = np.clip(recalibrator.predict(u), 0, 1)
        # 3. build inverse CDF
        Q = interp1d(F_cal, y, bounds_error=False,
                    fill_value=(y[0], y[-1]))
        # 4. p-grid and quantiles
        p = np.linspace(0.005, 0.995, P)
        q = Q(p)
        # 5. mean & variance
        m = np.trapz(q, p)
        v = np.trapz((q - m)**2, p)
        return v

    
    def quantify_improvement_with_holdout(
        self,
        mu_train: np.ndarray, var_train: np.ndarray, gt_train: np.ndarray, cdf_train:np.ndarray,
        mu_test: np.ndarray, var_test: np.ndarray, gt_test: np.ndarray, cdf_test:np.ndarray,
        model_name: str = None
    ):
        """Fit on train set, report calibration error and per-dimension sharpness on test set."""
        self._fit_models(gt_train, cdf_train)
        if model_name is None:
            raise ValueError("model_name must be provided to save recalibration models")
        self.save_models("recalibration_models", model_name)

        # --- Uncalibrated ---
        cdf_uncal = self._get_cdf_at_y(cdf_test, gt_test)
        unc_error = self._compute_calibration_error(cdf_uncal)
        unc_sharpness_vec = np.mean(np.sum(var_test.reshape(var_test.shape[0], 26, 3), axis = -1))  # per-dim average σ
        unc_sharpness_vec_keys = np.mean(np.sum(var_test.reshape(var_test.shape[0], 26, 3), -1), axis = 0)
        print(f"sharpness uncalibrated per keypoint", unc_sharpness_vec_keys)
        print(f"Calibration Error (Uncalibrated): {unc_error:.6f}")
        print(f"Sharpness (Uncalibrated): {unc_sharpness_vec}")

        # --- Calibrated ---
        for model_type in self.models:
            print("################################# ", model_type, " ###########################################")
            cdf_cal = self._apply_model(gt_test, cdf_test, model_type)
            err = self._compute_calibration_error(cdf_cal)
            print(f"Calibration Error (Recalibrated - {model_type}): {err:.6f}")

            # Compute sharpness as variance of predictive distribution (only if monotonic model)
            if model_type in ["isotonic"]:  # list of monotonic models
                sharpness_vals = []
                for d in range(self.num_dims):
                    model = self.models[model_type][d]
                    # Use dense y support for numerical integration (per dimension)
                    #y_support = np.linspace(np.min(gt_test[:, d]), np.max(gt_test[:, d]), 500)
                    q_low, q_high = np.quantile(gt_test[:, d], [0.01, 0.99])
                    y_support = np.linspace(q_low, q_high, 100)
                    var_d = self.fast_nongaussian_variance(
                            mu=np.mean(mu_test[:, d]),
                            sigma=np.mean(np.sqrt(var_test[:, d])),
                            recalibrator=self.models['isotonic'][d]
                        )#self._estimate_variance_from_isotonic(model, y_support)
                    sharpness_vals.append(var_d)
                sharpness_vec = np.stack(sharpness_vals, axis=0)
                print(f"Sharpness Vector (Recalibrated - {model_type}) per keypoint:")
                #print(np.sqrt(sharpness_vec))
                print(np.sum(sharpness_vec.reshape( 26,3), axis = -1))
                
                #print("mean ", np.mean(np.sqrt(sharpness_vec)))
                print("Sharpness Calibrated: ")
                print(np.mean(np.sum(sharpness_vec.reshape(26,3), axis = -1)))

            else:
                print(f"(Skipping sharpness: {model_type} is not a monotonic CDF model)")

    def fast_nongaussian_moments(mu, sigma, recalibrator,
                             L=4, G=200, P=200):
        """Return (mean_cal, var_cal) via quantile‐integration."""
        # build y‐grid
        y = np.linspace(mu - L*sigma, mu + L*sigma, G)
        # calibrated CDF
        u = norm.cdf((y - mu) / sigma)
        F_cal = np.clip(recalibrator.predict(u), 0.0, 1.0)
        # inverse CDF
        Q = interp1d(F_cal, y, bounds_error=False,
                    fill_value=(y[0], y[-1]))
        # quantile grid
        p = np.linspace(0.005, 0.995, P)
        q = Q(p)
        # mean & var
        m = np.trapz(q, p)
        v = np.trapz((q - m)**2, p)
        return m, v








