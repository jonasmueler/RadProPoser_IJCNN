import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent.parent))
from misc.constants import JOINT_NAMES, NUM_DIMS, NUM_JOINTS


class HuPRRecalibrator:
    """Isotonic regression recalibrator for HuPR 3D pose predictions."""

    def __init__(self, num_dims: int = NUM_DIMS) -> None:
        """Initialize recalibrator with isotonic regression models per dimension.

        Args:
            num_dims: Number of output dimensions (default 51 for 17 joints × 3)
        """
        self.num_dims = num_dims
        self.models: Dict[str, List[IsotonicRegression]] = {
            "isotonic": [IsotonicRegression(out_of_bounds="clip") for _ in range(num_dims)],
        }

    def save_models(self, directory: str, model_name: str) -> None:
        """Save fitted isotonic models to disk.

        Args:
            directory: Base directory name (e.g., "recalibration_models")
            model_name: Name of the model (e.g., "motionbert")
        """
        base_dir = Path(__file__).parent
        full_path = base_dir / directory / model_name
        full_path.mkdir(parents=True, exist_ok=True)

        for model_type, mdl_list in self.models.items():
            path = full_path / f"{model_type}.pkl"
            joblib.dump(mdl_list, path)

        print(f"Saved {len(self.models)} model types to '{directory}/{model_name}'")

    def load_models(self, directory: str, model_name: str) -> None:
        """Load fitted isotonic models from disk.

        Args:
            directory: Base directory name (e.g., "recalibration_models")
            model_name: Name of the model (e.g., "motionbert")

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        base_dir = Path(__file__).parent
        full_path = base_dir / directory / model_name

        for model_type in list(self.models.keys()):
            path = full_path / f"{model_type}.pkl"
            if not path.is_file():
                raise FileNotFoundError(f"Could not find calibration file '{path}'")
            self.models[model_type] = joblib.load(path)

        print(f"Loaded {len(self.models)} model types from '{full_path}'")

    def _get_cdf(self, mu: np.ndarray, sigma: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Compute CDF values under Gaussian assumption.

        Args:
            mu: Mean predictions of shape (N, D)
            sigma: Standard deviations of shape (N, D)
            gt: Ground truth values of shape (N, D)

        Returns:
            CDF values of shape (N, D)
        """
        return norm.cdf(gt, loc=mu, scale=sigma)

    def _fit_models(
        self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray, clip_percentile: float = 0.01
    ) -> None:
        """Fit isotonic regression models on training data.

        Args:
            mu: Mean predictions of shape (N, D)
            var: Variances of shape (N, D)
            gt: Ground truth values of shape (N, D)
            clip_percentile: Percentile for clipping extreme CDF values
        """
        sigma = np.sqrt(var)
        cdf = self._get_cdf(mu, sigma, gt)

        for model_type in self.models:
            print(f"Fitting model type: {model_type}")
            for d in range(self.num_dims):
                x = cdf[:, d].reshape(-1)
                y = np.array([(x <= v).mean() for v in x])

                q_low, q_high = np.quantile(x, [clip_percentile, 1 - clip_percentile])
                mask = (x >= q_low) & (x <= q_high)

                x_clip = x[mask].reshape(-1, 1)
                y_clip = y[mask]

                model = self.models[model_type][d]
                model.fit(x_clip, y_clip)

    def _apply_model(
        self, mu: np.ndarray, var: np.ndarray, gt: np.ndarray, model_type: str
    ) -> np.ndarray:
        """Apply fitted recalibration model to get calibrated CDF values.

        Args:
            mu: Mean predictions of shape (N, D)
            var: Variances of shape (N, D)
            gt: Ground truth values of shape (N, D)
            model_type: Type of model to apply (e.g., "isotonic")

        Returns:
            Recalibrated CDF values of shape (N, D)
        """
        sigma = np.sqrt(var)
        cdf = self._get_cdf(mu, sigma, gt)

        recalibrated = np.zeros_like(cdf)
        for d in range(self.num_dims):
            recalibrated[:, d] = self.models[model_type][d].predict(cdf[:, d].reshape(-1, 1))

        return recalibrated

    def _compute_calibration_error(self, cdf: np.ndarray, num_bins: int = 100) -> float:
        """Compute Expected Calibration Error (ECE).

        Args:
            cdf: CDF values of shape (N, D)
            num_bins: Number of bins for computing calibration error

        Returns:
            ECE value (lower is better, 0 is perfect)
        """
        expected_p = np.linspace(0.01, 0.99, num_bins)
        flat = cdf.flatten()
        observed_p = np.array([(flat <= p).mean() for p in expected_p])
        return float(np.mean(np.abs(observed_p - expected_p)))

    def _plot_calibration_coverage(
        self,
        cdf_uncal: np.ndarray,
        cdf_cal_dict: Dict[str, np.ndarray],
        save_path: Path,
        num_bins: int = 50,
        font_scale: float = 2,
    ) -> None:
        """Plot calibration curves comparing uncalibrated vs recalibrated.

        Args:
            cdf_uncal: Uncalibrated CDF values of shape (N, D)
            cdf_cal_dict: Dictionary mapping model type to calibrated CDF values
            save_path: Path to save the plot
            num_bins: Number of bins for the calibration curve
            font_scale: Font scaling factor
        """
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

        fig, ax = plt.subplots(figsize=(10, 10))

        def plot_cdf(cdf_vals: np.ndarray, label: str) -> None:
            """Plot a single calibration curve on the axes.

            Args:
                cdf_vals: CDF values of shape (N, D).
                label: Legend label for this curve.
            """
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

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved calibration plot to '{save_path}'")

    def _fast_nongaussian_variance(
        self,
        mu: float,
        sigma: float,
        recalibrator: IsotonicRegression,
        L: float = 4,
        G: int = 200,
        P: int = 200,
    ) -> float:
        """Estimate recalibrated variance via quantile integration.

        Args:
            mu: Predictive mean
            sigma: Predictive standard deviation
            recalibrator: Fitted isotonic regression model
            L: Range multiplier for the y-grid
            G: Number of points in the y-grid
            P: Number of points in the probability grid

        Returns:
            Estimated recalibrated variance
        """
        y = np.linspace(mu - L * sigma, mu + L * sigma, G)
        u = norm.cdf((y - mu) / sigma)
        F_cal = np.clip(recalibrator.predict(u), 0, 1)

        Q = interp1d(F_cal, y, bounds_error=False, fill_value=(y[0], y[-1]))
        p = np.linspace(0.005, 0.995, P)
        q = Q(p)

        m = np.trapz(q, p)
        v = np.trapz((q - m) ** 2, p)

        return float(v)

    def quantify_improvement_with_holdout(
        self,
        mu_train: np.ndarray,
        var_train: np.ndarray,
        gt_train: np.ndarray,
        mu_test: np.ndarray,
        var_test: np.ndarray,
        gt_test: np.ndarray,
        model_name: str,
    ) -> Dict[str, Any]:
        """Fit on train set, evaluate on test set, and save models.

        Args:
            mu_train: Training mean predictions of shape (N, D)
            var_train: Training variances of shape (N, D)
            gt_train: Training ground truth of shape (N, D)
            mu_test: Test mean predictions of shape (N, D)
            var_test: Test variances of shape (N, D)
            gt_test: Test ground truth of shape (N, D)
            model_name: Name for saving the model (e.g., "motionbert")

        Returns:
            Dictionary with ECE values (float) and per-joint sharpness arrays (np.ndarray).
        """
        self._fit_models(mu_train, var_train, gt_train)
        self.save_models("recalibration_models", model_name)

        cdf_uncal = self._get_cdf(mu_test, np.sqrt(var_test), gt_test)
        unc_error = self._compute_calibration_error(cdf_uncal)

        unc_sharpness = np.mean(np.sum(var_test.reshape(-1, NUM_JOINTS, 3), axis=-1))
        unc_sharpness_per_joint = np.mean(
            np.sum(var_test.reshape(-1, NUM_JOINTS, 3), axis=-1), axis=0
        )

        print(f"\nSharpness (Uncalibrated) per joint: {unc_sharpness_per_joint}")
        print(f"Sharpness (Uncalibrated) mean: {unc_sharpness:.6f}")
        print(f"ECE (Uncalibrated): {unc_error:.6f}")

        results = {
            "ece_uncalibrated": unc_error,
            "sharpness_uncalibrated": unc_sharpness,
            "sharpness_uncalibrated_per_joint": unc_sharpness_per_joint,
        }

        for model_type in self.models:
            print(f"\n{'='*60}")
            print(f"Model type: {model_type}")
            print("=" * 60)

            cdf_cal = self._apply_model(mu_test, var_test, gt_test, model_type)
            cal_error = self._compute_calibration_error(cdf_cal)
            print(f"ECE (Recalibrated - {model_type}): {cal_error:.6f}")

            sharpness_vals = []
            for d in range(self.num_dims):
                var_d = self._fast_nongaussian_variance(
                    mu=np.mean(mu_test[:, d]),
                    sigma=np.mean(np.sqrt(var_test[:, d])),
                    recalibrator=self.models[model_type][d],
                )
                sharpness_vals.append(var_d)

            sharpness_vec = np.array(sharpness_vals)
            sharpness_per_joint = np.sum(sharpness_vec.reshape(NUM_JOINTS, 3), axis=-1)
            cal_sharpness = np.mean(sharpness_per_joint)

            print(f"Sharpness (Recalibrated - {model_type}) per joint:")
            print(sharpness_per_joint)
            print(f"Sharpness (Recalibrated - {model_type}) mean: {cal_sharpness:.6f}")

            results[f"ece_{model_type}"] = cal_error
            results[f"sharpness_{model_type}"] = cal_sharpness
            results[f"sharpness_{model_type}_per_joint"] = sharpness_per_joint

        plots_dir = Path(__file__).parent / "calibration_plots"
        self._plot_calibration_coverage(
            cdf_uncal=cdf_uncal,
            cdf_cal_dict={mt: self._apply_model(mu_test, var_test, gt_test, mt) for mt in self.models},
            save_path=plots_dir / f"calibration_{model_name}.pdf",
        )

        return results
