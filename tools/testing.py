import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import sys
from config import *
import trainLoop

def get_model_checkpoint_path(model_name: str, testing_epoch: int = None) -> str:
    """
    Get the path to the model checkpoint for testing.
    
    Args:
        model_name: Name of the model (e.g., "RPPgaussianGaussianCov")
        testing_epoch: Specific epoch number to use, or None to use the latest checkpoint
    
    Returns:
        Full path to the checkpoint file
    """
    model_dir = os.path.join(PATHORIGIN, "trainedModels", model_name)
    
    if testing_epoch is not None:
        # Use specified epoch
        checkpoint_name = f"{model_name}{testing_epoch}"
        checkpoint_path = os.path.join(model_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path
    else:
        # Find latest checkpoint
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # List all files in the directory that start with model_name
        checkpoint_files = [
            f for f in os.listdir(model_dir) 
            if f.startswith(model_name) and os.path.isfile(os.path.join(model_dir, f))
        ]
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {model_dir}")
        
        # Extract epoch numbers and find the maximum
        epochs = []
        for f in checkpoint_files:
            # Remove model_name prefix to get epoch number
            suffix = f[len(model_name):]
            try:
                epoch = int(suffix)
                epochs.append((epoch, f))
            except ValueError:
                continue
        
        if not epochs:
            raise ValueError(f"Could not parse epoch numbers from checkpoints in {model_dir}")
        
        # Get the checkpoint with the highest epoch number
        latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
        checkpoint_path = os.path.join(model_dir, latest_file)
        print(f"Using latest checkpoint: {latest_file} (epoch {latest_epoch})")
        return checkpoint_path

class RadarPoseTester:
    def __init__(self, root_path: str, seq_len: int = SEQLEN, device: str = TRAINCONFIG["device"], single_frame: bool = False):
        self.root_path = root_path
        self.seq_len = seq_len
        self.device = device
        self.single_frame = single_frame

    @staticmethod
    def compute_mpjpe(preds: torch.Tensor, targets: torch.Tensor, keypoints: bool = False):
        preds = preds.view(preds.size(0), 26, 3)
        targets = targets.view(targets.size(0), 26, 3)

        diff = preds - targets
        dist = torch.norm(diff, dim=-1)

        if keypoints:
            return dist.mean(dim=0), dist.std(dim=0)
        return dist.mean(), dist.std()
    
    def compute_p_mpjpe(self, preds: torch.Tensor, targets: torch.Tensor, keypoints: bool = False):
        preds = preds.view(preds.size(0), 26, 3)
        targets = targets.view(targets.size(0), 26, 3)

        aligned_preds = torch.stack([
            self.procrustes_torch(tgt, pred) for pred, tgt in zip(preds, targets)
        ])

        diff = aligned_preds - targets
        dist = torch.norm(diff, dim=-1)

        if keypoints:
            return dist.mean(dim=0), dist.std(dim=0)
        return dist.mean(), dist.std()
    
    def procrustes_torch(self, 
                         X: torch.Tensor, 
                         Y: torch.Tensor, 
                         scaling: bool = True, 
                         reflection: str = 'best'):
        muX = X.mean(dim=0, keepdim=True)
        muY = Y.mean(dim=0, keepdim=True)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0 ** 2).sum()
        ssY = (Y0 ** 2).sum()

        normX = torch.sqrt(ssX)
        normY = torch.sqrt(ssY)

        X0 /= normX
        Y0 /= normY

        A = X0.T @ Y0
        U, S, Vt = torch.linalg.svd(A, full_matrices=False)
        V = Vt.T
        T = V @ U.T

        if reflection != 'best':
            have_reflection = torch.det(T) < 0
            if reflection != have_reflection:
                V[:, -1] *= -1
                S[-1] *= -1
                T = V @ U.T

        traceTA = S.sum()

        if scaling:
            b = traceTA * normX / normY
            Z = normX * traceTA * (Y0 @ T) + muX
        else:
            b = 1.0
            Z = normY * (Y0 @ T) + muX

        return Z

    def load_sequence(self, combination: list[str]):
        radar_path = os.path.join(self.root_path, "radar", f"data_cube_parsed_{'_'.join(combination)}.npz")
        skeleton_path = os.path.join(self.root_path, "skeletons", f"skeleton_{'_'.join(combination)}.npy")

        if not os.path.exists(radar_path) or not os.path.exists(skeleton_path):
            return None, None

        radar_npz = np.load(radar_path)
        radar_tensor = torch.from_numpy(radar_npz[radar_npz.files[0]]).to(self.device).to(torch.complex64)

        skeleton_tensor = torch.from_numpy(np.load(skeleton_path)).float()
        skeleton_tensor = skeleton_tensor.view(skeleton_tensor.shape[0], 26, 3) * 100

        return radar_tensor, skeleton_tensor

    def predict_sequence(self, combination: list[str], keypoints: bool = False, return_preds: bool = False):
        radar, target = self.load_sequence(combination)
        if radar is None or target is None:
            return (None,) * 6

        batch_size = 32
        preds, vars, ensembles = [], [], []

        self.model.eval()
        with torch.no_grad():
            if self.single_frame:
                # Single frame mode: process each frame individually
                loop_range = range(0, radar.size(0), batch_size)
            else:
                # Sequence mode: original logic
                loop_range = range(0, radar.size(0) - self.seq_len, batch_size)
            
            for i in tqdm(loop_range, desc=f"Predicting {'_'.join(combination)}"):
                if self.single_frame:
                    end = min(i + batch_size, radar.size(0))
                    batch = radar[i:end]
                else:
                    end = min(i + batch_size, radar.size(0) - self.seq_len)
                    batch = torch.stack([radar[j:j+self.seq_len] for j in range(i, end)])
                
                if MODELNAME in ("RPPgaussianGaussian", "RPPlaplaceLaplace", "RPPlaplaceGaussian", "RPPgaussianLaplace"):
                    ensemble, _, _, mu, var = self.model(batch, return_samples=True)
                elif MODELNAME == "RPPgaussianGaussianCov":
                    ensemble, _, _, mu, cov = self.model(batch, return_samples=True)
                    var = cov.diagonal(dim1=-2, dim2=-1)
                elif MODELNAME == "RPPnormalizingFlow":
                    mu, var, ensemble = self.model(batch, inference=True)
                elif MODELNAME == "RPPevidential":
                    out_uncertainty, ensemble = self.model(batch)
                    mu = out_uncertainty[0]
                    var = torch.var(ensemble, dim=-1)
                elif MODELNAME == "HoEtAlBaseline":
                    root, offsets = self.model(batch)
                    mu = root.unsqueeze(dim=1) + offsets
                    mu = torch.cat([root.unsqueeze(dim=1), mu], dim=1)
                    var = torch.zeros_like(mu)
                    ensemble = None
                else:
                    mu = self.model(batch)
                    var = torch.zeros_like(mu)
                    ensemble = None

                preds.append(mu)
                vars.append(var)
                if ensemble is not None:
                    ensembles.append(ensemble)

        preds = torch.cat(preds, dim=0)
        vars = torch.cat(vars, dim=0)
        if self.single_frame:
            gts = target.reshape(-1, 26 * 3).to(self.device)
        else:
            gts = target[self.seq_len:].reshape(-1, 26 * 3).to(self.device)
        
        if ensembles:
            ensembles = torch.cat(ensembles, dim=0)
        else:
            ensembles = None

        error, std = self.compute_p_mpjpe(preds, gts, keypoints)
        print(error)
        return error, std, preds, vars, gts, ensembles

    def evaluate(self, test_set: list[list[str]], keypoints: bool = False):
        losses, stds, preds, vars, gts, ensembles = [], [], [], [], [], []

        for comb in test_set:
            mean, std, pred, var, gt, ensemble = self.predict_sequence(comb, keypoints)
            if mean is not None:
                losses.append(mean)
                stds.append(std)
                preds.append(pred)
                vars.append(var)
                gts.append(gt)
                if ensemble is not None:
                    ensembles.append(ensemble)

        error = torch.mean(torch.stack(losses), dim=0)
        std = torch.mean(torch.stack(stds), dim=0)
        preds = torch.cat(preds, dim=0)
        vars = torch.cat(vars, dim=0)
        gts = torch.cat(gts, dim=0)
        
        if ensembles:
            ensembles = torch.cat(ensembles, dim=0)
        else:
            ensembles = None

        return error, std, preds, vars, gts, ensembles

    def evaluate_single(
        self,
        test_set: list[list[str]],
        keypoints: bool = False,
        batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        losses, stds = [], []
        all_preds, all_vars, all_gts = [], [], []

        for comb in test_set:
            radar, target = self.load_sequence(comb)
            if radar is None or target is None:
                continue

            preds_batches, gts_batches = [], []

            self.model.eval()
            with torch.no_grad():
                total = radar.size(0)
                for i in tqdm(range(0, total, batch_size), desc=f"Predicting {'_'.join(comb)}"):
                    end = min(i + batch_size, total)
                    batch_frames = radar[i:end].to(self.device)

                    if MODELNAME == "HoEtAlBaseline":
                        root, offset = self.model(batch_frames)
                        preds = root.unsqueeze(dim=1) + offset
                        preds = torch.cat([root.unsqueeze(dim=1), preds], dim=1)
                    else:
                        preds = self.model(batch_frames)
                    
                    preds_batches.append(preds)

                    gts_batch = target[i:end].reshape(-1, 26 * 3).to(self.device)
                    gts_batches.append(gts_batch)

            preds = torch.cat(preds_batches, dim=0)
            gts = torch.cat(gts_batches, dim=0)
            vars_ = torch.zeros_like(preds)

            error, std = self.compute_mpjpe(preds, gts, keypoints)

            losses.append(error)
            stds.append(std)
            all_preds.append(preds)
            all_vars.append(vars_)
            all_gts.append(gts)

        error = torch.mean(torch.stack(losses), dim=0)
        std = torch.mean(torch.stack(stds), dim=0)
        preds = torch.cat(all_preds, dim=0)
        vars = torch.cat(all_vars, dim=0)
        gts = torch.cat(all_gts, dim=0)

        return error, std, preds, vars, gts

    def run_evaluation(self, parts, angles, actions, reps, model_path: str):
        sys.path.append(MODELPATH)
        
        if MODELNAME in ("RPPgaussianGaussian", "RPPlaplaceLaplace", "RPPlaplaceGaussian", "RPPgaussianLaplace"):
            from vae_lstm_ho import RadProPoserVAE as Encoder
        elif MODELNAME == "RPPgaussianGaussianCov":
            from vae_lstm_ho import RadProPoserVAECov as Encoder
        elif MODELNAME == "RPPevidential":
            from evidential_pose_regression import RadProPoserEvidential as Encoder
        elif MODELNAME == "RPPnormalizingFlow":
            from normalizing_flow import RadProPoserVAENF as Encoder
        elif MODELNAME == "HoEtAlBaseline":
            from vae_lstm_ho import HRRadarPose as Encoder
        else:
            raise ValueError(f"Invalid MODELNAME: {MODELNAME}. Check config.py for valid model names.")

        self.model = Encoder().to(self.device)
        self.model = trainLoop.loadCheckpoint(self.model, None, model_path)
        self.model.eval()
        all_preds, all_gts, all_vars, all_ensembles = [], [], [], []
        val_preds, val_gts, val_vars, val_ensembles = [], [], [], []
        test_preds, test_gts, test_vars, test_ensembles = [], [], [], []
        results_by_participant = {}

        for part in parts:
            results_by_participant[part] = {}
            for angle in angles:
                combos = [[part, angle, act, rep] for act in actions for rep in reps]
                error, std, preds, vars, gts, ensembles = self.evaluate(combos)

                if isinstance(error, torch.Tensor):
                    error = error.item() if error.numel() == 1 else error.cpu().numpy()
                if isinstance(std, torch.Tensor):
                    std = std.item() if std.numel() == 1 else std.cpu().numpy()

                results_by_participant[part][angle] = {
                    "error": error,
                    "std": std
                }

                preds_np = preds.cpu().numpy()
                gts_np = gts.cpu().numpy()
                vars_np = vars.cpu().numpy()
                ensembles_np = ensembles.cpu().numpy() if ensembles is not None else None

                all_preds.append(preds_np)
                all_gts.append(gts_np)
                all_vars.append(vars_np)
                if ensembles_np is not None:
                    all_ensembles.append(ensembles_np)

                if part == "p1":
                    val_preds.append(preds_np)
                    val_gts.append(gts_np)
                    val_vars.append(vars_np)
                    if ensembles_np is not None:
                        val_ensembles.append(ensembles_np)
                else:
                    test_preds.append(preds_np)
                    test_gts.append(gts_np)
                    test_vars.append(vars_np)
                    if ensembles_np is not None:
                        test_ensembles.append(ensembles_np)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        calib_dir = os.path.join(base_dir, "calibration_analysis")
        os.makedirs(calib_dir, exist_ok=True)
        model_suffix = MODELNAME.lower() if MODELNAME else "default"

        if val_preds:
            np.save(os.path.join(calib_dir, f"mu_validation_{model_suffix}.npy"), np.concatenate(val_preds, axis=0))
            np.save(os.path.join(calib_dir, f"gt_validation_{model_suffix}.npy"), np.concatenate(val_gts, axis=0))
            np.save(os.path.join(calib_dir, f"var_validation_{model_suffix}.npy"), np.concatenate(val_vars, axis=0))
            if val_ensembles:
                np.save(os.path.join(calib_dir, f"ensemble_validation_{model_suffix}.npy"), np.concatenate(val_ensembles, axis=0))

        if test_preds:
            np.save(os.path.join(calib_dir, f"mu_testing_{model_suffix}.npy"), np.concatenate(test_preds, axis=0))
            np.save(os.path.join(calib_dir, f"gt_testing_{model_suffix}.npy"), np.concatenate(test_gts, axis=0))
            np.save(os.path.join(calib_dir, f"var_testing_{model_suffix}.npy"), np.concatenate(test_vars, axis=0))
            if test_ensembles:
                np.save(os.path.join(calib_dir, f"ensemble_testing_{model_suffix}.npy"), np.concatenate(test_ensembles, axis=0))

        return results_by_participant

if __name__ == "__main__":
    single_frame_mode = TRAINCONFIG.get("HR_SINGLE", False)
    tester = RadarPoseTester(root_path=PATHRAW, single_frame=single_frame_mode)

    # Get checkpoint path based on testing_epoch setting
    testing_epoch = TRAINCONFIG.get("testing_epoch", None)
    model_path = get_model_checkpoint_path(MODELNAME, testing_epoch)

    res = tester.run_evaluation(
        parts=["p1", "p2", "p12"],
        angles=["an0", "an1", "an2"],
        actions=["ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9"],
        reps=["r0", "r1"],
        model_path=model_path
    )

    print(res)
