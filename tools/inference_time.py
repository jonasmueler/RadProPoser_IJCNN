import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from config import MODELPATH, SEQLEN

# add model path
sys.path.append(MODELPATH)

# model configurations
MODELS = {
    "RPPgaussianGaussian": {"class": "RadProPoserVAE", "module": "vae_lstm_ho", "single_frame": False},
    "RPPlaplaceLaplace": {"class": "RadProPoserVAE", "module": "vae_lstm_ho", "single_frame": False},
    "RPPlaplaceGaussian": {"class": "RadProPoserVAE", "module": "vae_lstm_ho", "single_frame": False},
    "RPPgaussianLaplace": {"class": "RadProPoserVAE", "module": "vae_lstm_ho", "single_frame": False},
    "RPPgaussianGaussianCov": {"class": "RadProPoserVAECov", "module": "vae_lstm_ho", "single_frame": False},
    "RPPevidential": {"class": "RadProPoserEvidential", "module": "evidential_pose_regression", "single_frame": False},
    "RPPnormalizingFlow": {"class": "RadProPoserVAENF", "module": "normalizing_flow", "single_frame": False},
    "HoEtAlBaseline": {"class": "HRRadarPose", "module": "vae_lstm_ho", "single_frame": True},
}

N_RUNS = 50
DEVICE = "cuda"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model: nn.Module, input_tensor: torch.Tensor, n_runs: int = N_RUNS) -> float:
    """Measure average inference time in milliseconds."""
    model.eval()
    
    # warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model.forward_inference(input_tensor)
    
    # synchronize before timing
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    # time n_runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if DEVICE == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model.forward_inference(input_tensor)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                start = time.perf_counter()
                _ = model.forward_inference(input_tensor)
                times.append((time.perf_counter() - start) * 1000)
    
    return sum(times) / len(times)


def count_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """Count FLOPs using PyTorch's flop counter."""
    model.eval()
    with torch.no_grad():
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            _ = model.forward_inference(input_tensor)
    
    return flop_counter.get_total_flops()


def load_model(model_config: dict) -> nn.Module:
    """Load model from config."""
    module = __import__(model_config["module"], fromlist=[model_config["class"]])
    model_class = getattr(module, model_config["class"])
    return model_class()


def format_number(n: float, decimals: int = 3) -> str:
    """Format large numbers with M/G suffix."""
    if n >= 1e9:
        return f"{round(n / 1e9, decimals)}G"
    elif n >= 1e6:
        return f"{round(n / 1e6, decimals)}M"
    elif n >= 1e3:
        return f"{round(n / 1e3, decimals)}K"
    return str(round(n, decimals))


if __name__ == "__main__":
    print(f"Benchmarking {len(MODELS)} models on {DEVICE}")
    print(f"Running {N_RUNS} forward passes for timing\n")
    
    # input shapes
    seq_input = torch.rand(1, SEQLEN, 4, 4, 64, 128).to(DEVICE)
    single_input = torch.rand(1, 4, 4, 64, 128).to(DEVICE)
    
    results = []
    
    for model_name, config in MODELS.items():
        print(f"Benchmarking {model_name}...")
        
        model = load_model(config).float().to(DEVICE)
        input_tensor = single_input if config["single_frame"] else seq_input
        
        # count parameters
        total_params = count_parameters(model)
        trainable_params = count_trainable_parameters(model)
        
        # count flops
        flops = count_flops(model, input_tensor)
        
        # measure inference time
        avg_time = measure_inference_time(model, input_tensor)
        
        results.append({
            "model": model_name,
            "params": total_params,
            "trainable": trainable_params,
            "flops": flops,
            "time_ms": avg_time,
        })
        
        # free memory
        del model
        torch.cuda.empty_cache()
    
    # print results
    print("\n" + "=" * 80)
    print(f"{'Model':<25} {'Params':<12} {'FLOPs':<12} {'Time (ms)':<12}")
    print("=" * 80)
    
    for r in results:
        print(f"{r['model']:<25} {format_number(r['params']):<12} {format_number(r['flops']):<12} {round(r['time_ms'], 3):<12}")
    
    print("=" * 80)

