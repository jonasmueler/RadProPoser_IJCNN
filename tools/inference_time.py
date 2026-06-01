import os
import sys
import time

# Ensure tools/ directory is on the path so local imports resolve
# regardless of which directory the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from config import MODEL_PATH, MODEL_REGISTRY, SEQ_LEN

# add model path
sys.path.append(MODEL_PATH)

SINGLE_FRAME_MODELS = {"HoEtAlBaseline"}

N_RUNS = 50
DEVICE = "cuda"


def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters in a model.

    Args:
        model: PyTorch model to count parameters for.

    Returns:
        Total number of parameters (trainable and non-trainable).
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count number of trainable (requires_grad) parameters in a model.

    Args:
        model: PyTorch model to count trainable parameters for.

    Returns:
        Number of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(
    model: nn.Module, input_tensor: torch.Tensor, n_runs: int = N_RUNS
) -> float:
    """Measure average inference time in milliseconds.

    Args:
        model: Model to benchmark (must have ``forward_inference`` method).
        input_tensor: Input tensor for the forward pass.
        n_runs: Number of timed forward passes to average over.

    Returns:
        Average inference time in milliseconds.
    """
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
    """Count FLOPs using PyTorch's flop counter.

    Args:
        model: Model to count FLOPs for (must have ``forward_inference`` method).
        input_tensor: Input tensor for the forward pass.

    Returns:
        Total number of floating point operations.
    """
    model.eval()
    with torch.no_grad():
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            _ = model.forward_inference(input_tensor)
    
    return flop_counter.get_total_flops()


def load_model(model_name: str) -> nn.Module:
    """Load model from MODEL_REGISTRY by name.

    Args:
        model_name: Key in MODEL_REGISTRY (e.g., "RPPgaussianGaussianCov").

    Returns:
        Instantiated model (not yet moved to device).
    """
    module_name, class_name = MODEL_REGISTRY[model_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)()


def format_number(n: float, decimals: int = 3) -> str:
    """Format large numbers with M/G/K suffix for readable display.

    Args:
        n: Number to format.
        decimals: Number of decimal places to round to.

    Returns:
        Formatted string with appropriate suffix (e.g., "1.234M").
    """
    if n >= 1e9:
        return f"{round(n / 1e9, decimals)}G"
    elif n >= 1e6:
        return f"{round(n / 1e6, decimals)}M"
    elif n >= 1e3:
        return f"{round(n / 1e3, decimals)}K"
    return str(round(n, decimals))


if __name__ == "__main__":
    print(f"Benchmarking {len(MODEL_REGISTRY)} models on {DEVICE}")
    print(f"Running {N_RUNS} forward passes for timing\n")

    # input shapes
    seq_input = torch.rand(1, SEQ_LEN, 4, 4, 64, 128).to(DEVICE)
    single_input = torch.rand(1, 4, 4, 64, 128).to(DEVICE)

    results = []

    for model_name in MODEL_REGISTRY:
        print(f"Benchmarking {model_name}...")

        model = load_model(model_name).float().to(DEVICE)
        input_tensor = single_input if model_name in SINGLE_FRAME_MODELS else seq_input
        
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

