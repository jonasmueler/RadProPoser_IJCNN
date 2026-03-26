# Claude Coding Conventions

This document defines the coding standards and conventions for this project.

## Type Hints

**REQUIRED**: Always use type hints for all function parameters and return values.

```python
# Good
def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    pass

# Bad
def compute_metrics(predictions, targets):
    pass
```

## Documentation

### Function Documentation

**REQUIRED**: Every function must have:
1. A **one-sentence description** of what it does
2. **Google-style docstring** for parameters and returns

### Module/File Documentation

**AVOID**: Do NOT add shebang lines (`#!/usr/bin/env python3`) or verbose module-level docstrings.

Keep file headers minimal - the code should be self-explanatory.

```python
# Bad - verbose header with shebang and docstring
#!/usr/bin/env python3
"""Testing script for mmRadPose dataset model evaluation.

This script evaluates models exactly as they are used in the validation loop,
reporting performance metrics per participant, per radar angle, and overall averages.

Usage:
    python tests/test_model_eval_mmRadPose.py

Configuration:
    Set DATASET="mmRadPose" in config.py
    Configure TESTCONFIG with model_path and mode ("deterministic", "gaussian", or "fine_tune")
"""

import torch
...

# Good - minimal header, just imports
import torch
from typing import Dict, Tuple

from config import TRAINCONFIG
from model.build_model import ModelWrapper


def evaluate_model() -> None:
    """Run model evaluation on validation set and save metrics."""
    ...
```

### Function Documentation Examples

```python
# Good
def preprocess_batch(x: torch.Tensor, y: torch.Tensor, dataset: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess input and target tensors for the specified dataset.

    Args:
        x: Input radar point cloud tensor of shape (B, T, N, F)
        y: Ground truth pose tensor of shape (B, T, J, 3)
        dataset: Dataset name ("mmRadPose", "mmBody", "mmFI", "MRI")

    Returns:
        Tuple of (preprocessed_x, preprocessed_y) tensors
    """
    pass

# Bad - no docstring
def preprocess_batch(x, y, dataset):
    pass

# Bad - unclear/missing parameter descriptions
def preprocess_batch(x: torch.Tensor, y: torch.Tensor, dataset: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess batch."""
    pass
```

## Code Organization

### Import Statements

**REQUIRED**: All imports must be at the top of the file, never inside functions or classes.

```python
# Good - imports at top of file
import numpy as np
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression


class MyClass:
    def __init__(self) -> None:
        self.model = IsotonicRegression()

    def process(self, data: np.ndarray) -> np.ndarray:
        return interp1d(data, data)(data)

# Bad - imports inside functions/classes
class MyClass:
    def __init__(self) -> None:
        from sklearn.isotonic import IsotonicRegression  # DON'T DO THIS
        self.model = IsotonicRegression()

    def process(self, data: np.ndarray) -> np.ndarray:
        from scipy.interpolate import interp1d  # DON'T DO THIS
        return interp1d(data, data)(data)
```

Import order (following PEP 8):
1. Standard library imports (e.g., `os`, `sys`, `math`)
2. Related third-party imports (e.g., `numpy`, `torch`, `scipy`)
3. Local application imports (e.g., `from config import TRAINCONFIG`)

Separate each group with a blank line.

### Abstraction and Encapsulation

**REQUIRED**: Follow principles of abstraction and encapsulation:
- Extract repeated logic into reusable functions
- Keep functions focused on a single responsibility
- Hide implementation details behind clean interfaces
- Use helper functions to reduce code duplication

```python
# Good - abstracted and reusable
def preprocess_gt(y: torch.Tensor, num_keypoints: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert ground truth to standard format and return both flat and reshaped versions."""
    y = y * 100
    y_flat = y.flatten(start_dim=-2, end_dim=-1)[:, -1]
    y_reshaped = y_flat.view(y.size(0), num_keypoints, 3)
    return y_flat, y_reshaped

def evaluate_model(dataloader: DataLoader) -> Dict[str, float]:
    """Run evaluation loop and compute metrics."""
    for x, y in dataloader:
        y_flat, y_reshaped = preprocess_gt(y, NUM_KEYPOINTS)
        # ... rest of evaluation
    return metrics

# Bad - repeated inline logic
def evaluate_model(dataloader: DataLoader) -> Dict[str, float]:
    """Run evaluation loop and compute metrics."""
    for x, y in dataloader:
        y = y * 100
        y_flat = y.flatten(start_dim=-2, end_dim=-1)[:, -1]
        y_reshaped = y_flat.view(y.size(0), num_keypoints, 3)
        # ... rest of evaluation
    return metrics
```

### Readability and Formatting

**REQUIRED**: Code must be nicely formatted and neatly ordered:
- Use clear, descriptive variable names
- Group related code together
- Add blank lines between logical sections
- Keep line length reasonable (< 100 characters when possible)
- Order imports alphabetically within groups

```python
# Good - well organized and readable
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import TRAINCONFIG, NUM_KEYPOINTS
from model.build_model import ModelWrapper


def load_model(checkpoint_path: str, device: str) -> ModelWrapper:
    """Load trained model from checkpoint file.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on ("cuda" or "cpu")

    Returns:
        Loaded ModelWrapper instance
    """
    model = ModelWrapper(
        backbone=TRAINCONFIG["backbone"],
        temporal_model=TRAINCONFIG["temporal_model"],
        prediction_head=TRAINCONFIG["head"]
    ).to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

# Bad - cluttered and hard to read
def load_model(checkpoint_path,device):
    model=ModelWrapper(backbone=TRAINCONFIG["backbone"],temporal_model=TRAINCONFIG["temporal_model"],prediction_head=TRAINCONFIG["head"]).to(device)
    checkpoint=torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
```

## Error Handling

### Minimal Try-Catch

**AVOID**: Abundant try-catch statements that are not needed.

Only use try-catch when:
1. You expect a specific error and can handle it gracefully
2. You need to provide better error messages
3. You're dealing with external resources (files, network, etc.)

```python
# Good - only catch when needed
def load_checkpoint(path: str) -> Dict:
    """Load model checkpoint from file.

    Args:
        path: Path to checkpoint file

    Returns:
        Checkpoint dictionary

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path)

# Bad - unnecessary try-catch
def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss between prediction and target."""
    try:
        loss = torch.nn.functional.mse_loss(pred, target)
        return loss
    except Exception as e:
        print(f"Error computing loss: {e}")
        return None
```

### Minimal If-Else

**AVOID**: Redundant if-else clauses.

Simplify logic by:
1. Early returns instead of nested if-else
2. Guard clauses at the start of functions
3. Using dictionary lookups instead of long if-elif chains

```python
# Good - early returns
def determine_head_type(mode: str) -> Tuple[str, bool]:
    """Parse mode string to determine prediction head type and fine-tune flag.

    Args:
        mode: Mode string ("deterministic", "gaussian", or "fine_tune")

    Returns:
        Tuple of (head_type, is_fine_tune)
    """
    mode_lower = mode.lower()

    if "deterministic" in mode_lower:
        return "deterministic", False
    if "fine_tune" in mode_lower or "finetune" in mode_lower:
        return "gaussian", True
    if "gaussian" in mode_lower:
        return "gaussian", False

    raise ValueError(f"Unknown mode: {mode}")

# Bad - nested if-else
def determine_head_type(mode: str) -> Tuple[str, bool]:
    """Parse mode string to determine prediction head type and fine-tune flag."""
    mode_lower = mode.lower()

    if "deterministic" in mode_lower:
        head_type = "deterministic"
        is_fine_tune = False
    else:
        if "fine_tune" in mode_lower or "finetune" in mode_lower:
            head_type = "gaussian"
            is_fine_tune = True
        else:
            if "gaussian" in mode_lower:
                head_type = "gaussian"
                is_fine_tune = False
            else:
                raise ValueError(f"Unknown mode: {mode}")

    return head_type, is_fine_tune

# Good - dictionary lookup for simple mappings
MODE_MAPPING = {
    "deterministic": ("deterministic", False),
    "gaussian": ("gaussian", False),
    "fine_tune": ("gaussian", True),
}

def determine_head_type_simple(mode: str) -> Tuple[str, bool]:
    """Parse mode string to determine prediction head type and fine-tune flag.

    Args:
        mode: Mode string key

    Returns:
        Tuple of (head_type, is_fine_tune)
    """
    mode_lower = mode.lower()

    if mode_lower not in MODE_MAPPING:
        raise ValueError(f"Unknown mode: {mode}")

    return MODE_MAPPING[mode_lower]
```

## Code Changes and Reversibility

### Always Keep Backups

**REQUIRED**: When making code changes, always have a way to revert to the previous working state.

**Best Practices**:
1. **Read files before editing** - Always use the Read tool to see current code before making changes
2. **Create backup copies** - For major refactoring, create a temporary copy with `_new` or `_backup` suffix
3. **Fix root causes, not symptoms** - Address the actual problem instead of adding workarounds
4. **Listen to user feedback** - If asked to revert, do so immediately without arguing
5. **Keep solutions simple** - Prefer direct fixes over complex workarounds

```python
# Good - fix root cause
# Problem: MODEL_TYPE = "gaussian" but code checks MODEL_TYPE == "gaussian_nll"
# Solution: Change MODEL_TYPE to "gaussian_nll"
MODEL_TYPE = "gaussian_nll"

# Bad - workaround for symptom
# Problem: TRAINCONFIG not defined when MODEL_TYPE doesn't match
# "Solution": Add getattr() everywhere to handle missing TRAINCONFIG
device = getattr(config, 'TRAINCONFIG', {}).get("device", "cuda")  # DON'T DO THIS
```

### When to Revert

Revert changes immediately if:
1. User explicitly asks to revert
2. The change introduces new complexity without solving the root problem
3. The fix is a workaround instead of addressing the actual issue
4. The code worked before and now doesn't work after refactoring

```python
# Example scenario requiring revert:
# Before refactoring: Code worked fine with direct imports
from config import TRAINCONFIG
device = TRAINCONFIG["device"]

# After attempted "fix": Added unnecessary complexity
import config
device = getattr(config, 'TRAINCONFIG', {}).get("device", "cuda")  # TOO COMPLEX

# After revert: Back to simple, working code
from config import TRAINCONFIG
device = TRAINCONFIG["device"]  # CORRECT

# Actual fix: Ensure TRAINCONFIG is always defined in config.py
MODEL_TYPE = "gaussian_nll"  # Match the condition check
```

### Version Control

Use git to track changes and enable easy reverting:

```bash
# Before major refactoring, commit current state
git add .
git commit -m "Working state before refactoring"

# After refactoring, if something breaks:
git diff  # Review what changed
git checkout -- file.py  # Revert specific file
git reset --hard HEAD  # Revert all changes (use carefully)
```

## Configuration

### No Command Line Arguments

**REQUIRED**: Never use command line arguments (argparse, sys.argv, etc.) in scripts.

All configuration should be defined in config files (e.g., `config.py`, YAML files) that can be imported and modified.

```python
# Bad - using argparse for configuration
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    train(batch_size=args.batch_size, lr=args.learning_rate)

# Good - using config file
from config import TRAINCONFIG

def main():
    train(
        batch_size=TRAINCONFIG["batch_size"],
        lr=TRAINCONFIG["learning_rate"],
    )
```

Benefits:
1. Configuration is explicit and version-controlled
2. No hidden state from command line
3. Easier to reproduce experiments
4. IDE autocompletion and type checking work properly

## Code Simplicity

### No Excessive Abstractions

**AVOID**: Dataclasses, `@property` decorators, and other "production-ready" patterns when simple code suffices.

This is research code, not production software. Prioritize readability and simplicity over enterprise patterns.

```python
# Bad - over-engineered with dataclass and properties
from dataclasses import dataclass

@dataclass
class ModelConfig:
    batch_size: int = 32
    learning_rate: float = 0.001

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation

    @property
    def warmup_steps(self) -> int:
        return int(self.total_steps * 0.1)

# Good - simple dictionary
MODEL_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
}

# Bad - unnecessary property for simple attribute access
class Model:
    def __init__(self, hidden_dim: int) -> None:
        self._hidden_dim = hidden_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

# Good - direct attribute
class Model:
    def __init__(self, hidden_dim: int) -> None:
        self.hidden_dim = hidden_dim
```

Use `@property` only when you need computed values or validation. Use dataclasses only when you have many fields that benefit from auto-generated methods.

## Summary

Follow these principles:
1. ✅ **Always use type hints**
2. ✅ **Document all functions** (one-sentence description + Google-style docstring)
3. ✅ **No shebang lines or verbose module docstrings** (keep file headers minimal)
4. ✅ **Imports at top of file** (never inside functions or classes, follow PEP 8 order)
5. ✅ **Apply abstraction and encapsulation** (extract reusable functions, single responsibility)
6. ✅ **Write readable, well-formatted code** (clear names, organized structure)
7. ✅ **Minimize try-catch** (only when truly needed for error handling)
8. ✅ **Simplify if-else** (early returns, guard clauses, dictionary lookups)
9. ✅ **Keep backups and fix root causes** (always have a way to revert, address actual problems)
10. ✅ **No command line arguments** (use config files instead)
11. ✅ **Avoid excessive abstractions** (no unnecessary dataclasses, @property, or enterprise patterns)

## For Future Sessions

At the start of new conversations, remind Claude to:
```
Read and follow the coding conventions in .claude/conventions.md
```

This ensures consistent code quality across the project.
