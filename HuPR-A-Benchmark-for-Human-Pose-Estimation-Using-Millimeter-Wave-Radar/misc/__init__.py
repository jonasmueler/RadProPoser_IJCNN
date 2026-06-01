from typing import Any

from misc.constants import (
    BONES,
    HUPR_TO_H36M,
    JOINT_NAMES,
    NUM_DIMS,
    NUM_JOINTS,
    TEST_SEQUENCES,
)
from misc.losses import kl_loss, nll_from_cov
from misc.metrics import (
    compute_ece,
    compute_mpjpe,
    compute_pa_mpjpe,
    compute_sharpness,
    get_max_preds,
    print_pose_table,
    print_uncertainty_table,
    procrustes_align,
)
from misc.utils import generateTarget


# Lazy import for cv2-dependent modules
def plotHumanPose(*args: Any, **kwargs: Any) -> Any:
    """Plot human pose with lazy import to avoid cv2 dependency at import time.

    Args:
        *args: Positional arguments forwarded to misc.plot.plotHumanPose.
        **kwargs: Keyword arguments forwarded to misc.plot.plotHumanPose.

    Returns:
        Result from the underlying plotHumanPose function.
    """
    from misc.plot import plotHumanPose as _plotHumanPose
    return _plotHumanPose(*args, **kwargs)
