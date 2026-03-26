from misc.constants import (
    BONES,
    HUPR_TO_H36M,
    JOINT_NAMES,
    NUM_DIMS,
    NUM_JOINTS,
    TEST_SEQUENCES,
)
from misc.losses import KLLoss, nll_from_cov
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
def plotHumanPose(*args, **kwargs):
    """Plot human pose (lazy import to avoid cv2 dependency at import time)."""
    from misc.plot import plotHumanPose as _plotHumanPose
    return _plotHumanPose(*args, **kwargs)
