# H3.6M 17-joint format names
JOINT_NAMES = [
    "Hip", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle",
    "Spine", "Thorax", "Neck", "Head", "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist",
]

NUM_JOINTS = 17
NUM_DIMS = NUM_JOINTS * 3  # 51 dimensions

# Test sequence IDs
TEST_SEQUENCES = [
    15, 16, 17, 38, 39, 40, 41, 42,
    244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256,
]

# HuPR 14-joint to H3.6M 17-joint mapping
HUPR_TO_H36M = {
    0: 1,   # R_Hip -> RHip
    1: 2,   # R_Knee -> RKnee
    2: 3,   # R_Ankle -> RAnkle
    3: 4,   # L_Hip -> LHip
    4: 5,   # L_Knee -> LKnee
    5: 6,   # L_Ankle -> LAnkle
    6: 9,   # Neck -> Neck
    7: 10,  # Head -> Head
    8: 11,  # L_Shoulder -> LShoulder
    9: 12,  # L_Elbow -> LElbow
    10: 13,  # L_Wrist -> LWrist
    11: 14,  # R_Shoulder -> RShoulder
    12: 15,  # R_Elbow -> RElbow
    13: 16,  # R_Wrist -> RWrist
}

# Bone definitions for H3.6M skeleton (parent_idx, child_idx)
BONES = [
    (0, 1), (1, 2), (2, 3),           # Right leg
    (0, 4), (4, 5), (5, 6),           # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine
    (8, 11), (11, 12), (12, 13),      # Left arm
    (8, 14), (14, 15), (15, 16),      # Right arm
]
