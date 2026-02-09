import numpy as np
from pathlib import Path
from pybullet_utils  import transformations

URDF_FILENAME = str(Path(__file__).resolve().parents[1] / "bittle" / "urdf" / "bittle_toes.urdf")

REF_POS_SCALE = 2.3
INIT_POS = np.array([0, 0, 0])
INIT_ROT = transformations.quaternion_from_euler(
    ai=0.0, aj=0.0, ak=-np.pi / 2.0, axes="sxyz")

# Order is [LF, LB, RF, RB], same as retarget configs.
# Using link names is more stable than hardcoded PyBullet indices.
SIM_TOE_LINK_NAMES = [
    "toeLF",
    "toeLB",
    "toeRF",
    "toeRB",
]
SIM_HIP_LINK_NAMES = [
    "left-front-shoulder-link",
    "left-back-shoulder-link",
    "right-front-shoulder-link",
    "right-back-shoulder-link",
]

SIM_ROOT_OFFSET = np.array([0, 0, 0])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0]),
]

# 8 actuated joints in PyBullet order:
# [LB_shoulder, LB_knee, LF_shoulder, LF_knee, RB_shoulder, RB_knee, RF_shoulder, RF_knee]
# Use mirrored knee rest angles so IK prefers the same bend direction on left/right.
DEFAULT_JOINT_POSE = np.array([0, -0.67,
                               0, -0.67,
                               0,  0.67,
                               0,  0.67])
JOINT_DAMPING = [0.5, 0.05,
                 0.5, 0.05,
                 0.5, 0.05,
                 0.5, 0.05]

FORWARD_DIR_OFFSET = np.array([0, 0, 0.025])
