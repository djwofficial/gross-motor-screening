"""OpenPose BODY_25 point order, the order used by the dataset and by every array in this repo.

Owner: Daryn. Source: the dataset's JSON files and OpenPose's BODY_25 output format.
"""

NAMES = [
    "nose",  # 0
    "neck",  # 1
    "r_shoulder",  # 2
    "r_elbow",  # 3
    "r_wrist",  # 4
    "l_shoulder",  # 5
    "l_elbow",  # 6
    "l_wrist",  # 7
    "mid_hip",  # 8
    "r_hip",  # 9
    "r_knee",  # 10
    "r_ankle",  # 11
    "l_hip",  # 12
    "l_knee",  # 13
    "l_ankle",  # 14
    "r_eye",  # 15
    "l_eye",  # 16
    "r_ear",  # 17
    "l_ear",  # 18
    "l_big_toe",  # 19
    "l_small_toe",  # 20
    "l_heel",  # 21
    "r_big_toe",  # 22
    "r_small_toe",  # 23
    "r_heel",  # 24
]
N_POINTS = len(NAMES)  # 25

NOSE, NECK = 0, 1
R_SHOULDER, R_ELBOW, R_WRIST = 2, 3, 4
L_SHOULDER, L_ELBOW, L_WRIST = 5, 6, 7
MID_HIP = 8
R_HIP, R_KNEE, R_ANKLE = 9, 10, 11
L_HIP, L_KNEE, L_ANKLE = 12, 13, 14
R_EYE, L_EYE, R_EAR, L_EAR = 15, 16, 17, 18
L_BIG_TOE, L_SMALL_TOE, L_HEEL = 19, 20, 21
R_BIG_TOE, R_SMALL_TOE, R_HEEL = 22, 23, 24

HEAD = [NOSE, R_EYE, L_EYE, R_EAR, L_EAR]
LEFT_FOOT = [L_ANKLE, L_BIG_TOE, L_SMALL_TOE, L_HEEL]
RIGHT_FOOT = [R_ANKLE, R_BIG_TOE, R_SMALL_TOE, R_HEEL]

# Lines to draw for a stick figure.
BONES = [
    (NECK, NOSE),
    (NOSE, R_EYE),
    (R_EYE, R_EAR),
    (NOSE, L_EYE),
    (L_EYE, L_EAR),
    (NECK, R_SHOULDER),
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW, R_WRIST),
    (NECK, L_SHOULDER),
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST),
    (NECK, MID_HIP),
    (MID_HIP, R_HIP),
    (R_HIP, R_KNEE),
    (R_KNEE, R_ANKLE),
    (R_ANKLE, R_HEEL),
    (R_ANKLE, R_BIG_TOE),
    (R_BIG_TOE, R_SMALL_TOE),
    (MID_HIP, L_HIP),
    (L_HIP, L_KNEE),
    (L_KNEE, L_ANKLE),
    (L_ANKLE, L_HEEL),
    (L_ANKLE, L_BIG_TOE),
    (L_BIG_TOE, L_SMALL_TOE),
]

TASKS = {1: "stand on one leg", 2: "hop on one leg", 3: "standing long jump"}
LABELS = {0: "Bad", 1: "Good", 2: "Perfect"}
