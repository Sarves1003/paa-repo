"""Central configuration for the myCobot 280 JN inspection framework.

Target platform: myCobot 280 JN = NVIDIA Jetson Nano 4 GB (JetPack 4.6.x, Ubuntu 18.04,
4x Cortex-A57, 4 GB LPDDR4 shared by CPU and GPU). All on-robot code is NumPy + OpenCV
only; PyTorch is needed only for training (Colab).
"""
import os
import numpy as _np

if tuple(int(x) for x in _np.__version__.split(".")[:2]) < (1, 17):
    raise SystemExit("NumPy >= 1.17 is required (JetPack ships 1.13): pip3 install --user numpy==1.19.5")

JETSON_RAM_GB = 4      # the study targets the Jetson Nano 4 GB (jetson_check.py verifies it)

# ---------------- robot (myCobot 280 JN: Jetson-Nano UART) ----------------
PORT = "/dev/ttyTHS1"
BAUD = 1000000
Q_HOME_DEG = [-80.15, -0.87, -3.69, 1.14, -2.37, 146.33]   # from your collect_and_train.py
MOVE_SPEED = 20            # myCobot speed setting (1-100). Keep low near the part.
MAX_JOINT_STEP_DEG = 70.0  # safety: refuse any single move with a larger joint change
SETTLE_TOL_DEG = 0.3       # arm is "settled" when joints change less than this ...
SETTLE_COUNT = 3           # ... for this many consecutive polls
SETTLE_TIMEOUT_S = 6.0

# Kinematic model used by FK/IK ("alt" verified by fk_check.py: matches the firmware to 0.1 mm)
KIN_MODEL = "alt"
KIN_Z_OFFSET_MM = 8.1      # firmware z minus model z, measured by fk_check.py (8.14 mm, SD 0.03 mm)

# ---------------- camera ----------------
CAM_INDEX = 0
CAM_W, CAM_H = 640, 480
DEPLOY_DIR = "deploy"
INTRINSICS_PATH = os.path.join(DEPLOY_DIR, "intrinsics.npz")   # optional (cx, cy)

# ---------------- inspection grid ----------------
GRID_N = 5
GRID_HALF_MM = 30.0        # 60 x 60 mm grid, 15 mm pitch (table_grid.py may shrink it; then set its value here)
START_CELL = 12            # the locked survey pose is the grid centre
LOCKED_POSE_PATH = os.path.join(DEPLOY_DIR, "locked_pose.json")
GROUNDING_PATH = os.path.join(DEPLOY_DIR, "grounding.json")
COST_MODEL_PATH = os.path.join(DEPLOY_DIR, "cost_model_{ik}.npz")
POLICY_PATH = os.path.join(DEPLOY_DIR, "geo_attn_policy.npz")
SURVEY_DIR = os.path.join(DEPLOY_DIR, "surveys")

# ---------------- perception ----------------
CANNY_LO, CANNY_HI = 40, 120
DEFECT_THRESHOLD = 0.40    # edge-density flag (NOT a validated defect detector)

# ---------------- grounding: markers ----------------
GROUNDING_MODE = "track"   # "track" = follow ANY visible markers (1 is enough, no ruler) [default]
                           # "board" = 4 markers at measured rectangle corners; "shift" = no markers
MARKER_IDS = [0, 1]   # found by check_markers.py
FENCE_FROM_BOX = True      # the drawn box becomes the workspace limit when it is found
BOARD_W_MM = 140.2         # from measure_board.py (only used in "board" mode)
BOARD_H_MM = 138.0         # from measure_board.py (only used in "board" mode)

# ---------------- workspace limits ----------------
FENCE_MARGIN_MM = 5.0      # planned targets must stay this far inside the fence polygon
FENCE_HARD_TOL_MM = 1.0    # hard stop allows this much IK tolerance (arm stays >= margin - tol inside)
FENCE_PATH = os.path.join(DEPLOY_DIR, "fence.json")   # optional; if absent the automatic box is used
FENCE_AUTO_HALF_MM = 60.0  # automatic limit: +/- this many mm around the locked pose

# ---------------- part tracking ----------------
TRACK_PART = False         # the drawn box is the fixture; the grid is anchored to it (table_grid.py)
PART_THRESH = None         # None = automatic (Otsu, capped at 140); or a fixed grey level
PART_MIN_AREA_PX = 1500
REQUIRE_FENCE = True       # inspections need a workspace limit (fence.json or the automatic box)

# ---------------- active view selection (viewplan experiments) ----------------
# Survey image at your locked pose; inspection views BELOW it (from your pose the arm reaches every grid
# cell only lower down). Level 0 = servoed view, level 1 = close-up. Values in mm relative to the survey pose.
VIEW_LEVELS_DZ = (-60.0, -80.0)
VIEW_CAM_HEIGHT_MM = 224.0   # measured by measure_height.py (camera lens -> table at the reference)
PART_MAX_HEIGHT_MM = 60.0  # tallest the part stands (vertical pose); measure and correct if needed
VIEW_MIN_CLEARANCE_MM = 25.0   # lens must stay at least this far above the part
VIEW_LAP_REF = 6.0         # sharpness reference while collecting (rescaled from the data later)

# ---------------- logging ----------------
LOGS_DIR = "hw_logs"
