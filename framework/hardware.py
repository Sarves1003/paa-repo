"""Robot and camera interfaces for the myCobot 280 JN.

Arm        real robot (pymycobot), with the send_angles no-wait patch from your
           original hardware.py, robust reads, settling detection and a joint-step
           safety check.
DryArm     kinematic stand-in (no robot) for testing every script with --dry-run.
Camera     V4L2/MJPG capture with buffer flushing.
DryCamera  synthesises eye-in-hand views by translating a reference image according
           to the arm's FK position, so grounding calibration can be tested offline.
"""
import inspect, json, os, time, types
import numpy as np
import config as C
from kinematics import fk


# ------------------------------------------------------------------ arm
class Arm:
    def __init__(self, port=C.PORT, baud=C.BAUD):
        from pymycobot import MyCobot
        self.mc = MyCobot(port, baud)
        self._orig = None
        module = inspect.getmodule(MyCobot)
        protocol = getattr(module, "ProtocolCode", None)
        if protocol is not None and hasattr(protocol, "SEND_ANGLES"):   # no-wait patch
            code, orig = protocol.SEND_ANGLES, self.mc._res
            self._orig = orig
            def once(self_, real_command, has_reply, genre):
                if genre == code:
                    self_._write(self_._flatten(real_command)); return 0
                return orig(real_command, has_reply, genre)
            self.mc._res = types.MethodType(once, self.mc)
        self.mc.power_on(); time.sleep(0.5)

    def read_deg(self, tries=8):
        for k in range(tries):
            try:
                q = self.mc.get_angles()
                if isinstance(q, (list, tuple)) and len(q) == 6 and np.all(np.isfinite(q)):
                    return np.array(q, float)
            except Exception:
                pass
            time.sleep(0.03 * (k + 1))
        raise RuntimeError("get_angles failed")

    def read_coords(self):
        try:
            c = self.mc.get_coords()
            if c and len(c) == 6:
                return np.array(c, float)
        except Exception:
            pass
        return np.full(6, np.nan)

    def send_deg(self, q_deg, speed=C.MOVE_SPEED):
        self.mc.send_angles([float(x) for x in q_deg], int(speed))

    def close(self):
        try:
            if self._orig is not None:
                self.mc._res = self._orig
            self.mc.close()
        except Exception:
            pass


class DryArm:
    """Kinematic stand-in. Like the real myCobot (fk_check / calibration data), each joint lands with a
    REPEATABLE pose-dependent error (sys_err_deg amplitude) plus a small random part (noise_deg)."""
    def __init__(self, q0_deg, noise_deg=0.15, seed=0, sys_err_deg=0.0):
        self.q = np.array(q0_deg, float)
        self.rng = np.random.default_rng(seed)
        self.noise, self.sys = noise_deg, sys_err_deg
        self.ph = self.rng.uniform(0, 2 * np.pi, 6)

    def read_deg(self):
        return self.q.copy()

    def read_coords(self):
        return np.r_[fk(np.deg2rad(self.q))[:3, 3] * 1000, np.full(3, np.nan)]

    def send_deg(self, q_deg, speed=C.MOVE_SPEED):
        q = np.array(q_deg, float)
        self.q = q + self.sys * np.sin(np.deg2rad(q) * 3.0 + self.ph) + self.rng.normal(0, self.noise, 6)

    def close(self):
        pass


def wait_settled(arm, timeout=C.SETTLE_TIMEOUT_S, tol=C.SETTLE_TOL_DEG, need=C.SETTLE_COUNT, dt=0.12):
    """Poll joints until they stop changing. Returns (elapsed_s, q_deg_mean_of_3)."""
    t0 = time.time()
    last, still = arm.read_deg(), 0
    while time.time() - t0 < timeout:
        time.sleep(dt)
        q = arm.read_deg()
        still = still + 1 if np.max(np.abs(q - last)) < tol else 0
        last = q
        if still >= need:
            break
    q = np.mean([arm.read_deg() for _ in range(3)], axis=0)
    return time.time() - t0, q


# ---- workspace fence (fence.py). Every move goes through safe_move, so it is enforced here.
_FENCE, _P_LOCK = None, None
LAST_REFUSAL = ""


def set_fence(fence, p_lock_mm):
    """fence: fence.Fence in survey-relative workspace mm; p_lock_mm: FK position of the locked pose."""
    global _FENCE, _P_LOCK
    _FENCE, _P_LOCK = fence, (None if p_lock_mm is None else np.asarray(p_lock_mm, float))


def in_fence_q(q_deg):
    if _FENCE is None:
        return True
    p = fk(np.deg2rad(np.asarray(q_deg, float)))[:3, 3] * 1000
    return _FENCE.signed_dist(p[:2] - _P_LOCK[:2]) >= _FENCE.margin - C.FENCE_HARD_TOL_MM


def safe_move(arm, q_target_deg, speed=C.MOVE_SPEED, max_step=C.MAX_JOINT_STEP_DEG):
    """Refuses moves outside the workspace fence or with a joint change above max_step.
    Returns (moved, elapsed_s, q_meas); the reason for a refusal is in hardware.LAST_REFUSAL."""
    global LAST_REFUSAL
    q_now = arm.read_deg()
    if not in_fence_q(q_target_deg):
        LAST_REFUSAL = "outside workspace fence"
        return False, 0.0, q_now
    if np.max(np.abs(np.asarray(q_target_deg) - q_now)) > max_step:
        LAST_REFUSAL = "joint step > %.0f deg" % max_step
        return False, 0.0, q_now
    LAST_REFUSAL = ""
    t0 = time.time()
    arm.send_deg(q_target_deg, speed)
    _, q = wait_settled(arm)
    return True, time.time() - t0, q


def load_locked_pose(dry=False):
    if os.path.exists(C.LOCKED_POSE_PATH):
        return json.load(open(C.LOCKED_POSE_PATH))
    if dry:
        return {"angles_deg": [-74.5, -32.6, -77.0, 40.6, -22.4, 82.7], "coords_mm": None}
    raise SystemExit(f"missing {C.LOCKED_POSE_PATH} (run your live_align.py first)")


def make_arm(dry, q0_deg=None):
    return DryArm(q0_deg, sys_err_deg=float(os.environ.get("DRY_SYS_ERR_DEG", 0))) if dry else Arm()


# ------------------------------------------------------------------ camera
def open_camera(verbose=True):
    """Open the USB camera robustly: configured CAM_INDEX first, then every /dev/video* device,
    with the V4L2 backend and then OpenCV's default. Returns (cap, description) or (None, reasons)."""
    import cv2, glob, re
    idx = [C.CAM_INDEX] + sorted({int(re.sub(r"\D", "", p)) for p in glob.glob("/dev/video*")
                                  if re.sub(r"\D", "", p)} - {C.CAM_INDEX}) + [i for i in range(4)]
    tried, seen = [], set()
    for i in idx:
        for name, be in (("V4L2", cv2.CAP_V4L2), ("default", cv2.CAP_ANY)):
            if (i, name) in seen:
                continue
            seen.add((i, name))
            cap = cv2.VideoCapture(i, be)
            if not cap.isOpened():
                tried.append("%d/%s: cannot open" % (i, name)); cap.release(); continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_H)
            ok = False
            for _ in range(10):
                ok, f = cap.read()
                if ok and f is not None:
                    break
            if ok:
                desc = "camera index %d (%s backend), %dx%d" % (i, name, f.shape[1], f.shape[0])
                if verbose:
                    print(desc + ("" if i == C.CAM_INDEX else
                                  "   -> set CAM_INDEX = %d in config.py to skip the search" % i))
                return cap, desc
            tried.append("%d/%s: opens but gives no frames" % (i, name)); cap.release()
    return None, tried


class Camera:
    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.cap, info = open_camera()
        if self.cap is None:
            raise SystemExit("no working camera. Tried: %s\nCheck: ls /dev/video*  and that no other program "
                             "uses the camera (sudo fuser -v /dev/video*)" % "; ".join(info))

    def grab(self, flush=5, n=3):
        """Discard stale frames, then return the median of n frames (lower noise). The first call
        also lets auto-exposure settle (15 extra frames)."""
        if not getattr(self, "_warm", False):
            for _ in range(15):
                self.cap.read()
            self._warm = True
        for _ in range(flush):
            self.cap.read()
        frames = []
        for _ in range(n + 4):
            ok, f = self.cap.read()
            if ok and f is not None:
                frames.append(f)
                if len(frames) == n:
                    break
            else:
                time.sleep(0.05)
        if not frames:
            return None
        return frames[0] if len(frames) == 1 else np.median(np.stack(frames), axis=0).astype(np.uint8)

    def release(self):
        self.cap.release()


class DryCamera:
    """Synthetic eye-in-hand view: reference image translated by A_true @ (p - p0)."""
    def __init__(self, arm, ref_img_path, A_true=((-4.0, 0.0), (0.0, 4.0)), seed=0):
        import cv2
        self.cv2, self.arm = cv2, arm
        img = cv2.imread(ref_img_path) if ref_img_path and os.path.exists(ref_img_path) else None
        if img is None:   # plain synthetic part: dark grooved block on white paper
            img = np.full((C.CAM_H, C.CAM_W, 3), 225, np.uint8)
            cv2.rectangle(img, (230, 170), (420, 320), (60, 60, 60), -1)
            for x in range(245, 410, 28):
                cv2.line(img, (x, 170), (x, 320), (150, 150, 150), 4)
        self.img = cv2.resize(img, (C.CAM_W, C.CAM_H))
        for (x, y) in [(150, 120), (490, 120), (490, 360), (150, 360)]:   # 4 corner markers, 85 x 60 mm
            cv2.rectangle(self.img, (x - 28, y - 28), (x + 28, y + 28), (255, 255, 255), -1)
            cv2.rectangle(self.img, (x - 20, y - 20), (x + 20, y + 20), (20, 20, 20), -1)
            cv2.rectangle(self.img, (x - 8, y - 8), (x + 2, y + 6), (255, 255, 255), -1)
        self.A = np.array(A_true, float)
        self.p0 = fk(np.deg2rad(arm.read_deg()))[:3, 3] * 1000
        self.rng = np.random.default_rng(seed)

    def grab(self, flush=0):
        p = fk(np.deg2rad(self.arm.read_deg()))[:3, 3] * 1000
        s = self.A @ (p[:2] - self.p0[:2])
        M = np.float32([[1, 0, s[0]], [0, 1, s[1]]])
        f = self.cv2.warpAffine(self.img, M, (C.CAM_W, C.CAM_H), borderValue=(225, 225, 225))
        return np.clip(f.astype(np.int16) + self.rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8)

    def release(self):
        pass


def make_camera(dry, arm=None, ref_img=None):
    return DryCamera(arm, ref_img) if dry else Camera()
