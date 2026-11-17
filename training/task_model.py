#!/usr/bin/env python3
"""Task model for ASSEMBLY + CORROSION inspection of the ringed part (shared by the Jetson and Colab).

What is inspected (one "item" per question the inspection must answer):
  ring_hex   wear ring next to the hex / O-ring end      1 = present (correct), 0 = missing (defect)
  ring_open  wear ring next to the open (plain) end      1 = present (correct), 0 = missing (defect)
  oring      black O-ring on the hex                     1 = present (correct), 0 = missing (defect)
  rust       corrosion anywhere on the metal             1 = rust (defect),     0 = clean (correct)
A part is ACCEPTED only if both rings and the O-ring are present and there is no rust.

Conditions typed when a placement is recorded (collect_views.py --condition, combine with commas):
  ok | no_ring_hex | no_ring_open | no_rings | no_oring | rust        e.g.  --condition no_ring_open,rust
--rust-facing up|side|down tells where the rust patch faces; rust facing DOWN cannot be seen by any view, so it is
excluded from training and reported separately.

Everything here is NumPy + OpenCV only (runs on the Jetson's Python 3.6/3.8 and in Colab).
"""
import numpy as np
import cv2

ITEMS = ["ring_hex", "ring_open", "oring", "rust"]
GOOD = np.array([1, 1, 1, 0], np.int64)              # label of every item on an accepted part
CONDITIONS = ["ok", "no_ring_hex", "no_ring_open", "no_rings", "no_oring", "rust"]
CROP = 112                                           # classifier input (pixels, square)
PAD_MM = {"horizontal": 10.0, "vertical": 25.0}      # margin around the part box (vertical: the side walls lean out)


# ------------------------------------------------------------------ labels
def parse_condition(cond):
    parts = [c.strip().lower() for c in str(cond).split(",") if c.strip()]
    bad = [c for c in parts if c not in CONDITIONS]
    if not parts or bad:
        raise ValueError("condition %r: use one or more of %s (comma separated)" % (cond, ", ".join(CONDITIONS)))
    if "ok" in parts and len(parts) > 1:
        raise ValueError("'ok' cannot be combined with a defect")
    return parts


def labels(cond, rust_facing="none"):
    """(y, m): label of every item and whether it may be used (rust facing down is invisible -> m = 0)."""
    p = parse_condition(cond)
    y = GOOD.copy()
    if "no_ring_hex" in p or "no_rings" in p:
        y[0] = 0
    if "no_ring_open" in p or "no_rings" in p:
        y[1] = 0
    if "no_oring" in p:
        y[2] = 0
    if "rust" in p:
        y[3] = 1
    m = np.ones(len(ITEMS), np.int64)
    if y[3] == 1 and str(rust_facing).lower() == "down":
        m[3] = 0
    return y, m


def is_defective(y):
    return bool(np.any(np.asarray(y) != GOOD))


def describe(y):
    y = np.asarray(y)
    out = []
    if y[0] == 0 and y[1] == 0:
        out.append("both rings missing")
    elif y[0] == 0:
        out.append("hex-end ring missing")
    elif y[1] == 0:
        out.append("open-end ring missing")
    if y[2] == 0:
        out.append("O-ring missing")
    if y[3] == 1:
        out.append("rust")
    return "ACCEPT" if not out else "REJECT: " + ", ".join(out)


# ------------------------------------------------------------------ crops
def box_corners(bbox, pad_px):
    x, y, w, h = [float(v) for v in bbox]
    x0, y0, x1, y1 = x - pad_px, y - pad_px, x + w + pad_px, y + h + pad_px
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def part_crop(img, corners_px, out=CROP, min_inside=0.6):
    """Axis-aligned crop of the quadrilateral corners_px (pixels of img), padded to a square, resized to out x out.
    Returns (crop or None, fraction of the crop rectangle inside the image)."""
    if img is None or corners_px is None or not np.all(np.isfinite(corners_px)):
        return None, 0.0
    h, w = img.shape[:2]
    x0, y0 = np.min(corners_px, 0); x1, y1 = np.max(corners_px, 0)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, 0.0
    cx0, cy0, cx1, cy1 = max(0, int(x0)), max(0, int(y0)), min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None, 0.0
    frac = float((cx1 - cx0) * (cy1 - cy0)) / float((x1 - x0) * (y1 - y0))
    if frac < min_inside:
        return None, frac
    c = img[cy0:cy1, cx0:cx1]
    s = max(c.shape[0], c.shape[1])
    sq = np.full((s, s, 3), 128, np.uint8)
    oy, ox = (s - c.shape[0]) // 2, (s - c.shape[1]) // 2
    sq[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return cv2.resize(sq, (out, out), interpolation=cv2.INTER_AREA), frac


def view_corners(vs, k, corners_survey, aff=None):
    """Survey-pixel corners -> pixels in view k (k=None: the survey itself)."""
    if k is None:
        return np.asarray(corners_survey, float)
    return np.array([vs.point_px(k, c, aff) for c in corners_survey], float)


def crop_for_view(img, vs, k, bbox, pose, px_per_mm, aff=None):
    pad = PAD_MM.get(pose, 15.0) * px_per_mm
    return part_crop(img, view_corners(vs, k, box_corners(bbox, pad), aff))


# ------------------------------------------------------------------ classifier (NumPy inference, BN folded)
def _conv3(x, w, b, s):
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="constant")
    Ho, Wo = (H - 1) // s + 1, (W - 1) // s + 1
    cols = np.empty((C, 3, 3, Ho, Wo), np.float32)
    for i in range(3):
        for j in range(3):
            cols[:, i, j] = xp[:, i:i + s * (Ho - 1) + 1:s, j:j + s * (Wo - 1) + 1:s]
    return (w.reshape(w.shape[0], -1) @ cols.reshape(C * 9, -1) + b[:, None]).reshape(w.shape[0], Ho, Wo)


class CNNClassifier:
    """Small CNN (baseline): loads a part_clf npz written by task_colab.py --backbone cnn."""
    def __init__(self, path="deploy/part_clf.npz", z=None):
        z = np.load(path) if z is None else z
        self.arch = "cnn"
        self.n = int(z["n_conv"]); self.items = [str(v) for v in z["items"]]
        self.convs = [(z["c%d_w" % i].astype(np.float32), z["c%d_b" % i].astype(np.float32), int(z["c%d_s" % i]))
                      for i in range(self.n)]
        self.fw, self.fb = z["fc_w"].astype(np.float32), z["fc_b"].astype(np.float32)
        self.temp = z["temperature"].astype(np.float32) if "temperature" in z.files else np.ones(len(self.items), np.float32)

    def logits(self, crop_bgr):
        x = crop_bgr.astype(np.float32).transpose(2, 0, 1) / 255.0 - 0.5
        for w, b, s in self.convs:
            x = np.maximum(_conv3(x, w, b, s), 0.0)
        return self.fw @ x.mean((1, 2)) + self.fb

    def prob(self, crop_bgr):
        return 1.0 / (1.0 + np.exp(-self.logits(crop_bgr) / self.temp))


try:
    from scipy.special import erf as _erf
except Exception:                                      # Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7)
    def _erf(x):
        s = np.sign(x); x = np.abs(x); t = 1.0 / (1.0 + 0.3275911 * x)
        y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
        return s * y


def _ln(x, w, b, eps=1e-6):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


class ViTClassifier:
    """Vision transformer (DINOv2 ViT-S/14, self-supervised pre-training) + linear head on [CLS, mean patch token].
    Pure NumPy inference of the exported weights (task_colab.py --backbone vit), so it runs on the Jetson without
    PyTorch. A 112 x 112 crop = 8 x 8 patches of 14 px + the CLS token = 65 tokens, 12 blocks, 6 heads, width 384."""
    def __init__(self, path="deploy/part_clf.npz", z=None):
        z = np.load(path) if z is None else z
        f = lambda k: z[k].astype(np.float32)
        self.arch = "vit"; self.items = [str(v) for v in z["items"]]
        self.size = int(z["img_size"]); self.patch = int(z["patch"]); self.heads = int(z["heads"])
        self.mean, self.std = f("img_mean"), f("img_std")
        self.pw = f("patch_w").reshape(f("patch_w").shape[0], -1); self.pb = f("patch_b")
        self.cls, self.pos = f("cls_token").reshape(-1), f("pos_embed").reshape(-1, self.pw.shape[0])
        self.blocks = [dict((k, f("b%d_%s" % (i, k))) for k in ("n1_w", "n1_b", "qkv_w", "qkv_b", "proj_w", "proj_b", "ls1",
                                                                 "n2_w", "n2_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b", "ls2"))
                       for i in range(int(z["depth"]))]
        self.nw, self.nb = f("norm_w"), f("norm_b")
        self.hw, self.hb = f("head_w"), f("head_b")
        self.temp = f("temperature") if "temperature" in z.files else np.ones(len(self.items), np.float32)

    def features(self, crop_bgr):
        img = crop_bgr
        if img.shape[0] != self.size or img.shape[1] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = (img[..., ::-1].astype(np.float32) / 255.0 - self.mean) / self.std           # RGB, ImageNet normalisation
        x = x.transpose(2, 0, 1)                                                           # (3, S, S)
        g, P = self.size // self.patch, self.patch
        t = x.reshape(3, g, P, g, P).transpose(1, 3, 0, 2, 4).reshape(g * g, 3 * P * P)     # patches in (c, kh, kw) order
        x = np.vstack([self.cls[None], t @ self.pw.T + self.pb]) + self.pos               # (N, D)
        N, D = x.shape; h = self.heads; dh = D // h
        for b in self.blocks:
            y = _ln(x, b["n1_w"], b["n1_b"])
            qkv = (y @ b["qkv_w"].T + b["qkv_b"]).reshape(N, 3, h, dh).transpose(1, 2, 0, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]
            a = q @ k.transpose(0, 2, 1) * (dh ** -0.5)
            a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
            y = (a @ v).transpose(1, 0, 2).reshape(N, D) @ b["proj_w"].T + b["proj_b"]
            x = x + b["ls1"] * y
            y = _ln(x, b["n2_w"], b["n2_b"]) @ b["fc1_w"].T + b["fc1_b"]
            y = 0.5 * y * (1.0 + _erf(y / np.sqrt(2.0)))                                    # exact GELU
            x = x + b["ls2"] * (y @ b["fc2_w"].T + b["fc2_b"])
        x = _ln(x, self.nw, self.nb)
        return np.concatenate([x[0], x[1:].mean(0)])

    def logits(self, crop_bgr):
        return self.hw @ self.features(crop_bgr) + self.hb

    def prob(self, crop_bgr):
        return 1.0 / (1.0 + np.exp(-self.logits(crop_bgr) / self.temp))


def PartClassifier(path="deploy/part_clf.npz"):
    """The deployed part-state classifier: the vision transformer when the file holds one, else the CNN."""
    z = np.load(path)
    arch = str(z["arch"]) if "arch" in z.files else "cnn"
    return ViTClassifier(path, z) if arch == "vit" else CNNClassifier(path, z)


def correctness(p, y):
    """P(the item is judged correctly) for probabilities p of label 1 and true labels y."""
    p = np.asarray(p, float); y = np.asarray(y)
    return np.where(y == 1, p, 1.0 - p)


def confidence(p):
    p = np.asarray(p, float)
    return np.maximum(p, 1.0 - p)


def fuse(probs):
    """Per item, the decision of the MOST CONFIDENT view. probs: list of (J,) arrays (survey and captured views).
    Returns (p_fused (J,), index of the deciding view per item)."""
    P = np.asarray(probs, float)
    if P.ndim == 1:
        P = P[None]
    i = np.argmax(np.abs(P - 0.5), 0)
    return P[i, np.arange(P.shape[1])], i


# ------------------------------------------------------------------ evidence prior (what a view is EXPECTED to show)
PRIOR_FEATURES = ["bias", "fp_part", "fp_edge", "fp_glare", "rel_x", "rel_y", "rel_r", "rel_x2", "rel_y2",
                  "roll", "pitch", "yaw", "pitch_relx", "pitch_rely", "roll_relx", "roll_rely", "level", "pose",
                  "pose_rel_r", "pose_tilt"]


def prior_features(vs, w_cells, fp_part, fp_edge, fp_glare, pose):
    """(K, F) features of every candidate view relative to the part (computable from the survey alone)."""
    w = np.asarray(w_cells, float); xy0 = vs.xy[:vs.P]
    c = (w[:, None] * xy0).sum(0) / max(w.sum(), 1e-6)
    rel = vs.xy - c[None]
    r = np.sqrt((rel ** 2).sum(1))
    rpy = vs.rot_tok * 1.0                                         # (K,3) already scaled
    lev = vs.level / max(vs.L - 1, 1)
    pz = float(pose) * np.ones(vs.K)
    tilt = np.abs(rpy).max(1)
    return np.stack([np.ones(vs.K), fp_part, fp_edge, fp_glare, rel[:, 0], rel[:, 1], r, rel[:, 0] ** 2, rel[:, 1] ** 2,
                     rpy[:, 0], rpy[:, 1], rpy[:, 2], rpy[:, 1] * rel[:, 0], rpy[:, 1] * rel[:, 1], rpy[:, 0] * rel[:, 0],
                     rpy[:, 0] * rel[:, 1], lev, pz, pz * r, pz * tilt], 1)


def prior_predict(beta, Xf):
    """Expected correctness of every item in every view: sigmoid(X beta). beta (J, F) -> (K, J)."""
    z = np.clip(Xf @ np.asarray(beta, float).T, -12, 12)
    return 1.0 / (1.0 + np.exp(-z))


def load_prior(path="deploy/task_prior.npz"):
    """(beta (J,F), learnable item indices)"""
    z = np.load(path)
    return z["beta"], [int(v) for v in z["learn"]]
