"""Perception for eye-in-hand inspection.

Perception.observe(...)  your original tokenizer (kept unchanged for comparison runs).
                         It re-crops the part from whatever view the camera has, so
                         its grid is NOT tied to fixed workspace positions.

Grounded survey tokenizer (new):
  One survey image is taken at the locked pose. Using the calibrated image-motion
  matrix A (px per mm of camera translation, from grounding.py), the survey-image
  patch that the camera will see when centred on cell k is
        u_k = u_c - A d_k ,
  where d_k is the cell's workspace offset (mm) and u_c the optical centre (px).
  Tokens are computed from those patches once per episode, so cell k always
  refers to the same physical location.
"""
import json
import cv2
import numpy as np
import config as C

STATIC_DIM = 6   # [x_norm, y_norm, edge_density, grad_x, grad_y, observed]


# ------------------------------------------------------------------ shared ops
def edge_mask(gray):
    e = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), C.CANNY_LO, C.CANNY_HI)
    return cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def estimate_shift(img0, img1):
    """Content translation of img1 relative to img0 in px (dx, dy) and peak response."""
    a = to_gray(img0).astype(np.float32)
    b = to_gray(img1).astype(np.float32)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return np.array([dx, dy]), float(resp)


def patch(img, center, size):
    """Square patch (or None if it leaves the image)."""
    h, w = img.shape[:2]
    s = int(round(size / 2))
    x, y = int(round(center[0])), int(round(center[1]))
    if x - s < 0 or y - s < 0 or x + s > w or y + s > h:
        return None
    return img[y - s:y + s, x - s:x + s]


def roi_edge_density(frame, frac=0.35):
    """Edge density of the centre ROI of an image taken AT a cell (view-consistent signal)."""
    if frame is None:
        return float("nan")
    g = to_gray(frame)
    h, w = g.shape
    y0, x0 = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    return float(edge_mask(g)[y0:h - y0, x0:w - x0].mean()) / 255.0


def ncc(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


# ------------------------------------------------------------------ grid geometry
def cell_offsets_mm(n=C.GRID_N, half=C.GRID_HALF_MM, center=(0.0, 0.0)):
    """(n*n, 2) cell centres as workspace offsets from the LOCKED pose (mm), row-major.
    `center` is the grid centre relative to the locked pose (part tracking moves it)."""
    out = []
    for i in range(n):
        for j in range(n):
            out.append([center[0] + (j / (n - 1) - 0.5) * 2 * half, center[1] + (i / (n - 1) - 0.5) * 2 * half])
    return np.array(out)


def locate_part(frame, G=None, thresh=C.PART_THRESH, min_area=C.PART_MIN_AREA_PX, exclude=None, roi=None):
    """Find the part in a survey image: largest dark blob that does not touch the image border,
    is compact (aspect < 3.5) and solid (> 0.6). Rejects tape lines and table edges.
    Returns dict(centroid_px, bbox_px, area_px[, center_mm, size_mm]) or None."""
    g = cv2.GaussianBlur(to_gray(frame), (5, 5), 0).copy()
    for q in (exclude or []):                      # markers are dark squares: paint them white first
        qq = np.asarray(q, np.float32)
        c = qq.mean(0)
        cv2.fillConvexPoly(g, (c + 1.25 * (qq - c)).astype(np.int32), 255)
    if roi is not None:                            # keep only the inside of the drawn box (slightly enlarged)
        rr = np.asarray(roi, np.float32); c = rr.mean(0)
        keep = np.zeros_like(g); cv2.fillConvexPoly(keep, (c + 1.08 * (rr - c)).astype(np.int32), 255)
        g[keep == 0] = 255
    t = thresh if thresh else min(140.0, cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    _, m = cv2.threshold(g, t, 255, cv2.THRESH_BINARY_INV)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # fill specular highlights
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    found = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = found[0] if len(found) == 2 else found[1]                         # OpenCV 3/4 compatible
    H, W = g.shape
    best = None
    for c in cs:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if max(rw, rh) / max(min(rw, rh), 1) > 3.5:
            continue
        if a / max(cv2.contourArea(cv2.convexHull(c)), 1) < 0.6:
            continue
        if best is None or a > best[0]:
            best = (a, c, (x, y, w, h))
    if best is None:
        return None
    M = cv2.moments(best[1])
    u = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    out = dict(centroid_px=u.tolist(), bbox_px=list(best[2]), area_px=float(best[0]), thresh=float(t))
    if G is not None:
        out["center_mm"] = px_to_offset(G, u).tolist()
        if G.get("box_px"):
            out["box_center_mm"] = px_to_offset(G, np.asarray(G["box_px"], float).mean(0)).tolist()
        out["size_mm"] = (np.array(best[2][2:]) / G["px_per_mm"]).tolist()
    return out


def track_grid_center(part, fence, half=C.GRID_HALF_MM):
    """Grid centre on the part, clamped so every cell stays inside the fence (with margin)."""
    if part is None or "center_mm" not in part:
        return np.zeros(2), False
    if part.get("box_center_mm") is not None:     # drawn box known: centre the grid between part and box
        c = 0.5 * (np.array(part["center_mm"], float) + np.array(part["box_center_mm"], float))
        part = dict(part, center_mm=c.tolist())
    c = np.array(part["center_mm"], float)
    lo, hi = fence.poly.min(0) + fence.margin + half, fence.poly.max(0) - fence.margin - half
    lo, hi = np.minimum(lo, 0.0), np.maximum(hi, 0.0)       # the locked pose is always allowed
    cc = np.clip(c, lo, hi)
    return cc, bool(np.allclose(cc, c))


def load_grounding(path=C.GROUNDING_PATH):
    import os
    if not os.path.exists(path):
        raise SystemExit("%s is missing: create it first with\n    python3 grounding.py calibrate --mode track\n"
                         "(the arm makes small moves around the locked pose; keep the area clear)" % path)
    G = json.load(open(path))
    for k in ("A", "uc", "H0", "M", "c0"):
        if k in G:
            G[k] = np.array(G[k], float)
    return G


def offset_to_px(G, d):
    """Survey-image pixel that the camera centre sees after moving the camera by robot offset d (mm)."""
    d = np.atleast_2d(np.asarray(d, float))
    if G.get("mode") == "board":                    # exact: homography of the measured marker rectangle
        p = G["c0"][None] + d @ G["M"].T            # board point under the camera centre
        q = np.c_[p, np.ones(len(p))] @ np.linalg.inv(G["H0"]).T
        u = q[:, :2] / q[:, 2:3]
    else:                                           # affine image-shift model
        u = G["uc"][None] - d @ G["A"].T
    return u[0] if u.shape[0] == 1 else u


def px_to_offset(G, u):
    """Robot offset (mm) that brings the camera centre over survey-image pixel u."""
    u = np.atleast_2d(np.asarray(u, float))
    if G.get("mode") == "board":
        q = np.c_[u, np.ones(len(u))] @ G["H0"].T
        p = q[:, :2] / q[:, 2:3]
        d = (p - G["c0"][None]) @ np.linalg.inv(G["M"]).T
    else:
        d = (G["uc"][None] - u) @ np.linalg.inv(G["A"]).T
    return d[0] if d.shape[0] == 1 else d


# ------------------------------------------------------------------ grounded tokens
class Survey:
    """static (N*N, STATIC_DIM); density (N*N,) nan if unobserved; observed (N*N,) bool;
    centers_px (N*N, 2) predicted patch centres in the survey image; patch_px float."""
    def __init__(self, static, density, observed, centers_px, patch_px):
        self.static, self.density, self.observed = static, density, observed
        self.centers_px, self.patch_px = centers_px, patch_px


def grounded_survey(frame, G, n=C.GRID_N, half=C.GRID_HALF_MM, center=(0.0, 0.0)):
    gray = to_gray(frame)
    mask = edge_mask(gray)
    d = cell_offsets_mm(n, half, center)
    pitch = 2 * half / (n - 1)
    patch_px = float(G["px_per_mm"] * pitch)
    centers = offset_to_px(G, d)
    gx_all = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_all = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    S = np.zeros((n * n, STATIC_DIM), np.float32)
    dens = np.full(n * n, np.nan)
    obs = np.zeros(n * n, bool)
    for k in range(n * n):
        S[k, 0], S[k, 1] = (d[k] - np.asarray(center)) / half     # grid-relative, in [-1, 1]
        pm = patch(mask, centers[k], patch_px)
        if pm is None:
            continue
        gx, gy = patch(gx_all, centers[k], patch_px), patch(gy_all, centers[k], patch_px)
        mag = np.sqrt(gx * gx + gy * gy) + 1e-6
        dens[k] = float(pm.mean()) / 255.0
        obs[k] = True
        S[k, 2] = dens[k]
        S[k, 3] = float((gx / mag).mean())
        S[k, 4] = float((gy / mag).mean())
        S[k, 5] = 1.0
    return Survey(S, dens, obs, centers, patch_px)


def draw_survey(frame, sv, path):
    img = frame.copy()
    s = int(sv.patch_px / 2)
    for k, c in enumerate(sv.centers_px):
        x, y = int(c[0]), int(c[1])
        col = (0, 160, 0) if sv.observed[k] else (0, 0, 255)
        cv2.rectangle(img, (x - s, y - s), (x + s, y + s), col, 1)
        txt = f"{k}:{sv.density[k]:.2f}" if sv.observed[k] else f"{k}:n/o"
        cv2.putText(img, txt, (x - s + 2, y - s + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1)
    cv2.imwrite(path, img)


# ------------------------------------------------------------------ original tokenizer (unchanged)
class ObjectGeometry:
    def __init__(self, mask, tokens, grid_n, center_m, extent_m, reachable):
        self.mask, self.tokens, self.grid_n = mask, tokens, grid_n
        self.center_m, self.extent_m, self.reachable = center_m, extent_m, reachable


class Perception:
    def __init__(self, intr_npz=None, rect_px=240):
        self.rect = int(rect_px)

    def observe(self, bgr, grid_n=5, fixture_half_m=0.04, standoff_m=0.15):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        lc = max(contours, key=cv2.contourArea)
        if cv2.contourArea(lc) < 500:
            return None
        x, y, w, h = cv2.boundingRect(lc)
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
        top = cv2.resize(gray[y0:y1, x0:x1], (self.rect, self.rect))
        mask = edge_mask(top)
        N = int(grid_n); cell = self.rect // N
        T = np.zeros((N * N, 8), np.float32); reach = np.zeros((N, N), bool)
        for i in range(N):
            for j in range(N):
                yc, xc = i * cell, j * cell
                ed = float(mask[yc:yc + cell, xc:xc + cell].mean()) / 255.0
                reach[i, j] = ed > 0.02
                gx = cv2.Sobel(top[yc:yc + cell, xc:xc + cell], cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(top[yc:yc + cell, xc:xc + cell], cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(gx * gx + gy * gy) + 1e-6
                nx, ny, nz = -float((gx / mag).mean()), -float((gy / mag).mean()), 1.0
                nrm = np.sqrt(nx * nx + ny * ny + nz * nz)
                T[i * N + j] = [j / (N - 1), i / (N - 1), nx / nrm, ny / nrm, nz / nrm, ed, 0.0, standoff_m]
        return ObjectGeometry(mask, T, N, np.zeros(2), np.array([fixture_half_m] * 2), reach)


def locate_part_scene(frame, G=None):
    """Part detection that ignores the markers and, when known, stays inside the drawn box."""
    import board as BD
    dets = BD.detect_markers(frame)
    roi = G.get("box_px") if G else None
    return locate_part(frame, G, exclude=list(dets.values()), roi=roi)


# >>> table_grid override (added by table_grid.py) >>>
_cell_offsets_robot_grid = cell_offsets_mm
def cell_offsets_mm(*args, **kw):
    """Table-anchored grid from deploy/table_grid.json if present, else the original robot-space grid.
    A centre offset (part tracking) is added in robot mm, as before."""
    import json as _j, os as _o, numpy as _n, config as _C
    _p = _o.path.join(_C.DEPLOY_DIR, "table_grid.json")
    if not _o.path.exists(_p):
        return _cell_offsets_robot_grid(*args, **kw)
    base = _n.array(_j.load(open(_p))["offsets_mm"], float)
    c = args[0] if args else (next(iter(kw.values())) if kw else None)
    if c is not None:
        base = base + _n.asarray(c, float)[:2][None]
    return base
# <<< table_grid override <<<
