"""Four corner markers on the table as a measured rectangle ("board").

Why: on mostly white paper, image-shift (phase correlation) grounding is unreliable, and the
camera views the table obliquely (~33 deg). Four dark squares at the corners of a rectangle whose
size you measure with a ruler give an exact image -> table-plane mapping (homography) at every
pose, with no ArUco decoding, no intrinsics, and no hand-eye matrix.

Detection: cv2.aruco if available (any common dictionary), otherwise plain OpenCV contours:
dark, square, 4-cornered blobs of similar size. Markers need a white border (>= 5 mm) and must
NOT touch other dark things (tape lines, the part).

Frames
  board frame (mm): TL marker centre = (0,0), TR = (W,0), BR = (W,H), BL = (0,H), with TL/TR/BR/BL
                    as seen in the survey image. W, H = centre-to-centre distances (config).
  robot offsets (mm): flange x,y relative to the locked pose (what the planners use).
Camera centre on the table: c = H(u_c), the board point under the image centre. Translating
the camera by d (orientation held) moves c by M d (M: 2x2, ~rotation, |scale| ~ 1). Fitting M
from a few known moves gives the robot <-> board mapping, and its scale/residual is an
EXTERNAL check of the robot's motion (independent of the servo encoders).
"""
import numpy as np
import cv2
import config as C


# ------------------------------------------------------------------ detection
def _order_tl_tr_br_bl(pts):
    pts = np.asarray(pts, float)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    p = pts[np.argsort(ang)]                      # angle order (clockwise in image coordinates)
    k = np.argmin(p.sum(1))                       # top-left = smallest x + y
    return np.roll(p, -k, axis=0)                 # TL, TR, BR, BL


def _order_corners(c):
    """Four marker corners in TL, TR, BR, BL image order."""
    return _order_tl_tr_br_bl(np.asarray(c, float).reshape(4, 2))


def _aruco_params(relaxed=False):
    """Detector settings that work with OpenCV 4.1-4.13. minDistanceToBorder = 0 is essential:
    a marker near the image edge (your top-left one) is otherwise silently ignored."""
    p = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "ArucoDetector") else cv2.aruco.DetectorParameters_create()
    for k, v in (("minDistanceToBorder", 0), ("adaptiveThreshWinSizeMax", 53 if relaxed else 23),
                 ("errorCorrectionRate", 0.8 if relaxed else 0.6),
                 ("polygonalApproxAccuracyRate", 0.05 if relaxed else 0.03)):
        try:
            setattr(p, k, v)
        except Exception:
            pass
    return p


def _aruco_once(gray, dic, relaxed):
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(dic, _aruco_params(relaxed)).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dic, parameters=_aruco_params(relaxed))
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(float) for i, c in zip(ids.ravel(), corners)}


def _aruco_detect(gray):
    """{id: corners(4,2)} for the configured marker ids (DICT_4X4_50 first: your markers are 0-3).
    Pass 1: standard settings + markers allowed at the image border. If not all expected ids are
    found: pass 2 with more tolerant settings, pass 3 on a 1.5x enlarged image. Results are merged."""
    if not hasattr(cv2, "aruco"):
        return {}
    want = set(getattr(C, "MARKER_IDS", [0, 1, 2, 3]))
    for name in ("DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_ARUCO_ORIGINAL"):
        if not hasattr(cv2.aruco, name):
            continue
        d = getattr(cv2.aruco, name)
        dic = cv2.aruco.getPredefinedDictionary(d) if hasattr(cv2.aruco, "getPredefinedDictionary") \
            else cv2.aruco.Dictionary_get(d)
        found = {}
        for relaxed, scale in ((False, 1.0), (True, 1.0), (False, 1.5)):
            g = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for k, c in _aruco_once(g, dic, relaxed).items():
                if k in want and k not in found:
                    found[k] = c / scale
            if want <= set(found):
                break
        if found:
            return {k: _order_corners(c) for k, c in found.items()}
    return {}


def _contour_detect(gray, sat=None):
    """Dark squares with a bright border (no ArUco module needed): list of (area, corners(4,2)).
    sat: HSV saturation; markers are black/white, so coloured squares (icons) are rejected."""
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    bs = int(max(H, W) / 12) | 1
    m = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, 10)
    t_glob = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    m[g > min(t_glob, 150)] = 0          # locally dark AND truly dark: no grey halo next to white borders
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    found = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = found[0] if len(found) == 2 else found[1]
    cand = []
    for c in cs:
        a = cv2.contourArea(c)
        if a < 120 or a > 0.05 * H * W:
            continue
        poly = cv2.approxPolyDP(c, 0.06 * cv2.arcLength(c, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if max(rw, rh) / max(min(rw, rh), 1) > 1.6 or a / max(rw * rh, 1) < 0.75:
            continue
        x, y, w, h = cv2.boundingRect(c)
        pad = int(0.3 * max(w, h))
        if x - pad < 0 or y - pad < 0 or x + w + pad > W or y + h + pad > H:
            continue
        ring = np.zeros_like(gray); inner = np.zeros_like(gray)
        cv2.rectangle(ring, (x - pad, y - pad), (x + w + pad, y + h + pad), 255, -1)
        cv2.drawContours(inner, [c], -1, 255, -1)
        ring[inner > 0] = 0
        if g[ring > 0].mean() < g[inner > 0].mean() + 40:      # needs a white border (quiet zone)
            continue
        if sat is not None and sat[inner > 0].mean() > 70:      # coloured -> not a black/white marker
            continue
        cand.append((a, _order_corners(poly.reshape(4, 2))))
    return cand


def detect_markers(img, predicted=None, match_px=45.0):
    """{key: corners(4,2) in TL,TR,BR,BL image order}.
    ArUco ids when the aruco module works. Without it (contour fallback), squares are named by
    matching to `predicted` {key: expected centre px} when given; otherwise all four must be visible
    and are named 'c0'..'c3' in TL, TR, BR, BL order."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = _aruco_detect(gray)
    if d:
        return d
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1] if img.ndim == 3 else None
    cand = _contour_detect(gray, sat)
    if predicted:
        out = {}
        for k, pc in predicted.items():
            best = min(cand, key=lambda t: np.linalg.norm(t[1].mean(0) - pc), default=None)
            if best is not None and np.linalg.norm(best[1].mean(0) - pc) < match_px:
                out[k] = best[1]
        return out
    if len(cand) < 4:
        return {}
    import itertools
    cand.sort(key=lambda t: -t[0])
    areas = np.array([c[0] for c in cand]); cents = np.array([c[1].mean(0) for c in cand])
    best, best_area = None, -1
    for idx in itertools.combinations(range(min(len(cand), 8)), 4):
        if areas[list(idx)].max() / areas[list(idx)].min() > 2.5:
            continue
        q = _order_tl_tr_br_bl(cents[list(idx)])
        side = lambda a, b: np.linalg.norm(q[a] - q[b])
        r1, r2 = side(0, 1) / max(side(3, 2), 1e-6), side(0, 3) / max(side(1, 2), 1e-6)
        if not (0.6 < r1 < 1.67 and 0.6 < r2 < 1.67):          # the four must form a rough rectangle
            continue
        ar = cv2.contourArea(q.astype(np.float32))
        if ar > best_area:
            best, best_area = [idx[int(np.argmin(np.linalg.norm(cents[list(idx)] - p, axis=1)))] for p in q], ar
    return {} if best is None else {"c%d" % k: cand[best[k]][1] for k in range(4)}


def detect_markers_multi(get_frame, tries=6, want=4):
    """Stationary camera: grab up to `tries` frames, detect markers in each, merge by id (median
    corners). Stops as soon as `want` markers are found. Returns (last frame, merged detections)."""
    seen, frame = {}, None
    for _ in range(tries):
        frame = get_frame()
        if frame is None:
            continue
        for k, c in detect_markers(frame).items():
            seen.setdefault(k, []).append(c)
        if len(seen) >= want:
            break
    return frame, {k: np.median(np.stack(v), axis=0) for k, v in seen.items()}


def marker_shift(dets0, dets):
    """Image displacement of the markers seen in BOTH detections (all corners of common ids).
    Returns (mean shift px (2,), number of common markers) or (None, 0)."""
    common = [k for k in dets if k in dets0]
    if not common:
        return None, 0
    d = np.vstack([np.asarray(dets[k]) - np.asarray(dets0[k]) for k in common])
    return np.median(d, axis=0), len(common)


def centres(dets):
    return {k: v.mean(0) for k, v in dets.items()}


# ------------------------------------------------------------------ geometry
def board_corners_mm(w=None, h=None):
    w = C.BOARD_W_MM if w is None else w
    h = C.BOARD_H_MM if h is None else h
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], float)


def homography(img_pts, w=None, h=None):
    """Image (px) -> board (mm)."""
    Hm, _ = cv2.findHomography(np.asarray(img_pts, np.float32), board_corners_mm(w, h).astype(np.float32))
    return Hm


def apply_h(Hm, pts):
    p = np.c_[np.atleast_2d(pts), np.ones(len(np.atleast_2d(pts)))] @ Hm.T
    return p[:, :2] / p[:, 2:3]


def corner_map(dets, H0=None):
    """At calibration (all 4 visible): marker key -> board corner index (TL=0..BL=3), and, if H0 is given,
    the table positions (mm) of the marker's own 4 corners. With those, ANY single visible marker
    later gives a full homography (4 point pairs), so outer cells stay measurable."""
    keys = list(dets)
    cen = centres(dets)
    q = _order_tl_tr_br_bl([cen[k] for k in keys])
    out = {}
    for k in keys:
        e = dict(corner=int(np.argmin(np.linalg.norm(q - cen[k], axis=1))))
        if H0 is not None:
            e["pts_mm"] = apply_h(H0, dets[k]).tolist()
            e["survey_centre_px"] = cen[k].tolist()
        out[str(k)] = e
    return out


def camera_point(img, G, d_cmd=None):
    """Board point (mm) under the image centre at the current pose, from every visible known marker
    (each contributes its 4 corners). Returns (point or None, number of markers used).
    d_cmd: commanded robot offset, used to name squares when ArUco is unavailable."""
    cmap = G.get("corner_map", {})
    pred = None
    if d_cmd is not None and "M" in G:            # used only if ArUco decodes nothing
        from perception import offset_to_px
        shift = np.asarray(G["uc"], float) - offset_to_px(G, d_cmd)       # approx. image content shift
        pred = {k: np.asarray(v["survey_centre_px"]) + shift for k, v in cmap.items() if "survey_centre_px" in v}
    dets = detect_markers(img, pred)
    src, dst = [], []
    for k, c in dets.items():
        e = cmap.get(str(k))
        if e and "pts_mm" in e:
            src += list(c); dst += list(e["pts_mm"])
    if len(src) < 4:
        return None, len(src) // 4
    Hm, _ = cv2.findHomography(np.float32(src), np.float32(dst), 0)
    return apply_h(Hm, np.asarray(G["uc"], float))[0], len(src) // 4


def fit_motion(deltas_robot, deltas_board):
    """Least-squares 2x2 M with  board_delta = M @ robot_delta.  Returns M, scale, RMSE (mm)."""
    D = np.asarray(deltas_robot, float); B = np.asarray(deltas_board, float)
    X, *_ = np.linalg.lstsq(D, B, rcond=None)
    M = X.T
    res = B - D @ X
    rmse = float(np.sqrt((res ** 2).sum(1).mean()))
    scale = float(np.sqrt(abs(np.linalg.det(M))))
    return M, scale, rmse, res
