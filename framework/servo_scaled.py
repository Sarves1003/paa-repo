"""Closed-loop view verification at a LOWER height than the survey (your servo.py handles same-height views).

Below the survey height the image is magnified by s = H / (H + dz) (H = camera-to-table distance at the
survey pose, dz < 0). For a view centred on survey pixel u_t, every survey marker corner m0 must appear at
    uc + s (m0 - u_t)
and a camera move d shifts the content by s A d (A = survey-height image motion from grounding.py).
s is MEASURED in every frame from the apparent marker size (median over visible markers), so the view is
verified without knowing H exactly. Corrections are accumulated on the commanded pose (as in servo.py).
"""
import time
import numpy as np
import board as BD
from kinematics import fk
from ik import ik_hold


def marker_size(c):
    c = np.asarray(c, float)
    return float(np.mean([np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]))


def marker_scale(dets0, dets, allow_single=False):
    """Magnification vs the reference image from the distances between >= 2 marker centres (long baseline).
    A single marker's size is unreliable (edge blur, perspective: it gave 1.17-1.40 instead of ~2.06 on the robot),
    so it is used only when allow_single=True (height measurement at small descents)."""
    common = [k for k in dets if k in dets0]
    if not common:
        return None
    if len(common) >= 2:
        c0 = {k: np.asarray(dets0[k], float).mean(0) for k in common}
        c1 = {k: np.asarray(dets[k], float).mean(0) for k in common}
        r = [np.linalg.norm(c1[a] - c1[b]) / np.linalg.norm(c0[a] - c0[b])
             for i, a in enumerate(common) for b in common[i + 1:] if np.linalg.norm(c0[a] - c0[b]) > 30]
        if r:
            return float(np.median(r))
    if not allow_single:
        return None
    return float(np.median([marker_size(dets[k]) / max(marker_size(dets0[k]), 1e-6) for k in common]))


def view_error(dets0, dets, u_t, uc, s):
    common = [k for k in dets if k in dets0]
    if not common:
        return None, 0
    e = np.vstack([np.asarray(uc, float) + s * (np.asarray(dets0[k], float) - np.asarray(u_t, float))
                   - np.asarray(dets[k], float) for k in common])
    return np.median(e, axis=0), len(common)


_REG = {}


def part_mask(survey, G=None, pad=12):
    """Mask (255 = use) of the survey without the part: registration then relies on the table plane only,
    which avoids the parallax between the part's top and the table when the camera is lowered."""
    import cv2
    try:
        from perception import locate_part_scene
        part = locate_part_scene(survey, G)
    except Exception:
        part = None
    m = np.full(survey.shape[:2], 255, np.uint8)
    if part is not None:
        x, y, w, h = part["bbox_px"]
        cv2.rectangle(m, (int(x - pad), int(y - pad)), (int(x + w + pad), int(y + h + pad)), 0, -1)
    return m


def register(survey, frame, mask=None, min_inliers=20):
    """Similarity transform survey -> frame from ORB features + RANSAC (2x3), or None."""
    import cv2
    key = (id(survey), None if mask is None else id(mask))
    if key not in _REG:
        orb = cv2.ORB_create(2000)
        g = cv2.cvtColor(survey, cv2.COLOR_BGR2GRAY) if survey.ndim == 3 else survey
        if len(_REG) > 4:
            _REG.clear()
        _REG[key] = (orb, ) + orb.detectAndCompute(g, mask)
    orb, k0, d0 = _REG[key]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    k1, d1 = orb.detectAndCompute(g, None)
    if d0 is None or d1 is None or len(k0) < 10 or len(k1) < 10:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
    if len(m) < min_inliers:
        return None
    p0 = np.float32([k0[x.queryIdx].pt for x in m]); p1 = np.float32([k1[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inl is None or int(inl.sum()) < min_inliers:
        return None
    return M


def measure_view(frame, dets0, u_t, uc, s_guess, survey=None, mask=None):
    """Where is the wanted view? Returns (error px (2,), scale, n_markers, method) or (None, s, 0, 'none').
    Order: (1) >= 2 markers with a plausible magnification; (2) registration to the survey on the table plane
    (part masked), then on the whole image; (3) one marker with the model magnification; (4) nothing."""
    dets = BD.detect_markers(frame)
    common = [k for k in dets if k in dets0]
    if len(common) >= 2:
        s = marker_scale(dets0, dets)
        if s is not None and 0.75 < s / max(float(s_guess), 1e-6) < 1.33:
            e, n = view_error(dets0, dets, u_t, uc, s)
            return e, s, n, "markers"
    if survey is not None:
        for mk, how in ((mask, "registration (table)"), (None, "registration")):
            if mk is None and how == "registration (table)":
                continue
            M = register(survey, frame, mk)
            if M is None:
                continue
            sr = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
            if 0.75 < sr / max(float(s_guess), 1e-6) < 1.33:
                pred = M[:, :2] @ np.asarray(u_t, float) + M[:, 2]
                return np.asarray(uc, float) - pred, sr, len(common), how
    if len(common) == 1:
        e, n = view_error(dets0, dets, u_t, uc, float(s_guess))
        return e, float(s_guess), n, "marker (model scale)"
    return None, float(s_guess), 0, "none"


def _lateral(arm, q_cmd, d, R_ref, safe_move, speed, max_step_mm):
    """Move the commanded pose sideways by d (mm), orientation held, in steps <= max_step_mm."""
    n = max(1, int(np.ceil(np.linalg.norm(d) / max_step_mm)))
    for _ in range(n):
        p_cmd = fk(np.deg2rad(q_cmd))[:3, 3]
        q, pe, re, ok = ik_hold(p_cmd + np.r_[d / n, 0.0] / 1000.0, np.deg2rad(q_cmd), R_ref)
        if not ok or not safe_move(arm, np.degrees(q), speed)[0]:
            return q_cmd, False
        q_cmd = np.degrees(q)
    return q_cmd, True


def servo_to_view_scaled(arm, cam, A, uc, dets0, u_t, R_ref, safe_move, q_cmd_deg, s_guess,
                         speed=10, max_iter=10, tol_px=3.0, gain=0.7, max_step_mm=10.0, survey=None,
                         survey_mask=None, ff_px=None):
    """Verify/correct a view taken at any height.
    ff_px: calibrated descent drift for this view (survey px, = drift_px_per_mm * depth below the survey):
    applied first as a feed-forward lateral move (model correction), then the closed loop removes the rest.
    Returns dict with view_err_raw_mm (as planned), view_err_open_mm (after feed-forward), view_err_mm (after
    the closed loop), scale, verify (method of the last measurement), servo_iters, servo_s, frame, q_cmd."""
    q_cmd = np.asarray(q_cmd_deg, float)
    A = np.asarray(A, float)
    t0 = time.time()
    out = dict(servo_iters=0, servo_markers=0, view_err_raw_mm=np.nan, view_err_open_px=np.nan, view_err_open_mm=np.nan,
               view_err_px=np.nan, view_err_mm=np.nan, servo_s=0.0, q_cmd=q_cmd, scale=float(s_guess), verify="none",
               ff_mm=0.0)
    frame = cam.grab()
    e, s, n, how = measure_view(frame, dets0, u_t, uc, s_guess, survey, survey_mask)
    if e is not None:
        out["view_err_raw_mm"] = float(np.linalg.norm(np.linalg.inv(s * A) @ e))
    if ff_px is not None and np.linalg.norm(ff_px) > 0.5:
        d_ff = np.linalg.solve(A, np.asarray(ff_px, float))            # mm, cancels the calibrated drift
        q_cmd, ok = _lateral(arm, q_cmd, d_ff, R_ref, safe_move, speed, max_step_mm)
        out["ff_mm"] = float(np.linalg.norm(d_ff)) if ok else 0.0
        frame = cam.grab()
        e, s, n, how = measure_view(frame, dets0, u_t, uc, s_guess, survey, survey_mask)
    out.update(frame=frame, q_cmd=q_cmd, scale=s, verify=how, servo_markers=n)
    if e is None:
        return out                                     # nothing to verify against: keep the (feed-forward) pose
    Ainv = np.linalg.inv(s * A)
    out.update(view_err_open_px=float(np.linalg.norm(e)), view_err_open_mm=float(np.linalg.norm(Ainv @ e)))
    best = (float(np.linalg.norm(Ainv @ e)), q_cmd, frame, e, s, n, how)   # best verified pose so far
    stall = 0
    for it in range(max_iter):
        if np.linalg.norm(e) <= tol_px:
            break
        if stall >= 3:                                 # 3 corrections without improving: the arm is limit-cycling
            break                                      # (myCobot dead band / backlash on ~1 mm steps)
        d = gain * (Ainv @ e)
        if np.linalg.norm(d) > max_step_mm:
            d *= max_step_mm / np.linalg.norm(d)
        q_new, ok = _lateral(arm, q_cmd, d, R_ref, safe_move, speed, max_step_mm)
        if not ok:
            break
        q_cmd = q_new
        frame = cam.grab()
        e2, s2, n2, how2 = measure_view(frame, dets0, u_t, uc, s, survey, survey_mask)
        out["servo_iters"] = it + 1
        if e2 is None:
            break
        s = 0.5 * s + 0.5 * s2; Ainv = np.linalg.inv(s * A)
        e, n, how = e2, n2, how2
        err_mm = float(np.linalg.norm(Ainv @ e))
        if err_mm < best[0] - 0.05:
            best = (err_mm, q_cmd, frame, e, s, n, how); stall = 0
        else:
            stall += 1
    if float(np.linalg.norm(Ainv @ e)) > best[0] + 0.3:   # ended worse than an earlier pose: go back to the best one
        qb = best[1]
        if safe_move(arm, np.asarray(qb, float), speed)[0]:
            q_cmd = qb; frame = cam.grab()
            e2, s2, n2, how2 = measure_view(frame, dets0, u_t, uc, best[4], survey, survey_mask)
            if e2 is not None:
                e, s, n, how = e2, best[4], n2, how2 + ", returned to best pose"
            else:
                _, _, frame, e, s, n, how = best
        Ainv = np.linalg.inv(s * A)
    out.update(view_err_px=float(np.linalg.norm(e)), view_err_mm=float(np.linalg.norm(Ainv @ e)),
               servo_markers=n, servo_s=time.time() - t0, frame=frame, q_cmd=q_cmd, scale=s, verify=how)
    return out
