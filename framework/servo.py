"""Closed-loop viewpoint correction (image-based visual servoing, translation only).

Why: on the myCobot 280 each joint lands 0.5-1.4 deg from its command (fk_check). At ~200 mm
stand-off, 1 deg of camera tilt moves the view ~3.5 mm on the table, more than the arm's position
error. Open-loop kinematics therefore cannot place the view precisely. After every move the
camera measures where the markers are, compares them with where they must be for the wanted view,
and the arm makes small corrective moves (orientation held) until the view error is below a
tolerance. The view is then defined by the image, not by the imperfect kinematics.

Model: the image content shifts by s = A @ d (px) when the camera moves by d (mm). For a wanted
view whose centre is survey pixel u_t, every survey marker point m0 should appear at
m0 + (u_c - u_t). Error e = desired - measured (px); correction d = gain * A^-1 e.
"""
import time
import numpy as np
import board as BD
from kinematics import fk
from ik import ik_hold


def view_error(dets0, dets, u_target, uc):
    """(error px (2,), n common markers): how far the content is from the wanted view."""
    common = [k for k in dets if k in dets0]
    if not common:
        return None, 0
    shift = np.asarray(uc, float) - np.asarray(u_target, float)
    e = np.vstack([np.asarray(dets0[k]) + shift - np.asarray(dets[k]) for k in common])
    return np.median(e, axis=0), len(common)


def servo_to_view(arm, cam, A, uc, dets0, u_target, R_ref, safe_move, q_cmd_deg, speed=10,
                  max_iter=5, tol_px=2.5, gain=0.8, max_step_mm=12.0):
    """Correct the current view until the content matches the wanted view (u_target at the image
    centre). Corrections accumulate on the COMMANDED pose (q_cmd_deg), not the measured one: the arm's
    repeatable joint error then stays a constant bias that the image feedback cancels (computing from
    the measured pose would add that bias again at every step and the loop would stall off target).
    Returns dict with errors before/after (px and mm), iterations, time, final frame, final command."""
    q_cmd = np.asarray(q_cmd_deg, float)
    Ainv = np.linalg.inv(np.asarray(A, float))
    t0 = time.time()
    frame = cam.grab()
    e, n = view_error(dets0, BD.detect_markers(frame), u_target, uc)
    out = dict(servo_iters=0, servo_markers=n, view_err_open_px=np.nan, view_err_open_mm=np.nan,
               view_err_px=np.nan, view_err_mm=np.nan, servo_s=0.0, frame=frame, q_cmd=q_cmd)
    if e is None:
        return out                                     # no marker in view: keep the open-loop pose
    out.update(view_err_open_px=float(np.linalg.norm(e)), view_err_open_mm=float(np.linalg.norm(Ainv @ e)))
    for it in range(max_iter):
        if np.linalg.norm(e) <= tol_px:
            break
        d = gain * (Ainv @ e)
        nd = np.linalg.norm(d)
        if nd > max_step_mm:
            d *= max_step_mm / nd
        p_cmd = fk(np.deg2rad(q_cmd))[:3, 3]
        q, pe, re, ok = ik_hold(p_cmd + np.r_[d, 0.0] / 1000.0, np.deg2rad(q_cmd), R_ref)
        if not ok:
            break
        moved, _, _ = safe_move(arm, np.degrees(q), speed)
        if not moved:
            break
        q_cmd = np.degrees(q)
        frame = cam.grab()
        e2, n2 = view_error(dets0, BD.detect_markers(frame), u_target, uc)
        out["servo_iters"] = it + 1
        if e2 is None:
            break
        e, n = e2, n2
    out.update(view_err_px=float(np.linalg.norm(e)), view_err_mm=float(np.linalg.norm(Ainv @ e)),
               servo_markers=n, servo_s=time.time() - t0, frame=frame, q_cmd=q_cmd)
    return out
