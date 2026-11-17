"""Damped-least-squares inverse kinematics.

ik(target, q0)              position only (backward compatible: returns q, pos_err_m).
ik_hold(target, q0, R_ref)  position + full orientation held at R_ref (normally the
                            locked survey orientation) -> constant camera viewing direction
                            without needing to know which flange axis the camera uses.

Tolerances are 0.5 mm / 0.5 deg. The returned residual is a solver quantity and must
never be reported as positioning accuracy (use encoder read-back or an external sensor).
"""
import numpy as np
from kinematics import fk, clamp_joints


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _rot_err(R, Rg):
    return 0.5 * sum(np.cross(R[:, k], Rg[:, k]) for k in range(3))


def _solve(target, q0, R_ref, iters, tol_pos, tol_rot_deg, damping, w_rot):
    q = np.asarray(q0, float).copy()
    tx = np.asarray(target, float)
    hold = R_ref is not None
    m = 6 if hold else 3
    W = np.r_[np.ones(3), np.full(3, w_rot)] if hold else np.ones(3)

    def task(qq):
        T = fk(qq)
        e = tx - T[:3, 3]
        return np.r_[e, _rot_err(T[:3, :3], R_ref)] if hold else e

    converged = False
    for _ in range(iters):
        e = task(q)
        T = fk(q)
        pe = np.linalg.norm(tx - T[:3, 3])
        re = rot_angle_deg(T[:3, :3], R_ref) if hold else 0.0
        if pe < tol_pos and re < tol_rot_deg:
            converged = True
            break
        J = np.zeros((m, 6)); eps = 1e-5
        for k in range(6):
            qk = q.copy(); qk[k] += eps
            J[:, k] = -(task(qk) - e) / eps
        Jw, ew = J * W[:, None], e * W
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + damping**2 * np.eye(m), ew)
        q = clamp_joints(q + np.clip(dq, -0.2, 0.2))
    T = fk(q)
    pe = float(np.linalg.norm(tx - T[:3, 3]))
    re = rot_angle_deg(T[:3, :3], R_ref) if hold else float("nan")
    return q, pe, re, converged


def ik(target_xyz, q0, iters=300, tol=5e-4, damping=0.01):
    q, pe, _, _ = _solve(target_xyz, q0, None, iters, tol, 180.0, damping, 0.0)
    return q, pe


def ik_hold(target_xyz, q0, R_ref, iters=400, tol_pos=5e-4, tol_rot_deg=0.5,
            damping=0.01, w_rot=0.05):
    return _solve(target_xyz, q0, R_ref, iters, tol_pos, tol_rot_deg, damping, w_rot)
