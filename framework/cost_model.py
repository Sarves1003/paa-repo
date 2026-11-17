#!/usr/bin/env python3
"""Kinematics-aware cost model for inspection ordering.

build: solve IK once for every grid cell (orientation held, or position-only),
       store joint targets q_k, reachability, and the travel-time matrix
           T_ij = t0 + max_m |q_i,m - q_j,m| / omega
       (time of a joint move is dominated by the joint that moves furthest).
fit:   identify t0 and omega from measured move times logged by run_inspection.py
       (linear regression of move time on max joint change), then rebuild T.

Why this matters: with the camera orientation held, the wrist has to compensate
base rotation, so Cartesian distance is a poor proxy for motion time. Orderings
that are short in Cartesian space (boustrophedon) need not be fast on the robot.

  python3 cost_model.py build --ik hold
  python3 cost_model.py build --ik position
  python3 cost_model.py fit hw_logs/inspection/*/log.csv --ik hold
"""
import argparse, glob, itertools, json, os
import numpy as np
import config as C
from kinematics import fk
from ik import ik, ik_hold
from hardware import load_locked_pose
from perception import cell_offsets_mm

DEFAULT_T0, DEFAULT_OMEGA = 0.8, 25.0   # s, deg/s  (placeholders until `fit` is run)


def path_for(ik_mode):
    return C.COST_MODEL_PATH.format(ik=ik_mode)


def time_params(ik_mode):
    """Fitted move-time model (cost_model.py fit) if available, else the placeholders."""
    p = os.path.join(C.DEPLOY_DIR, "time_model_%s.json" % ik_mode)
    if os.path.exists(p):
        j = json.load(open(p))
        return float(j["t0"]), float(j["omega"])
    return DEFAULT_T0, DEFAULT_OMEGA


def build(ik_mode="hold", dz_mm=0.0, t0=None, omega=None, dry=False, center_mm=(0.0, 0.0),
          max_pos_mm=1.0, max_rot_deg=1.0, save=True, verbose=True):
    """Joint targets, reachability and move-time matrix for a 5x5 grid centred at `center_mm`
    (relative to the locked pose; part tracking moves it). The arm starts every episode at the
    cell nearest the locked pose (`start`)."""
    if t0 is None or omega is None:
        t0, omega = time_params(ik_mode)
    lock = load_locked_pose(dry)
    q_lock = np.deg2rad(np.array(lock["angles_deg"], float))
    T0 = fk(q_lock); R_ref, p_lock = T0[:3, :3], T0[:3, 3]
    off = cell_offsets_mm(center=center_mm)
    n = len(off)
    Q = np.zeros((n, 6)); reach = np.zeros(n, bool)
    pos_res = np.zeros(n); rot_res = np.full(n, np.nan)
    for k, d in enumerate(off):
        tgt = p_lock + np.array([d[0], d[1], dz_mm]) / 1000.0
        if ik_mode == "hold":
            q, pe, re, ok = ik_hold(tgt, q_lock, R_ref)
            reach[k] = ok and pe * 1000 < max_pos_mm and re < max_rot_deg
            rot_res[k] = re
        else:
            q, pe = ik(tgt, q_lock)
            reach[k] = pe * 1000 < max_pos_mm
        Q[k], pos_res[k] = q, pe * 1000
    Qd = np.degrees(Q)
    reach &= np.max(np.abs(Qd - np.degrees(q_lock)), axis=1) < C.MAX_JOINT_STEP_DEG
    from fence import load_fence
    fence = load_fence()                     # file if present, otherwise the automatic box
    in_fence = np.ones(n, bool)
    for k, d in enumerate(off):              # planned target inside the margin AND solved pose passes hard check
        pk = fk(np.deg2rad(Qd[k]))[:3, 3] * 1000 - p_lock * 1000
        in_fence[k] = fence.contains(d) and fence.signed_dist(pk[:2]) >= fence.margin - C.FENCE_HARD_TOL_MM
    reach &= in_fence
    in_box = np.ones(n, bool)
    if os.path.exists(C.GROUNDING_PATH):           # drawn box, checked in the survey image
        from perception import load_grounding, offset_to_px
        Gg = load_grounding()
        if Gg.get("box_px"):
            import cv2
            poly = np.asarray(Gg["box_px"], np.float32)
            margin_px = C.FENCE_MARGIN_MM * Gg["px_per_mm"]
            for k, d in enumerate(off):
                u = offset_to_px(Gg, d)
                in_box[k] = cv2.pointPolygonTest(poly, (float(u[0]), float(u[1])), True) >= margin_px
    reach &= in_box
    dq = np.abs(Qd[:, None, :] - Qd[None, :, :]).max(-1)
    T = t0 + dq / omega
    np.fill_diagonal(T, t0)                  # re-imaging the current cell still costs a capture
    dist0 = np.linalg.norm(off, axis=1)
    cand = np.flatnonzero(reach)
    start = int(cand[np.argmin(dist0[cand])]) if len(cand) else int(np.argmin(dist0))
    Dc = np.linalg.norm(off[:, None] - off[None], axis=-1)
    M = dict(ik_mode=ik_mode, q_deg=Qd, reach=reach, in_fence=in_fence, in_box=in_box, pos_res_mm=pos_res, rot_res_deg=rot_res,
             dq_max_deg=dq, T=T, D_cart_mm=Dc, xy_mm=off, t0=t0, omega=omega, start=start,
             center_mm=np.asarray(center_mm, float), q_lock_deg=np.degrees(q_lock), p_lock_mm=p_lock * 1000,
             dz_mm=dz_mm, fence_mode=fence.meta.get("mode", "file"))
    if save:
        os.makedirs(C.DEPLOY_DIR, exist_ok=True)
        np.savez(path_for(ik_mode), **M)
    return M


def build_many(ik_mode, centers, **kw):
    """Cost models for several grid centres (used to train a policy that follows the part)."""
    return [build(ik_mode, center_mm=c, save=False, **kw) for c in centers]


def load(ik_mode="hold"):
    z = np.load(path_for(ik_mode), allow_pickle=True)
    return {k: (z[k].item() if z[k].shape == () else z[k]) for k in z.files}


def fit(log_paths, ik_mode="hold", dry=False):
    import pandas as pd          # only needed here (run on PC if pandas is not on the Jetson)
    df = pd.concat([pd.read_csv(p) for p in log_paths], ignore_index=True)
    df = df[(df.moved == 1) & df.move_s.notna() & df.dq_max_deg.notna() & (df.dq_max_deg > 0.5)]
    if len(df) < 10:
        raise SystemExit("need >= 10 logged moves to fit the time model")
    X = np.c_[np.ones(len(df)), df.dq_max_deg.values]
    (b0, b1), *_ = np.linalg.lstsq(X, df.move_s.values, rcond=None)
    pred = X @ np.array([b0, b1])
    r2 = 1 - ((df.move_s - pred) ** 2).sum() / ((df.move_s - df.move_s.mean()) ** 2).sum()
    t0, omega = float(b0), float(1.0 / b1)
    print(f"time model: t = {t0:.3f} s + max|dq| / {omega:.1f} deg/s   (R^2 = {r2:.3f}, n = {len(df)})")
    M = build(ik_mode, t0=t0, omega=omega, dry=dry)
    json.dump(dict(t0=t0, omega=omega, r2=float(r2), n=int(len(df))),
              open(os.path.join(C.DEPLOY_DIR, f"time_model_{ik_mode}.json"), "w"), indent=2)
    return M


def order_time(order, T, start=C.START_CELL):
    seq = [start] + list(order)
    return float(sum(T[a, b] for a, b in zip(seq[:-1], seq[1:])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "fit"])
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--ik", choices=["hold", "position"], default="hold")
    ap.add_argument("--dz-mm", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.mode == "build":
        M = build(a.ik, a.dz_mm, dry=a.dry_run)
    else:
        M = fit(sorted(set(itertools.chain.from_iterable(glob.glob(p) for p in a.logs))), a.ik, a.dry_run)
    r = M["reach"]
    print(f"[{a.ik}] reachable cells: {r.sum()}/{len(r)}; max solver residual {M['pos_res_mm'][r].max():.2f} mm; "
          f"mean pairwise move time {M['T'][np.ix_(r, r)].mean():.2f} s; saved {path_for(a.ik)}")
