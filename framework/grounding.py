#!/usr/bin/env python3
"""Image <-> workspace grounding for the eye-in-hand camera.

Mode "board" (recommended; config.GROUNDING_MODE = "board"):
  Four dark square markers at the corners of a rectangle on the table, centre-to-centre
  distances BOARD_W_MM x BOARD_H_MM measured with a ruler. At every pose the four markers give
  a homography image -> table (mm), so the oblique camera view is handled exactly.
  calibrate: survey image at the locked pose + small known moves (orientation held). The board
             point under the image centre moves by M @ d for a robot move d; M is fitted.
             Its scale (ideally 1.00) and residual (mm) are an EXTERNAL check of the robot's
             motion against the printed markers. The marker rectangle also becomes the
             workspace limit: the camera centre is never sent outside it.
  validate:  visit every grid cell; the markers measure where the camera centre really is.
             Error = |measured - expected| on the table, in mm, independent of the encoders.

Mode "shift" (fallback, no markers): phase correlation of image shifts. Needs a textured
  scene; matches with low correlation peak or NCC are now discarded instead of reported.

  python3 grounding.py calibrate [--dry-run]
  python3 grounding.py validate  [--dry-run]
"""
import argparse, csv, json, os, time
import numpy as np
import cv2
import config as C
from kinematics import fk
from ik import ik_hold
from hardware import make_arm, make_camera, safe_move, wait_settled, load_locked_pose
from perception import (estimate_shift, patch, ncc, cell_offsets_mm, load_grounding, to_gray,
                        offset_to_px, px_to_offset)
import board as BD

MIN_PEAK, MIN_NCC = 0.3, 0.5          # shift mode: reliability gates


def move_offset(arm, q_lock, R_ref, p_lock_mm, dxy_mm, dz_mm=0.0):
    tgt = (p_lock_mm + np.array([dxy_mm[0], dxy_mm[1], dz_mm])) / 1000.0
    q, pe, re, ok = ik_hold(tgt, np.deg2rad(q_lock), R_ref)
    if not ok:
        return False, None
    moved, _, q_meas = safe_move(arm, np.degrees(q))
    return moved, q_meas


def optical_centre(w, h):
    if os.path.exists(C.INTRINSICS_PATH):
        K = np.load(C.INTRINSICS_PATH)["mtx"]
        return [float(K[0, 2]), float(K[1, 2])]
    return [w / 2.0, h / 2.0]


def board_debug(img, path):
    """Save what the square detector sees, to fix lighting/placement when detection fails."""
    gray = to_gray(img)
    _, m = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(path, np.hstack([gray, m]))


def setup(a):
    lock = load_locked_pose(a.dry_run)
    q_lock = np.array(lock["angles_deg"], float)
    T0 = fk(np.deg2rad(q_lock))
    arm = make_arm(a.dry_run, q_lock); arm.send_deg(q_lock); wait_settled(arm)
    cam = make_camera(a.dry_run, arm, a.ref_img)
    out = os.path.join(C.LOGS_DIR, "grounding"); os.makedirs(out, exist_ok=True)
    return q_lock, T0[:3, :3], T0[:3, 3] * 1000, arm, cam, out


def deltas(s):
    return [(s, 0), (-s, 0), (0, s), (0, -s), (s, s), (-s, -s), (s, -s), (-s, s)]


# ------------------------------------------------------------------ board mode
def _box_offsets(c0, M):
    """Drawn box (measure_board.py -> deploy/box.json) in robot offsets, or None."""
    p = os.path.join(C.DEPLOY_DIR, "box.json")
    if not os.path.exists(p):
        return None
    bb = np.array(json.load(open(p))["box_in_board_mm"], float)
    return ((bb - c0[None]) @ np.linalg.inv(M).T).tolist()


def calibrate_board(a):
    W, Hh = a.board_w, a.board_h
    if W <= 0 or Hh <= 0:
        raise SystemExit("set BOARD_W_MM and BOARD_H_MM in config.py (marker centre-to-centre, measured with a ruler)")
    q_lock, R_ref, p_lock, arm, cam, out = setup(a)
    try:
        I0 = cam.grab(); cv2.imwrite(os.path.join(out, "calib_survey.jpg"), I0)
        h, w = I0.shape[:2]; uc = np.array(optical_centre(w, h))
        I0, dets0 = BD.detect_markers_multi(cam.grab, tries=6)
        if len(dets0) < 4:
            board_debug(I0, os.path.join(out, "board_debug.jpg"))
            raise SystemExit(f"only {len(dets0)} of 4 corner markers found in the survey image; see "
                             f"{out}/board_debug.jpg (markers need a white border and must not touch tape/part)")
        cm0 = BD.corner_map(dets0)
        q0 = np.zeros((4, 2))
        for k, c in dets0.items():
            q0[cm0[str(k)]["corner"]] = c.mean(0)
        H0 = BD.homography(q0, W, Hh); c0 = BD.apply_h(H0, uc)[0]
        cmap = BD.corner_map(dets0, H0)                 # + each marker's own corners on the table
        Gb = dict(board_w_mm=W, board_h_mm=Hh, uc=uc, corner_map=cmap)
        D, B, E = [], [], []
        steps = [a.step_mm] * a.reps + [a.step_mm / 2.0]          # half-size moves if markers leave the view
        for k_rep, st in enumerate(steps):
            if k_rep == len(steps) - 1 and len(D) >= 6 and np.linalg.matrix_rank(np.array(D, float)) == 2:
                break
            for d in deltas(st):
                arm.send_deg(q_lock); wait_settled(arm)
                moved, q_meas = move_offset(arm, q_lock, R_ref, p_lock, d)
                if not moved:
                    print(f"  move {d}: not possible (IK/limits), skipped"); continue
                ck, nk = BD.camera_point(cam.grab(), Gb, None)
                if ck is None:
                    print(f"  move {d}: markers not all visible, skipped"); continue
                enc = fk(np.deg2rad(q_meas))[:3, 3] * 1000 - p_lock      # encoder-measured move
                D.append(d); B.append(ck - c0); E.append(enc[:2])
                print(f"  camera moved {d} mm -> camera point on table moved ({ck[0]-c0[0]:+.1f}, {ck[1]-c0[1]:+.1f}) mm")
        arm.send_deg(q_lock); wait_settled(arm)
    finally:
        cam.release(); arm.close()
    if len(D) < 4 or np.linalg.matrix_rank(np.array(D, float)) < 2:
        raise SystemExit("not enough usable moves in BOTH x and y: markers left the camera view. Place the markers "
                         "closer to the image centre (look at hw_logs/grounding/calib_survey.jpg) or use --step-mm 8")
    M, scale, rmse, res = BD.fit_motion(D, B)
    Me, scale_e, rmse_e, _ = BD.fit_motion(E, B)
    corners_off = (BD.board_corners_mm(W, Hh) - c0[None]) @ np.linalg.inv(M).T   # marker centres in robot offsets
    G = dict(mode="board", uc=uc.tolist(), image_wh=[w, h], board_w_mm=W, board_h_mm=Hh,
             H0=H0.tolist(), c0=c0.tolist(), M=M.tolist(), step_mm=a.step_mm, n_used=len(D),
             motion_scale=scale, motion_rmse_mm=rmse, motion_scale_encoder=scale_e, motion_rmse_encoder_mm=rmse_e,
             fence_offsets_mm=corners_off.tolist(), markers_px=q0.tolist(), corner_map=cmap,
             box_offsets_mm=_box_offsets(c0, M),
             p_lock_mm=p_lock.tolist(), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    Ga = dict(G, H0=H0, c0=c0, M=M, uc=uc)          # array version for the mapping functions
    eps = 1.0
    G["px_per_mm"] = Ga["px_per_mm"] = float(np.mean(
        [np.linalg.norm(offset_to_px(Ga, (eps, 0)) - offset_to_px(Ga, (0, 0))),
         np.linalg.norm(offset_to_px(Ga, (0, eps)) - offset_to_px(Ga, (0, 0)))]) / eps)
    os.makedirs(C.DEPLOY_DIR, exist_ok=True)
    json.dump(G, open(C.GROUNDING_PATH, "w"), indent=2)
    _report(Ga, I0, out)


def _report(G, I0, out):
    h, w = I0.shape[:2]
    cen = offset_to_px(G, cell_offsets_mm())
    half_patch = G["px_per_mm"] * 2 * C.GRID_HALF_MM / (C.GRID_N - 1) / 2
    inside = ((cen[:, 0] - half_patch >= 0) & (cen[:, 1] - half_patch >= 0) &
              (cen[:, 0] + half_patch <= w) & (cen[:, 1] + half_patch <= h))
    vis = I0.copy()
    if G.get("mode") == "board":
        print(f"\nrobot -> table motion: scale {G['motion_scale']:.3f} (1.000 = exact), residual RMSE "
              f"{G['motion_rmse_mm']:.2f} mm over {G['n_used']} moves   [external check of the arm's motion]")
        print(f"  using encoder-measured moves instead of commands: scale {G['motion_scale_encoder']:.3f}, "
              f"RMSE {G['motion_rmse_encoder_mm']:.2f} mm")
        if abs(G["motion_scale"] - 1) > 0.1:
            print("  WARNING: scale far from 1 -> check BOARD_W_MM / BOARD_H_MM (centre-to-centre, in mm)")
        poly = np.array([offset_to_px(G, d) for d in G["fence_offsets_mm"]], np.int32)
        cv2.polylines(vis, [poly], True, (255, 120, 0), 2)
        for p in G["markers_px"]:
            cv2.circle(vis, tuple(int(v) for v in p), 6, (0, 200, 255), 2)
        print(f"marker rectangle in robot offsets (mm): {np.round(G['fence_offsets_mm'], 1).tolist()}")
        if G.get("box_offsets_mm"):
            pb = np.array([offset_to_px(G, d) for d in G["box_offsets_mm"]], np.int32)
            cv2.polylines(vis, [pb], True, (255, 0, 255), 2)
            print(f"drawn box in robot offsets (mm):         {np.round(G['box_offsets_mm'], 1).tolist()}"
                  f"  -> workspace limit")
    else:
        print(f"\nA = {np.round(G['A'], 3).tolist()} px/mm ({G['px_per_mm']:.2f} px/mm, fit RMSE {G['fit_rmse_px']:.2f} px)")
    for k, c in enumerate(cen):
        cv2.circle(vis, tuple(int(v) for v in c), 4, (0, 170, 0) if inside[k] else (0, 0, 255), -1)
    cv2.imwrite(os.path.join(out, "grounding_overlay.jpg"), vis)
    print(f"grid cells inside the survey image: {inside.sum()}/{C.GRID_N ** 2}; overlay: {out}/grounding_overlay.jpg")
    print(f"saved {C.GROUNDING_PATH}")


def validate_board(a, G):
    import hardware as HW
    from fence import load_fence
    q_lock, R_ref, p_lock, arm, cam, out = setup(a)
    HW.set_fence(load_fence(), p_lock)
    rows = []
    try:
        for k, d in enumerate(cell_offsets_mm()):
            arm.send_deg(q_lock); wait_settled(arm)
            moved, q_meas = move_offset(arm, q_lock, R_ref, p_lock, d)
            row = dict(cell=k, dx_mm=d[0], dy_mm=d[1], moved=int(moved))
            if moved:
                ck, nk = BD.camera_point(cam.grab(), G, d)
                row["markers_seen"] = nk
                if ck is not None:
                    exp = G["c0"] + G["M"] @ d
                    e = ck - exp
                    row.update(meas_x_mm=ck[0], meas_y_mm=ck[1], exp_x_mm=exp[0], exp_y_mm=exp[1],
                               err_x_mm=e[0], err_y_mm=e[1], err_mm=float(np.linalg.norm(e)))
            rows.append(row)
            print(f"cell {k:2d}: " + (f"camera position error {row['err_mm']:.2f} mm (measured with the markers)"
                                      if "err_mm" in row else ("markers not all visible" if moved else "skipped (limits)")))
        arm.send_deg(q_lock); wait_settled(arm)
    finally:
        cam.release(); arm.close()
    return rows, out


# ------------------------------------------------------------------ track mode (default)
def calibrate_track(a):
    """Marker tracking: any visible markers (1 is enough) are followed while the camera makes small
    known moves (orientation held). Image shift = A @ move. No ruler, no 4-marker requirement."""
    q_lock, R_ref, p_lock, arm, cam, out = setup(a)
    try:
        I0, dets0 = BD.detect_markers_multi(cam.grab, tries=4, want=4)
        cv2.imwrite(os.path.join(out, "calib_survey.jpg"), I0)
        if not dets0:
            raise SystemExit("no marker visible at the survey pose (hw_logs/grounding/calib_survey.jpg)")
        print(f"survey: markers {sorted(map(str, dets0))} visible")
        D, S = [], []
        steps = [a.step_mm] * a.reps + [a.step_mm / 2.0]
        for k_rep, st in enumerate(steps):
            if k_rep == len(steps) - 1 and len(D) >= 6 and np.linalg.matrix_rank(np.array(D, float)) == 2:
                break
            for d in deltas(st):
                arm.send_deg(q_lock); wait_settled(arm)
                moved, _ = move_offset(arm, q_lock, R_ref, p_lock, d)
                if not moved:
                    print(f"  move {d}: not possible (IK/limits), skipped"); continue
                sh, n = BD.marker_shift(dets0, BD.detect_markers(cam.grab()))
                if sh is None:
                    print(f"  move {d}: no common marker in view, skipped"); continue
                D.append(d); S.append(sh)
                print(f"  camera moved {d} mm -> markers shifted ({sh[0]:+.1f}, {sh[1]:+.1f}) px  [{n} marker(s)]")
        arm.send_deg(q_lock); wait_settled(arm)
    finally:
        cam.release(); arm.close()
    if len(D) < 4 or np.linalg.matrix_rank(np.array(D, float)) < 2:
        raise SystemExit("not enough moves with a common marker in view (try --step-mm 8)")
    D, S = np.array(D, float), np.array(S, float)
    X, *_ = np.linalg.lstsq(D, S, rcond=None); A = X.T
    resid = S - D @ X
    h, w = I0.shape[:2]
    uc = np.array(optical_centre(w, h))
    rmse_px = float(np.sqrt((resid ** 2).sum(1).mean()))
    ppm = float(np.mean(np.linalg.norm(A, axis=0)))
    G = dict(mode="shift", method="marker tracking", A=A.tolist(), uc=uc.tolist(), image_wh=[w, h],
             step_mm=a.step_mm, fit_rmse_px=rmse_px, fit_rmse_mm=rmse_px / ppm, n_used=len(D), px_per_mm=ppm,
             markers0={str(k): v.tolist() for k, v in dets0.items()}, p_lock_mm=p_lock.tolist(),
             created=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:                                               # the drawn box, if found, becomes the workspace limit
        from measure_board import find_box
        box = find_box(I0, dets0 if len(dets0) >= 4 else None)
    except Exception:
        box = None
    if box is not None:
        G["box_px"] = BD._order_tl_tr_br_bl(box).tolist()        # limit is checked in the survey image
    os.makedirs(C.DEPLOY_DIR, exist_ok=True)
    json.dump(G, open(C.GROUNDING_PATH, "w"), indent=2)
    print(f"\nA = {np.round(A, 3).tolist()} px/mm  ({ppm:.2f} px/mm); fit residual {rmse_px:.2f} px = "
          f"{rmse_px / ppm:.2f} mm over {len(D)} moves")
    print("   (open-loop scatter from the arm's ~1 deg joint errors; every view is corrected by servoing)")
    if box is not None:
        print("drawn box found -> grid cells must lie inside it (checked in the survey image)")
    else:
        print(f"drawn box not found -> workspace limit: automatic +/-{C.FENCE_AUTO_HALF_MM:.0f} mm box")
    G["A"], G["uc"] = A, uc
    vis = I0.copy()
    for k, c in dets0.items():
        cv2.polylines(vis, [np.asarray(c, np.int32)], True, (0, 200, 255), 2)
    if box is not None:
        cv2.polylines(vis, [np.asarray(G["box_px"], np.int32)], True, (255, 0, 255), 2)
    for c in offset_to_px(G, cell_offsets_mm()):
        cv2.circle(vis, tuple(int(v) for v in c), 4, (0, 170, 0), -1)
    cv2.imwrite(os.path.join(out, "grounding_overlay.jpg"), vis)
    print(f"overlay: {out}/grounding_overlay.jpg; saved {C.GROUNDING_PATH}")


def validate_track(a, G):
    """Camera position check at every cell from the markers in view (independent of the encoders):
    measured marker shift vs the shift predicted for the commanded move, converted to mm."""
    import hardware as HW
    from fence import load_fence
    q_lock, R_ref, p_lock, arm, cam, out = setup(a)
    HW.set_fence(load_fence(), p_lock)
    dets0 = {int(k) if k.isdigit() else k: np.array(v) for k, v in G["markers0"].items()}
    rows = []
    try:
        for k, d in enumerate(cell_offsets_mm()):
            arm.send_deg(q_lock); wait_settled(arm)
            moved, _ = move_offset(arm, q_lock, R_ref, p_lock, d)
            row = dict(cell=k, dx_mm=d[0], dy_mm=d[1], moved=int(moved))
            if moved:
                from servo import servo_to_view
                u_t = offset_to_px(G, d)
                q_c = np.degrees(ik_hold((p_lock + np.r_[d, 0.0]) / 1000.0, np.deg2rad(q_lock), R_ref)[0])
                r = servo_to_view(arm, cam, G["A"], G["uc"], dets0, u_t, R_ref, safe_move, q_c)
                row.update(markers_seen=r["servo_markers"], servo_iters=r["servo_iters"],
                           open_loop_err_mm=r["view_err_open_mm"], servo_s=r["servo_s"])
                if np.isfinite(r["view_err_mm"]):
                    row["err_mm"] = r["view_err_mm"]
            rows.append(row)
            print(f"cell {k:2d}: " + (f"view error open-loop {row['open_loop_err_mm']:.2f} mm -> after servoing "
                                      f"{row['err_mm']:.2f} mm ({row['servo_iters']} corrections, {row['markers_seen']} marker(s))"
                                      if "err_mm" in row else ("no common marker in view" if moved else "skipped (limits)")))
        arm.send_deg(q_lock); wait_settled(arm)
    finally:
        cam.release(); arm.close()
    return rows, out


# ------------------------------------------------------------------ shift mode (fallback)
def calibrate_shift(a):
    q_lock, R_ref, p_lock, arm, cam, out = setup(a)
    try:
        I0 = cam.grab(); cv2.imwrite(os.path.join(out, "calib_survey.jpg"), I0)
        D, S = [], []
        for rep in range(a.reps):
            for d in deltas(a.step_mm)[:6]:
                arm.send_deg(q_lock); wait_settled(arm)
                moved, _ = move_offset(arm, q_lock, R_ref, p_lock, d)
                if not moved:
                    continue
                sh, r = estimate_shift(I0, cam.grab())
                ok = r >= MIN_PEAK
                print(f"  camera moved {d} mm -> image shift ({sh[0]:+.1f}, {sh[1]:+.1f}) px, peak {r:.2f}"
                      + ("" if ok else "  (UNRELIABLE, discarded)"))
                if ok:
                    D.append(d); S.append(sh)
        arm.send_deg(q_lock); wait_settled(arm)
    finally:
        cam.release(); arm.close()
    if len(D) < 4:
        raise SystemExit("too few reliable shifts: the scene lacks texture. Use the 4-marker board "
                         "(GROUNDING_MODE = 'board').")
    D, S = np.array(D, float), np.array(S, float)
    X, *_ = np.linalg.lstsq(D, S, rcond=None); A = X.T
    resid = S - D @ X
    h, w = I0.shape[:2]
    G = dict(mode="shift", A=A.tolist(), uc=optical_centre(w, h), image_wh=[w, h], step_mm=a.step_mm,
             fit_rmse_px=float(np.sqrt((resid ** 2).sum(1).mean())), n_used=len(D),
             px_per_mm=float(np.mean(np.linalg.norm(A, axis=0))), p_lock_mm=p_lock.tolist(),
             created=time.strftime("%Y-%m-%d %H:%M:%S"))
    if G["fit_rmse_px"] > 0.15 * G["px_per_mm"] * a.step_mm:
        print("WARNING: poor fit (RMSE > 15% of the shift); use the 4-marker board for reliable grounding")
    json.dump(G, open(C.GROUNDING_PATH, "w"), indent=2)
    G["A"], G["uc"] = np.array(G["A"]), np.array(G["uc"])
    _report(G, I0, out)


def validate_shift(a, G):
    import hardware as HW
    from fence import load_fence
    q_lock, R_ref, p_lock, arm, cam, out = setup(a)
    HW.set_fence(load_fence(), p_lock)
    size = G["px_per_mm"] * 2 * C.GRID_HALF_MM / (C.GRID_N - 1) * 1.5
    rows = []
    try:
        I0 = to_gray(cam.grab())
        for k, d in enumerate(cell_offsets_mm()):
            arm.send_deg(q_lock); wait_settled(arm)
            moved, _ = move_offset(arm, q_lock, R_ref, p_lock, d)
            row = dict(cell=k, dx_mm=d[0], dy_mm=d[1], moved=int(moved))
            if moved:
                P = patch(I0, offset_to_px(G, d), size); Q = patch(to_gray(cam.grab()), G["uc"], size)
                if P is not None and Q is not None:
                    sh, r = estimate_shift(P, Q); nc = ncc(P, Q)
                    row.update(peak=r, ncc=nc, reliable=int(r >= MIN_PEAK and nc >= MIN_NCC))
                    if row["reliable"]:
                        row["err_mm"] = float(np.linalg.norm(np.linalg.solve(G["A"], sh)))
            rows.append(row)
            print(f"cell {k:2d}: " + (f"grounding error {row['err_mm']:.2f} mm (peak {row['peak']:.2f}, NCC {row['ncc']:.2f})"
                                      if "err_mm" in row else "UNRELIABLE match, not counted"))
        arm.send_deg(q_lock); wait_settled(arm)
    finally:
        cam.release(); arm.close()
    return rows, out


def validate(a):
    G = load_grounding()
    if G.get("mode") == "board":
        rows, out = validate_board(a, G)
    elif "markers0" in G:
        rows, out = validate_track(a, G)
    else:
        rows, out = validate_shift(a, G)
    path = os.path.join(out, f"validate_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, restval=""); wr.writeheader(); wr.writerows(rows)
    e = np.array([r["err_mm"] for r in rows if "err_mm" in r])
    eo = np.array([r["open_loop_err_mm"] for r in rows if np.isfinite(r.get("open_loop_err_mm", np.nan))])
    if eo.size:
        print(f"\nOPEN-LOOP view error (kinematics only): median {np.median(eo):.2f} mm, max {eo.max():.2f} mm")
    print(f"\nmode {G.get('mode', 'shift')}: {e.size}/{len(rows)} cells measured")
    if e.size:
        print(f"error: median {np.median(e):.2f} mm, 95th pct {np.percentile(e, 95):.2f} mm, max {e.max():.2f} mm")
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "validate"])
    ap.add_argument("--step-mm", type=float, default=15.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--board-w", type=float, default=C.BOARD_W_MM)
    ap.add_argument("--board-h", type=float, default=C.BOARD_H_MM)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))
    ap.add_argument("--mode", dest="cal_mode", choices=["track", "board", "shift"], default=C.GROUNDING_MODE,
                    help="calibration method: track = follow any visible markers (default, no ruler needed)")
    a = ap.parse_args()
    if a.mode == "calibrate":
        print("calibration method: %s" % a.cal_mode)
        {"board": calibrate_board, "track": calibrate_track}.get(a.cal_mode, calibrate_shift)(a)
    else:
        validate(a)
