#!/usr/bin/env python3
"""EXPERIMENT E2: record the REAL view library (the data-driven twin for RL).

Your setup stays as it is: the survey image is taken at your locked pose (high: whole box and all markers
in view). From that pose the arm cannot move outward at the same height, but it reaches every grid cell
lower down, so every inspection view is taken BELOW the survey pose:
  level 0: VIEW_LEVELS_DZ[0] (default -60 mm), centred on the cell by scale-aware visual servoing
  level 1: VIEW_LEVELS_DZ[1] (default -80 mm), close-up straight below the level-0 view
Before any motion the camera clearance above the part is checked (VIEW_CAM_HEIGHT_MM must be measured).

  python3 collect_views.py --pose horizontal --n 3 --condition ok
  python3 collect_views.py --pose horizontal --n 3 --condition rust --rust-facing up
  (conditions: ok | no_ring_hex | no_ring_open | no_rings | no_oring | rust, comma-combined; see task_model.py)
  options: --speed 15, --all-cells, --dry-run
Output: deploy/views/<pose>_<NNN>/ survey.jpg, v<k>.jpg, meta.json

Unified mode (default when deploy/lattice_views.json exists, built by lattice_views.py from refcover_updated.py's
lattice): the candidate views, their heights and the safe routes come from that file; the heights are the SMALL
descents allowed by refcover_updated.py (--max-descent), not VIEW_LEVELS_DZ. Run  python3 refcover_updated.py plan
and  python3 lattice_views.py  (same options) first. --no-lattice = old two-level mode (VIEW_LEVELS_DZ).
"""
import argparse, glob, json, os, time
import numpy as np
import cv2
import config as C
import hardware as HW
from hardware import make_arm, make_camera, safe_move, wait_settled, load_locked_pose
from kinematics import fk
from ik import ik_hold
from fence import auto_fence
from perception import load_grounding, locate_part_scene
from views import ViewSet, survey_features, levels_from_config, scale_guess, register_to_survey, registered_view_check
from servo_scaled import servo_to_view_scaled

VIEW_DIR = os.path.join(C.DEPLOY_DIR, "views")


def check_clearance(levels):
    """Refuse levels that bring the camera too close to the part (needs the measured camera height)."""
    H = getattr(C, "VIEW_CAM_HEIGHT_MM", None)
    if not H:
        raise SystemExit("set VIEW_CAM_HEIGHT_MM in config.py: measure with a ruler the distance from the camera lens "
                         "to the table with the arm at the locked pose (needed for the collision check)")
    part_h = float(getattr(C, "PART_MAX_HEIGHT_MM", 60.0)); margin = float(getattr(C, "VIEW_MIN_CLEARANCE_MM", 25.0))
    low = H + min(levels)
    if low < part_h + margin:
        raise SystemExit("lowest view would put the lens %.0f mm above the table, the part is up to %.0f mm tall: "
                         "clearance %.0f mm < %.0f mm. Use smaller VIEW_LEVELS_DZ in config.py."
                         % (low, part_h, low - part_h, margin))
    print("camera clearance above the tallest part: %.0f mm at the lowest view (min %.0f mm)" % (low - part_h, margin))
    return float(H)


def load_viewset(levels=None):
    """levels: inspection heights (mm below the survey pose). Default VIEW_LEVELS_DZ from config.py (old mode);
    the unified mode passes the heights of deploy/lattice_views.json."""
    levels = list(levels) if levels is not None else levels_from_config()
    H = check_clearance(levels)
    tg = json.load(open(os.path.join(C.DEPLOY_DIR, "table_grid.json")))
    G = load_grounding()
    return ViewSet(tg, levels, scales=scale_guess(levels, H), uc=G["uc"]), G, H


def plan_q(vs, q_lock, R_ref, p_lock):
    """Joint angles of every view (planned, camera orientation held): used for moving and the time model."""
    Q = np.full((vs.K, 6), np.nan); ok = np.zeros(vs.K, bool)
    for k in range(vs.K):
        tgt = (p_lock + np.r_[vs.off[vs.cell[k]], vs.dz[vs.level[k]]]) / 1000.0
        q, pe, re, good = ik_hold(tgt, np.deg2rad(q_lock), R_ref)
        if not good:
            q, pe, re, good = ik_hold(tgt, np.deg2rad(q_lock), R_ref, iters=1500)
        if good:
            Q[k] = np.degrees(q); ok[k] = True
    return Q, ok


def servo_view(arm, cam, G, m0, vs, k, R_ref, q_cmd, speed, survey=None, mask=None):
    """Scale-aware servo for a view below the survey pose, with the calibrated descent-drift feed-forward."""
    drift = np.asarray(G.get("descent_drift_px_per_mm", [0.0, 0.0]), float)
    return servo_to_view_scaled(arm, cam, G["A"], G["uc"], m0, vs.u[vs.cell[k]], R_ref, safe_move, q_cmd,
                                vs.scales[vs.level[k]], speed=speed, survey=survey, survey_mask=mask,
                                ff_px=drift * (-vs.dz[vs.level[k]]))


def capture_tilted(arm, cam, vs, k, survey, G, px_per_mm, speed=8):
    """Tilted view: the marker servo assumes the reference orientation, so the view is centred by a REGISTRATION
    servo instead: register the image to the survey, move the camera (orientation held) until the aimed cell is at
    the image centre, and keep the measured survey -> view homography for scoring."""
    from perception import px_to_offset
    from views import registration_servo, table_mask
    t = time.time()
    mask = table_mask(survey, G)

    def grab():
        wait_settled(arm); time.sleep(0.4)
        for _ in range(2):
            cam.grab()
        return cam.grab()

    def move_by(dd):
        q = np.radians(arm.read_deg()); T = fk(q)
        p = T[:3, 3] * 1000 + np.r_[np.asarray(dd, float), 0.0]
        q2, pe, re, ok = ik_hold(p / 1000.0, q, T[:3, :3], iters=800)
        return bool(ok) and safe_move(arm, np.degrees(q2), speed, max_step=12.0)[0]

    r = registration_servo(grab, move_by, survey, vs.u[vs.cell[k]],
                           lambda u: np.asarray(px_to_offset(G, np.asarray(u, float)), float).ravel()[:2],
                           np.asarray(G["uc"], float), px_per_mm, mask=mask)
    return dict(frame=r["frame"], servo_s=time.time() - t, servo_iters=r["iters"], servo_markers=0,
                view_err_open_mm=r["err_first_mm"], view_err_raw_mm=r["err_first_mm"], ff_mm=np.nan,
                view_err_mm=r["err_mm"],
                scale=float(r["scale"]) if np.isfinite(r["scale"]) else float(vs.scales[vs.level[k]]),
                verify=r["verify"], H_meas=r["H"].tolist() if r["H"] is not None else None, reg_inliers=r["inliers"])


def survey_servo(arm, cam, G, m0, R_ref, q_lock):
    """Survey at the locked pose; same-height servo (your servo.py) to repeat the calibration view."""
    from servo import servo_to_view
    return servo_to_view(arm, cam, G["A"], G["uc"], m0, G["uc"], R_ref, safe_move, q_lock, speed=10)


def move_to(arm, q_target, R_ref, speed, max_joint_deg=35.0, step_mm=12.0):
    """Safe move to a joint target. Small joint changes: one guarded move. Large changes (survey <-> lowered
    views, the elbow folds by ~60 deg): go there along a straight Cartesian path in short steps, camera
    orientation held, so every single command stays small and the camera never goes below the lower of the two heights.
    Returns (moved, elapsed_s, q_meas)."""
    q_target = np.asarray(q_target, float)
    q_now = arm.read_deg()
    if np.max(np.abs(q_target - q_now)) <= max_joint_deg:
        return safe_move(arm, q_target, speed)
    t0 = time.time()
    p0 = fk(np.deg2rad(q_now))[:3, 3]; p1 = fk(np.deg2rad(q_target))[:3, 3]
    # sideways motion always happens at the LOWER of the two heights: from the survey pose the arm cannot
    # move outward at the survey height, but it can lower down
    if p1[2] < p0[2]:                       # going down: straight down first, then sideways
        corner = np.r_[p0[:2], p1[2]]
    else:                                   # going up: sideways first (still low), then straight up
        corner = np.r_[p1[:2], p0[2]]
    q = np.deg2rad(q_now)
    for a_, b_ in ((p0, corner), (corner, p1)):
        n = max(1, int(np.ceil(np.linalg.norm(b_ - a_) * 1000 / step_mm)))
        for i in range(1, n + 1):
            qi, pe, re, ok = ik_hold(a_ + (b_ - a_) * i / n, q, R_ref, iters=800)
            if not ok:
                HW.LAST_REFUSAL = "path IK failed"
                return False, time.time() - t0, arm.read_deg()
            moved, _, _ = safe_move(arm, np.degrees(qi), speed)
            if not moved:
                return False, time.time() - t0, arm.read_deg()
            q = qi
    moved, _, qm = safe_move(arm, q_target, speed)      # final small correction to the exact target
    return moved, time.time() - t0, qm


def descend(arm, cam, dz_mm, R_ref, speed):
    """Close-up: straight down from the current pose (camera orientation held)."""
    q_now = arm.read_deg(); p_now = fk(np.deg2rad(q_now))[:3, 3]
    q, _, _, good = ik_hold(p_now + np.r_[0, 0, dz_mm / 1000.0], np.deg2rad(q_now), R_ref)
    if not good:
        q, _, _, good = ik_hold(p_now + np.r_[0, 0, dz_mm / 1000.0], np.deg2rad(q_now), R_ref, iters=1500)
    if not good or not safe_move(arm, np.degrees(q), speed)[0]:
        return None
    wait_settled(arm); time.sleep(0.3)
    for _ in range(2):
        cam.grab()
    return cam.grab()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", choices=["horizontal", "vertical"], required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--speed", type=int, default=15)
    ap.add_argument("--all-cells", action="store_true", help="also record cells without part (slower)")
    ap.add_argument("--no-lattice", action="store_true", help="ignore deploy/lattice_views.json (old two-level mode)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))
    ap.add_argument("--condition", required=True,
                    help="state of the part you place: ok | no_ring_hex | no_ring_open | no_rings | no_oring | rust "
                         "(combine with commas, e.g. no_ring_open,rust). Same for all --n placements of this command.")
    ap.add_argument("--rust-facing", dest="rust_facing", choices=["none", "up", "side", "down"], default="none",
                    help="where the rust patch faces (needed when the condition contains rust)")
    a = ap.parse_args()
    import task_model as TM
    try:
        y_task, m_task = TM.labels(a.condition, a.rust_facing)
    except ValueError as e:
        raise SystemExit(str(e))
    if "rust" in a.condition and a.rust_facing == "none":
        raise SystemExit("--condition contains rust: give --rust-facing up, side or down")
    print("part condition for these placements: %s -> %s" % (a.condition, TM.describe(y_task)))

    import lattice_views as LVm
    LV = None if a.no_lattice else LVm.load_views()
    if LV is not None and "lattice_file" not in LV:
        raise SystemExit("deploy/lattice_views.json was built by an old lattice_views.py (no lattice file recorded; "
                         "heights %s mm). Delete it and rebuild:  python3 lattice_views.py <same options as the plan>"
                         % LV.get("levels_dz"))
    if LV is not None:
        print("candidate views from %s (lattice %s): heights %s mm" % ("deploy/lattice_views.json", LV["lattice_file"],
                                                                         LV["levels_dz"]))
    vs, G, H = load_viewset(None if LV is None else LV["levels_dz"])
    m0 = {int(k) if str(k).isdigit() else k: np.array(v) for k, v in G["markers0"].items()}
    lock = load_locked_pose(a.dry_run); q_lock = np.array(lock["angles_deg"], float)
    T0 = fk(np.deg2rad(q_lock)); R_ref, p_lock = T0[:3, :3], T0[:3, 3] * 1000
    HW.set_fence(auto_fence(), p_lock)
    if LV is not None:                                   # UNIFIED MODE: candidates and routes from Part A
        vs = ViewSet(json.load(open(os.path.join(C.DEPLOY_DIR, "table_grid.json"))), LV["levels_dz"],
                     scales=LV["scales"], uc=G["uc"], via_hi=False, level_H=LV.get("levels_H"),
                     level_rpy=LV.get("levels_rpy"))
        if vs.rot.any():
            print("tilted candidate levels: %s (verified by registration to the survey)"
                  % [tuple(vs.rpy[l]) for l in range(vs.L) if vs.rot[l]])
        Qplan = np.array([v["q_deg"] if v["ok"] else [np.nan] * 6 for v in LV["views"]], float)
        okplan = np.array([v["ok"] for v in LV["views"]])
        print("unified mode: candidate views and safe routes from the Part A lattice (deploy/lattice_views.json)")
    else:
        Qplan, okplan = plan_q(vs, q_lock, R_ref, p_lock)
    print("%d candidate views (%d cells x levels %s mm below the survey); reachable %d/%d"
          % (vs.K, vs.P, [int(v) for v in vs.dz], okplan.sum(), vs.K))
    lap_ref = float(getattr(C, "VIEW_LAP_REF", 6.0))
    tg_ppm = float(json.load(open(os.path.join(C.DEPLOY_DIR, "table_grid.json")))["box_px_per_mm"])
    import robot_trace as RT
    RT.install(os.path.join(C.LOGS_DIR, "trace", "collect_%s_%s" % (a.pose, time.strftime("%Y%m%d_%H%M%S"))))
    arm = make_arm(a.dry_run, q_lock); arm.send_deg(q_lock); wait_settled(arm)
    cam = make_camera(a.dry_run, arm, a.ref_img)
    mover = LVm.LatticeMover(arm, LV, a.speed) if LV is not None else None
    os.makedirs(VIEW_DIR, exist_ok=True)
    start = len(glob.glob(os.path.join(VIEW_DIR, a.pose + "_*")))
    try:
        for n in range(start, start + a.n):
            if not a.dry_run:
                input(f"\n[{a.pose} {n}] place the part ({a.condition}"
                      f"{', rust facing ' + a.rust_facing if a.rust_facing != 'none' else ''}) {a.pose.upper()} inside "
                      f"the box (new position/rotation), hands out, Enter ...")
            d = os.path.join(VIEW_DIR, f"{a.pose}_{n:03d}"); os.makedirs(d, exist_ok=True)
            if mover is not None:
                mover.goto_reference(q_lock)
            else:
                move_to(arm, q_lock, R_ref, 20)
            RT.event("survey", placement=n, pose=a.pose)
            survey = survey_servo(arm, cam, G, m0, R_ref, q_lock)["frame"]
            from servo_scaled import part_mask
            s_mask = part_mask(survey, G)
            cv2.imwrite(os.path.join(d, "survey.jpg"), survey)
            sf = survey_features(survey, vs, lap_ref, lambda im: locate_part_scene(im, G))
            todo = [k for k in range(vs.K) if okplan[k] and (a.all_cells or (sf["fp_part"][vs.hi[k]] > 0.02
                                                                             and sf["w"][vs.cell[k]] > 0.02))]
            todo = sorted(todo, key=lambda k: (vs.cell[k], vs.level[k]))
            print(f"  part {'found' if sf['bbox'] else 'NOT found (whole box used)'}; recording {len(todo)} views")
            rec, t_start = {}, time.time()
            for k in todo:
                c, l = vs.cell[k], vs.level[k]
                r = dict(k=int(k), cell=int(c), level=int(l), ok=0)
                RT.event("view", placement=n, view=int(k))
                t = time.time()
                if mover is not None:                    # unified: lattice route, every view verified directly
                    if not mover.goto_view(k):
                        r["refusal"] = HW.LAST_REFUSAL; rec[k] = r; continue
                    if vs.rot[l]:
                        s = capture_tilted(arm, cam, vs, k, survey, G, tg_ppm)
                    else:
                        s = servo_view(arm, cam, G, m0, vs, k, R_ref, Qplan[k], min(a.speed, 12), survey, s_mask)
                    img = s["frame"]
                    r.update(ok=1, move_s=time.time() - t - s["servo_s"], servo_s=s["servo_s"], servo_iters=s["servo_iters"],
                             servo_markers=s["servo_markers"], view_err_open_mm=s["view_err_open_mm"],
                             view_err_raw_mm=s["view_err_raw_mm"], ff_mm=s["ff_mm"],
                             view_err_mm=s["view_err_mm"], scale=s["scale"], verify=s["verify"], total_s=time.time() - t)
                    if s.get("H_meas") is not None:
                        r["H_meas"] = s["H_meas"]; r["reg_inliers"] = s["reg_inliers"]
                    r["rpy"] = [float(x) for x in vs.rpy[l]]
                elif l == 0:
                    moved, mv_s, _ = move_to(arm, Qplan[k], R_ref, a.speed)
                    if not moved:
                        r["refusal"] = HW.LAST_REFUSAL; rec[k] = r; continue
                    s = servo_view(arm, cam, G, m0, vs, k, R_ref, Qplan[k], min(a.speed, 15), survey, s_mask)
                    img = s["frame"]
                    r.update(ok=1, move_s=mv_s, servo_s=s["servo_s"], servo_iters=s["servo_iters"],
                             servo_markers=s["servo_markers"], view_err_open_mm=s["view_err_open_mm"],
                             view_err_mm=s["view_err_mm"], scale=s["scale"], total_s=time.time() - t)
                else:
                    h = vs.hi[k]
                    if not rec.get(h, {}).get("ok"):
                        rec[k] = r; continue
                    img = descend(arm, cam, vs.dz[l] - vs.dz[0], R_ref, a.speed)
                    if img is None:
                        r["refusal"] = "close-up move not possible"; rec[k] = r
                        move_to(arm, Qplan[h], R_ref, a.speed); continue
                    r.update(ok=1, servo_s=0.0, total_s=time.time() - t)
                r["q_meas"] = [float(v) for v in arm.read_deg()]
                cv2.imwrite(os.path.join(d, f"v{k:02d}.jpg"), img)
                rec[k] = r
                print(f"    view {k:2d} (cell {c:2d}, level {l}) ok" +
                      (f", view error {r['view_err_mm']:.2f} mm, scale {r['scale']:.2f}, verified by {r.get('verify', 'markers')}"
                       if np.isfinite(r.get('view_err_mm', np.nan)) else
                       (" (not verified: open-loop)" if (l == 0 or mover is not None) else "")))
                if l > 0 and mover is None:
                    move_to(arm, Qplan[vs.hi[k]], R_ref, a.speed)     # back up before the next cell
            if mover is not None:
                mover.goto_reference(q_lock)
            else:
                move_to(arm, q_lock, R_ref, 20)                       # back to the survey pose (vertical path)
            json.dump(dict(pose=a.pose, placement=n, condition=a.condition, rust_facing=a.rust_facing,
                           task_items=TM.ITEMS, task_labels=[int(v) for v in y_task], task_mask=[int(v) for v in m_task],
                           levels_dz=[float(v) for v in vs.dz], dz=float(vs.dz[-1] - vs.dz[0]),
                           lattice=LV is not None, via_hi=vs.via_hi, T_route=LV["T"] if LV is not None else None,
                           levels_rpy=vs.rpy.tolist(), levels_H=LV.get("levels_H") if LV is not None else None,
                           scales_plan=vs.scales.tolist(),
                           cam_height_mm=H, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                           q_lock=q_lock.tolist(), q_plan=np.nan_to_num(Qplan).tolist(), plan_ok=okplan.tolist(),
                           part_bbox=sf["bbox"], views=list(rec.values()), wall_s=time.time() - t_start,
                           grounding=G.get("created")),
                      open(os.path.join(d, "meta.json"), "w"), indent=2, default=float)
            print(f"  saved {d} ({sum(r['ok'] for r in rec.values())} views, {time.time() - t_start:.0f} s)")
    finally:
        try:
            if mover is not None:
                mover.goto_reference(q_lock)
            else:
                move_to(arm, q_lock, R_ref, 15)
        finally:
            cam.release(); arm.close()


if __name__ == "__main__":
    main()
