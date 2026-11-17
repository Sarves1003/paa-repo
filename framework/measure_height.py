#!/usr/bin/env python3
"""Measure, at the reference pose: (1) the camera-to-table distance H, (2) the DESCENT DRIFT: how far the view
slides sideways per mm the camera is lowered (camera not looking exactly straight down). No ruler needed.

  python3 measure_height.py            # moves straight down 10, 20, 30, 45 mm and back; prints H and the drift
  python3 measure_height.py --write    # also writes VIEW_CAM_HEIGHT_MM into config.py and the drift into grounding.json
  python3 measure_height.py --write --height 265   # RECOMMENDED: your ruler height; the robot measures the drift
  options: --steps 10 20 30 45  --speed 8  --dry-run
The drift is used by every inspection run as a feed-forward correction before the visual servo.

Principle: lowering the camera by d (orientation held) magnifies the table by s = H / (H - d), so
1/s = 1 - d/H is linear in d with slope -1/H. The magnification s is measured as the median size ratio of
the markers visible in both images; H comes from a least-squares fit over all steps (through 1/s = 1 at d = 0).
Only straight-down moves are used (reachable from your reference). Keep the area under the camera clear.
"""
import argparse, os, re, time
import numpy as np
import config as C
import hardware as HW
from hardware import make_arm, make_camera, safe_move, wait_settled, load_locked_pose
from kinematics import fk
from ik import ik_hold
from fence import auto_fence
from servo_scaled import marker_scale, register, part_mask


def write_config(H):
    s = open("config.py").read()
    s, n = re.subn(r"^VIEW_CAM_HEIGHT_MM\s*=.*$",
                   "VIEW_CAM_HEIGHT_MM = %.1f   # measured by measure_height.py (camera lens -> table at the reference)" % H,
                   s, flags=re.M)
    if n == 0:
        s += "\nVIEW_CAM_HEIGHT_MM = %.1f\n" % H
    open("config.py", "w").write(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, nargs="+", default=[10.0, 20.0, 30.0, 45.0])
    ap.add_argument("--speed", type=int, default=8)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--height", type=float, default=None,
                    help="your ruler value (lens -> table, mm): used and written instead of the fitted height")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))
    a = ap.parse_args()
    import board as BD
    q_ref = np.array(load_locked_pose(a.dry_run)["angles_deg"], float)
    T0 = fk(np.radians(q_ref)); R, p0 = T0[:3, :3], T0[:3, 3]
    HW.set_fence(auto_fence(), p0 * 1000)
    arm = make_arm(a.dry_run, q_ref); arm.send_deg(q_ref, 10); wait_settled(arm)
    cam = make_camera(a.dry_run, arm, a.ref_img)
    rows = []
    try:
        time.sleep(0.5)
        img0 = cam.grab(); d0 = BD.detect_markers(img0)
        if not d0:
            raise SystemExit("no marker visible at the reference: cannot measure")
        try:
            from perception import load_grounding
            Gg = load_grounding()
        except BaseException:
            Gg = None
        mask0 = part_mask(img0, Gg); uc = np.array([img0.shape[1] / 2.0, img0.shape[0] / 2.0])
        drift = []                                         # (depth mm, survey px now at the image centre)
        q = np.radians(q_ref)
        for d in sorted(a.steps):
            qn, pe, re, ok = ik_hold(p0 - np.r_[0, 0, d / 1000.0], q, R, iters=1500)
            if not ok or not safe_move(arm, np.degrees(qn), a.speed)[0]:
                print("  %.0f mm down: move not possible (%s), stopping" % (d, HW.LAST_REFUSAL or "IK")); break
            q = qn; wait_settled(arm); time.sleep(0.4)
            img = cam.grab(); dets = BD.detect_markers(img)
            s = marker_scale(d0, dets, allow_single=True)
            common = [k for k in dets if k in d0]
            u_star = None
            if len(common) >= 2 and s is not None:            # survey pixel now under the image centre, from markers
                u_star = np.mean([np.asarray(d0[k], float).mean(0) - (np.asarray(dets[k], float).mean(0) - uc) / s
                                  for k in common], 0)
            else:
                for mk in (mask0, None):                       # from registration on the table plane
                    M = register(img0, img, mk)
                    if M is not None:
                        s_r = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
                        s = s if s is not None else s_r
                        u_star = np.linalg.solve(M[:, :2], uc - M[:, 2]); break
            if s is None:
                print("  %.0f mm down: no markers and no registration, stopping" % d); break
            rows.append((d, s))
            if u_star is not None:
                drift.append((d, u_star - uc))
            print("  %.0f mm down: image %.3f x larger%s" % (d, s, "" if u_star is None else
                  ", view centre moved (%+.1f, %+.1f) survey px" % tuple(u_star - uc)))
    finally:
        try:
            for d, _ in reversed(rows[:-1]):                       # back up the same straight line
                qn, *_ = ik_hold(p0 - np.r_[0, 0, d / 1000.0], np.radians(arm.read_deg()), R, iters=1500)
                safe_move(arm, np.degrees(qn), a.speed)
            safe_move(arm, q_ref, a.speed)
        finally:
            cam.release(); arm.close()
    if not rows:
        raise SystemExit("no measurement")
    D = np.array([r[0] for r in rows]); S = np.array([r[1] for r in rows])
    if np.all(np.abs(S - 1) < 0.005):
        raise SystemExit("no magnification measured (markers did not grow): check that the camera looks at the table")
    # Fits WITH an offset: the first move after the reference image takes up gear backlash and makes a one-time
    # jump (on the robot: ~10 mm, ~2 % scale) that is not depth-dependent; a fit through zero would turn it into
    # a false slope. 1/s = c (1 - d/H)  ->  linear in d with intercept c and slope -c/H.
    if len(D) >= 3:
        b1, b0 = np.polyfit(D, 1 / S, 1)
        H = float(b0 / -b1) if b1 < 0 else float("nan")
        res_s = float(np.sqrt(np.mean((1 / S - (b0 + b1 * D)) ** 2)))
        print("camera lens -> table at the reference: H = %.0f mm (fit with offset: residual %.4f, reference-image "
              "scale offset %.1f %%)" % (H, res_s, 100 * (1 / b0 - 1)))
    else:
        k = float(np.sum(D * (1 - 1 / S)) / np.sum(D * D)); H = 1.0 / k
        print("camera lens -> table at the reference: H = %.0f mm (only %d steps: fit through zero)" % (H, len(D)))
    if not np.isfinite(H) or H < 80 or H > 600:
        raise SystemExit("implausible height %.0f mm: check that markers stay visible, re-run" % H)
    part_h = float(getattr(C, "PART_MAX_HEIGHT_MM", 60.0))
    print("clearance above a %.0f mm part at the reference: %.0f mm" % (part_h, H - part_h))
    g = None
    if len(drift) >= 3:
        Dd = np.array([x[0] for x in drift]); Uu = np.array([x[1] for x in drift])
        X = np.c_[np.ones(len(Dd)), Dd]
        coef, *_ = np.linalg.lstsq(X, Uu, rcond=None)                  # rows: offset (px), slope (px per mm)
        jump, g = coef[0], coef[1]
        res = float(np.sqrt(((Uu - X @ coef) ** 2).sum(1).mean()))
        ppm = None
        try:
            import json as _j
            ppm = float(_j.load(open(os.path.join(C.DEPLOY_DIR, "table_grid.json")))["box_px_per_mm"])
        except Exception:
            pass
        mm100 = 100 * np.linalg.norm(g) / ppm if ppm else float("nan")
        print("descent drift: (%+.3f, %+.3f) survey px per mm lowered, fit residual %.1f px%s"
              % (g[0], g[1], res, "" if not ppm else "; lowering 100 mm slides the view %.1f mm (camera tilt about "
                 "%.1f deg)" % (mm100, np.degrees(np.arctan(mm100 / 100)))))
        print("one-time jump at the first move (backlash, not depth-dependent, removed by the visual servo): "
              "(%+.1f, %+.1f) px%s" % (jump[0], jump[1], "" if not ppm else " = %.1f mm" % (np.linalg.norm(jump) / ppm)))
        if res > 5.0:
            print("WARNING: drift fit residual > 5 px: the drift is not well determined; the feed-forward will be weak")
    else:
        print("descent drift: not measured (needs markers or registration at >= 3 steps)")
    if a.height:
        print("using your ruler height %.0f mm (fitted %.0f mm, %+.0f %%)" % (a.height, H, 100 * (H / a.height - 1)))
        H = float(a.height)
    if a.write:
        write_config(H); print("written VIEW_CAM_HEIGHT_MM = %.0f into config.py" % H)
        if g is not None:
            import json as _j
            Gj = _j.load(open(C.GROUNDING_PATH))
            Gj["descent_drift_px_per_mm"] = [float(g[0]), float(g[1])]
            _j.dump(Gj, open(C.GROUNDING_PATH, "w"), indent=2)
            print("written descent_drift_px_per_mm into %s (used as feed-forward by every inspection run)" % C.GROUNDING_PATH)
    else:
        print("put this in config.py:  VIEW_CAM_HEIGHT_MM = %.1f   (or re-run with --write)" % H)


if __name__ == "__main__":
    main()
