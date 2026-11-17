#!/usr/bin/env python3
"""Fine alignment around the pose YOU locked with live_align.py. The robot never wanders: it refines
the view within a small radius (default 25 mm) of your hand-locked pose and saves the refined pose.

Workflow (your setup and hand pose stay as they are)
  0. python3 auto_align.py --accept   keep your hand pose exactly (no motion); use this when the arm is at its
                                    reach limit at the survey height. Inspection views are taken lower down.
  1. python3 live_align.py          jog the arm until the box (or the part) is roughly in the middle of
                                    the image, all markers visible  ->  press "Lock pose"
  2. python3 auto_align.py --check  NO motion: how far off is the view, and which way to jog if too far
  3. python3 auto_align.py          probe image motion, then centre the target in small steps; saves
                                    deploy/locked_pose.json (your hand pose is kept as locked_pose_manual.json)
  4. python3 grounding.py calibrate --mode track ; python3 table_grid.py --install

Options: --target auto|box|part (default auto: drawn box, else part, else marker centre)
         --max-travel 25 (mm from your hand pose)  --max-step 3  --tol-px 4  --speed 10  --show  --dry-run
Every run starts from the hand pose (locked_pose_manual.json), so repeated runs never drift.
Re-lock with live_align.py at any time: the next run uses the new hand pose.
"""
import argparse, json, os, shutil, time
import numpy as np
import cv2
import config as C
import board as BD
import hardware as HW
from hardware import make_arm, make_camera, wait_settled, safe_move
from kinematics import fk
from ik import ik_hold
from fence import auto_fence
from perception import locate_part

MANUAL = os.path.join(C.DEPLOY_DIR, "locked_pose_manual.json")
LAST_FAIL = ""


# ------------------------------------------------------------------ target and camera helpers (also used by serial_inspect.py)
def find_target(img, mode):
    dets = BD.detect_markers(img)
    if mode in ("box", "auto"):
        try:
            from measure_board import find_box
            box = find_box(img, dets if len(dets) >= 4 else None)
        except Exception:
            box = None
        if box is not None:
            return BD._order_tl_tr_br_bl(box).mean(0), "drawn box", dets, box
        if mode == "box":
            return None, "none (drawn box not found)", dets, None
    if mode in ("part", "auto"):
        part = locate_part(img, exclude=list(dets.values()))
        if part is not None:
            return np.array(part["centroid_px"]), "part", dets, None
    if dets and mode == "auto":
        return np.mean([np.asarray(c).mean(0) for c in dets.values()], 0), "marker centre", dets, None
    return None, "none", dets, None


class LiveCam:
    """Camera wrapper: every grabbed frame is annotated and published to <LOGS_DIR>/live.jpg (+ window with --show)."""

    def __init__(self, cam, show=False):
        self.cam = cam
        self.show = bool(show and os.environ.get("DISPLAY"))
        self.state = dict(c=None, msg="starting")
        os.makedirs(C.LOGS_DIR, exist_ok=True)
        self.path = os.path.join(C.LOGS_DIR, "live.jpg")
        self.tmp = os.path.join(C.LOGS_DIR, "live_tmp.jpg")
        with open(os.path.join(C.LOGS_DIR, "live.html"), "w") as f:
            f.write("<!doctype html><meta charset=utf-8><title>live</title>"
                    "<body style='margin:0;background:#111'><img id=i style='width:100%;max-width:960px'>"
                    "<script>setInterval(()=>{document.getElementById('i').src='live.jpg?'+Date.now()},250)</script>")

    def grab(self):
        img = self.cam.grab()
        try:
            self._publish(img)
        except Exception:
            pass
        return img

    def _publish(self, img):
        h, w = img.shape[:2]; uc = (w // 2, h // 2)
        v = img.copy()
        for k, corners in BD.detect_markers(img).items():
            cv2.polylines(v, [np.asarray(corners, np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2)
        cv2.line(v, (uc[0] - 25, uc[1]), (uc[0] + 25, uc[1]), (0, 255, 0), 2)
        cv2.line(v, (uc[0], uc[1] - 25), (uc[0], uc[1] + 25), (0, 255, 0), 2)
        c = self.state.get("c")
        if c is not None:
            cp = (int(c[0]), int(c[1]))
            err = float(np.hypot(c[0] - uc[0], c[1] - uc[1]))
            col = (0, 255, 0) if err <= 5 else ((0, 165, 255) if err <= 20 else (0, 0, 255))
            cv2.circle(v, cp, 6, (0, 0, 255), -1); cv2.line(v, uc, cp, col, 1)
        cv2.rectangle(v, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(v, self.state.get("msg", ""), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imwrite(self.tmp, v); os.replace(self.tmp, self.path)
        if self.show:
            cv2.imshow("auto_align live", v); cv2.waitKey(1)

    def release(self):
        try:
            self.cam.release()
        finally:
            if self.show:
                cv2.destroyAllWindows()

    def __getattr__(self, name):
        return getattr(self.cam, name)


def fresh(cam, n=3):
    """Discard buffered frames so the image reflects the arm's current pose."""
    for _ in range(n):
        cam.grab()
    return cam.grab()


def move_by(arm, q_cmd, d_mm, R_ref, speed, settle_s=0.4):
    """Move the COMMANDED pose by d_mm in x/y (camera orientation held). Returns the new command, or None
    with the reason in auto_align.LAST_FAIL. A failed IK is retried from the current command with more
    iterations and looser tolerance."""
    global LAST_FAIL
    p = fk(np.deg2rad(q_cmd))[:3, 3]
    tgt = p + np.r_[d_mm, 0.0] / 1000.0
    q, pe, re, ok = ik_hold(tgt, np.deg2rad(q_cmd), R_ref)
    if not ok:
        q, pe, re, ok = ik_hold(tgt, np.deg2rad(q_cmd), R_ref, iters=1200, tol_pos=8e-4, tol_rot_deg=0.8)
    if not ok:
        LAST_FAIL = "IK did not converge (position residual %.2f mm, orientation %.2f deg)" % (pe * 1000, re)
        return None
    moved, _, _ = safe_move(arm, np.degrees(q), speed)
    if not moved:
        LAST_FAIL = "safety check: " + HW.LAST_REFUSAL
        return None
    wait_settled(arm)
    time.sleep(settle_s)
    LAST_FAIL = ""
    return np.degrees(q)


# ------------------------------------------------------------------ anchor (your hand-locked pose)
def load_anchor(dry):
    """The pose you locked in live_align.py. A locked_pose.json NOT written by auto_align is a new hand pose:
    it is copied to locked_pose_manual.json and becomes the anchor."""
    lp = C.LOCKED_POSE_PATH
    if os.path.exists(lp):
        j = json.load(open(lp))
        if j.get("source") != "auto_align.py":
            os.makedirs(C.DEPLOY_DIR, exist_ok=True)
            shutil.copy(lp, MANUAL)
            print("hand-locked pose from live_align found -> anchor (%s)" % MANUAL)
    if os.path.exists(MANUAL):
        return json.load(open(MANUAL))
    if dry:
        return {"angles_deg": [-74.5, -32.6, -77.0, 40.6, -22.4, 82.7]}
    raise SystemExit("no locked pose: run  python3 live_align.py  and press 'Lock pose' first")


def jog_hint(img_dir):
    """Plain-language direction from the image offset (target minus image centre)."""
    dx, dy = img_dir
    h = ("left" if dx < 0 else "right") if abs(dx) > 8 else ""
    v = ("up" if dy < 0 else "down") if abs(dy) > 8 else ""
    side = " and ".join(s for s in (h, v) if s)
    return "the target is %s of the image centre: jog the arm so that the target moves toward the centre (green cross in live.jpg)" % (side or "near")


# ------------------------------------------------------------------ reachability of the experiment moves
def reach_report(q0_deg, max_travel):
    """Which moves the experiments need are possible from this pose (camera orientation held)?
    Returns (can_align, inspection_ok): fine alignment at the survey height, and the inspection views at
    every height in VIEW_LEVELS_DZ (these are required)."""
    q0 = np.deg2rad(q0_deg); T = fk(q0); R, p = T[:3, :3], T[:3, 3]
    h = float(C.GRID_HALF_MM); r = min(max_travel, 20.0)
    levels = [float(v) for v in getattr(C, "VIEW_LEVELS_DZ", (-60.0, -80.0))]

    def n_ok(moves):
        bad = []
        for d in moves:
            d = np.asarray(d, float)
            res = ik_hold(p + d / 1000.0, q0, R)
            if not res[3]:
                res = ik_hold(p + d / 1000.0, q0, R, iters=1500)
            if not res[3]:
                bad.append(d)
        return len(moves) - len(bad), len(moves)

    grid = lambda dz: [(x, y, dz) for x in (-h, -h / 2, 0, h / 2, h) for y in (-h, -h / 2, 0, h / 2, h)]
    print("pose: flange height %.0f mm, distance from base %.0f mm, J3 %.0f deg"
          % (p[2] * 1000, np.hypot(p[0], p[1]) * 1000, q0_deg[2]))
    pr = n_ok([(4, 0, 0), (-4, 0, 0), (0, 4, 0), (0, -4, 0), (r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0)])
    gm = n_ok([(15, 0, 0), (-15, 0, 0), (0, 15, 0), (0, -15, 0), (15, 15, 0), (-15, -15, 0), (15, -15, 0), (-15, 15, 0)])
    print("  survey height: fine-alignment moves %d/%d, calibration moves %d/%d" % (pr + gm))
    insp_ok = True
    for dz in levels:
        n, N = n_ok(grid(dz))
        insp_ok &= n == N
        print("  inspection views %+.0f mm below the survey: %d/%d grid cells reachable" % (dz, n, N))
    can_align = pr[0] == pr[1]
    if gm[0] < 4:
        print("  WARNING: fewer than 4 calibration moves possible; grounding.py calibrate may fail")
    if not insp_ok:
        print("-> some inspection views are unreachable: use smaller VIEW_LEVELS_DZ steps or a smaller GRID_HALF_MM")
    return can_align, insp_ok


def accept_pose(anchor, img, what, err_px):
    """Use your hand pose unchanged as the survey pose (no robot motion)."""
    if os.path.exists(C.LOCKED_POSE_PATH):
        shutil.copy(C.LOCKED_POSE_PATH, C.LOCKED_POSE_PATH + ".bak_%s" % time.strftime("%Y%m%d_%H%M%S"))
    j = dict(anchor); j.update(source="auto_align.py", accepted_hand_pose=True, target=what, final_error_px=err_px,
                               saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(j, open(C.LOCKED_POSE_PATH, "w"), indent=2)
    os.makedirs(os.path.join(C.DEPLOY_DIR, "reference"), exist_ok=True)
    cv2.imwrite(os.path.join(C.DEPLOY_DIR, "reference", "reference.jpg"), img)
    print("ACCEPTED your hand pose unchanged as the survey pose (view offset %.0f px; the inspection grid is anchored"
          " to the drawn box, so the survey view does not need to be centred)." % err_px)
    print("saved %s; next: python3 grounding.py calibrate --mode track ; python3 table_grid.py --install" % C.LOCKED_POSE_PATH)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["auto", "box", "part"], default="auto")
    ap.add_argument("--check", action="store_true", help="no motion: report the offset and exit")
    ap.add_argument("--accept", action="store_true", help="no motion: use your hand pose unchanged as the survey pose")
    ap.add_argument("--max-travel", type=float, default=25.0, help="largest refinement from your hand pose (mm)")
    ap.add_argument("--tol-px", type=float, default=4.0)
    ap.add_argument("--max-iter", type=int, default=25)
    ap.add_argument("--probe-mm", type=float, default=4.0)
    ap.add_argument("--max-step", type=float, default=3.0)
    ap.add_argument("--gain", type=float, default=0.6)
    ap.add_argument("--speed", type=int, default=10)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))
    a = ap.parse_args()

    anchor = load_anchor(a.dry_run)
    q0 = np.array(anchor["angles_deg"], float)
    can_align, insp_ok = reach_report(q0, a.max_travel)
    if not insp_ok and not a.check:
        raise SystemExit("inspection views not reachable from this pose (see above)")
    T0 = fk(np.deg2rad(q0)); R_ref, p0 = T0[:3, :3], T0[:3, 3] * 1000
    HW.set_fence(auto_fence(a.max_travel + C.FENCE_MARGIN_MM + 2.0), p0)    # hard limit around YOUR pose
    arm = make_arm(a.dry_run, q0); arm.send_deg(q0, 15); wait_settled(arm)
    cam = LiveCam(make_camera(a.dry_run, arm, a.ref_img), a.show)
    out = os.path.join(C.LOGS_DIR, "align"); os.makedirs(out, exist_ok=True)
    print("live view: %s (serve %s and open live.html, or --show)" % (cam.path, C.LOGS_DIR))
    saved = False
    try:
        time.sleep(0.4)
        img0 = fresh(cam); cv2.imwrite(os.path.join(out, "align_before.jpg"), img0)
        c0, what, dets0, _ = find_target(img0, a.target)
        h, w = img0.shape[:2]; uc = np.array([w / 2.0, h / 2.0])
        if c0 is None:
            raise SystemExit("no target found (%s); markers %s. Jog with live_align so the box/part is in view."
                             % (what, sorted(map(str, dets0))))
        cam.state.update(c=c0, msg="target: " + what)
        e0 = c0 - uc
        print("target: %s at (%.0f, %.0f) px, image centre (%.0f, %.0f): offset %.0f px; markers %s"
              % (what, c0[0], c0[1], uc[0], uc[1], np.linalg.norm(e0), sorted(map(str, dets0))))

        def target_now(img, retries=2):
            """Target position via the markers (re-captures if markers are missing: fewer markers = less reliable)."""
            for r in range(retries + 1):
                dets = BD.detect_markers(img)
                sh, n = BD.marker_shift(dets0, dets) if dets0 else (None, 0)
                if sh is not None and (n >= min(2, len(dets0)) or r == retries):
                    c = c0 + np.asarray(sh, float); break
                if r < retries:
                    img = fresh(cam); continue
                c, _, _, _ = find_target(img, a.target); n = 0
            cam.state["c"] = c
            return c, n

        # rough reach check BEFORE any motion (uses the calibrated scale if available)
        ppm = 2.45
        if os.path.exists(C.GROUNDING_PATH):
            try:
                ppm = float(json.load(open(C.GROUNDING_PATH)).get("px_per_mm", ppm))
            except Exception:
                pass
        est_mm = np.linalg.norm(e0) / ppm
        print("estimated correction about %.0f mm (at ~%.2f px/mm); allowed %.0f mm" % (est_mm, ppm, a.max_travel))
        if a.accept:
            if not insp_ok:
                raise SystemExit("cannot accept: inspection views not reachable from this pose")
            if len(dets0) == 0:
                raise SystemExit("cannot accept: no marker visible at this pose")
            accept_pose(anchor, img0, what, float(np.linalg.norm(e0))); saved = True
            return
        if a.check or est_mm > a.max_travel or not can_align:
            if np.linalg.norm(e0) <= a.tol_px:
                print("view already centred.")
            elif not can_align:
                print("fine alignment is not possible at this height (the arm is at its reach limit outward).")
                print("-> the view is %.0f px off centre: if all markers are visible, use it as it is:  python3 auto_align.py --accept"
                      % np.linalg.norm(e0))
            elif est_mm > a.max_travel:
                print("TOO FAR for fine alignment. " + jog_hint(e0) + ", then press 'Lock pose' again.")
            else:
                print("view within reach: run  python3 auto_align.py  to refine (or --accept to keep it as it is).")
            print("inspection views: " + ("all reachable" if insp_ok else "NOT all reachable (see above)"))
            return

        # (1) probe image motion with +/- moves on both axes (central differences cancel backlash)
        cam.state["msg"] = "probing image motion"
        D, S = [], []
        for d in (np.array([a.probe_mm, 0.0]), np.array([-a.probe_mm, 0.0]),
                  np.array([0.0, a.probe_mm]), np.array([0.0, -a.probe_mm])):
            arm.send_deg(q0, 15); wait_settled(arm); time.sleep(0.3)
            if move_by(arm, q0, d, R_ref, a.speed) is None:
                raise SystemExit("probe move %s mm failed: %s" % (d.tolist(), LAST_FAIL))
            c, n = target_now(fresh(cam))
            if c is None:
                raise SystemExit("target lost while probing: use --probe-mm 2")
            D.append(d); S.append(c - c0)
        D, S = np.array(D), np.array(S)
        X, *_ = np.linalg.lstsq(D, S, rcond=None); A = X.T                 # content shift = A @ camera move
        fit = float(np.sqrt(((S - D @ X) ** 2).sum(1).mean()))
        print("image motion A = %s px/mm (axes %.2f / %.2f px/mm; probe fit residual %.1f px)"
              % (np.round(A, 2).tolist(), *np.linalg.norm(A, axis=0), fit))
        if abs(np.linalg.det(A)) < 0.3:
            raise SystemExit("image barely moves with the camera: check that the camera is fixed to the arm")
        need = np.linalg.solve(A, uc - c0)
        print("correction needed: (%+.1f, %+.1f) mm = %.1f mm" % (need[0], need[1], np.linalg.norm(need)))
        if np.linalg.norm(need) > 0.8 * a.max_travel:                      # keep headroom for the arm's joint errors
            arm.send_deg(q0, 15); wait_settled(arm)
            raise SystemExit("needs %.0f mm, more than 80%% of --max-travel %.0f mm. %s, lock again in live_align, and re-run."
                             % (np.linalg.norm(need), a.max_travel, jog_hint(e0)))

        # (2) servo in small steps from YOUR pose; Broyden update of A from what the image really did
        arm.send_deg(q0, 15); wait_settled(arm); time.sleep(0.3)
        q_cmd = q0.copy(); travel = np.zeros(2)
        c_prev = d_prev = None; n_prev = -1
        err = np.inf; best = (np.inf, q0.copy(), np.zeros(2))
        for it in range(a.max_iter):
            c, n = target_now(fresh(cam))
            if c is None:
                raise SystemExit("target lost; re-lock closer with live_align")
            if d_prev is not None and np.linalg.norm(d_prev) > 1.0 and n == n_prev and n > 0:
                pred, obs = A @ d_prev, c - c_prev
                if np.linalg.norm(obs - pred) < 0.6 * np.linalg.norm(pred):    # plausible: learn from it
                    A_new = A + 0.4 * np.outer(obs - pred, d_prev) / float(d_prev @ d_prev)
                    if np.linalg.det(A_new) * np.linalg.det(A) > 0:
                        A = A_new
            n_prev = n
            e = uc - c; err = float(np.linalg.norm(e))
            if err < best[0]:
                best = (err, q_cmd.copy(), travel.copy())
            msg = "step %d: error %.1f px (%.1f mm) [%d marker(s)], travel %.1f mm" % (
                it, err, err / max(np.linalg.norm(A, axis=0).mean(), 1e-6), n, np.linalg.norm(travel))
            print("  " + msg); cam.state["msg"] = msg
            if err <= a.tol_px:
                break
            d = a.gain * np.linalg.solve(A, e)
            if np.linalg.norm(d) > a.max_step:
                d *= a.max_step / np.linalg.norm(d)
            if np.linalg.norm(travel + d) > a.max_travel:
                print("  stop: next step would exceed --max-travel %.0f mm from your hand pose" % a.max_travel); break
            q_new = None
            for shrink in (1.0, 0.5, 0.25):                                 # smaller step if IK/safety refuses
                q_new = move_by(arm, q_cmd, d * shrink, R_ref, a.speed)
                if q_new is not None:
                    d = d * shrink; break
            if q_new is None:
                print("  stop: " + LAST_FAIL); break
            c_prev, d_prev = c, d
            q_cmd = q_new; travel = travel + d

        if err > best[0] + 2.0:                                             # return to the best view reached
            print("  returning to the best pose seen (error %.1f px)" % best[0])
            safe_move(arm, best[1], a.speed); q_cmd, travel = best[1], best[2]
        wait_settled(arm); time.sleep(0.3)
        img1 = fresh(cam); cv2.imwrite(os.path.join(out, "align_after.jpg"), img1)
        c_end, _ = target_now(img1)
        if c_end is not None:
            err = float(np.linalg.norm(uc - c_end))
        n_mk = len(BD.detect_markers(img1))
        if err > 3 * a.tol_px:
            print("NOT saved: final error %.1f px. Your hand pose is unchanged; jog closer in live_align and lock again." % err)
            return
        coords = None
        try:
            coords = arm.read_coords().tolist()
        except Exception:
            pass
        if os.path.exists(C.LOCKED_POSE_PATH):
            shutil.copy(C.LOCKED_POSE_PATH, C.LOCKED_POSE_PATH + ".bak_%s" % time.strftime("%Y%m%d_%H%M%S"))
        json.dump(dict(angles_deg=[float(v) for v in q_cmd], coords_mm=coords, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                       source="auto_align.py", target=what, final_error_px=err, travel_mm=travel.tolist(),
                       anchor_angles_deg=[float(v) for v in q0], image_motion_px_per_mm=A.tolist()),
                  open(C.LOCKED_POSE_PATH, "w"), indent=2)
        os.makedirs(os.path.join(C.DEPLOY_DIR, "reference"), exist_ok=True)
        cv2.imwrite(os.path.join(C.DEPLOY_DIR, "reference", "reference.jpg"), img1)
        saved = True
        print("\naligned on the %s: error %.1f px after %.1f mm of refinement; %d marker(s) visible"
              % (what, err, np.linalg.norm(travel), n_mk))
        print("saved %s (your hand pose stays in %s)" % (C.LOCKED_POSE_PATH, MANUAL))
        print("next: python3 grounding.py calibrate --mode track ; python3 table_grid.py --install")
    finally:
        if not saved:
            try:
                arm.send_deg(q0, 15); wait_settled(arm)
            except Exception:
                pass
        cam.release(); arm.close()


if __name__ == "__main__":
    main()
