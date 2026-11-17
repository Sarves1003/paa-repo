#!/usr/bin/env python3
"""Move the camera to the reference position recommended by reference_advisor.py and make it the new reference.

  python3 goto_reference.py                # move (straight line, constant height, orientation held, 5 mm steps)
  python3 goto_reference.py --save         # move and save it as deploy/locked_pose.json (old pose backed up)
  python3 goto_reference.py --back         # return to the backed-up old reference
The camera moves horizontally at the reference height only, so it never descends toward the part.
Afterwards slide the sheet (box + markers) by the same amount toward the robot, until
  python3 auto_align.py --check   reports the box centred, then run  python3 auto_align.py  (fine centring).
"""
import argparse, json, os, shutil, time
import numpy as np
import config as C
import hardware as HW
from hardware import make_arm, wait_settled, safe_move
from kinematics import fk
from ik import ik_hold

ADV = os.path.join("results_views", "reference_advisor.json")
BAK = os.path.join(C.DEPLOY_DIR, "locked_pose_before_advisor.json")


def line_move(arm, p_from, p_to, R, q_seed, speed, step_mm=5.0):
    n = max(1, int(np.ceil(np.linalg.norm(p_to - p_from) / step_mm)))
    q = np.radians(q_seed)
    for i in range(1, n + 1):
        p = p_from + (p_to - p_from) * i / n
        qi, pe, re, ok = ik_hold(p / 1000.0, q, R, iters=1500)
        if not ok:
            raise SystemExit("IK failed %.0f mm along the path: stopped (arm is safe where it is)" % (np.linalg.norm(p - p_from)))
        moved, _, _ = safe_move(arm, np.degrees(qi), speed, max_step=12.0)
        if not moved:
            raise SystemExit("move refused: %s" % HW.LAST_REFUSAL)
        q = qi
        print("  %3d/%d  camera at (%.0f, %.0f, %.0f) mm" % (i, n, *p), flush=True)
    return np.degrees(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="save the reached pose as the new reference")
    ap.add_argument("--back", action="store_true", help="return to the backed-up old reference")
    ap.add_argument("--speed", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    lock = json.load(open(C.LOCKED_POSE_PATH))
    q_now = np.array(lock["angles_deg"], float)
    T = fk(np.radians(q_now)); R, p_now = T[:3, :3], T[:3, 3] * 1000
    if a.back:
        if not os.path.exists(BAK):
            raise SystemExit("no backup %s" % BAK)
        old = np.array(json.load(open(BAK))["angles_deg"], float); p_to = fk(np.radians(old))[:3, 3] * 1000
    else:
        if not os.path.exists(ADV):
            raise SystemExit("run  python3 reference_advisor.py  first")
        best = json.load(open(ADV))["best"]
        p_to = p_now + np.r_[np.asarray(best["move_box_mm"], float), 0.0]
        print("advisor: move %.0f mm toward the robot base and %.0f mm sideways -> %.0f%% of the views reachable "
              "(kappa %.3f)" % (-best["shift_radial_mm"], best["shift_lateral_mm"], best["reachable_pct"], best["kappa_ref"]))
    if np.linalg.norm(p_to - p_now) < 1.0:
        raise SystemExit("already there")
    HW.set_fence(None, None)                                  # this move is checked by IK + joint-step limit only
    arm = make_arm(a.dry_run, q_now); arm.send_deg(q_now, a.speed); wait_settled(arm)
    print("moving the camera %.0f mm at constant height %.0f mm ..." % (np.linalg.norm(p_to - p_now), p_now[2]))
    q_end = line_move(arm, p_now, p_to, R, arm.read_deg(), a.speed)
    wait_settled(arm); time.sleep(0.5)
    q_meas = np.asarray(arm.read_deg(), float)
    print("reached joints (deg):", np.round(q_meas, 2).tolist())
    if a.save or a.back:
        if not a.back and not os.path.exists(BAK):
            shutil.copy(C.LOCKED_POSE_PATH, BAK)
        out = dict(lock); out["angles_deg"] = [float(x) for x in q_meas]
        out["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S"); out["source"] = "goto_reference.py" + (" --back" if a.back else "")
        json.dump(out, open(C.LOCKED_POSE_PATH, "w"), indent=2)
        print("saved %s (old reference kept as %s)" % (C.LOCKED_POSE_PATH, BAK))
    arm.close()


if __name__ == "__main__":
    main()
