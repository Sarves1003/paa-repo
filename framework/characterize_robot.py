#!/usr/bin/env python3
"""Robot characterization around YOUR reference pose (kinematic and motion-dynamic parameters).

  python3 characterize_robot.py all            # static + steps + repeat + backlash   (~15 min)
  python3 characterize_robot.py steps          # or one phase: static | steps | repeat | backlash
  options: --speeds 10 25 50   --repeats 15   --max-drop 30   --dry-run
Every move is small, checked before it is sent (camera never lower than --max-drop mm below the reference,
inside the workspace limit, joint step <= 20 deg), and recorded by robot_trace (commands, ~20 Hz joint
trajectories, servo status where the firmware provides it).

Phases (output folder hw_logs/characterize/<stamp>/):
  static    joint limits from the firmware; FK model vs firmware coordinates at the reference (10 reads)
            -> static_robot.json
  steps     step responses: each joint, 3 amplitudes, +/- direction, each speed; out and back
            -> trace files (moves.csv / samples.csv); Colab derives rise time, settling time, overshoot,
               steady-state error, peak velocity / acceleration, duration vs amplitude and speed (time model)
  repeat    N departures to random nearby poses and returns to the reference; at every return: read-back
            joints, firmware coordinates, and the marker positions in the camera image
            -> repeat.csv (encoder repeatability, camera-measured repeatability in px and mm)
  backlash  each joint approaches the reference alternately from + and - (5 deg) several times
            -> backlash.csv (joint read-back and image position by approach direction)
These are measured motion-dynamic properties; inertial parameters (masses, inertias) are NOT identifiable
without joint torque measurement and are not claimed.
"""
import argparse, csv, json, os, time
import numpy as np
import config as C
import hardware as HW
import robot_trace as RT
from hardware import make_arm, make_camera, safe_move, wait_settled, load_locked_pose
from kinematics import fk
from fence import auto_fence


def safe_target(q, q_ref, z_ref, max_drop):
    p = fk(np.radians(q))[:3, 3] * 1000
    return (p[2] >= z_ref - max_drop and HW.in_fence_q(q) and np.all(np.abs(q) < 160)
            and np.max(np.abs(q - q_ref)) <= 20.0 + 1e-9)


def markers_px(cam):
    import board as BD
    img = cam.grab()
    d = BD.detect_markers(img)
    return {str(k): np.asarray(v, float).mean(0) for k, v in d.items()}, img


def px_per_mm():
    for p, key in ((os.path.join(C.DEPLOY_DIR, "table_grid.json"), "box_px_per_mm"), (C.GROUNDING_PATH, "px_per_mm")):
        if os.path.exists(p):
            try:
                return float(json.load(open(p))[key])
            except Exception:
                pass
    return float("nan")


def phase_static(arm, q_ref, out):
    reads = []
    for _ in range(10):
        q = arm.read_deg(); c = arm.read_coords()
        reads.append(dict(q=q.tolist(), fw=c[:3].tolist(), fk=(fk(np.radians(q))[:3, 3] * 1000).tolist()))
        time.sleep(0.1)
    fw = np.array([r["fw"] for r in reads]); fkp = np.array([r["fk"] for r in reads])
    d = dict(joint_limits_deg=RT.joint_limits(arm), reference_deg=q_ref.tolist(),
             fk_minus_firmware_mm=np.nanmean(fkp - fw, 0).tolist(),
             fk_firmware_dist_mm=float(np.nanmean(np.linalg.norm(fkp - fw, axis=1))),
             encoder_readback_sd_deg=np.std([r["q"] for r in reads], 0).tolist(), reads=reads)
    json.dump(d, open(os.path.join(out, "static_robot.json"), "w"), indent=2)
    print("static: |FK - firmware| = %.2f mm at the reference; joint limits %s"
          % (d["fk_firmware_dist_mm"], d["joint_limits_deg"] or "not reported by this firmware"))


def phase_steps(arm, q_ref, z_ref, a):
    amps = {1: (3, 6, 10), 2: (3, 6, 10), 3: (3, 6, 10), 4: (5, 10, 20), 5: (5, 10, 20), 6: (5, 10, 20)}
    n = done = 0
    for sp in a.speeds:
        for j in range(1, 7):
            for amp in amps[j]:
                for sg in (1, -1):
                    q = q_ref.copy(); q[j - 1] += sg * amp; n += 1
                    if not safe_target(q, q_ref, z_ref, a.max_drop):
                        continue
                    if safe_move(arm, q, sp)[0]:
                        safe_move(arm, q_ref, sp); done += 1
        print("  steps at speed %d done" % sp)
    print("steps: %d of %d step pairs executed (others skipped by the safety check)" % (done, n))


def phase_repeat(arm, cam, q_ref, z_ref, a, out):
    rng = np.random.default_rng(1)
    safe_move(arm, q_ref, 15); time.sleep(0.5)
    m0, _ = markers_px(cam); ppm = px_per_mm()
    rows = []
    for i in range(a.repeats):
        for _ in range(30):
            dq = rng.uniform(-1, 1, 6) * np.array([8, 4, 4, 8, 8, 8])
            if safe_target(q_ref + dq, q_ref, z_ref, a.max_drop):
                break
        safe_move(arm, q_ref + dq, 20); safe_move(arm, q_ref, 20); time.sleep(0.4)
        q = arm.read_deg(); c = arm.read_coords(); m, _ = markers_px(cam)
        common = [k for k in m if k in m0]
        sh = np.median([m[k] - m0[k] for k in common], 0) if common else np.full(2, np.nan)
        rows.append(dict(i=i, **{"q_%d" % (j + 1): q[j] for j in range(6)}, fw_x=c[0], fw_y=c[1], fw_z=c[2],
                         img_dx_px=sh[0], img_dy_px=sh[1], img_err_mm=float(np.linalg.norm(sh) / ppm),
                         n_markers=len(common), **{"dep_%d" % (j + 1): dq[j] for j in range(6)}))
    with open(os.path.join(out, "repeat.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    Q = np.array([[r["q_%d" % (j + 1)] for j in range(6)] for r in rows])
    e = np.array([r["img_err_mm"] for r in rows])
    print("repeat: joint read-back SD %s deg; camera-measured return error median %.2f mm (max %.2f)"
          % (np.round(Q.std(0), 3).tolist(), np.nanmedian(e), np.nanmax(e)))


def phase_backlash(arm, cam, q_ref, z_ref, a, out):
    safe_move(arm, q_ref, 15); time.sleep(0.4)
    m0, _ = markers_px(cam); rows = []
    for j in range(1, 7):
        for rep in range(a.backlash_reps):
            for sg in (1, -1):
                q = q_ref.copy(); q[j - 1] += sg * 5.0
                if not safe_target(q, q_ref, z_ref, a.max_drop):
                    continue
                safe_move(arm, q, 15); safe_move(arm, q_ref, 15); time.sleep(0.3)
                qr = arm.read_deg(); m, _ = markers_px(cam)
                common = [k for k in m if k in m0]
                sh = np.median([m[k] - m0[k] for k in common], 0) if common else np.full(2, np.nan)
                rows.append(dict(joint=j, rep=rep, approach=sg, q_read=qr[j - 1], q_err=qr[j - 1] - q_ref[j - 1],
                                 img_dx_px=sh[0], img_dy_px=sh[1]))
    with open(os.path.join(out, "backlash.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import collections
    B = collections.defaultdict(dict)
    for j in range(1, 7):
        for sg in (1, -1):
            v = [r["q_err"] for r in rows if r["joint"] == j and r["approach"] == sg]
            if v:
                B[j][sg] = np.mean(v)
    print("backlash (joint read-back, approach from + minus from -, deg): " +
          ", ".join("J%d %.2f" % (j, B[j][1] - B[j][-1]) for j in B if 1 in B[j] and -1 in B[j]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["all", "static", "steps", "repeat", "backlash"])
    ap.add_argument("--speeds", type=int, nargs="+", default=[10, 25, 50])
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--backlash-reps", type=int, default=4)
    ap.add_argument("--max-drop", type=float, default=30.0, help="camera never lower than this below the reference (mm)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))
    a = ap.parse_args()
    out = os.path.join(C.LOGS_DIR, "characterize", time.strftime("%Y%m%d_%H%M%S"))
    RT.install(out)
    q_ref = np.array(load_locked_pose(a.dry_run)["angles_deg"], float)
    z_ref = fk(np.radians(q_ref))[2, 3] * 1000
    HW.set_fence(auto_fence(), fk(np.radians(q_ref))[:3, 3] * 1000)
    arm = make_arm(a.dry_run, q_ref); arm.send_deg(q_ref, 15); wait_settled(arm)
    cam = None
    try:
        if a.phase in ("all", "static"):
            RT.event("phase_static")
            phase_static(arm, q_ref, out)
        if a.phase in ("all", "steps"):
            RT.event("phase_steps")
            phase_steps(arm, q_ref, z_ref, a)
        if a.phase in ("all", "repeat", "backlash"):
            cam = make_camera(a.dry_run, arm, a.ref_img)
        if a.phase in ("all", "repeat"):
            RT.event("phase_repeat")
            phase_repeat(arm, cam, q_ref, z_ref, a, out)
        if a.phase in ("all", "backlash"):
            RT.event("phase_backlash")
            phase_backlash(arm, cam, q_ref, z_ref, a, out)
    finally:
        try:
            safe_move(arm, q_ref, 15)
        finally:
            if cam is not None:
                cam.release()
            arm.close()
    print("saved in", out)


if __name__ == "__main__":
    main()
