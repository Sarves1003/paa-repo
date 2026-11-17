#!/usr/bin/env python3
"""Motion study for camera-in-hand inspection: trajectory profiles, jerk, image settling, servo dynamics.

Runs on the Jetson with the arm, the camera and your existing deploy/ files (locked_pose.json,
lattice_views.json, fence). Every target is a pose that is already in your certified lattice, or a small
offset (<= 6 deg) from the locked reference pose; every streamed waypoint is checked before the move starts.

  python3 motion_study.py profiles          # 5 inspection moves x 5 profiles x 3 repeats   (~25 min)
  python3 motion_study.py excite            # small sinusoids per joint -> servo frequency response (~15 min)
  python3 motion_study.py hold              # static holds at 9 configurations -> servo currents vs pose (~5 min)
  python3 motion_study.py all               # all three
  options: --repeats 3  --rate 20  --vmax 30  --max-drop 30  --dry-run

Output: hw_logs/motion/<stamp>/  trials.csv, frames/<trial>.npz, plus robot_trace files
(stream.csv joints ~20 Hz, status.csv servo status incl. currents where the firmware gives them,
commands.csv every command sent, events.csv). Zip the whole folder and upload it.

Profiles compared on the same start/goal poses:
  native20, native40  one send_angles at speed 20 / 40 (the firmware's own interpolation)
  quintic             minimum-jerk  s(u) = 10u^3 - 15u^4 + 6u^5, streamed at --rate Hz
  septic              zero-jerk ends s(u) = 35u^4 - 84u^5 + 70u^6 - 20u^7, streamed
  quintic_fast        quintic with half the duration (to test the speed-jerk-settling trade-off)
Duration of streamed profiles: T = c * max|dq| / vmax  (c = 1.875 quintic, 2.1875 septic), at least 0.8 s.
"""
import argparse, csv, json, os, threading, time
import numpy as np
import config as C
import hardware as HW
import robot_trace as RT
from hardware import make_arm, make_camera, safe_move, wait_settled, load_locked_pose
from kinematics import fk
from fence import auto_fence

PROFILES = ["native20", "native40", "quintic", "septic", "quintic_fast"]


def s_profile(u, kind):
    u = np.clip(u, 0.0, 1.0)
    if kind.startswith("quintic"):
        return 10 * u**3 - 15 * u**4 + 6 * u**5
    if kind == "septic":
        return 35 * u**4 - 84 * u**5 + 70 * u**6 - 20 * u**7
    raise ValueError(kind)


def safe_q(q, q_ref, z_ref, max_drop):
    p = fk(np.radians(q))[:3, 3] * 1000
    return bool(p[2] >= z_ref - max_drop - 1e-6 and HW.in_fence_q(q) and np.all(np.abs(q) < 165))


# ------------------------------------------------------------------ camera recorder
class FrameRecorder:
    """Grabs frames continuously in a thread; keeps time stamps, full-resolution sharpness
    (variance of the Laplacian) and a 160x120 grey copy for offline image-motion analysis."""

    def __init__(self, cam):
        self.cam, self.on, self.th = cam, False, None
        import cv2
        self.cv2 = cv2

    def start(self):
        self.t, self.sharp, self.small = [], [], []
        self.on = True
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()

    def _run(self):
        cap = getattr(self.cam, "cap", None)
        while self.on:
            if cap is not None:
                ok, f = cap.read()
                if not ok or f is None:
                    time.sleep(0.01); continue
            else:                                       # dry-run camera
                f = self.cam.grab(); time.sleep(0.04)
            t = time.time()
            g = self.cv2.cvtColor(f, self.cv2.COLOR_BGR2GRAY)
            self.t.append(t)
            self.sharp.append(float(self.cv2.Laplacian(g, self.cv2.CV_64F).var()))
            self.small.append(self.cv2.resize(g, (160, 120), interpolation=self.cv2.INTER_AREA))

    def stop(self, path):
        self.on = False
        if self.th is not None:
            self.th.join(timeout=2.0)
        np.savez_compressed(path, t=np.array(self.t), sharp=np.array(self.sharp, np.float32),
                            small=np.array(self.small, np.uint8) if self.small else np.zeros((0, 120, 160), np.uint8))
        return len(self.t)


# ------------------------------------------------------------------ motions
def stream_profile(arm, qa, qb, kind, a):
    dq = np.max(np.abs(qb - qa))
    c = 1.875 if kind.startswith("quintic") else 2.1875
    T = max(0.8, c * dq / a.vmax)
    if kind == "quintic_fast":
        T = max(0.5, T / 2)
    n = max(2, int(np.ceil(T * a.rate)))
    W = [qa + s_profile(i / n, kind) * (qb - qa) for i in range(1, n + 1)]
    t0 = time.time()
    for i, q in enumerate(W, 1):
        arm.send_deg(q, a.stream_speed)
        dt = t0 + i / a.rate - time.time()
        if dt > 0:
            time.sleep(dt)
    arm.send_deg(qb, a.stream_speed)
    return T, n


def check_path(qa, qb, kind, q_ref, z_ref, a):
    pts = [qa + s * (qb - qa) for s in np.linspace(0, 1, 41)]
    return all(safe_q(q, q_ref, z_ref, a.max_drop) for q in pts)


def pick_moves(LV, q_ref):
    """Representative inspection moves from the certified lattice: survey -> view, lateral and diagonal
    moves on one level, straight -> tilted on one cell, and a tilt switch."""
    V = {(v["level"], v["cell"]): np.array(v["q_deg"], float) for v in LV["views"] if v.get("ok")}
    want = [("survey_to_L1c12", None, (1, 12)), ("lateral_L1c12_c13", (1, 12), (1, 13)),
            ("diagonal_L1c6_c18", (1, 6), (1, 18)), ("straight_to_tilt_L1c12_L3c12", (1, 12), (3, 12)),
            ("tilt_switch_L2c12_L3c12", (2, 12), (3, 12))]
    alt = {(1, 13): [(1, 11), (1, 7), (1, 17)], (1, 6): [(1, 8), (1, 16)], (1, 18): [(1, 16), (1, 8)],
           (3, 12): [(3, 7), (3, 17), (3, 11)], (2, 12): [(2, 7), (2, 17), (2, 13)], (1, 12): [(1, 7), (1, 17)]}
    moves = []
    for name, A, Bk in want:
        def get(key):
            if key is None:
                return q_ref.copy()
            for k in [key] + alt.get(key, []):
                if k in V:
                    return V[k]
            return None
        qa, qb = get(A), get(Bk)
        if qa is not None and qb is not None:
            moves.append((name, qa, qb))
        else:
            print("  move %s skipped: lattice view not certified" % name)
    return moves


def phase_profiles(arm, cam, q_ref, z_ref, a, out):
    import lattice_views as LVmod
    LV = LVmod.load_views()
    if LV is None:
        raise SystemExit("deploy/lattice_views.json not found (run lattice_views.py first)")
    moves = pick_moves(LV, q_ref)
    rec = FrameRecorder(cam); os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    rows, tid = [], 0
    for rep in range(a.repeats):
        for name, qa0, qb0 in moves:
            for direction in (+1, -1):
                qa, qb = (qa0, qb0) if direction > 0 else (qb0, qa0)
                for kind in (PROFILES if rep % 2 == 0 else PROFILES[::-1]):
                    if not check_path(qa, qb, kind, q_ref, z_ref, a):
                        print("  %s %s refused by the safety check" % (name, kind)); continue
                    ok = safe_move(arm, q_ref, 15)[0] and safe_move(arm, qa, 15)[0]
                    if not ok:
                        print("  could not reach start of", name); continue
                    time.sleep(0.6)
                    tid += 1; label = "trial_%04d" % tid
                    RT.event(label, move=name, profile=kind, direction=direction, rep=rep)
                    rec.start(); time.sleep(0.4)
                    t_cmd = time.time()
                    if kind.startswith("native"):
                        arm.send_deg(qb, int(kind[6:])); T_plan, n_wp = float("nan"), 1
                    else:
                        T_plan, n_wp = stream_profile(arm, qa, qb, kind, a)
                    t_cmd_end = time.time()
                    time.sleep(a.post)                          # keep recording while the arm and image settle
                    nfr = rec.stop(os.path.join(out, "frames", label + ".npz"))
                    q_end = np.mean([arm.read_deg() for _ in range(3)], 0)
                    RT.event("idle")
                    rows.append(dict(trial=label, move=name, profile=kind, direction=direction, rep=rep,
                                     t_cmd=t_cmd, t_cmd_end=t_cmd_end, T_plan=T_plan, n_waypoints=n_wp,
                                     dq_max=float(np.max(np.abs(qb - qa))), n_frames=nfr,
                                     **{"qa_%d" % (j + 1): qa[j] for j in range(6)},
                                     **{"qb_%d" % (j + 1): qb[j] for j in range(6)},
                                     **{"qend_%d" % (j + 1): q_end[j] for j in range(6)}))
                    print("  %s %-28s %-12s dir %+d  frames %d" % (label, name, kind, direction, nfr))
            write_csv(os.path.join(out, "trials.csv"), rows)
    safe_move(arm, q_ref, 15)


def phase_excite(arm, q_ref, z_ref, a, out):
    """Small sinusoids q_j(t) = q_ref_j + A sin(2 pi f t) streamed at --rate Hz; the response from
    stream.csv gives gain and phase lag per joint and frequency (servo bandwidth and delay)."""
    amps = {1: 4.0, 2: 2.0, 3: 2.0, 4: 4.0, 5: 4.0, 6: 4.0}
    rows = []
    for j in range(1, 7):
        for f in a.freqs:
            T = max(4.0 / f, 3.0); n = int(T * a.rate)
            W = []
            for i in range(n + 1):
                q = q_ref.copy(); q[j - 1] += amps[j] * np.sin(2 * np.pi * f * i / a.rate); W.append(q)
            if not all(safe_q(q, q_ref, z_ref, a.max_drop) for q in W):
                print("  joint %d f=%.2f refused" % (j, f)); continue
            safe_move(arm, q_ref, 15); time.sleep(0.5)
            label = "excite_j%d_f%.2f" % (j, f)
            RT.event(label, joint=j, freq=f, amp=amps[j])
            t0 = time.time()
            for i, q in enumerate(W):
                arm.send_deg(q, a.stream_speed)
                dt = t0 + (i + 1) / a.rate - time.time()
                if dt > 0:
                    time.sleep(dt)
            time.sleep(0.8); RT.event("idle")
            rows.append(dict(label=label, joint=j, freq=f, amp=amps[j], t0=t0, T=T, rate=a.rate))
            print("  %s done" % label)
        write_csv(os.path.join(out, "excite.csv"), rows)
    safe_move(arm, q_ref, 15)


def phase_hold(arm, q_ref, z_ref, a, out):
    """Holds at 9 configurations (joint 2 and joint 3 offsets) and reads the servo status for 3 s each."""
    rows = []
    for d2 in (-6, 0, 6):
        for d3 in (-6, 0, 6):
            q = q_ref.copy(); q[1] += d2; q[2] += d3
            if not safe_q(q, q_ref, z_ref, a.max_drop):
                print("  hold d2=%d d3=%d refused" % (d2, d3)); continue
            safe_move(arm, q, 15); time.sleep(1.0)
            label = "hold_d2_%+d_d3_%+d" % (d2, d3)
            RT.event(label, d2=d2, d3=d3)
            vals = {}
            t_end = time.time() + 3.0
            mc = getattr(arm, "mc", None)
            while time.time() < t_end:
                for name in ("get_servo_currents", "get_servo_voltages", "get_servo_temps"):
                    fn = getattr(mc, name, None) if mc is not None else None
                    if fn is None:
                        continue
                    try:
                        v = fn()
                        if isinstance(v, (list, tuple)) and len(v) >= 6:
                            vals.setdefault(name, []).append([float(x) for x in v[:6]])
                    except Exception:
                        pass
                time.sleep(0.1)
            RT.event("idle")
            q_meas = np.mean([arm.read_deg() for _ in range(3)], 0)
            r = dict(label=label, d2=d2, d3=d3, **{"q_%d" % (j + 1): q_meas[j] for j in range(6)})
            for name, L in vals.items():
                m = np.median(np.array(L), 0)
                for j in range(6):
                    r["%s_%d" % (name.replace("get_servo_", ""), j + 1)] = m[j]
            rows.append(r); print("  %s: %s" % (label, ", ".join(sorted(vals)) or "no servo status from firmware"))
    write_csv(os.path.join(out, "hold.csv"), rows)
    safe_move(arm, q_ref, 15)


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["all", "profiles", "excite", "hold"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rate", type=float, default=20.0, help="waypoint rate for streamed profiles (Hz)")
    ap.add_argument("--vmax", type=float, default=30.0, help="peak joint speed of streamed profiles (deg/s)")
    ap.add_argument("--stream-speed", type=int, default=60, help="myCobot speed used for each streamed waypoint")
    ap.add_argument("--post", type=float, default=2.5, help="seconds recorded after the last command")
    ap.add_argument("--freqs", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5])
    ap.add_argument("--max-drop", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))
    a = ap.parse_args()
    out = os.path.join(C.LOGS_DIR, "motion", time.strftime("%Y%m%d_%H%M%S"))
    RT.install(out, stream_hz=25.0, status_hz=5.0)
    q_ref = np.array(load_locked_pose(a.dry_run)["angles_deg"], float)
    z_ref = fk(np.radians(q_ref))[2, 3] * 1000
    HW.set_fence(auto_fence(), fk(np.radians(q_ref))[:3, 3] * 1000)
    arm = make_arm(a.dry_run, q_ref)
    try:
        RT.start_stream(arm)
    except Exception:
        pass
    arm.send_deg(q_ref, 15); wait_settled(arm)
    json.dump(dict(args=vars(a), q_ref=q_ref.tolist(), z_ref=z_ref, start=time.time()),
              open(os.path.join(out, "run.json"), "w"), indent=1)
    cam = None
    try:
        if a.phase in ("all", "profiles"):
            cam = make_camera(a.dry_run, arm, a.ref_img)
            RT.event("phase_profiles"); phase_profiles(arm, cam, q_ref, z_ref, a, out)
        if a.phase in ("all", "excite"):
            RT.event("phase_excite"); phase_excite(arm, q_ref, z_ref, a, out)
        if a.phase in ("all", "hold"):
            RT.event("phase_hold"); phase_hold(arm, q_ref, z_ref, a, out)
    finally:
        try:
            RT.stop_stream()
        except Exception:
            pass
        if cam is not None:
            cam.release()
        arm.close()
    print("\nDone. Zip and upload:  cd %s/.. && zip -qr motion_%s.zip %s" % (out, os.path.basename(out), os.path.basename(out)))


if __name__ == "__main__":
    main()
