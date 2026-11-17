"""Trajectory and robot-state recorder. One call at the start of a script records EVERY robot command and
the joint trajectory while the arm moves, without changing hardware.py:

    import robot_trace as RT; RT.install("hw_logs/<experiment>/<run>")      # before the first robot move

Files written to that folder (CSV, one header line):
  commands.csv  cmd_id, t, q_cmd_1..6 (deg), speed, source        every command sent to the arm
  samples.csv   cmd_id, t, q_1..6 (deg)                            joint angles polled while the arm moves/settles (~20 Hz)
  moves.csv     cmd_id, t_cmd, t_end, duration_s, settle_s, q_from_1..6, q_cmd_1..6, q_end_1..6, dq_max_deg,
                speed, ss_err_max_deg, refused, refusal          one row per completed or refused move
  status.csv    cmd_id, t, kind, v_1..v_6                         servo status after each settled move, when the
                firmware provides it (kinds: encoders, speeds, currents, voltages, temps, is_moving, error)
  static.json   joint limits, firmware/library versions, kinematic model, time of start
  stream.csv    CONTINUOUS, in parallel with the inspection (background thread, default 15 Hz, also while the arm
                is idle, capturing or the AI is deciding): t, cmd_id, event, q_1..6 (deg), qd_1..6 (deg/s),
                qdd_1..6 (deg/s^2), x/y/z_mm (flange, FK), v_cart_mm_s, kappa (conditioning), moving
  events.csv    t, name, info (JSON): inspection events (survey, decision, view reached, verified, captured, ...)
                written by the scripts through robot_trace.event(), to align dynamics with inspection steps
All serial access to the arm goes through one lock, so the background reader never collides with commands.
Derived quantities (velocity, acceleration, jerk, rise/settling time, overshoot, FK position, conditioning
kappa) are computed later in Colab (export_for_colab.py adds FK and kappa to the samples).
"""
import csv, json, os, sys, threading, time
import numpy as np
import hardware as HW

_STATE = dict(on=False)
_OPTIONAL = ["get_encoders", "get_servo_speeds", "get_servo_currents", "get_servo_voltages", "get_servo_temps",
             "is_moving", "get_error_information"]


class _Writer:
    def __init__(self, path, header):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline=""); self.w = csv.writer(self.f); self.lk = threading.Lock()
        if new:
            self.w.writerow(header)

    def row(self, r):
        with self.lk:
            self.w.writerow(r); self.f.flush()


def _q(arm):
    try:
        return np.asarray(arm.read_deg(), float)
    except Exception:
        return np.full(6, np.nan)


def _status(arm, cid):
    mc = getattr(arm, "mc", None)
    if mc is None:
        return
    for name in list(_STATE["optional"]):
        fn = getattr(mc, name, None)
        if fn is None:
            _STATE["optional"].discard(name); continue
        try:
            with _STATE["lock"]:
                v = fn()
        except TypeError:                                  # per-joint signature in some versions
            try:
                with _STATE["lock"]:
                    v = [fn(j) for j in range(1, 7)]
            except Exception:
                _STATE["optional"].discard(name); continue
        except Exception:
            _STATE["optional"].discard(name); continue
        if v is None or v == -1:
            _STATE["optional"].discard(name); continue
        vals = list(v) if isinstance(v, (list, tuple)) else [v]
        _STATE["status"].row([cid, time.time(), name.replace("get_", "")] + [float(x) if isinstance(x, (int, float)) else str(x)
                                                                            for x in vals[:6]])


def install(run_dir, dt=0.05, status=True, stream_hz=15.0, status_hz=1.0):
    """Start recording into run_dir (created). Safe to call once per script; later calls switch the folder.
    stream_hz > 0 starts the continuous parallel sampler when the arm is created (0 = off)."""
    os.makedirs(run_dir, exist_ok=True)
    S = _STATE
    S.update(dir=run_dir, dt=dt, status=status, cid=0, last=None, optional=set(_OPTIONAL) if status else set())
    S["commands"] = _Writer(os.path.join(run_dir, "commands.csv"),
                            ["cmd_id", "t"] + ["q_cmd_%d" % i for i in range(1, 7)] + ["speed", "source"])
    S["samples"] = _Writer(os.path.join(run_dir, "samples.csv"), ["cmd_id", "t"] + ["q_%d" % i for i in range(1, 7)])
    S["moves"] = _Writer(os.path.join(run_dir, "moves.csv"),
                         ["cmd_id", "t_cmd", "t_end", "duration_s", "settle_s"] + ["q_from_%d" % i for i in range(1, 7)]
                         + ["q_cmd_%d" % i for i in range(1, 7)] + ["q_end_%d" % i for i in range(1, 7)]
                         + ["dq_max_deg", "speed", "ss_err_max_deg", "refused", "refusal"])
    S["status"] = _Writer(os.path.join(run_dir, "status.csv"), ["cmd_id", "t", "kind"] + ["v_%d" % i for i in range(1, 7)])
    S["events"] = _Writer(os.path.join(run_dir, "events.csv"), ["t", "name", "info"])
    S["stream"] = _Writer(os.path.join(run_dir, "stream.csv"),
                          ["t", "cmd_id", "event"] + ["q_%d" % i for i in range(1, 7)] + ["qd_%d" % i for i in range(1, 7)]
                          + ["qdd_%d" % i for i in range(1, 7)] + ["x_mm", "y_mm", "z_mm", "v_cart_mm_s", "kappa", "moving"])
    S.update(stream_hz=stream_hz, status_hz=status_hz, label="")
    if S["on"]:
        return run_dir
    S["on"] = True
    S["lock"] = threading.RLock()
    import config as C
    _patch_serial()
    for cls in (HW.Arm, HW.DryArm):
        orig_send = cls.send_deg

        def send(self, q_deg, speed=C.MOVE_SPEED, _o=orig_send):
            S["cid"] += 1
            q_from = _q(self)
            S["last"] = dict(cid=S["cid"], t=time.time(), q_cmd=np.asarray(q_deg, float), q_from=q_from, speed=speed)
            frame = sys._getframe(1)
            S["commands"].row([S["cid"], S["last"]["t"]] + list(np.round(S["last"]["q_cmd"], 3)) +
                              [speed, "%s:%s" % (os.path.basename(frame.f_code.co_filename), frame.f_code.co_name)])
            with S["lock"]:
                return _o(self, q_deg, speed)
        cls.send_deg = send

    orig_wait = HW.wait_settled

    def wait_settled(arm, timeout=None, tol=None, need=None, dt=0.12):
        t0 = time.time()
        timeout = C.SETTLE_TIMEOUT_S if timeout is None else timeout
        tol = C.SETTLE_TOL_DEG if tol is None else tol
        need = C.SETTLE_COUNT if need is None else need
        dt = min(dt, S["dt"]); cid = S["cid"]
        last, still = arm.read_deg(), 0
        S["samples"].row([cid, time.time()] + list(np.round(last, 3)))
        while time.time() - t0 < timeout:
            time.sleep(dt)
            q = arm.read_deg()
            S["samples"].row([cid, time.time()] + list(np.round(q, 3)))
            still = still + 1 if np.max(np.abs(q - last)) < tol else 0
            last = q
            if still >= need:
                break
        q = np.mean([arm.read_deg() for _ in range(3)], axis=0)
        L = S.get("last")
        if L and L["cid"] == cid:
            t_end = time.time()
            S["moves"].row([cid, L["t"], t_end, t_end - L["t"], t_end - t0] + list(np.round(L["q_from"], 3))
                           + list(np.round(L["q_cmd"], 3)) + list(np.round(q, 3))
                           + [float(np.max(np.abs(L["q_cmd"] - L["q_from"]))), L["speed"],
                              float(np.max(np.abs(q - L["q_cmd"]))), 0, ""])
            if S["status"]:
                _status(arm, cid)
        return time.time() - t0, q

    orig_safe = HW.safe_move

    def safe_move(arm, q_target_deg, speed=C.MOVE_SPEED, max_step=C.MAX_JOINT_STEP_DEG):
        res = orig_safe(arm, q_target_deg, speed, max_step)
        if not res[0]:
            q = res[2]
            S["moves"].row([S["cid"], time.time(), time.time(), 0, 0] + list(np.round(q, 3))
                           + list(np.round(np.asarray(q_target_deg, float), 3)) + list(np.round(q, 3))
                           + [float(np.max(np.abs(np.asarray(q_target_deg) - q))), speed, "", 1, HW.LAST_REFUSAL])
        return res

    HW.wait_settled, HW.safe_move = wait_settled, safe_move
    for m in list(sys.modules.values()):                   # rebind names already imported elsewhere
        if m is None or m is HW:
            continue
        if getattr(m, "wait_settled", None) is orig_wait:
            m.wait_settled = wait_settled
        if getattr(m, "safe_move", None) is orig_safe:
            m.safe_move = safe_move
    _static(run_dir)
    print("trace: recording robot commands and trajectories in", run_dir)
    return run_dir


# ------------------------------------------------------------------ parallel sampler
def _kappa(q, L=0.1, eps=1e-6):
    from kinematics import fk
    T = fk(q); p, R = T[:3, 3], T[:3, :3]; J = np.zeros((6, 6))
    for i in range(6):
        q2 = np.array(q, float); q2[i] += eps; T2 = fk(q2); dR = T2[:3, :3] @ R.T
        J[:3, i] = (T2[:3, 3] - p) / eps / L
        J[3:, i] = 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / eps
    sv = np.linalg.svd(J, compute_uv=False)
    return float(sv[-1] / sv[0])


def _patch_serial():
    """Every serial access of the arm (reads, commands, close) goes through one re-entrant lock; creating an arm
    starts the parallel sampler, closing it stops the sampler first."""
    S = _STATE
    for cls in (HW.Arm, HW.DryArm):
        for name in ("read_deg", "read_coords"):
            orig = getattr(cls, name)

            def locked(self, *a, _o=orig, **k):
                with S["lock"]:
                    return _o(self, *a, **k)
            setattr(cls, name, locked)
        S.setdefault("raw_read", {})[cls] = cls.read_deg
        o_init, o_close = cls.__init__, cls.close

        def init(self, *a, _o=o_init, **k):
            _o(self, *a, **k)
            start_stream(self)

        def close(self, *a, _o=o_close, **k):
            stop_stream()
            with S["lock"]:
                return _o(self, *a, **k)
        cls.__init__, cls.close = init, close


def start_stream(arm):
    S = _STATE
    if not S.get("stream_hz") or S.get("thread") is not None:
        return
    S["stop"] = threading.Event()
    th = threading.Thread(target=_sampler, args=(arm,), daemon=True); S["thread"] = th; th.start()


def stop_stream():
    S = _STATE
    if S.get("thread") is not None:
        S["stop"].set(); S["thread"].join(timeout=2.0); S["thread"] = None


def _sampler(arm):
    from kinematics import fk
    S = _STATE
    period = 1.0 / float(S["stream_hz"]); n_status = max(1, int(round(S["stream_hz"] / max(S["status_hz"], 1e-6))))
    prev = None; prev_qd = np.zeros(6); i = 0
    while not S["stop"].is_set():
        t0 = time.time()
        try:
            q = np.asarray(arm.read_deg(), float)          # locked read
        except Exception:
            time.sleep(period); continue
        t = time.time()
        if prev is not None and t > prev[0]:
            dt = t - prev[0]; qd = (q - prev[1]) / dt; qdd = (qd - prev_qd) / dt
        else:
            qd = np.zeros(6); qdd = np.zeros(6)
        T = fk(np.radians(q)); p = T[:3, 3] * 1000
        v = float(np.linalg.norm(p - prev[2]) / (t - prev[0])) if prev is not None and t > prev[0] else 0.0
        try:
            kap = _kappa(np.radians(q))
        except Exception:
            kap = float("nan")
        S["stream"].row([t, S["cid"], S["label"]] + list(np.round(q, 3)) + list(np.round(qd, 2)) + list(np.round(qdd, 1))
                        + list(np.round(p, 2)) + [round(v, 2), round(kap, 4), int(np.max(np.abs(qd)) > 1.0)])
        if S["status"] and i % n_status == 0:
            try:
                _status(arm, S["cid"])
            except Exception:
                pass
        prev, prev_qd, i = (t, q, p), qd, i + 1
        time.sleep(max(0.0, period - (time.time() - t0)))


def event(name, **info):
    """Mark an inspection event (also labels the stream samples that follow). No-op if recording is off."""
    S = _STATE
    if not S.get("on"):
        return
    S["label"] = name
    S["events"].row([time.time(), name, json.dumps(info, default=str)])


def _static(run_dir):
    import config as C
    info = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), kin_model=C.KIN_MODEL, kin_z_offset_mm=C.KIN_Z_OFFSET_MM,
                move_speed_default=C.MOVE_SPEED, max_joint_step_deg=C.MAX_JOINT_STEP_DEG, numpy=np.__version__)
    try:
        import pymycobot; info["pymycobot"] = getattr(pymycobot, "__version__", "?")
    except Exception:
        pass
    json.dump(info, open(os.path.join(run_dir, "static.json"), "w"), indent=2)


def joint_limits(arm):
    """Firmware joint limits (deg) when available."""
    mc = getattr(arm, "mc", None); out = {}
    for j in range(1, 7):
        try:
            out[j] = [float(mc.get_joint_min_angle(j)), float(mc.get_joint_max_angle(j))]
        except Exception:
            pass
    return out
