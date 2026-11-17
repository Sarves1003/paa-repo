#!/usr/bin/env python3
"""Platform check and on-device measurements for the Jetson Nano 4 GB (run on the Nano).

Verifies: board model, total RAM (~3.9 GB visible on the 4 GB Nano; the 2 GB model shows
~1.9 GB), L4T/JetPack release, power mode, Python/NumPy/OpenCV versions.
Measures (no robot needed):
  * NumPy policy decision latency (deployed weights if present, else random weights)
  * grounded survey tokenisation time on a 640x480 frame
  * orientation-held IK time per cell
  * peak resident memory of this process
  * energy per decision from the Nano's INA3221 power monitor
    (board input rail, active minus idle power x time per decision)
Writes hw_logs/jetson_check.json (cite these numbers, not assumed ones).

  python3 jetson_check.py [--n 2000]
"""
import argparse, glob, json, os, platform, resource, subprocess, sys, threading, time
import numpy as np

import config as C


def read(path):
    try:
        return open(path).read().strip("\x00\n ")
    except Exception:
        return None


def total_ram_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) / 1024 ** 2
    return float("nan")


def power_mode():
    try:
        out = subprocess.run(["nvpmodel", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=5).stdout
        return " ".join(out.split())
    except Exception:
        return None


def find_power_rail():
    """INA3221 on the Nano developer kit; channel 0 = board input (POM_5V_IN), in mW."""
    for p in sorted(glob.glob("/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power0_input")):
        return p
    for p in sorted(glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/")):
        cur, volt = os.path.join(p, "curr1_input"), os.path.join(p, "in1_input")
        if os.path.exists(cur) and os.path.exists(volt):
            return (cur, volt)
    return None


def sample_power(rail):
    if isinstance(rail, tuple):
        i, v = float(read(rail[0])), float(read(rail[1]))      # mA, mV
        return i * v / 1000.0
    return float(read(rail))                                      # mW


class PowerLogger(threading.Thread):
    def __init__(self, rail, dt=0.02):
        super().__init__(daemon=True); self.rail, self.dt, self.vals, self.on, self.err = rail, dt, [], True, None

    def run(self):
        while self.on:
            try:
                self.vals.append(sample_power(self.rail))
            except Exception as ex:
                self.err = repr(ex)
            time.sleep(self.dt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2000); a = ap.parse_args()
    R = {}
    R["board"] = read("/proc/device-tree/model")
    R["l4t"] = read("/etc/nv_tegra_release")
    R["ram_total_gb"] = round(total_ram_gb(), 2)
    R["power_mode"] = power_mode()
    R["python"] = platform.python_version()
    R["numpy"] = np.__version__
    try:
        import cv2; R["opencv"] = cv2.__version__
    except Exception:
        R["opencv"] = None
    print(json.dumps(R, indent=1))

    if R["board"] and "Nano" not in R["board"]:
        print("WARNING: this does not look like a Jetson Nano")
    if R["ram_total_gb"] < 3.0:
        print("WARNING: RAM < 3 GB -> this is the 2 GB Nano; the study targets the 4 GB model")
    if tuple(int(x) for x in np.__version__.split(".")[:2]) < (1, 17):
        sys.exit("NumPy >= 1.17 required:  pip3 install --user numpy==1.19.5")

    from geo_attn import forward_np, load_np, TOK_DIM, CTX_DIM
    rng = np.random.default_rng(0)
    if os.path.exists(C.POLICY_PATH):
        W = load_np(C.POLICY_PATH); R["policy"] = C.POLICY_PATH
    else:   # random weights with the deployed shapes (latency does not depend on values)
        d, L, ff = 32, 2, 64
        W = {"meta_d": np.array(d), "meta_L": np.array(L), "meta_clip": np.array(10.0)}
        def lin(n, i, o):
            W[n + ".weight"] = rng.normal(0, .1, (o, i)).astype(np.float32); W[n + ".bias"] = np.zeros(o, np.float32)
        lin("emb_t", TOK_DIM, d); lin("emb_c", CTX_DIM, d); lin("dec_q", d, d); lin("dec_k", d, d)
        for l in range(L):
            for nm in ("q", "k", "v"):
                lin(f"{nm}.{l}", d, d)
            lin(f"ff1.{l}", d, ff); lin(f"ff2.{l}", ff, d)
            for ln in ("ln1", "ln2"):
                W[f"{ln}.{l}.weight"] = np.ones(d, np.float32); W[f"{ln}.{l}.bias"] = np.zeros(d, np.float32)
        R["policy"] = "random weights (deployed architecture)"
    n_par = sum(v.size for k, v in W.items() if not k.startswith(("meta_", "info_")))
    R["policy_params"] = int(n_par)
    tok = rng.normal(size=(25, TOK_DIM)).astype(np.float32); ctx = rng.normal(size=CTX_DIM).astype(np.float32)
    for _ in range(50):
        forward_np(W, tok, ctx)
    lat = []
    for _ in range(a.n):
        t = time.perf_counter(); forward_np(W, tok, ctx); lat.append((time.perf_counter() - t) * 1e3)
    lat = np.array(lat)
    R["decision_latency_ms"] = dict(mean=float(lat.mean()), sd=float(lat.std()), p50=float(np.median(lat)),
                                    p95=float(np.percentile(lat, 95)), max=float(lat.max()), n=a.n)

    from perception import grounded_survey
    img = os.path.join("data", "raw_camera.jpg")
    import cv2
    frame = cv2.imread(img) if os.path.exists(img) else np.full((C.CAM_H, C.CAM_W, 3), 200, np.uint8)
    G = {"mode": "shift", "A": np.array([[-4.0, 0.0], [0.0, 4.0]]), "uc": np.array([C.CAM_W / 2, C.CAM_H / 2]),
         "px_per_mm": 4.0}
    ts = []
    for _ in range(30):
        t = time.perf_counter(); grounded_survey(frame, G); ts.append((time.perf_counter() - t) * 1e3)
    R["survey_tokenisation_ms"] = dict(mean=float(np.mean(ts)), p95=float(np.percentile(ts, 95)))

    from kinematics import fk
    from ik import ik_hold
    q0 = np.deg2rad([-74.5, -32.6, -77.0, 40.6, -22.4, 82.7]); T0 = fk(q0)
    ti = []
    for dx, dy in [(0.0175, 0), (0, 0.0175), (0.035, 0.035), (-0.035, 0), (0, -0.035)]:
        t = time.perf_counter(); ik_hold(T0[:3, 3] + [dx, dy, 0], q0, T0[:3, :3]); ti.append((time.perf_counter() - t) * 1e3)
    R["ik_hold_ms_per_cell"] = dict(mean=float(np.mean(ti)), max=float(np.max(ti)))

    rail = find_power_rail()
    if rail is not None:
        try:
            sample_power(rail)
        except Exception as ex:
            R["power"] = ("power sensor not readable (%r). Run once:  sudo chmod a+r "
                          "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_*  then re-run" % ex)
            rail = None
    if rail is not None:
        idle = PowerLogger(rail); idle.start(); time.sleep(3.0); idle.on = False; idle.join()
        act = PowerLogger(rail); act.start(); t0 = time.perf_counter(); k = 0
        while time.perf_counter() - t0 < 3.0:
            forward_np(W, tok, ctx); k += 1
        act.on = False; act.join()
        p_idle, p_act = float(np.mean(idle.vals)), float(np.mean(act.vals))
        t_dec = 3.0 / k
        R["power"] = dict(rail=str(rail), idle_mW=p_idle, active_mW=p_act, decisions=k,
                          incremental_energy_per_decision_mJ=max(p_act - p_idle, 0.0) * t_dec,
                          total_energy_per_decision_mJ=p_act * t_dec,
                          note="board input rail; incremental = (active - idle) x time per decision")
    elif "power" not in R:
        R["power"] = "INA3221 power monitor not found"

    R["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    R["peak_rss_fraction_of_ram"] = round(R["peak_rss_mb"] / (R["ram_total_gb"] * 1024), 4)
    os.makedirs(C.LOGS_DIR, exist_ok=True)
    out = os.path.join(C.LOGS_DIR, "jetson_check.json")
    json.dump(R, open(out, "w"), indent=2)
    print(json.dumps({k: R[k] for k in R if k not in ("board", "l4t", "power_mode", "python", "numpy", "opencv")}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
