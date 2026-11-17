#!/usr/bin/env python3
"""Fair on-board timing of the view policies (Jetson Nano), no robot motion.

  sudo nvpmodel -m 0 && sudo jetson_clocks        # once: maximum, fixed clocks
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 time_view_policy2.py
  python3 time_view_policy2.py --threads default  # optional second run with NumPy's default threading

Every policy file in deploy/view_policy_*.npz is timed in 5 interleaved rounds (A B C A B C ...), so clock
ramp-up or thermal drift affects all models equally. Writes hw_logs/view_policy_timing2.json.
"""
import argparse, glob, json, os, platform, time
ap = argparse.ArgumentParser(); ap.add_argument("--threads", default="1"); ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()
if a.threads != "default":
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = a.threads
import numpy as np
from geo_attn import forward_np, load_np


def sh(cmd):
    try:
        return os.popen(cmd + " 2>/dev/null").read().strip()
    except Exception:
        return ""


K = len(json.load(open("deploy/lattice_views.json"))["views"])
paths = sorted(glob.glob("deploy/view_policy_*.npz"))
models = {}
rng = np.random.default_rng(0)
for p in paths:
    W = load_np(p); td, cd = int(W["meta_tok_dim"]), int(W["meta_ctx_dim"])
    models[p] = (W, rng.normal(size=(K, td)).astype(np.float32), rng.normal(size=cd).astype(np.float32),
                 str(W.get("meta_arch", "attn")),
                 int(sum(v.size for k, v in W.items() if not k.startswith(("meta_", "info_")))))
mask = np.zeros(K, bool)
for W, tok, ctx, _, _ in models.values():          # warm-up
    for _ in range(50):
        forward_np(W, tok, ctx, mask)
T = {p: [] for p in models}
for rnd in range(5):
    for p, (W, tok, ctx, _, _) in models.items():
        for _ in range(a.n // 5):
            t = time.perf_counter(); forward_np(W, tok, ctx, mask); T[p].append((time.perf_counter() - t) * 1e3)
R = dict(K=K, threads=a.threads, machine=platform.machine(), numpy=np.__version__,
         nvpmodel=sh("nvpmodel -q | tail -1"), cpu_mhz=sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
         governor=sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"), temp_c=sh("cat /sys/devices/virtual/thermal/thermal_zone0/temp"))
print("threads=%s  nvpmodel=%s  cpu0=%s kHz  governor=%s  temp=%s" % (a.threads, R["nvpmodel"], R["cpu_mhz"], R["governor"], R["temp_c"]))
for p, ts in T.items():
    ts = np.array(ts); arch, npar = models[p][3], models[p][4]
    R[p] = dict(arch=arch, n_params=npar, median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)),
                max_ms=float(ts.max()), mean_ms=float(ts.mean()))
    print("%-40s %-4s %6d par  median %7.2f ms  p95 %7.2f ms  max %7.2f ms" % (p, arch, npar, R[p]["median_ms"], R[p]["p95_ms"], R[p]["max_ms"]))
os.makedirs("hw_logs", exist_ok=True)
json.dump(R, open("hw_logs/view_policy_timing2_threads%s.json" % a.threads, "w"), indent=2)
print("saved hw_logs/view_policy_timing2_threads%s.json" % a.threads)
