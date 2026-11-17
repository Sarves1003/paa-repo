#!/usr/bin/env python3
"""E5: on-board decision time of the deployed view policies on the Jetson Nano (no robot motion).

  python3 time_view_policy.py              # -> hw_logs/view_policy_timing.json
Times the NumPy forward pass of deploy/view_policy_attn.npz and deploy/view_policy_mlp.npz with the real
number of candidate views K (deploy/lattice_views.json) and the token width stored in the weights.
"""
import json, os, platform, resource, time
import numpy as np
from geo_attn import forward_np, load_np


def main():
    K = len(json.load(open("deploy/lattice_views.json"))["views"]) if os.path.exists("deploy/lattice_views.json") else 50
    R = dict(K=K, machine=platform.machine(), numpy=np.__version__)
    rng = np.random.default_rng(0)
    for name in ("attn", "mlp"):
        p = "deploy/view_policy_%s.npz" % name
        if not os.path.exists(p):
            print("skip %s (not found)" % p); continue
        W = load_np(p)
        td, cd = int(W["meta_tok_dim"]), int(W["meta_ctx_dim"])
        tok = rng.normal(size=(K, td)).astype(np.float32); ctx = rng.normal(size=cd).astype(np.float32)
        mask = np.zeros(K, bool)
        for _ in range(20):
            forward_np(W, tok, ctx, mask)
        ts = []
        for _ in range(500):
            t = time.perf_counter(); forward_np(W, tok, ctx, mask); ts.append((time.perf_counter() - t) * 1e3)
        ts = np.array(ts)
        R[name] = dict(median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)), tok_dim=td,
                       n_params=int(sum(v.size for k, v in W.items() if not k.startswith(("meta_", "info_")))))
        print("%-4s: %.2f ms median, %.2f ms p95 per decision (K = %d views, %d parameters)"
              % (name, R[name]["median_ms"], R[name]["p95_ms"], K, R[name]["n_params"]))
    R["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    os.makedirs("hw_logs", exist_ok=True); json.dump(R, open("hw_logs/view_policy_timing.json", "w"), indent=2)
    print("saved hw_logs/view_policy_timing.json")


if __name__ == "__main__":
    main()
