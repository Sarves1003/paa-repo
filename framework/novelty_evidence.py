#!/usr/bin/env python3
"""Evidence for each novelty claim, computed from the data the robot already recorded (Colab or any PC).

  python3 novelty_evidence.py            # in the mycobot_framework folder -> results_novelty/ (figures, claims table)
  python3 novelty_evidence.py --ik-nodes 300    # C4: how many lattice nodes to re-solve with naive IK (default 250)

Each claim is answered by a measurement, with n and a test, and is stated against the ASSUMPTION it challenges
(what learned or classical view planners usually take for granted), so a reviewer can check every number:
  C1  the obtained view differs from the commanded one, and the joint encoders do not predict the difference
  C2  image-based verification closes the gap (markers or registration to the survey)
  C3  verification works only in a bounded descent range -> the safe view set must be limited by verifiability,
      not only by collision
  C4  branch-continuous lattice vs naive per-view IK from the reference: joint jumps, branch changes, singularity
  C5  planner: proposed vs singularity-blind ablation vs hand-programmed raster (largest command, depth needed)
  C6  room for learning in the real-image library: how far the survey-based greedy is from the oracle, and how
      well the survey predicts view quality (if it predicted perfectly, no learning would be needed)
Nothing is simulated except C4/C5 (kinematic planning on the arm's own model) and C6 (replay of recorded images).
"""
import argparse, csv, glob, json, os, sys
import numpy as np

OUT = "results_novelty"
CLAIMS = []


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name + ".png"), dpi=200); fig.savefig(os.path.join(OUT, name + ".pdf"))
    print("  figure", os.path.join(OUT, name + ".png"))


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon
        d = np.asarray(a) - np.asarray(b)
        return float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else np.nan
    except Exception:
        return np.nan


def claim(cid, text, assumption, metric, value, n, test=""):
    CLAIMS.append(dict(claim=cid, statement=text, assumption_challenged=assumption, metric=metric, value=value, n=n,
                       test=test))
    print("  [%s] %s: %s (n=%s) %s" % (cid, metric, value, n, test))


# ------------------------------------------------------------------ C1
def c1_encoders():
    print("\nC1  commanded vs obtained view; can the joint encoders predict the error?")
    from kinematics import fk
    H = None
    try:
        import config as C
        H = float(getattr(C, "VIEW_CAM_HEIGHT_MM", 0) or 0) or None
    except Exception:
        pass
    H = H or 265.0
    pred, meas, mvec, pvec = [], [], [], []
    for d in sorted(glob.glob("hw_logs/characterize/*")):
        f = os.path.join(d, "repeat.csv"); s = os.path.join(d, "static_robot.json")
        if not (os.path.exists(f) and os.path.exists(s)):
            continue
        ref = np.array(json.load(open(s))["reference_deg"], float)
        T0 = fk(np.radians(ref)); p0, a0 = T0[:3, 3] * 1000, T0[:3, 2]; zt = p0[2] - H
        hit = lambda p, a: p + (zt - p[2]) / a[2] * a
        g0 = hit(p0, a0)
        for r in csv.DictReader(open(f)):
            q = np.array([_num(r["q_%d" % i]) for i in range(1, 7)])
            e = _num(r.get("img_err_mm"))
            if not np.isfinite(e) or not np.all(np.isfinite(q)):
                continue
            T = fk(np.radians(q)); g = hit(T[:3, 3] * 1000, T[:3, 2])
            pv = (g - g0)[:2]; pvec.append(pv); pred.append(float(np.linalg.norm(pv))); meas.append(e)
            mvec.append([_num(r["img_dx_px"]), _num(r["img_dy_px"])])
    ss = []
    for f in glob.glob("hw_logs/**/moves.csv", recursive=True):
        for r in csv.DictReader(open(f)):
            v = _num(r.get("ss_err_max_deg"))
            if np.isfinite(v) and r.get("refused", "0") in ("0", "", "False"):
                ss.append(v)
    ss = np.array(ss)
    plt = _plt(); fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    if len(meas) >= 5:
        pred, meas, mvec, pvec = map(np.array, (pred, meas, mvec, pvec))
        r = float(np.corrcoef(pred, meas)[0, 1])
        # leave-one-out: best linear map from the encoder-predicted shift (mm) to the measured image shift (px)
        res, base = [], []
        for i in range(len(meas)):
            m = np.ones(len(meas), bool); m[i] = False
            X = np.c_[pvec[m], np.ones(m.sum())]; W, *_ = np.linalg.lstsq(X, mvec[m], rcond=None)
            res.append(np.linalg.norm(mvec[i] - np.r_[pvec[i], 1] @ W)); base.append(np.linalg.norm(mvec[i] - mvec[m].mean(0)))
        res, base = np.array(res), np.array(base)
        ax[0].scatter(pred, meas, s=10)
        ax[0].set_xlabel("view shift predicted from joint encoders (mm)"); ax[0].set_ylabel("view error measured in the image (mm)")
        ax[0].set_title("return to the same commanded pose (r = %.2f)" % r, fontsize=8)
        claim("C1", "Returning to the same commanded pose, the camera view lands %.1f mm (median) from where it "
              "should" % np.median(meas), "the commanded viewpoint is reached",
              "measured view error on return to pose, median [IQR] mm",
              "%.1f [%.1f-%.1f]" % (np.median(meas), *np.percentile(meas, [25, 75])), len(meas))
        claim("C1", "Correlation between encoder-predicted and measured view error", "", "Pearson r", "%.2f" % r, len(meas))
        frac = float(np.median(res) / max(np.median(base), 1e-9))
        txt = ("The joint readback EXPLAINS most of the view error once a linear map is fitted (encoder-based correction "
               "is a strong baseline that must be reported; image verification must beat it, and the map was fitted at "
               "ONE pose only)" if frac < 0.5 else
               "Correcting the view from the encoders (cross-validated linear map) leaves most of the error: "
               "proprioception cannot replace image verification")
        claim("C1", txt, "the error can be corrected from proprioception" if frac >= 0.5 else "",
              "image error left after encoder correction / without correction (px, median)",
              "%.1f / %.1f" % (np.median(res), np.median(base)), len(meas), "leave-one-out")
    if len(ss):
        ax[1].hist(np.clip(ss, 0, 10), bins=40)
        ax[1].set_xlabel("worst-joint error after a move (deg)"); ax[1].set_ylabel("moves")
        ax[1].set_title("all logged moves", fontsize=8)
        claim("C1", "Steady-state joint error after a move (worst joint)", "", "median / p95 (deg)",
              "%.2f / %.2f" % (np.median(ss), np.percentile(ss, 95)), len(ss))
    _save(fig, "C1_encoders")


# ------------------------------------------------------------------ C2, C3
def _all_views():
    """Every view with an error before and after verification: coverage runs and the recorded library."""
    V = []
    for f in glob.glob("hw_logs/refcover*/*/views.csv"):
        run = os.path.basename(os.path.dirname(f))
        for r in csv.DictReader(open(f)):
            V.append(dict(src="coverage run " + run, dz=_num(r.get("dz")), open=_num(r.get("view_err_open_mm")),
                          ver=_num(r.get("view_err_mm")), verify=r.get("verify", ""), scale=_num(r.get("scale_meas")),
                          markers=_num(r.get("markers"))))
    for f in glob.glob("deploy/views/*/meta.json"):
        m = json.load(open(f)); lv = m.get("levels_dz", [])
        for v in m.get("views", []):
            if not v.get("ok"):
                continue
            V.append(dict(src="library " + os.path.basename(os.path.dirname(f)),
                          dz=float(lv[v["level"]]) if v.get("level") is not None and v["level"] < len(lv) else np.nan,
                          open=_num(v.get("view_err_open_mm")), ver=_num(v.get("view_err_mm")),
                          verify=v.get("verify", "markers"), scale=_num(v.get("scale")),
                          markers=_num(v.get("servo_markers"))))
    return V


def c2_c3_verification():
    print("\nC2  verification closes the gap;  C3  verification range bounds the safe view set")
    V = _all_views()
    if not V:
        print("  skip: no views.csv / meta.json found"); return
    o = np.array([v["open"] for v in V]); c = np.array([v["ver"] for v in V]); dz = np.array([v["dz"] for v in V])
    both = np.isfinite(o) & np.isfinite(c)
    plt = _plt(); fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    if both.sum() >= 5:
        for x, lab in ((o[both], "open loop (as moved)"), (c[both], "verified")):
            xs = np.sort(x); ax[0].plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=lab)
        ax[0].set_xlabel("view error (mm)"); ax[0].set_ylabel("fraction of views"); ax[0].legend(frameon=False)
        ax[0].set_xlim(0, max(15, np.percentile(o[both], 95))); ax[0].set_title("all verified views", fontsize=8)
        p = _wilcoxon(o[both], c[both])
        claim("C2", "Closed-loop verification reduces the view error", "the commanded viewpoint is reached",
              "median open-loop -> verified error (mm)", "%.1f -> %.2f" % (np.median(o[both]), np.median(c[both])),
              int(both.sum()), "Wilcoxon p=%.2g" % p)
        claim("C2", "Views verified to <= 2 mm", "", "fraction", "%.0f%%" % (100 * np.mean(c[both] <= 2)), int(both.sum()))
        meth = {}
        for v in V:
            if np.isfinite(v["ver"]):
                k = v["verify"].split(" (")[0] if "registration" in v["verify"] else v["verify"]
                meth[k] = meth.get(k, 0) + 1
        claim("C2", "Verification methods used (views without a marker are verified by registration)", "",
              "views by method", json.dumps(meth), int(sum(meth.values())))
    # C3: by descent
    lv = sorted(set(np.round(dz[np.isfinite(dz)])))
    rows = []
    for h in lv:
        sel = np.isfinite(dz) & (np.round(dz) == h)
        ok = sel & np.isfinite(c) & (c <= 2.0)
        rows.append((h, int(sel.sum()), 100 * ok.sum() / max(sel.sum(), 1),
                     float(np.nanmedian(c[sel])) if np.isfinite(c[sel]).any() else np.nan))
    if rows:
        H_ = [-r[0] for r in rows]
        ax[1].bar(range(len(rows)), [r[2] for r in rows], tick_label=["%d\n(n=%d)" % (h, r[1]) for h, r in zip(H_, rows)])
        ax[1].set_xlabel("descent below the reference (mm)"); ax[1].set_ylabel("% of views verified to <= 2 mm")
        ax[1].set_title("verification vs descent", fontsize=8)
        for h, n, pct, med in rows:
            claim("C3", "Verifiable views at %d mm below the reference" % -h,
                  "any collision-free view is usable", "% verified <= 2 mm (median verified error)",
                  "%.0f%% (%.2f mm)" % (pct, med) if np.isfinite(med) else "%.0f%% (not verifiable)" % pct, n)
    _save(fig, "C2_C3_verification")


# ------------------------------------------------------------------ C4
def _load_lattice():
    for f in ["deploy/refcover_lattice.npz"] + sorted(glob.glob("deploy/refcover_lattice_*.npz")):
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            if "q" in z.files and "D" in z.files:
                return f, {k: (z[k].item() if z[k].shape == () else z[k]) for k in z.files}
    return None, None


def c4_lattice(n_ik=250, seed=0):
    print("\nC4  branch-continuous lattice vs naive per-view IK from the reference")
    f, L = _load_lattice()
    if L is None:
        print("  skip: no deploy/refcover_lattice*.npz"); return
    from ik import ik_hold
    from kinematics import fk
    lock = json.load(open("deploy/locked_pose.json"))
    q_ref = np.radians(np.array(lock["angles_deg"], float)); T0 = fk(q_ref); R0, p0 = T0[:3, :3], T0[:3, 3] * 1000
    solved = np.asarray(L["solved"], bool); kap = np.asarray(L["kappa"], float); ok = np.asarray(L["ok"], bool)
    nxy = int(L["nx"]) * int(L["ny"]); N = len(solved)
    Rl = None
    if "Rt_layers" in L:
        Rl = [np.asarray(R) @ R0 for R in L["Rt_layers"]]
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(solved); idx = rng.choice(idx, min(n_ik, len(idx)), replace=False)
    jump_lat, jump_naive, branch, fail = [], [], 0, 0
    for n in idx:
        k = n // nxy; R = Rl[k] if Rl is not None else R0
        tgt = (p0 + np.r_[np.asarray(L["D"])[n], float(np.asarray(L["dz"])[n])]) / 1000.0
        qn, pe, re, good = ik_hold(tgt, q_ref, R, iters=1200)            # what a hand-written planner does
        ql = np.asarray(L["q"])[n]
        jump_lat.append(np.degrees(np.max(np.abs(ql - q_ref))))
        if not good:
            fail += 1; continue
        jump_naive.append(np.degrees(np.max(np.abs(qn - q_ref))))
        branch += int(np.degrees(np.max(np.abs(qn - ql))) > 20.0)
    jump_lat, jump_naive = np.array(jump_lat), np.array(jump_naive)
    plt = _plt(); fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    ax[0].hist(kap[solved], bins=30); ax[0].axvline(0.04, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel("conditioning kappa of reachable poses"); ax[0].set_ylabel("lattice nodes")
    ax[0].set_title("%s: %d nodes, %d reachable, %d usable" % (os.path.basename(f), N, solved.sum(), ok.sum()), fontsize=7)
    if len(jump_naive):
        ax[1].hist([jump_lat, jump_naive], bins=25, label=["lattice (continuation)", "naive IK from reference"])
        ax[1].set_xlabel("largest joint change from the reference (deg)"); ax[1].set_ylabel("views"); ax[1].legend(frameon=False)
    _save(fig, "C4_lattice")
    claim("C4", "Share of the box workspace that is unreachable or near-singular from the taught pose",
          "every pose over the part is reachable and well-conditioned",
          "nodes not usable (unreachable or kappa < 0.04)", "%.0f%%" % (100 * (1 - ok.mean())), N)
    if len(jump_naive):
        txt = ("Naive per-view IK from the reference lands on a different joint branch for some views"
               if branch + fail > 0 else
               "Naive per-view IK from the reference finds the SAME branch for every sampled view: on this arm and "
               "workspace branch continuity is not where the lattice helps; its value is the reachability/conditioning "
               "map and the bounded steps of its routes (C5)")
        claim("C4", txt, "per-view IK is enough" if branch + fail > 0 else "",
              "views on another branch (>20 deg from the continuation solution) / IK failures",
              "%d / %d of %d" % (branch, fail, len(idx)), len(idx))
        claim("C4", "Largest joint change from the reference: continuation vs naive", "",
              "median (p95) deg", "%.1f (%.1f) vs %.1f (%.1f)" % (np.median(jump_lat), np.percentile(jump_lat, 95),
                                                                np.median(jump_naive), np.percentile(jump_naive, 95)), len(idx))


# ------------------------------------------------------------------ C5
def c5_planner():
    print("\nC5  planner comparison over required resolutions")
    fs = [f for f in sorted(glob.glob("results_views/refcover_sweep*.json"))]
    best = None
    for f in fs:                                     # the sweep whose raster needs the widest depth range
        S = json.load(open(f)); S = S if isinstance(S, list) else S.get("rows", [])
        dz = [r.get("dz") for r in S if "raster" in r.get("label", "") and r.get("dz") is not None]
        if S and (best is None or (dz and min(dz) < best[2])):
            best = (f, S, min(dz) if dz else 0)
    if best is None:
        print("  skip: no results_views/refcover_sweep*.json"); return
    f, S, _ = best
    key = lambda lab: "proposed" if lab.startswith("proposed") else "ablation" if lab.startswith("ablation") else "raster"
    plt = _plt(); fig, ax = plt.subplots(1, 3, figsize=(8, 2.6))
    for m, st in (("proposed", "o-"), ("ablation", "s--"), ("raster", "^:")):
        R = sorted([r for r in S if key(r["label"]) == m], key=lambda r: r["rho_factor"])
        x = [r["rho_factor"] for r in R]
        ax[0].plot(x, [r["n_views"] for r in R], st, label=m)
        ax[1].plot(x, [r["max_cmd_joint_change_deg"] for r in R], st, label=m)
        ax[2].plot(x, [r.get("n_cmds_over_safety", 0) for r in R], st, label=m)
    for a_, t in zip(ax, ("views", "largest joint command (deg)", "commands over the safety limit")):
        a_.set_xlabel("required resolution (x survey)"); a_.set_ylabel(t)
    ax[0].legend(frameon=False)
    _save(fig, "C5_planner")
    pr = [r for r in S if key(r["label"]) == "proposed"]; ra = [r for r in S if key(r["label"]) == "raster"]
    claim("C5", "Largest single joint command over the sweep (%s)" % os.path.basename(f),
          "a hand-programmed raster at the required height is safe", "proposed vs raster (deg)",
          "%.0f vs %.0f" % (max(r["max_cmd_joint_change_deg"] for r in pr), max(r["max_cmd_joint_change_deg"] for r in ra)),
          len(pr))
    claim("C5", "Raster descent needed for the required resolution", "", "mm below the reference",
          ", ".join("%gx: %d" % (r["rho_factor"], -r["dz"]) for r in sorted(ra, key=lambda r: r["rho_factor"])), len(ra))
    claim("C5", "Commands over the safety limit", "", "proposed vs raster (sum over sweep)",
          "%d vs %d" % (sum(r.get("n_cmds_over_safety", 0) for r in pr), sum(r.get("n_cmds_over_safety", 0) for r in ra)),
          len(pr))


# ------------------------------------------------------------------ C6
def c6_learning_room():
    print("\nC6  room for learning in the real-image library")
    if not os.path.exists("data/view_library.npz"):
        print("  skip: no data/view_library.npz yet (collect_views.py, then build_view_dataset.py)"); return
    from views import ViewSet, rollout
    z = np.load("data/view_library.npz"); meta = json.loads(str(z["meta"]))
    vs = ViewSet.from_meta(meta) if hasattr(ViewSet, "from_meta") else ViewSet(
        meta["table_grid"], meta.get("levels_dz"), scales=meta["scales"], uc=meta["uc"], via_hi=meta.get("via_hi", True))
    keys = ["Q", "F", "qpred", "qs", "w", "T", "avail", "fp_part", "fp_edge", "fp_glare", "pose"]
    D = {k: z[k].astype(bool if k == "avail" else float) for k in keys}
    T = D["T"]; full = np.array([np.nansum(np.where(D["avail"][i], np.minimum(T[i][-1], 999), 0)) for i in range(len(T))])
    res = {}
    for f in (0.2, 0.35, 0.5):
        for m in ("greedy_pred", "setcover", "nn_time", "raster_low", "oracle"):
            res.setdefault(m, []).append(rollout(D, vs, f * full, m)[0])
    ora = np.concatenate(res["oracle"])
    pct = {m: 100 * np.concatenate(v).sum() / max(ora.sum(), 1e-9) for m, v in res.items()}
    sel = D["avail"][:, :, None] & (D["F"] > 0) & (D["w"][:, None, :] > 0.2)
    r = float(np.corrcoef(D["Q"][sel], D["qpred"][sel])[0, 1]) if sel.sum() > 10 else np.nan
    plt = _plt(); fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    ms = [m for m in pct if m != "oracle"]
    ax[0].bar(ms, [pct[m] for m in ms]); ax[0].axhline(100, color="k", lw=0.8, ls="--")
    ax[0].set_ylabel("% of oracle improvement"); ax[0].set_title("baselines on the recorded library", fontsize=8)
    if sel.sum() > 10:
        ax[1].scatter(D["qpred"][sel], D["Q"][sel], s=3, alpha=0.4)
        ax[1].set_xlabel("quality predicted from the survey"); ax[1].set_ylabel("quality measured in the view")
        ax[1].set_title("survey -> view predictability (r = %.2f)" % r, fontsize=8)
    _save(fig, "C6_learning_room")
    claim("C6", "Headroom for a learned policy: best hand-made planner vs the oracle on the real library",
          "hand-made heuristics are enough", "best baseline % of oracle",
          "%.0f%% (%s)" % (max(pct[m] for m in ms), max(ms, key=lambda m: pct[m])), int(len(full)),
          "budgets 20/35/50 %")
    claim("C6", "How well the survey predicts view quality", "the survey (or a renderer) predicts what a view shows",
          "Pearson r", "%.2f" % r, int(sel.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ik-nodes", type=int, default=250)
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    for fn, args in ((c1_encoders, ()), (c2_c3_verification, ()), (c4_lattice, (a.ik_nodes,)), (c5_planner, ()),
                     (c6_learning_room, ())):
        try:
            fn(*args)
        except Exception as e:
            print("  %s failed: %s" % (fn.__name__, e))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "claims_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CLAIMS[0]) if CLAIMS else ["claim"]); w.writeheader(); w.writerows(CLAIMS)
    with open(os.path.join(OUT, "claims_table.md"), "w") as f:
        f.write("| claim | statement | assumption challenged | metric | value | n | test |\n|---|---|---|---|---|---|---|\n")
        for c in CLAIMS:
            f.write("| %s |\n" % " | ".join(str(c[k]) for k in ("claim", "statement", "assumption_challenged", "metric",
                                                                  "value", "n", "test")))
    print("\nsaved %s/claims_table.csv and claims_table.md (%d entries)" % (OUT, len(CLAIMS)))


if __name__ == "__main__":
    main()
