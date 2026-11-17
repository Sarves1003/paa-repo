#!/usr/bin/env python3
"""Fill the manuscript with YOUR results: every number in paper/main.tex is a macro computed here.

  python3 make_paper.py            # -> paper/results_macros.tex, paper/table_*.tex, paper/main.pdf (if LaTeX exists)
  python3 make_paper.py --no-pdf
Values that do not exist yet print as a red [--] in the PDF, so nothing is ever invented.
Run plot_all.py first (figures), analyze_views.py optional.
"""
import argparse, glob, json, os, subprocess
import numpy as np
import pandas as pd

M = {}


def put(name, val, fmt="%.2f"):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return
    M[name] = (fmt % val) if isinstance(val, (float, np.floating, int, np.integer)) and fmt else str(val)


def wil(d):
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    return float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else float("nan")


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); out = np.empty(m); run = 0
    for r, k in enumerate(o):
        run = max(run, (m - r) * p[k]); out[k] = min(1, run)
    return out


def pfmt(p):
    return "<0.001" if p < 0.001 else "%.3f" % p


# ------------------------------------------------------------------ E1
def e1():
    import plot_all as PA
    d = PA.servo_data()
    if len(d):
        put("ServoN", len(d), "%d")
        put("ServoOpenMed", d.open.median()); put("ServoOpenNF", np.percentile(d.open, 95))
        put("ServoClosedMed", d.closed.median()); put("ServoClosedNF", np.percentile(d.closed, 95))
        put("ServoReduction", 100 * (1 - d.closed.median() / max(d.open.median(), 1e-9)), "%.0f")
        p = wil(d.open - d.closed)
        if np.isfinite(p):
            M["ServoP"] = pfmt(p)
    lv = PA.library_views()
    if len(lv):
        put("ServoIters", lv.iters.mean(), "%.1f"); put("ServoSec", lv.servo_s.mean(), "%.1f")
    if os.path.exists("deploy/table_grid.json") and os.path.exists("deploy/grounding.json"):
        t = json.load(open("deploy/table_grid.json")); A = np.asarray(json.load(open("deploy/grounding.json"))["A"], float)
        n = np.linalg.norm(A, axis=0)
        put("AxisXppm", n[0]); put("AxisYppm", n[1]); put("BoxPpm", t["box_px_per_mm"])
        put("AxisAngle", np.degrees(np.arccos(abs(A[:, 0] @ A[:, 1]) / np.prod(n))), "%.0f")
        put("GridPitch", t["pitch_mm"], "%.1f"); put("GridHalf", t["half_mm"], "%.1f")


# ------------------------------------------------------------------ E2
def e2():
    if not os.path.exists("data/view_library.npz"):
        return
    z = np.load("data/view_library.npz"); meta = json.loads(str(z["meta"]))
    put("NPlace", len(z["names"]), "%d")
    put("NHoriz", int((z["poses"] == "horizontal").sum()), "%d"); put("NVert", int((z["poses"] == "vertical").sum()), "%d")
    put("NViewsRec", int(z["avail"].sum()), "%d"); put("NTest", int((z["split"] == 1).sum()), "%d")
    put("CloseScale", meta["scales"][1]); put("CloseDz", abs(meta["dz"]), "%.0f")
    lv = meta.get("levels_dz", [0.0, meta["dz"]])
    put("LevelZeroDz", abs(lv[0]), "%.0f"); put("LevelOneDz", abs(lv[-1]), "%.0f")
    put("ScaleZero", meta["scales"][0]); put("ScaleOne", meta["scales"][-1])
    put("NCand", z["avail"].shape[1], "%d"); put("FullTime", np.mean(z["T"][:, -1].sum(1)) if False else None)
    P = z["w"].shape[1]
    for pose, tag in (("horizontal", "H"), ("vertical", "V")):
        idx = np.flatnonzero(z["poses"] == pose)
        if not len(idx):
            continue
        for lev, lt in ((0, "Hi"), (1, "Lo")):
            vals = []
            for i in idx:
                Q = z["Q"][i][lev * P:(lev + 1) * P][z["avail"][i][lev * P:(lev + 1) * P]]
                w = z["w"][i] > 0.2
                vals += list(Q[:, w][Q[:, w] > 0])
            if vals:
                put("Qual%s%s" % (lt, tag), float(np.mean(vals)))
        x, y = [], []
        for i in idx:
            m = z["avail"][i][:, None] & (z["F"][i] > 0) & (z["w"][i][None] > 0.2) & (z["Q"][i] > 0)
            x += list(z["qpred"][i][m]); y += list(z["Q"][i][m])
        if len(x) > 3:
            put("PredCorr%s" % tag, float(np.corrcoef(x, y)[0, 1]))
    if os.path.exists("deploy/pose_clf.json"):
        c = json.load(open("deploy/pose_clf.json"))
        put("PoseLOO", 100 * c["loo_accuracy"], "%.0f"); put("PoseN", c["n"], "%d")


# ------------------------------------------------------------------ E3
def e3():
    import plot_all as PA
    R = PA.train_runs()
    if not R:
        return
    T = PA.offline_table()
    a = T[T.pose == "all"].groupby("method").pct.agg(["mean", "std", "count"])
    key = {"attn": "Attn", "mlp": "Mlp", "attn+nopose": "NoPose", "attn+notime": "NoTime", "greedy_pred": "Greedy",
           "nn_time": "Nearest", "raster_low": "RasterLo", "raster_high": "RasterHi", "random": "Random"}
    for m, k in key.items():
        if m in a.index:
            put(k + "Pct", a.loc[m, "mean"], "%.1f"); put(k + "Sd", a.loc[m, "std"] if a.loc[m, "count"] > 1 else 0.0, "%.1f")
    put("NSeeds", int(a.loc["attn", "count"]) if "attn" in a.index else None, "%d")
    for pose, tag in (("horizontal", "H"), ("vertical", "V")):
        t = T[T.pose == pose].groupby("method").pct.mean()
        for m in ("attn", "mlp", "greedy_pred"):
            if m in t.index:
                put(key[m] + "Pct" + tag, t[m], "%.1f")
    rows = {}
    for r in R:
        for x in json.load(open(os.path.join(r["dir"], "test_rows.json"))):
            rows.setdefault(r["variant"], {}).setdefault((x["placement"], x["frac"]), []).append(x["policy"])
            if r["variant"] == "attn":
                for m in ("greedy_pred", "nn_time"):
                    rows.setdefault(m, {}).setdefault((x["placement"], x["frac"]), []).append(x[m])
    if "attn" in rows:
        A_ = {k: np.mean(v) for k, v in rows["attn"].items()}
        for other, tag in (("mlp", "Mlp"), ("greedy_pred", "Greedy"), ("nn_time", "Nearest"),
                           ("attn+nopose", "NoPose"), ("attn+notime", "NoTime")):
            if other in rows:
                B = {k: np.mean(v) for k, v in rows[other].items()}
                ks = sorted(set(A_) & set(B)); d = np.array([A_[k] - B[k] for k in ks])
                p = wil(d)
                if np.isfinite(p):
                    M["AttnVs%sP" % tag] = pfmt(p)
                put("AttnVs%sWin" % tag, 100 * np.mean(d > 1e-9), "%.0f"); put("NCases", len(d), "%d")
    # table
    L = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & All & Horizontal & Vertical \\", r"\midrule"]
    for m in ["attn", "mlp", "attn+nopose", "attn+notime", "greedy_pred", "nn_time", "raster_low", "raster_high", "random"]:
        cells = []
        for pose in ("all", "horizontal", "vertical"):
            t = T[(T.pose == pose) & (T.method == m)].pct
            cells.append("%.1f $\\pm$ %.1f" % (t.mean(), t.std() if len(t) > 1 else 0.0) if len(t) else "--")
        if any(c != "--" for c in cells):
            L.append("%s & %s \\\\" % (PA.NAMES.get(m, m).replace("(ours)", "(ours)"), " & ".join(cells)))
    L += [r"\bottomrule", r"\end{tabular}"]
    open("paper/table_offline.tex", "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ E4
def e4():
    import plot_all as PA
    df = PA.live()
    if df is None:
        return
    L = [r"\begin{tabular}{llccc}", r"\toprule",
         r"Pose & Method & Improvement (mean $\pm$ SD) & Policy better & Holm $p$ \\", r"\midrule"]
    for pose, e in [("all", df)] + list(df.groupby("pose")):
        piv = e.pivot_table(index=["run", "trial"], columns="method", values="improvement").dropna()
        tag = {"all": "", "horizontal": "H", "vertical": "V"}[pose]
        put("LiveN" + tag, len(piv), "%d")
        if "policy" not in piv:
            continue
        others = [c for c in PA.ORDER if c in piv.columns and c != "policy"]
        ps = [wil(piv["policy"] - piv[c]) for c in others]
        ph = holm(np.nan_to_num(ps, nan=1.0)) if ps else []
        put("LivePolicy" + tag, piv["policy"].mean(), "%.3f")
        for c in ["policy"] + others:
            win, pp = "--", "--"
            if c != "policy":
                i = others.index(c)
                win = "%.0f\\%%" % (100 * np.mean(piv["policy"] - piv[c] > 1e-9)); pp = pfmt(ph[i]) if np.isfinite(ps[i]) else "n/a"
                nm = {"policy_mlp": "Mlp", "greedy_pred": "Greedy", "nn_time": "Nearest", "raster_low": "Raster"}.get(c)
                if nm:
                    put("Live%s%s" % (nm, tag), piv[c].mean(), "%.3f")
                    if np.isfinite(ps[i]):
                        M["LiveVs%s%sP" % (nm, tag)] = pfmt(ph[i])
            L.append("%s & %s & %.3f $\\pm$ %.3f & %s & %s \\\\" % (pose, PA.NAMES.get(c, c), piv[c].mean(),
                                                                    piv[c].std() if len(piv) > 1 else 0, win, pp))
        L.append(r"\midrule")
    if "pose_auto" in df:
        ok = df.dropna(subset=["pose_auto"])
        ok = ok[ok.pose_auto.astype(str) != ""]
        if len(ok):
            put("LivePoseAcc", 100 * (ok.pose_auto == ok.pose).mean(), "%.0f")
    L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
    open("paper/table_live.tex", "w").write("\n".join(L) + "\n")
    vf = glob.glob("hw_logs/viewplan/*/views.csv")
    if vf:
        v = pd.concat([pd.read_csv(f) for f in vf])
        im = v[v.method == "policy"].infer_ms.dropna()
        if len(im):
            put("InferMs", im.mean()); put("InferNF", np.percentile(im, 95))


# ------------------------------------------------------------------ E5 / E6
def e5_e6():
    for p in ("hw_logs/jetson_check.json",):
        if os.path.exists(p):
            j = json.load(open(p))
            put("JetsonRam", j.get("ram_total_gb"), "%.2f")
            put("PeakMem", j.get("peak_rss_mb") or j.get("peak_memory_mb"), "%.0f")
            dl = j.get("decision_latency_ms") or {}
            if "InferMs" not in M:
                put("InferMs", dl.get("mean")); put("InferNF", dl.get("p95"))
    p = "results_views/defect_results.json"
    if os.path.exists(p):
        R = json.load(open(p))
        put("DefN", R.get("n_defects"), "%d"); put("DefHidden", R.get("n_defects_not_visible_in_survey"), "%d")
        put("DefAUC", R.get("auc_quality_predicts_visibility")); put("DefTiles", R.get("n_tiles"), "%d")
        H = R.get("capture_rate_defects_not_visible_in_survey") or {}
        for m, k in (("policy", "Policy"), ("policy_mlp", "Mlp"), ("greedy_pred", "Greedy"), ("raster_low", "Raster"),
                     ("oracle", "Oracle")):
            v = (H.get(m) or {}).get("0.35")
            if v is not None:
                put("DefCap" + k, 100 * v, "%.0f")


def write_macros():
    names = ["ServoN", "ServoOpenMed", "ServoOpenNF", "ServoClosedMed", "ServoClosedNF", "ServoReduction", "ServoP",
             "ServoIters", "ServoSec", "AxisXppm", "AxisYppm", "BoxPpm", "AxisAngle", "GridPitch", "GridHalf",
             "NPlace", "NHoriz", "NVert", "NViewsRec", "NTest", "CloseScale", "CloseDz", "NCand",
             "LevelZeroDz", "LevelOneDz", "ScaleZero", "ScaleOne",
             "QualHiH", "QualLoH", "QualHiV", "QualLoV", "PredCorrH", "PredCorrV", "PoseLOO", "PoseN",
             "AttnPct", "AttnSd", "MlpPct", "MlpSd", "NoPosePct", "NoPoseSd", "NoTimePct", "NoTimeSd", "GreedyPct",
             "GreedySd", "NearestPct", "NearestSd", "RasterLoPct", "RasterHiPct", "RandomPct", "NSeeds",
             "AttnPctH", "AttnPctV", "MlpPctH", "MlpPctV", "GreedyPctH", "GreedyPctV",
             "AttnVsMlpP", "AttnVsMlpWin", "AttnVsGreedyP", "AttnVsGreedyWin", "AttnVsNearestP", "AttnVsNoPoseP",
             "AttnVsNoTimeP", "NCases",
             "LiveN", "LiveNH", "LiveNV", "LivePolicy", "LivePolicyH", "LivePolicyV", "LiveMlp", "LiveGreedy",
             "LiveNearest", "LiveRaster", "LiveVsMlpP", "LiveVsGreedyP", "LiveVsNearestP", "LiveVsRasterP",
             "LiveVsMlpHP", "LiveVsMlpVP", "LiveVsGreedyHP", "LiveVsGreedyVP", "LivePoseAcc",
             "InferMs", "InferNF", "JetsonRam", "PeakMem",
             "DefN", "DefHidden", "DefAUC", "DefTiles", "DefCapPolicy", "DefCapMlp", "DefCapGreedy", "DefCapRaster", "DefCapOracle"]
    L = ["% generated by make_paper.py -- do not edit; re-run after new experiments",
         r"\providecommand{\Missing}{\textcolor{red}{[--]}}"]
    for n in sorted(set(names) | set(M)):
        L.append(r"\newcommand{\R%s}{%s}" % (n, M.get(n, r"\Missing")))
    open("paper/results_macros.tex", "w").write("\n".join(L) + "\n")
    print("macros: %d filled, %d still missing" % (len(M), len(set(names) - set(M))))
    for t in ("table_offline.tex", "table_live.tex"):
        p = os.path.join("paper", t)
        if not os.path.exists(p):
            open(p, "w").write("\\textcolor{red}{[table filled after the experiments]}\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-pdf", action="store_true"); a = ap.parse_args()
    os.makedirs("paper/figs", exist_ok=True)
    for f in (e1, e2, e3, e4, e5_e6):
        try:
            f()
        except Exception as e:
            print("  %s: %s" % (f.__name__, e))
    write_macros()
    if a.no_pdf:
        return
    if not os.path.exists("paper/main.tex"):
        return print("paper/main.tex missing (copy it from the package)")
    try:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd="paper",
                           stdout=subprocess.DEVNULL, timeout=120)
        subprocess.run(["bibtex", "main"], cwd="paper", stdout=subprocess.DEVNULL, timeout=60)
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd="paper",
                           stdout=subprocess.DEVNULL, timeout=120)
        print("paper/main.pdf" if os.path.exists("paper/main.pdf") else "LaTeX failed: see paper/main.log")
    except FileNotFoundError:
        print("no pdflatex here: upload paper/ to Overleaf (it contains main.tex, results_macros.tex, figs/)")


if __name__ == "__main__":
    main()
