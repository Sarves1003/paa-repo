#!/usr/bin/env python3
"""Statistics for the paper (Colab or Jetson).

  python3 analyze_views.py                         # both parts, whatever exists
Reads  results_views/train_*/test_summary.json, test_rows.json   (E3: held-out placements, seeds)
       hw_logs/viewplan/*/episodes.csv                            (E4: live robot)
Writes results_views/offline_summary.txt, live_tests.txt, fig_live.pdf
"""
import glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); out = np.empty(m); run = 0
    for r, k in enumerate(o):
        run = max(run, (m - r) * p[k]); out[k] = min(1, run)
    return out


def wil(d):
    d = np.asarray(d, float)
    return float(wilcoxon(d).pvalue) if np.any(d != 0) and len(d) > 5 else float("nan")


def offline(L):
    S = [dict(json.load(open(p)), dir=os.path.dirname(p))
         for p in sorted(glob.glob("results_views/train_*/test_summary.json"))]
    if not S:
        L.append("E3: no training results yet"); return
    df = pd.DataFrame([dict(arch=s["arch"], seed=s["seed"], no_pose=s.get("no_pose"), no_time=s.get("no_time"),
                            pct=s["policy_pct_of_oracle"], **{"pct_" + c["method"]: c["pct_of_oracle"] for c in s["comparisons"]})
                       for s in S])
    df["variant"] = df.arch + np.where(df.no_pose, "+nopose", "") + np.where(df.no_time, "+notime", "")
    L.append("E3 held-out placements, % of the oracle (mean +/- SD over seeds):")
    L.append(df.groupby("variant").pct.agg(["mean", "std", "count"]).round(1).to_string())
    base = [c for c in df.columns if c.startswith("pct_")]
    L.append("baselines (% of oracle): " + ", ".join("%s %.1f" % (c[4:], df[c].mean()) for c in base))
    # transformer vs ablation, paired by (placement, budget), averaged over seeds
    rows = {}
    for s in S:
        v = s["arch"] + ("+nopose" if s.get("no_pose") else "") + ("+notime" if s.get("no_time") else "")
        for r in json.load(open(os.path.join(s["dir"], "test_rows.json"))):
            rows.setdefault(v, {}).setdefault((r["placement"], r["frac"]), []).append(r["policy"])
    if "attn" in rows:
        a = {k: np.mean(v) for k, v in rows["attn"].items()}
        for v in sorted(rows):
            if v == "attn":
                continue
            b = {k: np.mean(x) for k, x in rows[v].items()}
            keys = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in keys])
            L.append("attn - %-12s: mean %+.4f, attn better in %.0f%% of %d cases, Wilcoxon p=%.3g"
                     % (v, d.mean(), 100 * np.mean(d > 1e-9), len(d), wil(d)))


def live(L):
    files = glob.glob("hw_logs/viewplan/*/episodes.csv")
    if not files:
        L.append("E4: no live robot runs yet"); return
    df = pd.concat([pd.read_csv(f).assign(run=os.path.dirname(f)) for f in files], ignore_index=True)
    for pose, e in [("all", df)] + list(df.groupby("pose")):
        piv = e.pivot_table(index=["run", "trial"], columns="method", values="improvement").dropna()
        L.append("\nE4 live robot, pose=%s, paired placements=%d (improvement in inspection quality, higher=better)"
                 % (pose, len(piv)))
        L.append(piv.agg(["mean", "std", "median"]).T.round(4).to_string())
        if piv.shape[1] >= 3 and len(piv) >= 3:
            fr = friedmanchisquare(*[piv[c] for c in piv.columns])
            L.append("Friedman: chi2=%.2f, p=%.3g" % (fr.statistic, fr.pvalue))
        if "policy" in piv:
            others = [c for c in piv.columns if c != "policy"]
            ps = [wil(piv["policy"] - piv[c]) for c in others]
            for c, p, ph in zip(others, ps, holm(np.nan_to_num(ps, nan=1.0))):
                d = piv["policy"] - piv[c]
                L.append("policy - %-11s: median %+.4f, better in %.0f%%, Wilcoxon p=%.3g, Holm p=%.3g"
                         % (c, d.median(), 100 * np.mean(d > 1e-9), p, ph))
        if len(piv) < 6:
            L.append("NOTE: fewer than 6 paired placements cannot reach p < 0.05; run more trials")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ms = sorted(df.method.unique())
    fig, axs = plt.subplots(1, df.pose.nunique(), figsize=(3.5 * df.pose.nunique(), 2.8), squeeze=False)
    for ax, (pose, e) in zip(axs[0], df.groupby("pose")):
        ax.boxplot([e[e.method == m].improvement for m in ms], labels=ms)
        ax.set_title(pose); ax.set_ylabel("quality improvement"); ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig("results_views/fig_live.pdf")


def main():
    os.makedirs("results_views", exist_ok=True)
    L1, L2 = [], []
    offline(L1); live(L2)
    open("results_views/offline_summary.txt", "w").write("\n".join(L1) + "\n")
    open("results_views/live_tests.txt", "w").write("\n".join(L2) + "\n")
    print("\n".join(L1 + L2))


if __name__ == "__main__":
    main()
