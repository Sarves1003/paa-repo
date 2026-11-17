#!/usr/bin/env python3
"""Statistics for run_inspection.py / grounding.py logs (the evidence for each claim).

  python3 analyze_experiments.py "hw_logs/inspection/*/log.csv" --grounding "hw_logs/grounding/validate_*.csv"

Outputs (results_hw/):
  episodes.csv          one row per (trial, method, task, ik)
  paired_tests.txt      Friedman + Wilcoxon signed-rank (policy vs each method, Holm),
                        paired by trial (same survey), with median differences
  time_model.txt        planned vs measured move time (R^2, MAE)   -> cost-model claim
  view_consistency.txt  drift and per-cell ROI-density CV, hold vs position -> view claim
  grounding.txt         grounding error per cell (mm)              -> grounding claim
  fig_*.pdf
"""
import argparse, glob, itertools, os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare, mannwhitneyu


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); out = np.empty(m); run = 0
    for r, k in enumerate(o):
        run = max(run, (m - r) * p[k]); out[k] = min(1, run)
    return out


def files(patterns):
    return sorted(set(itertools.chain.from_iterable(glob.glob(p) for p in patterns)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--grounding", nargs="*", default=[])
    ap.add_argument("--out", default="results_hw")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.concat([pd.read_csv(f).assign(run=os.path.dirname(f)) for f in files(a.logs)], ignore_index=True)
    mv = df[df.moved == 1].copy()
    for col in ("servo_s", "servo_iters", "view_err_open_mm", "view_err_mm"):
        if col not in mv.columns:
            mv[col] = np.nan
    mv["servo_s"] = mv["servo_s"].fillna(0.0)

    # ---------------- episodes
    g = mv.groupby(["run", "task", "ik_mode", "trial", "method"])
    ep = g.agg(n_cells=("cell", "size"), measured_s=("move_s", "sum"), servo_s=("servo_s", "sum"), planned_s=("planned_move_s", "sum"),
               value=("value", "sum"), joint_dq_sum=("dq_max_deg", "sum"), drift_max=("view_drift_deg", "max"),
               drift_mean=("view_drift_deg", "mean"), rb_err_mean=("rb_err_mm", "mean"),
               rb_z_mean=("rb_z_err_mm", "mean"), ground_res_med=("ground_res_mm", "median"),
               infer_ms=("infer_ms", "mean")).reset_index()
    cart = []
    for key, gg in g:
        gg = gg.sort_values("step"); xy = gg[["tx_mm", "ty_mm"]].values.astype(float)
        cart.append(float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0)
    ep["cart_mm"] = cart
    ep.to_csv(os.path.join(a.out, "episodes.csv"), index=False)

    # ---------------- paired tests
    L = []
    for (task, ik), e in ep.groupby(["task", "ik_mode"]):
        metric, better = ("measured_s", "lower") if task == "coverage" else ("value", "higher")
        piv = e.pivot_table(index=["run", "trial"], columns="method", values=metric).dropna()
        L.append(f"== task={task} ik={ik}  metric={metric} ({better} is better)  paired trials={len(piv)}")
        L.append(piv.describe().loc[["mean", "std", "50%"]].round(3).to_string())
        if len(piv) < 6:
            L.append("NOTE: with < 6 paired trials a two-sided Wilcoxon test cannot reach p < 0.05; run >= 10 trials.")
        if piv.shape[1] >= 3 and len(piv) >= 3:
            fr = friedmanchisquare(*[piv[c] for c in piv.columns])
            L.append(f"Friedman across {piv.shape[1]} methods: chi2={fr.statistic:.2f}, p={fr.pvalue:.4f}")
        if "policy" in piv.columns:
            others = [c for c in piv.columns if c != "policy"]
            ps, lines = [], []
            for c in others:
                d = piv["policy"] - piv[c]
                p = wilcoxon(d).pvalue if np.any(d != 0) and len(d) > 1 else 1.0
                ps.append(p)
                rel = 100 * d.median() / max(abs(piv[c].median()), 1e-9)
                lines.append((c, d.median(), rel, p))
            for (c, md, rel, p), ph in zip(lines, holm(ps)):
                L.append(f"policy - {c:9s}: median diff {md:+.3f} ({rel:+.1f}%), Wilcoxon p={p:.4f}, Holm p={ph:.4f}")
        for c1, c2 in [("boustro", "twoopt"), ("nn_cart", "nn_time")]:
            if c1 in piv.columns and c2 in piv.columns:
                d = piv[c2] - piv[c1]
                p = wilcoxon(d).pvalue if np.any(d != 0) and len(d) > 1 else 1.0
                L.append(f"{c2} - {c1} (Cartesian vs joint-time planning): median {d.median():+.3f}, p={p:.4f}")
    open(os.path.join(a.out, "paired_tests.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    # ---------------- time model validation
    tm = mv.dropna(subset=["planned_move_s", "move_s"])
    T = []
    for ik, t in tm.groupby("ik_mode"):
        r = np.corrcoef(t.planned_move_s, t.move_s)[0, 1]
        mae = float(np.abs(t.planned_move_s - t.move_s).mean())
        T.append(f"ik={ik}: n={len(t)} moves, Pearson r={r:.3f} (R^2={r*r:.3f}), MAE={mae:.3f} s, "
                 f"mean measured {t.move_s.mean():.2f} s vs planned {t.planned_move_s.mean():.2f} s")
    T.append("If MAE is large, run `python3 cost_model.py fit <logs>` and repeat the trials.")
    open(os.path.join(a.out, "time_model.txt"), "w").write("\n".join(T) + "\n"); print("\n".join(T))
    fig, ax = plt.subplots(figsize=(3.3, 3))
    for ik, t in tm.groupby("ik_mode"):
        ax.scatter(t.planned_move_s, t.move_s, s=6, label=ik)
    lim = [0, max(tm.move_s.max(), tm.planned_move_s.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlabel("planned move time (s)"); ax.set_ylabel("measured (s)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_time_model.pdf")); plt.close(fig)

    # ---------------- view consistency
    V = []
    if mv.ik_mode.nunique() == 2:
        x = mv[mv.ik_mode == "position"].view_drift_deg; y = mv[mv.ik_mode == "hold"].view_drift_deg
        V.append(f"view drift (deg): position median {x.median():.2f} (max {x.max():.1f}) vs hold median "
                 f"{y.median():.2f} (max {y.max():.1f}); Mann-Whitney p={mannwhitneyu(x, y).pvalue:.2g}")
        cv = mv.groupby(["ik_mode", "cell"]).roi_density.agg(["mean", "std", "count"])
        cv = cv[cv["count"] >= 3]; cv["cv"] = cv["std"] / cv["mean"]
        cp, ch = cv.loc["position", "cv"].dropna(), cv.loc["hold", "cv"].dropna()
        common = cp.index.intersection(ch.index)
        if len(common) >= 5:
            w = wilcoxon(cp[common], ch[common])
            V.append(f"per-cell ROI edge-density CV across visits: position median {cp[common].median():.3f} vs "
                     f"hold {ch[common].median():.3f} (paired by cell, n={len(common)}, Wilcoxon p={w.pvalue:.3g})")
        rp, rh = mv[mv.ik_mode == "position"].rb_err_mm, mv[mv.ik_mode == "hold"].rb_err_mm
        V.append(f"read-back error (mm): position {rp.mean():.2f} vs hold {rh.mean():.2f}")
        jt = ep.groupby("ik_mode").measured_s.mean()
        V.append(f"mean episode time (s): " + ", ".join(f"{k} {v:.1f}" for k, v in jt.items()))
        fig, ax = plt.subplots(figsize=(3.3, 2.5))
        ax.boxplot([x, y], labels=["position-only IK", "orientation hold"]); ax.set_ylabel("view drift (deg)")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_view_drift.pdf")); plt.close(fig)
    else:
        V.append("run the same methods with --ik position and --ik hold to test view consistency")
    open(os.path.join(a.out, "view_consistency.txt"), "w").write("\n".join(V) + "\n"); print("\n".join(V))

    X = []
    if "in_fence" in df.columns:
        moved_out = int(((df.moved == 1) & (df.in_fence == 0)).sum())
        refused = df[df.refusal.astype(str).str.contains("fence", na=False)]
        X.append(f"fence compliance: {moved_out} moves outside the workspace fence (must be 0); "
                 f"{len(refused)} targets refused by the fence")
    if "view_err_mm" in mv.columns and mv.view_err_mm.notna().any():
        vo, vc = mv.view_err_open_mm.dropna(), mv.view_err_mm.dropna()
        sv = ("closed-loop view servoing: view error open-loop median %.2f mm (95th %.2f) -> after servoing median "
              "%.2f mm (95th %.2f); %.1f corrections/cell, %.2f s/cell extra"
              % (vo.median(), np.percentile(vo, 95), vc.median(), np.percentile(vc, 95),
                 mv.servo_iters.mean(), mv.servo_s.mean()))
        if len(vo) == len(vc) and len(vc) > 5:
            sv += "; Wilcoxon p=%.2g" % wilcoxon(vo.values, vc.values).pvalue
        X.append(sv)
    open(os.path.join(a.out, "fence_compliance.txt"), "w").write("\n".join(X) + "\n"); print("\n".join(X))

    # ---------------- grounding
    gf = files(a.grounding)
    if gf:
        gd = pd.concat([pd.read_csv(f).assign(file=os.path.basename(f)) for f in gf])
        e = gd.err_mm.dropna() if "err_mm" in gd.columns else pd.Series(dtype=float)
        if len(e):
            s = (f"grounding/camera-position error over {len(e)} cell visits ({len(gf)} validation runs): "
                 f"median {e.median():.2f} mm, 95th pct {np.percentile(e, 95):.2f} mm, max {e.max():.2f} mm")
            if "ncc" in gd.columns:
                s += f"; NCC median {gd.ncc.median():.3f}"
            if "markers_seen" in gd.columns:
                s += f"; markers used per cell: median {gd.markers_seen.median():.0f}"
            for fn, gg in gd.groupby("file"):
                if gg.err_mm.notna().any():
                    s += f"\n  {fn}: median {gg.err_mm.median():.2f} mm (n={gg.err_mm.notna().sum()})"
        else:
            s = "no reliable grounding measurements in the validation files"
        open(os.path.join(a.out, "grounding.txt"), "w").write(s + "\n"); print(s)
    gr = mv[mv.ik_mode == "hold"].ground_res_mm.dropna()
    if len(gr):
        print(f"in-run grounding residual (hold): median {gr.median():.2f} mm, 95th pct {np.percentile(gr, 95):.2f} mm")
    if mv.infer_ms.notna().any():
        print(f"policy inference on robot: {mv.infer_ms.mean():.2f} ms mean, p95 {mv.infer_ms.quantile(.95):.2f} ms")

    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))
    for ax, (task, e) in zip(axs, ep.groupby("task")):
        met = "measured_s" if task == "coverage" else "value"
        order = sorted(e.method.unique())
        ax.boxplot([e[e.method == m][met] for m in order], labels=order)
        ax.set_title(task); ax.set_ylabel(met); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_methods.pdf")); plt.close(fig)
    print(f"\nsaved results in {a.out}/")


if __name__ == "__main__":
    main()
