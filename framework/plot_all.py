#!/usr/bin/env python3
"""Every figure for the paper, from the files the experiments write. Missing data -> that figure is skipped.

  python3 plot_all.py            # -> paper/figs/*.pdf (+ .png previews)
Figures
  fig1_servo        E1  open-loop vs servoed view error (validation runs + every recorded library view)
  fig2_jacobian     E1  measured image motion of the two robot axes vs a pure translation
  fig3_library      E2  measured patch quality by level and pose; survey-predicted vs measured quality
  fig4_learning     E3  validation % of oracle during training, transformer vs MLP (mean +/- SD over seeds)
  fig5_offline      E3  held-out % of oracle per method (seeds as error bars), by pose
  fig6_budget       E3  quality improvement vs time budget, held-out placements
  fig7_live         E4  live robot: improvement per method and pose, paired by placement
  fig8_defects      E6  defect visibility vs view quality; capture of defects not visible in the survey
  fig9_episode      E4  one live episode: survey with the views the policy chose
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper/figs"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150})
NAMES = {"policy": "Transformer (ours)", "attn": "Transformer (ours)", "policy_mlp": "MLP (no attention)",
         "mlp": "MLP (no attention)", "greedy_pred": "Survey greedy", "setcover": "Set cover (Part A)", "nn_time": "Nearest view",
         "raster_high": "Raster (high)", "raster_low": "Raster (close-up)", "random": "Random", "oracle": "Oracle"}
ORDER = ["attn", "policy", "mlp", "policy_mlp", "greedy_pred", "setcover", "nn_time", "raster_low", "raster_high", "random", "oracle"]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig); print("  made", name)


def library_views():
    rows = []
    for d in sorted(glob.glob("deploy/views/*_*/meta.json")):
        m = json.load(open(d))
        for v in m["views"]:
            if v.get("ok") and v["level"] == 0 and v.get("view_err_open_mm") is not None:
                rows.append(dict(pose=m["pose"], open=v["view_err_open_mm"], closed=v.get("view_err_mm"),
                                 iters=v.get("servo_iters"), servo_s=v.get("servo_s")))
    return pd.DataFrame(rows)


def servo_data():
    parts = []
    for f in glob.glob("hw_logs/grounding/validate_*.csv"):
        g = pd.read_csv(f)
        if "open_loop_err_mm" in g:
            parts.append(pd.DataFrame(dict(open=g.open_loop_err_mm, closed=g.err_mm, src="validation")))
    lv = library_views()
    if len(lv):
        parts.append(pd.DataFrame(dict(open=lv.open, closed=lv.closed, src="view library")))
    return pd.concat(parts, ignore_index=True).dropna() if parts else pd.DataFrame()


def fig1():
    d = servo_data()
    if not len(d):
        return print("  skip fig1 (no validation csv / library)")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bins = np.linspace(0, max(d.open.quantile(0.98), 2), 30)
    ax.hist(d.open, bins, alpha=0.6, label="open loop (kinematics)")
    ax.hist(d.closed, bins, alpha=0.6, label="after visual servoing")
    ax.axvline(d.open.median(), ls="--", c="C0", lw=0.8); ax.axvline(d.closed.median(), ls="--", c="C1", lw=0.8)
    ax.set_xlabel("view error on the table (mm)"); ax.set_ylabel("views"); ax.legend(frameon=False)
    ax.set_title("n = %d views" % len(d)); save(fig, "fig1_servo")


def fig2():
    p = "deploy/table_grid.json"
    if not os.path.exists(p):
        return print("  skip fig2 (no table_grid.json)")
    G = json.load(open("deploy/grounding.json")); A = np.asarray(G["A"], float)
    t = json.load(open(p)); b = t["box_px_per_mm"]
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(b * np.cos(th), b * np.sin(th), "k:", lw=0.8, label="pure translation (%.2f px/mm)" % b)
    for i, (lab, c) in enumerate((("robot x", "C0"), ("robot y", "C3"))):
        ax.annotate("", xy=(A[0, i], A[1, i]), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
        ax.text(A[0, i] * 1.08, A[1, i] * 1.08, "%s %.2f" % (lab, np.linalg.norm(A[:, i])), color=c, fontsize=7)
    r = 1.3 * max(np.abs(A).max(), b)
    ax.set_xlim(-r, r); ax.set_ylim(r, -r); ax.set_aspect("equal")
    ax.set_xlabel("image u (px per mm)"); ax.set_ylabel("image v (px per mm)"); ax.legend(frameon=False, loc="lower left")
    ang = np.degrees(np.arccos(abs(A[:, 0] @ A[:, 1]) / np.prod(np.linalg.norm(A, axis=0))))
    ax.set_title("axes %.0f deg apart" % ang); save(fig, "fig2_jacobian")


def lib():
    p = "data/view_library.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p); meta = json.loads(str(z["meta"]))
    return z, meta


def fig3():
    L = lib()
    if L is None:
        return print("  skip fig3 (no view library)")
    z, meta = L
    P = z["w"].shape[1]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    data, labels = [], []
    for pose in ("horizontal", "vertical"):
        for lev, nm in ((0, "survey height"), (1, "close-up")):
            vals = []
            for i in np.flatnonzero(z["poses"] == pose):
                sel = z["avail"][i][lev * P:(lev + 1) * P]
                Q = z["Q"][i][lev * P:(lev + 1) * P][sel]
                w = z["w"][i] > 0.2
                vals += list(Q[:, w][Q[:, w] > 0])
            if vals:
                data.append(vals); labels.append("%s\n%s" % (pose[:5] + ".", nm))
    if data:
        axs[0].boxplot(data, labels=labels, showfliers=False); axs[0].set_ylabel("measured patch quality")
    for pose, c in (("horizontal", "C0"), ("vertical", "C3")):
        x, y = [], []
        for i in np.flatnonzero(z["poses"] == pose):
            m = z["avail"][i][:, None] & (z["F"][i] > 0) & (z["w"][i][None] > 0.2) & (z["Q"][i] > 0)
            x += list(z["qpred"][i][m]); y += list(z["Q"][i][m])
        if x:
            r = np.corrcoef(x, y)[0, 1]
            axs[1].scatter(x, y, s=2, alpha=0.3, c=c, label="%s (r = %.2f)" % (pose, r))
    axs[1].plot([0, 1], [0, 1], "k--", lw=0.6)
    axs[1].set_xlabel("quality predicted from the survey"); axs[1].set_ylabel("quality measured in the view")
    axs[1].legend(frameon=False, markerscale=4); save(fig, "fig3_library")


def train_runs():
    out = []
    for f in sorted(glob.glob("results_views/train_*/test_summary.json")):
        s = json.load(open(f)); d = os.path.dirname(f)
        s["variant"] = s["arch"] + ("+nopose" if s.get("no_pose") else "") + ("+notime" if s.get("no_time") else "")
        s["dir"] = d; out.append(s)
    return out


def fig4():
    R = [r for r in train_runs() if r["variant"] in ("attn", "mlp")]
    if not R:
        return print("  skip fig4 (no training runs)")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for v, c in (("attn", "C0"), ("mlp", "C1")):
        cs = [pd.read_csv(os.path.join(r["dir"], "curve.csv")) for r in R if r["variant"] == v
              and os.path.exists(os.path.join(r["dir"], "curve.csv"))]
        if not cs:
            continue
        n = min(len(c_) for c_ in cs); Y = np.stack([c_.val_pct_of_oracle.values[:n] for c_ in cs])
        it = cs[0]["iter"].values[:n]
        ax.plot(it, Y.mean(0), c=c, label="%s (%d seeds)" % (NAMES[v], len(cs)))
        ax.fill_between(it, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color=c, alpha=0.2)
    ax.set_xlabel("training iteration"); ax.set_ylabel("validation, % of oracle"); ax.legend(frameon=False)
    save(fig, "fig4_learning")


def offline_table():
    """rows: method, variant/seed, pose, pct_of_oracle (one row per seed)."""
    rows = []
    for r in train_runs():
        for pose in ("all", "horizontal", "vertical"):
            S = r if pose == "all" else r.get("by_pose_" + pose)
            if not S:
                continue
            rows.append(dict(method=r["variant"], seed=r["seed"], pose=pose, pct=S["policy_pct_of_oracle"]))
            if r["variant"] == "attn":
                for c in S["comparisons"]:
                    rows.append(dict(method=c["method"], seed=r["seed"], pose=pose, pct=c["pct_of_oracle"]))
    return pd.DataFrame(rows)


def fig5():
    T = offline_table()
    if not len(T):
        return print("  skip fig5 (no training runs)")
    T = T[T.method != "oracle"]
    poses = [p for p in ("horizontal", "vertical") if p in set(T.pose)] or ["all"]
    fig, axs = plt.subplots(1, len(poses), figsize=(3.4 * len(poses), 2.6), squeeze=False)
    for ax, pose in zip(axs[0], poses):
        t = T[T.pose == pose].groupby("method").pct.agg(["mean", "std"])
        ms = [m for m in ORDER + sorted(set(t.index) - set(ORDER)) if m in t.index]
        ax.bar(range(len(ms)), t.loc[ms, "mean"], yerr=t.loc[ms, "std"].fillna(0), capsize=2,
               color=["C0" if m.startswith("attn") else "C1" if m == "mlp" else "0.6" for m in ms])
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([NAMES.get(m, m) for m in ms], rotation=40, ha="right")
        ax.set_ylabel("held-out, % of oracle"); ax.set_title(pose); ax.axhline(100, c="k", lw=0.6, ls=":")
    save(fig, "fig5_offline")


def fig6():
    R = [r for r in train_runs() if r["variant"] in ("attn", "mlp")]
    rows = []
    for r in R:
        for x in json.load(open(os.path.join(r["dir"], "test_rows.json"))):
            rows.append(dict(frac=x["frac"], method=r["variant"], seed=r["seed"], v=x["policy"]))
            if r["variant"] == "attn":
                for m in ("greedy_pred", "nn_time", "raster_low", "random", "oracle"):
                    rows.append(dict(frac=x["frac"], method=m, seed=r["seed"], v=x[m]))
    if not rows:
        return print("  skip fig6 (no training runs)")
    D = pd.DataFrame(rows).groupby(["method", "frac"]).v.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for m in [m for m in ORDER if m in D.columns]:
        ax.plot(D.index * 100, D[m], marker="o", ms=3, label=NAMES[m],
                lw=1.6 if m in ("attn", "mlp") else 0.9, ls="--" if m == "oracle" else "-")
    ax.set_xlabel("time budget (% of a full inspection)"); ax.set_ylabel("quality improvement")
    ax.legend(frameon=False, ncol=2); save(fig, "fig6_budget")


def live():
    fs = glob.glob("hw_logs/viewplan/*/episodes.csv")
    return pd.concat([pd.read_csv(f).assign(run=os.path.dirname(f)) for f in fs], ignore_index=True) if fs else None


def fig7():
    df = live()
    if df is None:
        return print("  skip fig7 (no live runs)")
    poses = sorted(df.pose.unique())
    fig, axs = plt.subplots(1, len(poses), figsize=(3.4 * len(poses), 2.6), squeeze=False)
    for ax, pose in zip(axs[0], poses):
        e = df[df.pose == pose]
        piv = e.pivot_table(index=["run", "trial"], columns="method", values="improvement")
        ms = [m for m in ORDER if m in piv.columns]
        for _, row in piv.iterrows():
            ax.plot(range(len(ms)), row[ms].values, c="0.8", lw=0.5, zorder=1)
        ax.boxplot([piv[m].dropna() for m in ms], positions=range(len(ms)), showfliers=False, widths=0.5)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([NAMES[m] for m in ms], rotation=40, ha="right")
        ax.set_ylabel("quality improvement (live)"); ax.set_title("%s, %d placements" % (pose, len(piv)))
    save(fig, "fig7_live")


def fig8():
    p = "results_views/defect_results.json"
    if not os.path.exists(p) or not os.path.exists("results_views/defect_labels.csv"):
        return print("  skip fig8 (no defect labels)")
    R = json.load(open(p))
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4))
    H = R.get("capture_rate_defects_not_visible_in_survey") or {}
    fr = "0.35"
    ms = [m for m in ORDER if m in H and H[m].get(fr) is not None]
    axs[1].bar(range(len(ms)), [100 * H[m][fr] for m in ms], color=["C0" if m == "policy" else "C1" if m == "policy_mlp" else "0.6" for m in ms])
    axs[1].set_xticks(range(len(ms))); axs[1].set_xticklabels([NAMES[m] for m in ms], rotation=40, ha="right")
    axs[1].set_ylabel("hidden defects captured (%)"); axs[1].set_title("budget 35 %%, n = %s defects" % R.get("n_defects_not_visible_in_survey"))
    q = R.get("mean_quality_visible"), R.get("mean_quality_not_visible")
    axs[0].bar([0, 1], [q[0] or 0, q[1] or 0], color=["C2", "0.6"])
    axs[0].set_xticks([0, 1]); axs[0].set_xticklabels(["defect visible", "not visible"])
    axs[0].set_ylabel("view quality at the defect"); axs[0].set_title("AUC = %.2f, %d tiles" % (R["auc_quality_predicts_visibility"], R["n_tiles"]))
    save(fig, "fig8_defects")


def fig9():
    import cv2
    fs = sorted(glob.glob("hw_logs/viewplan/*/views.csv"))
    if not fs:
        return print("  skip fig9 (no live runs)")
    f = fs[-1]; d = os.path.dirname(f); v = pd.read_csv(f)
    v = v[(v.method == "policy") & (v.trial == v[v.method == "policy"].trial.min())]
    if not len(v):
        return print("  skip fig9 (no policy episode)")
    t = int(v.trial.iloc[0])
    sv = cv2.imread(os.path.join(d, "img", "t%d_policy_survey.jpg" % t))
    tg = json.load(open("deploy/table_grid.json")); U = np.asarray(tg["image_px"])
    s = cv2.cvtColor(sv, cv2.COLOR_BGR2RGB)
    imgs = [cv2.cvtColor(cv2.imread(os.path.join(d, "img", n)), cv2.COLOR_BGR2RGB) for n in v.img.head(5)]
    fig, axs = plt.subplots(1, 1 + len(imgs), figsize=(1.45 * (1 + len(imgs)), 1.5))
    axs[0].imshow(s); axs[0].set_title("survey + plan")
    for n, (_, r) in enumerate(v.iterrows()):
        c = U[int(r.cell)]
        axs[0].plot(c[0], c[1], "o", ms=4 if r.level == 0 else 7, mfc="none", mec="C1" if r.level else "C0")
        axs[0].text(c[0] + 4, c[1] - 4, str(n), color="y", fontsize=6)
    for ax, im, (_, r) in zip(axs[1:], imgs, v.head(5).iterrows()):
        ax.imshow(im); ax.set_title("%d: cell %d, %s" % (r.step, r.cell, "close" if r.level else "high"), fontsize=6)
    for ax in axs:
        ax.axis("off")
    save(fig, "fig9_episode")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        try:
            f()
        except Exception as e:
            print("  %s failed: %s" % (f.__name__, e))
    print("figures in", OUT)
