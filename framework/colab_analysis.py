"""Plots and statistics from the exported data (used by colab_plots.ipynb; also runs on a PC).

    import colab_analysis as CA
    CA.run_all(".")                 # every analysis whose data exists -> figs_colab/*.png + figs_colab/summary.json

Sections (each skipped if its data is missing):
  robot_steps        step responses: duration vs joint change per speed (time model t = t0 + dq/omega),
                     rise / settling time, overshoot, steady-state error, peak joint velocity and acceleration
  robot_repeat       repeatability at the reference: joint read-back and camera-measured error
  robot_backlash     read-back and image offset by approach direction per joint
  robot_static       FK vs firmware coordinates, joint limits
  trajectories       joint trajectories, flange height and conditioning kappa over time for every recorded run
  refcover_atlas     reachability / conditioning lattice per height with the planned route
  refcover_compare   proposed vs singularity-blind vs hand-programmed raster over required resolutions
  refcover_runs      view error open-loop vs servoed, conditioning executed, mosaics
  stream_runs        PARALLEL tracking during inspection (stream.csv + events.csv): joints, joint speeds,
                     flange height and Cartesian speed, conditioning, with inspection phases shaded; per-run
                     kinematic/dynamic summary and time breakdown of the loop (deciding, moving, verifying, capturing)
Polling is ~20 Hz, so velocities/accelerations are coarse estimates of the motion, stated as such.
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figs_colab"
SUM = {}
plt.rcParams.update({"font.size": 8, "figure.dpi": 130})


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name + ".png"), dpi=200); plt.close(fig)
    print("  figure", name)


def _runs(root, kind):
    return sorted(d for d in glob.glob(os.path.join(root, "hw_logs", kind, "*")) if os.path.isdir(d))


# ------------------------------------------------------------------ motion metrics per command
def step_metrics(run):
    mv = pd.read_csv(os.path.join(run, "moves.csv")); sm = pd.read_csv(os.path.join(run, "samples.csv"))
    mv = mv[mv.refused == 0]
    rows = []
    for _, m in mv.iterrows():
        s = sm[sm.cmd_id == m.cmd_id].sort_values("t")
        if len(s) < 3:
            continue
        qf = np.array([m["q_from_%d" % i] for i in range(1, 7)]); qc = np.array([m["q_cmd_%d" % i] for i in range(1, 7)])
        j = int(np.argmax(np.abs(qc - qf))); A = qc[j] - qf[j]
        if abs(A) < 0.5:
            continue
        t = s.t.values - m.t_cmd; y = s["q_%d" % (j + 1)].values
        p = (y - qf[j]) / A
        def first(cond):
            i = np.flatnonzero(cond); return float(t[i[0]]) if len(i) else np.nan
        yf = y[-1]
        outside = np.flatnonzero(np.abs(y - yf) > 0.3)
        v = np.gradient(y, t) if len(np.unique(t)) == len(t) else np.full(len(t), np.nan)
        acc = np.gradient(v, t) if np.isfinite(v).all() else v
        rows.append(dict(run=os.path.basename(run), cmd_id=m.cmd_id, joint=j + 1, amp_deg=float(A), speed=m.speed,
                         duration_s=m.duration_s, dq_max_deg=m.dq_max_deg, rise_s=first(p >= 0.9) - first(p >= 0.1),
                         settle_s=float(t[outside[-1] + 1]) if len(outside) and outside[-1] + 1 < len(t) else float(t[0]),
                         overshoot_pct=float(max(0.0, (np.max(p) - 1) * 100)), ss_err_deg=float(yf - qc[j]),
                         peak_vel_dps=float(np.nanmax(np.abs(v))), peak_acc_dps2=float(np.nanmax(np.abs(acc)))))
    return pd.DataFrame(rows)


def robot_steps(root):
    runs = _runs(root, "characterize")
    D = pd.concat([step_metrics(r) for r in runs if os.path.exists(os.path.join(r, "samples.csv"))], ignore_index=True) \
        if runs else pd.DataFrame()
    if not len(D):
        return print("  skip robot_steps (no characterization data)")
    os.makedirs(OUT, exist_ok=True); D.to_csv(os.path.join(OUT, "robot_step_metrics.csv"), index=False)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    fits = {}
    for sp, g in D.groupby("speed"):
        axs[0].scatter(g.dq_max_deg, g.duration_s, s=8, label="speed %d" % sp)
        if len(g) > 3:
            b1, b0 = np.polyfit(g.dq_max_deg, g.duration_s, 1); fits[int(sp)] = dict(t0_s=b0, omega_dps=1 / b1 if b1 > 0 else np.nan)
            x = np.linspace(0, g.dq_max_deg.max(), 10); axs[0].plot(x, b0 + b1 * x, lw=0.8)
    axs[0].set_xlabel("largest joint change (deg)"); axs[0].set_ylabel("move duration (s)"); axs[0].legend(frameon=False)
    D.boxplot(column="peak_vel_dps", by="speed", ax=axs[1]); axs[1].set_title(""); axs[1].set_ylabel("peak joint velocity (deg/s)")
    D.boxplot(column="settle_s", by="joint", ax=axs[2]); axs[2].set_title(""); axs[2].set_ylabel("settling time (s)")
    plt.suptitle(""); _save(fig, "robot_steps")
    # one example step response per speed
    r = runs[-1]; sm = pd.read_csv(os.path.join(r, "samples.csv")); mv = pd.read_csv(os.path.join(r, "moves.csv"))
    fig, ax = plt.subplots(figsize=(4, 2.8))
    for sp, g in D[D.run == os.path.basename(r)].groupby("speed"):
        m = g.iloc[(g.amp_deg.abs() - 10).abs().argmin()]
        mm = mv[mv.cmd_id == m.cmd_id].iloc[0]; s = sm[sm.cmd_id == m.cmd_id]
        ax.plot(s.t - mm.t_cmd, s["q_%d" % m.joint] - mm["q_from_%d" % m.joint], marker=".", label="J%d, speed %d" % (m.joint, sp))
    ax.set_xlabel("time after command (s)"); ax.set_ylabel("joint change (deg)"); ax.legend(frameon=False)
    _save(fig, "robot_step_examples")
    tab = D.groupby(["joint", "speed"])[["rise_s", "settle_s", "overshoot_pct", "ss_err_deg", "peak_vel_dps", "peak_acc_dps2"]].median()
    tab.to_csv(os.path.join(OUT, "robot_step_table.csv"))
    SUM["robot_time_model_per_speed"] = fits
    SUM["robot_steps"] = dict(n_moves=int(len(D)), ss_err_abs_median_deg=float(D.ss_err_deg.abs().median()),
                              overshoot_median_pct=float(D.overshoot_pct.median()), settle_median_s=float(D.settle_s.median()))


def robot_repeat(root):
    fs = [f for r in _runs(root, "characterize") for f in [os.path.join(r, "repeat.csv")] if os.path.exists(f)]
    if not fs:
        return print("  skip robot_repeat")
    R = pd.concat([pd.read_csv(f) for f in fs])
    Q = R[["q_%d" % i for i in range(1, 7)]]
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
    axs[0].bar(range(1, 7), Q.std().values); axs[0].set_xlabel("joint"); axs[0].set_ylabel("read-back SD at the reference (deg)")
    axs[1].hist(R.img_err_mm.dropna(), 15); axs[1].set_xlabel("camera-measured return error (mm)"); axs[1].set_ylabel("returns")
    _save(fig, "robot_repeatability")
    SUM["robot_repeat"] = dict(n=int(len(R)), joint_sd_deg=Q.std().round(3).tolist(),
                               img_err_median_mm=float(R.img_err_mm.median()), img_err_max_mm=float(R.img_err_mm.max()),
                               fw_xyz_sd_mm=R[["fw_x", "fw_y", "fw_z"]].std().round(3).tolist())


def robot_backlash(root):
    fs = [f for r in _runs(root, "characterize") for f in [os.path.join(r, "backlash.csv")] if os.path.exists(f)]
    if not fs:
        return print("  skip robot_backlash")
    B = pd.concat([pd.read_csv(f) for f in fs])
    g = B.groupby(["joint", "approach"]).agg(q_err=("q_err", "mean"), dx=("img_dx_px", "mean"), dy=("img_dy_px", "mean")).reset_index()
    bl = g.pivot(index="joint", columns="approach", values="q_err")
    bl = (bl[1] - bl[-1]) if 1 in bl and -1 in bl else pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=(3.6, 2.6)); ax.bar(bl.index, bl.values)
    ax.set_xlabel("joint"); ax.set_ylabel("read-back: approach + minus - (deg)"); _save(fig, "robot_backlash")
    SUM["robot_backlash_deg"] = {int(k): float(v) for k, v in bl.items()}


def robot_static(root):
    fs = glob.glob(os.path.join(root, "hw_logs", "characterize", "*", "static_robot.json"))
    if fs:
        d = json.load(open(sorted(fs)[-1]))
        SUM["robot_static"] = {k: d[k] for k in ("fk_firmware_dist_mm", "fk_minus_firmware_mm", "joint_limits_deg",
                                                 "encoder_readback_sd_deg")}
    fk = sorted(glob.glob(os.path.join(root, "hw_logs", "fk_check_*.csv")))
    if fk:
        F = pd.read_csv(fk[-1]); SUM["fk_check"] = dict(repo_mean_mm=float(F.repo_err.mean()), alt_mean_mm=float(F.alt_err.mean()))


def trajectories(root, max_runs=6):
    fs = sorted(glob.glob(os.path.join(root, "hw_logs", "**", "samples_fk.csv"), recursive=True))
    if not fs:
        return print("  skip trajectories (run export_for_colab.py on the Jetson first)")
    stats = []
    for f in fs[-max_runs:]:
        S = pd.read_csv(f); name = os.path.relpath(os.path.dirname(f), os.path.join(root, "hw_logs")).replace(os.sep, "_")
        t = S.t - S.t.iloc[0]
        fig, axs = plt.subplots(3, 1, figsize=(7, 5.5), sharex=True)
        for j in range(1, 7):
            axs[0].plot(t, S["q_%d" % j], lw=0.8, label="J%d" % j)
        axs[0].set_ylabel("joint (deg)"); axs[0].legend(ncol=6, frameon=False, fontsize=6)
        axs[1].plot(t, S.z_mm, lw=0.8); axs[1].set_ylabel("flange z (mm)")
        axs[2].plot(t, S.kappa, lw=0.8); axs[2].axhline(0.04, c="r", ls=":", lw=0.8); axs[2].set_ylabel("kappa")
        axs[2].set_xlabel("time (s)"); axs[0].set_title(name); _save(fig, "traj_" + name)
        stats.append(dict(run=name, n_samples=len(S), duration_s=float(t.iloc[-1]), kappa_min=float(S.kappa.min()),
                          z_range_mm=float(S.z_mm.max() - S.z_mm.min())))
    SUM["trajectories"] = stats


def refcover_atlas(root):
    p = os.path.join(root, "deploy", "refcover_lattice.npz")
    if not os.path.exists(p):
        return print("  skip refcover_atlas")
    z = np.load(p, allow_pickle=True)
    dzs, nx, ny = z["dzs"], int(z["nx"]), int(z["ny"]); kap = np.where(z["solved"], z["kappa"], np.nan)
    D = z["D"]
    fig = plt.figure(figsize=(6, 4.5)); ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(D[:, 0], D[:, 1], z["dz"], c=kap, s=6, cmap="viridis", vmin=0, vmax=0.15)
    pl = os.path.join(root, "deploy", "refcover_plan.json")
    if os.path.exists(pl):
        P = json.load(open(pl)); n = [r["node"] for r in P["route"]]
        ax.plot(D[n, 0], D[n, 1], z["dz"][n], "r-", lw=1.5, label="planned route")
        v = [x["node"] for x in P["views"]]; ax.scatter(D[v, 0], D[v, 1], z["dz"][v], c="r", s=40, marker="*", label="views")
        ax.legend(frameon=False)
    ax.set_xlabel("dx (mm)"); ax.set_ylabel("dy (mm)"); ax.set_zlabel("dz (mm)"); fig.colorbar(sc, shrink=0.6, label="kappa")
    _save(fig, "refcover_atlas3d")
    SUM["refcover_lattice"] = dict(nodes=int(len(kap)), solved=int(np.isfinite(kap).sum()),
                                   usable=int(np.asarray(z["ok"]).sum()), heights=dzs.tolist())


def refcover_kappa_maps(root):
    """Per-height reachability/conditioning maps (same figure refcover.py makes when matplotlib works)."""
    p = os.path.join(root, "deploy", "refcover_lattice.npz")
    if not os.path.exists(p):
        return print("  skip refcover_kappa_maps")
    z = np.load(p, allow_pickle=True)
    nx, ny, dzs, xs, ys = int(z["nx"]), int(z["ny"]), z["dzs"], z["xs"], z["ys"]
    box = [100.0, 100.0]
    tgp = os.path.join(root, "deploy", "table_grid.json")
    if os.path.exists(tgp):
        box = json.load(open(tgp)).get("box_mm", box)
    nz = len(dzs); cols = min(nz, 4); rows = int(np.ceil(nz / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.3 * rows), squeeze=False)
    for k in range(nz):
        ax = axs[k // cols][k % cols]; sl = slice(k * nx * ny, (k + 1) * nx * ny)
        K = np.where(z["solved"][sl], z["kappa"][sl], np.nan).reshape(ny, nx)
        im = ax.imshow(K, origin="upper", vmin=0, vmax=0.15, cmap="viridis", extent=[xs[0], xs[-1], ys[-1], ys[0]])
        ax.add_patch(plt.Rectangle((0, 0), box[0], box[1], fill=False, ec="w", lw=0.8))
        ax.set_title("dz %+d mm" % dzs[k], fontsize=7); ax.tick_params(labelsize=6)
    for k in range(nz, rows * cols):
        axs[k // cols][k % cols].axis("off")
    fig.colorbar(im, ax=axs, shrink=0.8, label="conditioning kappa")
    os.makedirs(OUT, exist_ok=True); fig.savefig(os.path.join(OUT, "refcover_kappa_maps.png"), dpi=200); plt.close(fig)
    print("  figure refcover_kappa_maps")


def refcover_compare(root):
    p = os.path.join(root, "results_views", "refcover_sweep.json")
    if not os.path.exists(p):
        return print("  skip refcover_compare")
    S = pd.DataFrame(json.load(open(p))); S["method"] = S.label.str.split(":").str[0].str.split(" ").str[0]
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
    for m, g in S.groupby("label"):
        lab = m.split("(")[0].strip()
        axs[0].plot(g.rho_factor, g.max_cmd_joint_change_deg, marker="o", label=lab)
        axs[1].plot(g.rho_factor, g.n_views, marker="o", label=lab)
        axs[2].plot(g.rho_factor, g.coverage_pct, marker="o", label=lab)
    axs[0].axhline(70, c="r", ls=":", lw=0.8); axs[0].set_ylabel("largest single command (deg)")
    axs[1].set_ylabel("views"); axs[2].set_ylabel("coverage (%)")
    for ax in axs:
        ax.set_xlabel("required resolution (x survey)")
    axs[0].legend(frameon=False, fontsize=6); _save(fig, "refcover_compare")
    SUM["refcover_sweep"] = S.drop(columns=["method"]).to_dict("records")


def refcover_runs(root):
    fs = sorted(glob.glob(os.path.join(root, "hw_logs", "refcover", "*", "views.csv")))
    if not fs:
        return print("  skip refcover_runs")
    V = pd.concat([pd.read_csv(f).assign(run=os.path.basename(os.path.dirname(f))) for f in fs])
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
    stages = [("as planned", "view_err_raw_mm"), ("after feed-forward", "view_err_open_mm"), ("verified", "view_err_mm")]
    stages = [(lab, c) for lab, c in stages if c in V and V[c].notna().any()]
    axs[0].boxplot([V[c].dropna() for _, c in stages], labels=[lab for lab, _ in stages])
    axs[0].set_ylabel("view error (mm)"); axs[0].tick_params(axis="x", labelsize=7)
    if "verify" in V:
        SUM["refcover_verified_by"] = V["verify"].value_counts().to_dict()
    axs[1].scatter(V.kappa_plan, V.kappa_meas, s=10); axs[1].plot([0, .15], [0, .15], "k:", lw=0.8)
    axs[1].set_xlabel("planned kappa"); axs[1].set_ylabel("kappa at the reached pose"); _save(fig, "refcover_runs")
    ms = sorted(glob.glob(os.path.join(root, "hw_logs", "refcover", "*", "mosaic.png")))[-6:]
    if ms:
        import matplotlib.image as mpimg
        fig, axs = plt.subplots(1, len(ms), figsize=(2.6 * len(ms), 2.8), squeeze=False)
        for ax, m in zip(axs[0], ms):
            ax.imshow(mpimg.imread(m)); ax.set_title(os.path.basename(os.path.dirname(m)), fontsize=7); ax.axis("off")
        _save(fig, "refcover_mosaics")
    SUM["refcover_runs"] = dict(n_runs=len(fs), n_views=int(len(V)),
                                view_err_raw_median=float(V.view_err_raw_mm.median()) if "view_err_raw_mm" in V else None,
                                view_err_open_median=float(V.view_err_open_mm.median()),
                                view_err_servoed_median=float(V.view_err_mm.median()), kappa_meas_min=float(V.kappa_meas.min()))


def time_model_all(root):
    fs = glob.glob(os.path.join(root, "hw_logs", "**", "moves.csv"), recursive=True)
    if not fs:
        return
    M = pd.concat([pd.read_csv(f) for f in fs]); M = M[(M.refused == 0) & (M.dq_max_deg > 0.5)]
    if len(M) < 10:
        return
    b1, b0 = np.polyfit(M.dq_max_deg, M.duration_s, 1)
    r = np.corrcoef(M.dq_max_deg, M.duration_s)[0, 1]
    SUM["time_model_all_moves"] = dict(n=int(len(M)), t0_s=float(b0), omega_dps=float(1 / b1) if b1 > 0 else None, r=float(r))


PHASE_COL = {"survey": "#9ecae1", "deciding": "#fdae6b", "decision": "#fdae6b", "moving_to_view": "#c7e9c0",
             "route": "#c7e9c0", "verifying_view": "#bcbddc", "captured": "#fcbba1", "view": "#c7e9c0",
             "episode_end": "#d9d9d9"}


def stream_runs(root, max_runs=8):
    fs = sorted(glob.glob(os.path.join(root, "hw_logs", "**", "stream.csv"), recursive=True))
    if not fs:
        return print("  skip stream_runs (no parallel tracking data)")
    rows, phases = [], []
    for f in fs[-max_runs:]:
        S = pd.read_csv(f)
        if len(S) < 5:
            continue
        d = os.path.dirname(f); name = os.path.relpath(d, os.path.join(root, "hw_logs")).replace(os.sep, "_")
        E = pd.read_csv(os.path.join(d, "events.csv")) if os.path.exists(os.path.join(d, "events.csv")) else pd.DataFrame(columns=["t", "name", "info"])
        t0 = S.t.iloc[0]; t = S.t - t0
        qd = S[["qd_%d" % j for j in range(1, 7)]].abs()
        fig, axs = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
        for j in range(1, 7):
            axs[0].plot(t, S["q_%d" % j], lw=0.8, label="J%d" % j)
            axs[1].plot(t, S["qd_%d" % j].abs(), lw=0.6)
        axs[2].plot(t, S.z_mm, lw=0.8, c="k"); ax2 = axs[2].twinx(); ax2.plot(t, S.v_cart_mm_s, lw=0.6, c="C3")
        axs[3].plot(t, S.kappa, lw=0.8); axs[3].axhline(0.04, c="r", ls=":", lw=0.8)
        axs[0].set_ylabel("joint (deg)"); axs[0].legend(ncol=6, frameon=False, fontsize=6)
        axs[1].set_ylabel("|joint speed| (deg/s)"); axs[2].set_ylabel("flange z (mm)"); ax2.set_ylabel("Cartesian speed (mm/s)", color="C3")
        axs[3].set_ylabel("kappa"); axs[3].set_xlabel("time (s)"); axs[0].set_title(name)
        if len(E):
            te = list(E.t - t0) + [t.iloc[-1]]
            for i, n in enumerate(E.name):
                for ax in axs:
                    ax.axvspan(te[i], te[i + 1], color=PHASE_COL.get(n, "#f0f0f0"), alpha=0.35, lw=0)
                phases.append(dict(run=name, phase=n, seconds=te[i + 1] - te[i]))
        _save(fig, "stream_" + name)
        dp = np.sqrt(np.diff(S.x_mm) ** 2 + np.diff(S.y_mm) ** 2 + np.diff(S.z_mm) ** 2)
        dq = np.abs(np.diff(S[["q_%d" % j for j in range(1, 7)]].values, axis=0)).sum(0)
        dec = E[E.name == "decision"] if len(E) else E
        inf = [json.loads(x).get("infer_ms") for x in dec["info"]] if len(dec) else []
        rows.append(dict(run=name, duration_s=float(t.iloc[-1]), samples=len(S), rate_hz=float(len(S) / max(t.iloc[-1], 1e-6)),
                         moving_pct=float(100 * S.moving.mean()), cart_path_mm=float(dp.sum()),
                         joint_travel_deg=dq.round(1).tolist(), max_joint_speed_dps=qd.max().round(1).tolist(),
                         max_cart_speed_mm_s=float(S.v_cart_mm_s.max()), kappa_min=float(S.kappa.min()),
                         kappa_median=float(S.kappa.median()), z_min_mm=float(S.z_mm.min()),
                         n_decisions=int(len(dec)), infer_ms_mean=float(np.mean([x for x in inf if x is not None])) if inf else None))
    if rows:
        R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "stream_summary.csv"), index=False)
        SUM["stream_runs"] = rows
    if phases:
        P = pd.DataFrame(phases).groupby(["run", "phase"]).seconds.sum().unstack(fill_value=0)
        P.to_csv(os.path.join(OUT, "stream_time_breakdown.csv"))
        fig, ax = plt.subplots(figsize=(7, 0.5 + 0.35 * len(P)))
        left = np.zeros(len(P))
        for ph in P.columns:
            ax.barh(range(len(P)), P[ph], left=left, color=PHASE_COL.get(ph, "#bdbdbd"), label=ph); left += P[ph].values
        ax.set_yticks(range(len(P))); ax.set_yticklabels(P.index, fontsize=6); ax.set_xlabel("time (s)")
        ax.legend(frameon=False, fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
        _save(fig, "stream_time_breakdown")
        SUM["stream_time_breakdown_s"] = P.round(2).to_dict("index")


def run_all(root="."):
    for f in (robot_static, robot_steps, robot_repeat, robot_backlash, time_model_all, trajectories,
              refcover_atlas, refcover_kappa_maps, refcover_compare, refcover_runs, stream_runs):
        try:
            f(root)
        except Exception as e:
            print("  %s failed: %s" % (f.__name__, e))
    os.makedirs(OUT, exist_ok=True)
    json.dump(SUM, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)
    print("summary -> %s/summary.json" % OUT)
    return SUM
