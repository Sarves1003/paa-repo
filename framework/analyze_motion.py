#!/usr/bin/env python3
"""Analyse a motion_study.py run (desktop, Colab or Jetson).

  python3 analyze_motion.py hw_logs/motion/<stamp>

Writes into <stamp>/analysis/:
  trial_metrics.csv    per trial: motion time, kinematic settling, image settling (time to a still, sharp
                       image), overshoot, final error, peak velocity / acceleration / jerk, RMS jerk
  profile_summary.csv  medians per move x profile, and paired tests of every profile against native20
  excite_fit.csv       per joint and frequency: gain, phase lag, delay; first-order-plus-delay fit per joint
  hold_summary.csv     servo status (currents etc.) per held configuration, when the firmware provides it
  fig_*.png            time-to-image vs jerk, trajectories, image motion, frequency response
"""
import json, os, sys
import numpy as np
import pandas as pd

V_STILL_DEG = 0.3          # kinematic settle: within this of the final pose for the rest of the trial
IMG_STILL_PX = 1.0         # image settle: frame-to-frame motion below this (full-resolution pixels) ...
SHARP_FRAC = 0.9           # ... and sharpness at least this fraction of the settled sharpness


def deriv(t, q, fs=50.0):
    """Uniform resampling and Savitzky-Golay derivatives (velocity, acceleration, jerk) per joint."""
    from scipy.signal import savgol_filter
    if len(t) < 9:
        z = np.zeros((0, q.shape[1])); return np.array([]), z, z, z, z
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    qu = np.stack([np.interp(tu, t, q[:, j]) for j in range(q.shape[1])], 1)
    w = min(len(tu) - (1 - len(tu) % 2), 11)
    if w < 5:
        z = np.zeros_like(qu); return tu, qu, z, z, z
    d = 1.0 / fs
    v = savgol_filter(qu, w, 3, deriv=1, delta=d, axis=0)
    acc = savgol_filter(qu, w, 3, deriv=2, delta=d, axis=0)
    jerk = savgol_filter(qu, w, 3, deriv=3, delta=d, axis=0)
    return tu, qu, v, acc, jerk


def image_motion(fr):
    import cv2
    small, t, sharp = fr["small"], fr["t"], fr["sharp"]
    sh = np.zeros(len(t))
    for i in range(1, len(t)):
        a = np.float32(small[i - 1]); b = np.float32(small[i])
        (dx, dy), _ = cv2.phaseCorrelate(a, b)
        sh[i] = 4.0 * np.hypot(dx, dy)                 # 160x120 -> 640x480 pixels
    return t, sh, sharp


def trial_metrics(run, tr, S):
    lab = tr["trial"]; st = S[S["event"] == lab].sort_values("t")
    qcols = ["q_%d" % j for j in range(1, 7)]
    qb = np.array([tr["qb_%d" % j] for j in range(1, 7)]); qa = np.array([tr["qa_%d" % j] for j in range(1, 7)])
    m = dict(trial=lab, move=tr["move"], profile=tr["profile"], direction=tr["direction"], rep=tr["rep"],
             dq_max=tr["dq_max"])
    t_cmd = tr["t_cmd"]
    if len(st) >= 9:
        t = st["t"].values; q = st[qcols].values.astype(float)
        tu, qu, v, acc, jerk = deriv(t, q)
        qfin = q[-3:].mean(0)
        dev = np.max(np.abs(qu - qfin), 1)
        moving = np.flatnonzero(dev > V_STILL_DEG)
        t_kin = (tu[moving[-1]] - t_cmd) if len(moving) else 0.0
        win = (tu >= t_cmd) & (tu <= t_cmd + max(t_kin, 0.1) + 0.3)
        sgn = np.sign(qb - qa)
        over = np.max(np.maximum(0, sgn * (qu[win] - qb)), 0) if win.any() else np.zeros(6)
        m.update(t_kinematic_s=t_kin, final_err_deg=float(np.max(np.abs(qfin - qb))),
                 overshoot_deg=float(np.max(over)), vel_peak=float(np.abs(v[win]).max()) if win.any() else np.nan,
                 acc_peak=float(np.abs(acc[win]).max()) if win.any() else np.nan,
                 jerk_peak=float(np.abs(jerk[win]).max()) if win.any() else np.nan,
                 jerk_rms=float(np.sqrt(np.mean(np.sum(jerk[win] ** 2, 1)))) if win.any() else np.nan,
                 jerk_integral=float(np.sum(np.sum(jerk[win] ** 2, 1)) / 50.0) if win.any() else np.nan,
                 stream_hz=float((len(t) - 1) / (t[-1] - t[0])) if t[-1] > t[0] else np.nan)
    fp = os.path.join(run, "frames", lab + ".npz")
    if os.path.exists(fp):
        fr = np.load(fp)
        if len(fr["t"]) > 5:
            t, sh, sharp = image_motion(fr)
            tail = t >= t[-1] - 0.5
            s_ref = np.median(sharp[tail]) if tail.any() else np.median(sharp)
            bad = (sh > IMG_STILL_PX) | (sharp < SHARP_FRAC * s_ref)
            bad &= t >= t_cmd
            idx = np.flatnonzero(bad)
            t_img = (t[idx[-1]] - t_cmd) if len(idx) else 0.0
            during = (t >= t_cmd) & (t <= t_cmd + max(t_img, 0.1))
            m.update(t_image_s=t_img, img_motion_peak_px=float(sh[t >= t_cmd].max()) if (t >= t_cmd).any() else np.nan,
                     sharp_min_frac=float(sharp[during].min() / s_ref) if during.any() and s_ref > 0 else np.nan,
                     fps=float((len(t) - 1) / (t[-1] - t[0])) if t[-1] > t[0] else np.nan)
    return m


def paired_tests(M, metric):
    from scipy.stats import wilcoxon
    rows = []
    key = ["move", "direction", "rep"]
    base = M[M.profile == "native20"].set_index(key)[metric]
    for p in sorted(M.profile.unique()):
        if p == "native20":
            continue
        other = M[M.profile == p].set_index(key)[metric]
        j = pd.concat([base, other], axis=1, keys=["b", "o"]).dropna()
        if len(j) >= 5 and (j.o - j.b).abs().sum() > 0:
            pv = wilcoxon(j.o, j.b).pvalue
        else:
            pv = np.nan
        rows.append(dict(metric=metric, profile=p, n=len(j), median_native20=j.b.median() if len(j) else np.nan,
                         median_profile=j.o.median() if len(j) else np.nan,
                         median_diff=(j.o - j.b).median() if len(j) else np.nan, p_wilcoxon=pv))
    return rows


def excite_fit(run, S):
    from scipy.optimize import least_squares
    ex = pd.read_csv(os.path.join(run, "excite.csv"))
    cmd = pd.read_csv(os.path.join(run, "commands.csv"))
    rows = []
    for _, r in ex.iterrows():
        j = int(r.joint); f = float(r.freq)
        c = cmd[(cmd.t >= r.t0) & (cmd.t <= r.t0 + r["T"])]
        s = S[(S.event == r.label) & (S.t >= r.t0) & (S.t <= r.t0 + r["T"])]
        if len(c) < 10 or len(s) < 10:
            continue
        def fit(t, y):
            X = np.c_[np.sin(2 * np.pi * f * (t - r.t0)), np.cos(2 * np.pi * f * (t - r.t0)), np.ones_like(t)]
            b = np.linalg.lstsq(X, y, rcond=None)[0]; return np.hypot(b[0], b[1]), np.arctan2(b[1], b[0])
        Ac, pc = fit(c.t.values, c["q_cmd_%d" % j].values)
        Ar, pr = fit(s.t.values, s["q_%d" % j].values)
        dphi = (pr - pc + np.pi) % (2 * np.pi) - np.pi
        rows.append(dict(joint=j, freq=f, gain=Ar / Ac if Ac > 0 else np.nan, phase_deg=np.degrees(dphi),
                         delay_s=-dphi / (2 * np.pi * f)))
    E = pd.DataFrame(rows)
    fits = []
    for j, g in E.groupby("joint"):
        w = 2 * np.pi * g.freq.values; G = g.gain.values; P = np.radians(g.phase_deg.values)
        def res(x):
            T, tau = np.abs(x)
            return np.r_[1 / np.sqrt(1 + (w * T) ** 2) - G, (-np.arctan(w * T) - w * tau) - P]
        try:
            x = np.abs(least_squares(res, [0.1, 0.05]).x)
            fits.append(dict(joint=j, time_constant_s=x[0], dead_time_s=x[1],
                             bandwidth_hz=1 / (2 * np.pi * x[0]) if x[0] > 0 else np.nan))
        except Exception:
            pass
    return E, pd.DataFrame(fits)


def figures(M, out, S, run, T):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = dict(native20="#9a9994", native40="#4a4945", quintic="#2a78d6", septic="#1baf7a", quintic_fast="#eb6834")
    if "t_image_s" in M and "jerk_rms" in M:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for p, g in M.groupby("profile"):
            ax[0].scatter(g.jerk_rms, g.t_image_s, s=12, color=cols.get(p, "k"), label=p, alpha=0.7)
        ax[0].set_xscale("log"); ax[0].set_xlabel("RMS joint jerk (deg/s$^3$)"); ax[0].set_ylabel("Time to still, sharp image (s)")
        ax[0].legend(fontsize=7)
        med = M.groupby(["move", "profile"]).t_image_s.median().unstack()
        med.plot.bar(ax=ax[1], color=[cols.get(c, "k") for c in med.columns], width=0.8, legend=False)
        ax[1].set_ylabel("Median time to image (s)"); ax[1].tick_params(axis="x", labelsize=6, rotation=20)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_time_vs_jerk.png"), dpi=160); plt.close(fig)
    mv = M.move.iloc[0]
    fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    for p in M.profile.unique():
        r = T[(T.move == mv) & (T.profile == p) & (T.direction == 1)]
        if not len(r):
            continue
        r = r.iloc[0]; st = S[S.event == r.trial].sort_values("t")
        if len(st) < 9:
            continue
        tu, qu, v, acc, jerk = deriv(st.t.values, st[["q_%d" % j for j in range(1, 7)]].values.astype(float))
        ax[0].plot(tu - r.t_cmd, np.linalg.norm(v, axis=1), color=cols.get(p, "k"), label=p)
        fp = os.path.join(run, "frames", r.trial + ".npz")
        if os.path.exists(fp):
            t, sh, _ = image_motion(np.load(fp)); ax[1].plot(t - r.t_cmd, sh, color=cols.get(p, "k"))
    ax[0].set_ylabel("|joint velocity| (deg/s)"); ax[0].legend(fontsize=7); ax[0].set_title(mv, fontsize=9)
    ax[1].set_ylabel("Image motion (px/frame)"); ax[1].set_xlabel("Time from command (s)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_trajectories.png"), dpi=160); plt.close(fig)


def main():
    run = sys.argv[1].rstrip("/")
    out = os.path.join(run, "analysis"); os.makedirs(out, exist_ok=True)
    S = pd.read_csv(os.path.join(run, "stream.csv"))
    if os.path.exists(os.path.join(run, "trials.csv")):
        T = pd.read_csv(os.path.join(run, "trials.csv"))
        M = pd.DataFrame([trial_metrics(run, tr, S) for _, tr in T.iterrows()])
        M.to_csv(os.path.join(out, "trial_metrics.csv"), index=False)
        num = [c for c in M.columns if c not in ("trial", "move", "profile", "direction", "rep")]
        summ = M.groupby(["move", "profile"])[num].median().reset_index()
        summ.to_csv(os.path.join(out, "profile_summary.csv"), index=False)
        tests = []
        for metric in ("t_image_s", "t_kinematic_s", "jerk_rms", "jerk_peak", "overshoot_deg", "final_err_deg"):
            if metric in M:
                tests += paired_tests(M, metric)
        pd.DataFrame(tests).to_csv(os.path.join(out, "paired_tests.csv"), index=False)
        print("\nMedian over all moves:")
        show = [c for c in ("t_kinematic_s", "t_image_s", "jerk_rms", "jerk_peak", "acc_peak", "overshoot_deg",
                            "final_err_deg", "stream_hz", "fps") if c in M]
        print(M.groupby("profile")[show].median().round(3).to_string())
        if tests:
            print("\nPaired against native20 (Wilcoxon):")
            print(pd.DataFrame(tests).round(4).to_string(index=False))
        try:
            figures(M, out, S, run, T)
        except Exception as e:
            print("figures skipped:", e)
    if os.path.exists(os.path.join(run, "excite.csv")):
        E, F = excite_fit(run, S)
        E.to_csv(os.path.join(out, "excite_fit.csv"), index=False); F.to_csv(os.path.join(out, "excite_model.csv"), index=False)
        print("\nServo frequency response (first-order + dead time per joint):"); print(F.round(3).to_string(index=False))
    if os.path.exists(os.path.join(run, "hold.csv")):
        H = pd.read_csv(os.path.join(run, "hold.csv"))
        H.to_csv(os.path.join(out, "hold_summary.csv"), index=False)
        cur = [c for c in H.columns if c.startswith("currents_")]
        print("\nHolds: %d configurations; servo currents %s" % (len(H), "recorded" if cur else "NOT provided by firmware"))


if __name__ == "__main__":
    main()
