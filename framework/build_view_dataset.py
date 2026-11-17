#!/usr/bin/env python3
"""Score every recorded view and build the data-driven twin (Jetson or Colab).

  python3 build_view_dataset.py                     # deploy/views/* -> data/view_library.npz
  options: --test-frac 0.25 --t0 0.5 --omega 30 --seed 0
Also writes results_views/library_stats.txt (E2 evidence): close-up gain, glare by pose, and how well
the survey PREDICTS view quality (if it predicted it perfectly, a hand-made greedy would be enough).
"""
import argparse, glob, json, os
import numpy as np
import cv2
from views import (ViewSet, patch_stats, quality, survey_features, estimate_affine, time_matrix, to_gray, scale_guess)


def raw_stats(img, k, vs, aff):
    g = to_gray(img)
    lap, glare = np.zeros(vs.P), np.zeros(vs.P); vis = np.zeros(vs.P, bool)
    for p in range(vs.P):
        s = patch_stats(g, vs.view_px(k, p, aff), vs.half(k))
        if s is not None:
            lap[p], glare[p], _ = s; vis[p] = True
    return lap, glare, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="deploy/views")
    ap.add_argument("--table-grid", default="deploy/table_grid.json")
    ap.add_argument("--grounding", default="deploy/grounding.json")
    ap.add_argument("--out", default="data/view_library.npz")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--t0", type=float, default=0.5)
    ap.add_argument("--omega", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tg = json.load(open(a.table_grid)); G = json.load(open(a.grounding))
    dirs = sorted(d for d in glob.glob(os.path.join(a.views, "*_*")) if os.path.exists(os.path.join(d, "meta.json")))
    if not dirs:
        raise SystemExit("no placements in %s (run collect_views.py)" % a.views)
    metas = [json.load(open(os.path.join(d, "meta.json"))) for d in dirs]
    levels = metas[0].get("levels_dz", [0.0, metas[0]["dz"]])
    dz = levels[-1] - levels[0]
    H = metas[0].get("cam_height_mm") or 250.0

    # 1) view scales measured from the images
    lattice = bool(metas[0].get("lattice"))
    affs = {}
    if lattice:                                     # unified mode: every view servoed -> scale measured at every view
        sc = [float(np.median([v["scale"] for m in metas for v in m["views"]
                               if v.get("ok") and v["level"] == l and v.get("scale")] or [scale_guess(levels, H)[l]]))
              for l in range(len(levels))]
        vs = ViewSet(tg, levels, scales=sc, uc=G["uc"], via_hi=False, level_H=metas[0].get("levels_H"),
                     level_rpy=metas[0].get("levels_rpy"))
        st_ = np.flatnonzero(~vs.rot)
        s0, s_lo = sc[st_[0]], sc[st_[-1]] / sc[st_[0]]
        print("unified mode: measured view scales vs the survey per level: %s" % np.round(sc, 3).tolist())
        n_reg = 0
        for d, m in zip(dirs, metas):                 # tilted views: scored with their MEASURED survey->view mapping
            for v in m["views"]:
                if v.get("ok") and v.get("H_meas") is not None:
                    affs[(d, v["k"])] = np.asarray(v["H_meas"], float); n_reg += 1
        if vs.rot.any():
            n_t = sum(1 for m in metas for v in m["views"] if v.get("ok") and vs.rot[v["level"]])
            print("tilted views: %d recorded, %d registered to the survey (the others use the planned mapping)"
                  % (n_t, n_reg))
    else:                                           # level 0 from the servo, close-up by feature matching
        s0s = [v["scale"] for m in metas for v in m["views"] if v.get("ok") and v["level"] == 0 and v.get("scale")]
        s0 = float(np.median(s0s)) if s0s else scale_guess(levels, H)[0]
        vs0 = ViewSet(tg, levels, scales=[s0, s0], uc=G["uc"])
        scales = []
        for d, m in zip(dirs, metas):
            ok = {v["k"]: v for v in m["views"] if v.get("ok")}
            for k, v in ok.items():
                if v["level"] > 0 and vs0.hi[k] in ok:
                    A = estimate_affine(cv2.imread(os.path.join(d, "v%02d.jpg" % vs0.hi[k])),
                                        cv2.imread(os.path.join(d, "v%02d.jpg" % k)))
                    if A is not None:
                        s = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
                        if 1.0 < s < 3.0:
                            affs[(d, k)] = A; scales.append(s)
        g = scale_guess(levels, H)
        s_lo = float(np.median(scales)) if scales else g[1] / g[0]
        vs = ViewSet(tg, levels, scales=[s0, s0 * s_lo], uc=G["uc"])
        print("view scales vs the survey: level 0 %.2f (%d servo measurements), close-up %.2f (%d image pairs); "
              "survey resolution factor %.2f" % (s0, len(s0s), s0 * s_lo, len(scales), vs.res_survey))

    # 2) raw stats of every recorded view; library-wide sharpness reference
    raws = []
    for d, m in zip(dirs, metas):
        ok = {v["k"]: v for v in m["views"] if v.get("ok")}
        R = {}
        for k in ok:
            R[k] = raw_stats(cv2.imread(os.path.join(d, "v%02d.jpg" % k)), k, vs, affs.get((d, k)))
        raws.append(R)
    all_lap = np.concatenate([r[0][r[2]] for R in raws for r in R.values()] or [np.ones(1)])
    lap_ref = float(np.percentile(all_lap, 95))
    print("sharpness reference (95th pct log-Laplacian variance): %.2f" % lap_ref)

    # 3) instances
    insts, names, poses, stats = [], [], [], []
    for d, m, R in zip(dirs, metas, raws):
        survey = cv2.imread(os.path.join(d, "survey.jpg"))
        sf = survey_features(survey, vs, lap_ref, None)
        if m.get("part_bbox"):
            from views import part_weights
            sf["w"] = part_weights(survey, vs, lambda img: {"bbox_px": m["part_bbox"]})[0]
        K, P = vs.K, vs.P
        Q = np.zeros((K, P)); avail = np.zeros(K, bool); servo = np.zeros(K)
        ok = {v["k"]: v for v in m["views"] if v.get("ok")}
        for k, (lap, glare, vis) in R.items():
            Q[k] = [quality(lap[p], glare[p], vs.res[vs.level[k]], lap_ref) if vis[p] else 0.0 for p in range(P)]
            avail[k] = True
            servo[k] = ok[k].get("servo_s", 0.0) or 0.0
        if lattice:                                 # safe lattice route time + measured servo time of the target view
            mean_servo = float(np.mean([v for k, v in enumerate(servo) if avail[k]] or [3.0]))
            servo = np.where(avail, servo, mean_servo)
            T = np.asarray(m["T_route"], float) + servo[None, :]
        else:
            mean_servo = float(np.mean([v for k, v in enumerate(servo) if avail[k] and vs.level[k] == 0] or [3.0]))
            servo = np.where((vs.level == 0) & ~avail, mean_servo, servo)
            qd = np.vstack([np.asarray(m["q_plan"], float), np.asarray(m["q_lock"], float)[None]])
            T = time_matrix(qd, servo, vs, a.t0, a.omega)
        pose = 1.0 if m["pose"] == "vertical" else 0.0
        insts.append(dict(Q=Q, F=sf["F"], qpred=sf["qpred"], qs=sf["qs"], w=sf["w"], T=T, avail=avail,
                          fp_part=sf["fp_part"], fp_edge=sf["fp_edge"], fp_glare=sf["fp_glare"], pose=pose,
                          servo_mean=mean_servo))
        names.append(os.path.basename(d)); poses.append(m["pose"])
        sel = avail[:, None] & (sf["F"] > 0) & (sf["w"][None] > 0.2)
        if sel.any():
            hiK = slice(vs.lv_high * P, (vs.lv_high + 1) * P); loK = slice(vs.lv_low * P, (vs.lv_low + 1) * P)
            rotK = np.repeat(vs.rot, P)
            stats.append(dict(pose=m["pose"], hi=Q[hiK][sel[hiK]].mean() if sel[hiK].any() else np.nan,
                              lo=Q[loK][sel[loK]].mean() if sel[loK].any() else np.nan,
                              tilt=Q[rotK][sel[rotK]].mean() if sel[rotK].any() else np.nan,
                              r=float(np.corrcoef(Q[sel], (sf["qpred"])[sel])[0, 1]) if sel.sum() > 3 else np.nan,
                              glare=float(np.mean([R[k][1][R[k][2]].mean() for k in R])), survey_v=float(
                                  (sf["w"] * sf["qs"]).sum() / max(sf["w"].sum(), 1e-6))))

    # 4) split by placement, stratified by pose
    rng = np.random.default_rng(a.seed)
    split = np.zeros(len(insts), int)                         # 0 train, 1 test
    for pz in set(poses):
        idx = [i for i, p in enumerate(poses) if p == pz]; rng.shuffle(idx)
        nt = int(round(a.test_frac * len(idx))) if len(idx) >= 4 else 0
        split[idx[:nt]] = 1
    keys = ["Q", "F", "qpred", "qs", "w", "T", "avail", "fp_part", "fp_edge", "fp_glare", "pose", "servo_mean"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, names=np.array(names), poses=np.array(poses), split=split,
             meta=np.array(json.dumps(dict(dz=dz, levels_dz=list(levels), scales=vs.scales.tolist(), via_hi=vs.via_hi,
                                           levels_rpy=vs.rpy.tolist(), levels_H=vs.Hl and [h.tolist() for h in vs.Hl],
                                           lattice=lattice, lap_ref=lap_ref, t0=a.t0, omega=a.omega,
                                           uc=list(G["uc"]), table_grid=tg))),
             **{k: np.stack([np.asarray(d[k]) for d in insts]) for k in keys})
    print("%d placements (%s) -> %s; test placements: %s" % (
        len(insts), ", ".join("%s %d" % (p, poses.count(p)) for p in sorted(set(poses))), a.out,
        [n for n, s in zip(names, split) if s]))

    os.makedirs("results_views", exist_ok=True)
    L = ["E2 view library: %d placements, view scales %.2f / %.2f, lap_ref %.2f" % (len(insts), s0, s0 * s_lo, lap_ref)]
    for pz in sorted(set(poses)):
        S = [s for s in stats if s["pose"] == pz]
        if S:
            f = lambda key: np.nanmean([s[key] for s in S])
            L.append("%s: mean patch quality high %.3f vs low %.3f vs tilted %.3f; mean glare %.3f; survey value %.3f; "
                     "survey->view quality correlation r=%.2f (n=%d placements)"
                     % (pz, f("hi"), f("lo"), f("tilt"), f("glare"), f("survey_v"), f("r"), len(S)))
    L.append("r well below 1 = the survey cannot predict view quality (glare/specular changes): "
             "this is what the learned planner must handle")
    open("results_views/library_stats.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
