#!/usr/bin/env python3
"""EXPERIMENT E6: does the planner find real defects? (ground truth made by you)

Make 3-6 small real marks on the part (scribe line, punch dot, file mark). For the placements you collect
WITH defects, mark where each defect is in the survey image, label in which recorded views it is
visible, then measure (a) visibility vs measured view quality and (b) the fraction of defects each planner
captures in at least one view within the time budget.

  python3 label_defects.py mark  --placement horizontal_003 --points "312,240;350,262"   # survey pixels
  python3 label_defects.py mark  --placement horizontal_003                             # click (display)
  python3 label_defects.py sheet                       # tile sheets: results_views/defect_sheets/*.jpg
  python3 label_defects.py label                       # type the visible tile numbers for every sheet
  python3 label_defects.py analyze [--policy deploy/view_policy_attn.npz --policy-mlp deploy/view_policy_mlp.npz]
Outputs: results_views/defect_labels.csv, defect_results.json
"""
import argparse, csv, glob, json, os
import numpy as np
import cv2
from views import ViewSet, patch_stats, quality, estimate_affine, rollout, to_gray

SHEETS = "results_views/defect_sheets"
LABELS = "results_views/defect_labels.csv"
TILE = 96


def load_lib(path="data/view_library.npz"):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    vs = ViewSet.from_meta(meta)
    return z, meta, vs


def defect_px(vs, k, u_d, aff=None):
    """Survey pixel u_d -> pixel in view k (as views.ViewSet.view_px, for an arbitrary point)."""
    if (aff is not None and np.shape(aff) == (3, 3)) or vs.rot[vs.level[k]]:
        return vs.point_px(k, u_d, aff)
    c, l = vs.cell[k], vs.level[k]
    u0 = vs.uc + (np.asarray(u_d, float) - vs.u[c])
    u_l0 = vs.uc + vs.scales[0] * (u0 - vs.uc)
    if l == 0:
        return u_l0
    return aff[:, :2] @ u_l0 + aff[:, 2] if aff is not None else vs.uc + vs.scales[l] * (u0 - vs.uc)


def _hmeas(d, k):
    """Measured survey -> view homography of a tilted view (collect_views.py), or None."""
    m = json.load(open(os.path.join(d, "meta.json")))
    for v in m["views"]:
        if v.get("k") == k and v.get("H_meas") is not None:
            return np.asarray(v["H_meas"], float)
    return None


def views_of(d):
    m = json.load(open(os.path.join(d, "meta.json")))
    return {v["k"]: v for v in m["views"] if v.get("ok")}


def cmd_mark(a):
    d = os.path.join("deploy/views", a.placement)
    img = cv2.imread(os.path.join(d, "survey.jpg"))
    if img is None:
        raise SystemExit("no survey.jpg in " + d)
    if a.points:
        pts = [[float(v) for v in p.split(",")] for p in a.points.split(";") if p.strip()]
    else:
        pts = []
        def on(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN:
                pts.append([float(x), float(y)])
        cv2.namedWindow("click defects, Enter = done"); cv2.setMouseCallback("click defects, Enter = done", on)
        while True:
            v = img.copy()
            for p in pts:
                cv2.circle(v, tuple(int(c) for c in p), 6, (0, 0, 255), 2)
            cv2.imshow("click defects, Enter = done", v)
            if cv2.waitKey(20) & 0xFF in (13, 10):
                break
        cv2.destroyAllWindows()
    json.dump([dict(id=i, u=p[0], v=p[1]) for i, p in enumerate(pts)], open(os.path.join(d, "defects.json"), "w"), indent=2)
    print("%d defect(s) marked in %s" % (len(pts), a.placement))


def crops(d, vs, u_d):
    """[(k or -1 for the survey, crop image, pixel)] for every recorded view that contains the defect."""
    out = []
    sv = cv2.imread(os.path.join(d, "survey.jpg"))
    out.append((-1, sv, np.asarray(u_d, float), 1.0))
    ok = views_of(d)
    for k in sorted(ok):
        img = cv2.imread(os.path.join(d, "v%02d.jpg" % k))
        aff = _hmeas(d, k)
        if aff is None and vs.via_hi and vs.level[k] > 0 and vs.hi[k] in ok:
            aff = estimate_affine(cv2.imread(os.path.join(d, "v%02d.jpg" % vs.hi[k])), img)
        p = defect_px(vs, k, u_d, aff)
        s = vs.scales[vs.level[k]]
        h = TILE / 2 * s / max(vs.scales)
        if h <= p[0] < img.shape[1] - h and h <= p[1] < img.shape[0] - h:
            out.append((k, img, p, s))
    return out


def cmd_sheet(a):
    z, meta, vs = load_lib()
    os.makedirs(SHEETS, exist_ok=True)
    rows = []
    for d in sorted(glob.glob("deploy/views/*_*")):
        fj = os.path.join(d, "defects.json")
        if not os.path.exists(fj):
            continue
        for df in json.load(open(fj)):
            tiles = []
            for n, (k, img, p, s) in enumerate(crops(d, vs, (df["u"], df["v"]))):
                h = int(round(TILE / 2 * s / max(vs.scales))) + 1
                c = img[int(p[1]) - h:int(p[1]) + h, int(p[0]) - h:int(p[0]) + h]
                c = cv2.resize(c, (TILE * 2, TILE * 2), interpolation=cv2.INTER_CUBIC)
                cv2.putText(c, str(n), (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                tiles.append(c)
                rows.append(dict(placement=os.path.basename(d), defect=df["id"], tile=n, k=k, visible=""))
            cols = 6
            while len(tiles) % cols:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
            cv2.imwrite(os.path.join(SHEETS, "%s_d%d.jpg" % (os.path.basename(d), df["id"])), sheet)
    if not rows:
        raise SystemExit("no defects.json found: run `label_defects.py mark` first")
    with open(LABELS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("%d tiles in %s; open the sheets and run `label_defects.py label` (or fill the visible column in %s)"
          % (len(rows), SHEETS, LABELS))


def cmd_label(a):
    rows = list(csv.DictReader(open(LABELS)))
    keys = sorted(set((r["placement"], r["defect"]) for r in rows))
    for p, dfi in keys:
        sel = [r for r in rows if r["placement"] == p and r["defect"] == dfi]
        if all(r["visible"] != "" for r in sel):
            continue
        ans = input("%s defect %s (sheet %s/%s_d%s.jpg, tiles 0-%d): visible tiles (e.g. 0,3,5 | a = all | n = none): "
                    % (p, dfi, SHEETS, p, dfi, len(sel) - 1)).strip().lower()
        vis = set(range(len(sel))) if ans == "a" else set() if ans in ("n", "") else {int(x) for x in ans.split(",")}
        for r in sel:
            r["visible"] = "1" if int(r["tile"]) in vis else "0"
        with open(LABELS, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("labels saved in", LABELS)


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def cmd_analyze(a):
    z, meta, vs = load_lib()
    rows = [r for r in csv.DictReader(open(LABELS)) if r["visible"] in ("0", "1")]
    names = [str(n) for n in z["names"]]
    lap_ref = meta["lap_ref"]
    qs, vis = [], []
    found = {}                                          # (placement, defect) -> set of views where visible
    for r in rows:
        d = os.path.join("deploy/views", r["placement"]); k = int(r["k"])
        df = [x for x in json.load(open(os.path.join(d, "defects.json"))) if x["id"] == int(r["defect"])][0]
        if k < 0:
            img, p, res, s = cv2.imread(os.path.join(d, "survey.jpg")), np.array([df["u"], df["v"]]), vs.res_survey, 1.0
        else:
            img = cv2.imread(os.path.join(d, "v%02d.jpg" % k)); res = vs.res[vs.level[k]]
            aff = _hmeas(d, k)
            if aff is None and vs.via_hi and vs.level[k] > 0:
                aff = estimate_affine(cv2.imread(os.path.join(d, "v%02d.jpg" % vs.hi[k])), img)
            p = defect_px(vs, k, (df["u"], df["v"]), aff); s = vs.scales[vs.level[k]]
        st = patch_stats(to_gray(img), p, vs.half_px * s / 2)
        if st is None:
            continue
        qs.append(quality(st[0], st[1], res, lap_ref)); vis.append(int(r["visible"]))
        if r["visible"] == "1":
            found.setdefault((r["placement"], int(r["defect"])), set()).add(k)
        else:
            found.setdefault((r["placement"], int(r["defect"])), set())
    qs, vis = np.array(qs), np.array(vis)
    out = dict(n_tiles=int(len(vis)), visible_rate=float(vis.mean()) if len(vis) else None,
               auc_quality_predicts_visibility=auc(qs[vis == 1], qs[vis == 0]),
               mean_quality_visible=float(qs[vis == 1].mean()) if (vis == 1).any() else None,
               mean_quality_not_visible=float(qs[vis == 0].mean()) if (vis == 0).any() else None)
    # (b) defects captured by each planner within the budget (library replay)
    keys = ["Q", "F", "qpred", "qs", "w", "T", "avail", "fp_part", "fp_edge", "fp_glare", "pose"]
    from geo_attn import load_np
    W = {}
    for m, p in (("policy", a.policy), ("policy_mlp", a.policy_mlp)):
        if p and os.path.exists(p):
            W[m] = load_np(p)
    methods = list(W) + ["greedy_pred", "setcover", "nn_time", "raster_high", "raster_low", "random", "oracle"]
    rates, rates_hard = {}, {}
    plc = sorted(set(p for p, _ in found))
    idx = [names.index(p) for p in plc if p in names]
    if idx:
        D = {k: z[k][idx].astype(bool if k == "avail" else float) for k in keys}
        _, tfull, _ = rollout(D, vs, np.inf, "greedy_pred")
        for frac in (0.2, 0.35, 0.5):
            for m in methods:
                _, _, seqs = rollout(D, vs, frac * tfull, m, np.random.default_rng(0), W.get(m))
                hit, hard = [], []
                for i, pname in enumerate([plc[j] for j in range(len(plc)) if plc[j] in names]):
                    taken = set(seqs[i]) | {vs.hi[k] for k in seqs[i]} | {-1}
                    for (pp, dd), ks in found.items():
                        if pp == pname:
                            hit.append(len(ks & taken) > 0)
                            if -1 not in ks:                       # not visible in the survey image
                                hard.append(len(ks & taken) > 0)
                rates.setdefault(m, {})[str(frac)] = float(np.mean(hit)) if hit else None
                rates_hard.setdefault(m, {})[str(frac)] = float(np.mean(hard)) if hard else None
    out["defect_capture_rate_by_method_and_budget"] = rates
    out["capture_rate_defects_not_visible_in_survey"] = rates_hard
    out["n_defects_not_visible_in_survey"] = int(sum(1 for v in found.values() if -1 not in v))
    out["n_defects"] = len(found)
    out["n_defects_visible_anywhere"] = int(sum(1 for v in found.values() if v))
    json.dump(out, open("results_views/defect_results.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mark", "sheet", "label", "analyze"])
    ap.add_argument("--placement")
    ap.add_argument("--points")
    ap.add_argument("--policy", default="deploy/view_policy_attn.npz")
    ap.add_argument("--policy-mlp", default="deploy/view_policy_mlp.npz")
    a = ap.parse_args()
    {"mark": cmd_mark, "sheet": cmd_sheet, "label": cmd_label, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
