#!/usr/bin/env python3
"""Axisymmetric normality model of the part surface, learned from a FEW GOOD parts (no CAD, no defect examples).

A lying axisymmetric part may be rolled about its axis and turned end-for-end at every placement, and its estimated
length varies by a few mm, so its unrolled surface (s along the axis, phi around it; soga.unwrap from the real images)
is only defined up to (i) a shift in phi (roll), (ii) a flip s -> -s, (iii) an axial shift and (iv) a small stretch.
The model therefore describes the surface by what is invariant to roll: for every axial bin s, the distribution of
colour/texture over phi. Fitted from K good parts:
  - every part is first ALIGNED on its ring pattern (fraction of ring-coloured pixels along the axis): flip, stretch
    and shift that best overlay the rings (normalised cross-correlation);
  - mu(s), Sigma(s) of the per-pixel feature f = (L*, a*, b*, |grad L*|), pooled over phi and over the K parts;
  - anomaly score per pixel d2 = (f - mu(s))^T Sigma(s)^-1 (f - mu(s)) (Mahalanobis); threshold tau from the good
    parts themselves (leave-one-part-out, 99.9th percentile);
  - ring bands LEARNED without assuming a colour: axial bins whose mean a* (red-green) stands clearly above the
    part's own bare-metal level; pixel threshold midway between the two.
A new surface is aligned the same way, then scored: connected regions above tau = defects (position, area, colour);
a learned ring band with too little ring colour = missing ring.

  python3 axisym.py fit         # from every run saved with  autoinspect.py run --learn-good
  python3 axisym.py profile     # diagnostic: good parts aligned + ring-colour profile -> results_auto/axisym_profile.png
  python3 axisym.py score --run hw_logs/autoinspect/<tag>/pass_0
Model: deploy/axisym_model.npz
"""
import argparse, glob, json, os
import numpy as np
import cv2

MODEL = os.path.join("deploy", "axisym_model.npz")
NS = 96                         # axial bins of the model
PHI_MAX = 55.0                  # only the surface within +-55 deg of the top is modelled (frontal views, little
                                # perspective error; beyond it the unrolling can sample the table at the silhouette)
SCALES = np.round(np.arange(0.86, 1.1401, 0.02), 3)     # part length estimates vary by about +-8 % between placements


def phi_rows(nphi):
    return -180.0 + (np.arange(nphi) + 0.5) * 360.0 / nphi


def band_valid(valid):
    """valid pixels restricted to |phi| <= PHI_MAX."""
    return valid & (np.abs(phi_rows(valid.shape[0])) <= PHI_MAX)[:, None]


def redness(F, thr):
    """Ring material: a* (red-green) above a threshold LEARNED from the good parts. No fixed colour is assumed."""
    return (F[..., 1] - 128.0) > thr


def rustness(F):
    a, b = F[..., 1] - 128.0, F[..., 2] - 128.0
    return (b > 12.0) & (b > a)


def features(tex):
    """(nphi, ns, 4) features: L*, a*, b* (0..255 OpenCV scale) and gradient magnitude of L*."""
    lab = cv2.cvtColor(cv2.GaussianBlur(tex, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    gx = cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1, ksize=3)
    return np.concatenate([lab, np.sqrt(gx ** 2 + gy ** 2)[..., None]], -1)


def to_bins(F, valid, ns_out=NS):
    """Resample the axial direction to ns_out bins (nearest)."""
    ns = F.shape[1]
    idx = np.minimum((np.arange(ns_out) + 0.5) / ns_out * ns, ns - 1).astype(int)
    return F[:, idx], valid[:, idx]


def profile(F, valid):
    """Median feature of every axial bin over the valid pixels (NaN where nothing valid)."""
    P = np.full((F.shape[1], F.shape[2]), np.nan, np.float32)
    for j in range(F.shape[1]):
        v = valid[:, j]
        if v.sum() >= 5:
            P[j] = np.median(F[v, j], 0)
    return P


def _rescale_idx(k, n=NS):
    """Index map that stretches the axial bins by factor k about the centre (nearest bin), -1 = outside."""
    c = (n - 1) / 2.0
    src = np.round(c + (np.arange(n) - c) / k).astype(int)
    return np.where((src >= 0) & (src < n), src, -1)


def _transform_1d(x, flip, k, sh):
    """Flip, stretch by k, shift by sh a per-bin array (NaN outside)."""
    q = x[::-1] if flip else x
    idx = _rescale_idx(k)
    q = np.where(idx >= 0, q[np.maximum(idx, 0)], np.nan)
    return np.roll(q, sh)


def apply_align(F, valid, flip, shift, scale=1.0):
    if flip:
        F, valid = F[:, ::-1], valid[:, ::-1]
    if scale != 1.0:
        idx = _rescale_idx(scale, F.shape[1])
        F = F[:, np.maximum(idx, 0)]; valid = valid[:, np.maximum(idx, 0)] & (idx >= 0)[None]
    F, valid = np.roll(F, shift, 1), np.roll(valid, shift, 1)
    if shift > 0:
        valid[:, :shift] = False
    elif shift < 0:
        valid[:, shift:] = False
    return F, valid


def to_original_bin(j, flip, shift, scale=1.0):
    """Aligned (model) bin j -> bin of the surface as recorded (inverse of apply_align)."""
    c = (NS - 1) / 2.0
    src = c + ((j - shift) - c) / scale
    return (NS - 1 - src) if flip else src


def align(P, Pm, max_shift=14):
    """Fallback alignment on the colour/brightness profile. Returns (flip, shift, cost, scale)."""
    best = (False, 0, np.inf, 1.0)
    w = np.array([1.0, 1.5, 1.5, 0.5])
    for flip in (False, True):
        Q0 = P[::-1] if flip else P
        for k in SCALES:
            idx = _rescale_idx(k)
            Q1 = np.where((idx >= 0)[:, None], Q0[np.maximum(idx, 0)], np.nan)
            for sh in range(-max_shift, max_shift + 1):
                Qs = np.roll(Q1, sh, 0)
                ok = np.isfinite(Qs[:, 0]) & np.isfinite(Pm[:, 0])
                if sh > 0:
                    ok[:sh] = False
                elif sh < 0:
                    ok[sh:] = False
                if ok.sum() < 0.6 * len(ok):
                    continue
                c = float(np.mean(((Qs[ok] - Pm[ok]) * w) ** 2))
                if c < best[2]:
                    best = (flip, sh, c, float(k))
    return best


def ring_profile(F, valid, thr=None):
    """Fraction of ring-coloured pixels per axial bin (a* above thr; thr = this part's own metal level + 3 if None)."""
    a = F[..., 1] - 128.0
    if thr is None:
        cnt = valid.sum(0)
        a_bin = np.where(cnt > 0, (a * valid).sum(0) / np.maximum(cnt, 1), np.nan)
        thr = float(np.nanmedian(a_bin)) + 3.0 if np.isfinite(a_bin).any() else 3.0
    return ((a > thr) & valid).sum(0) / np.maximum(valid.sum(0), 1), thr


def align_rings(rp, rp_ref, max_shift=12):
    """Flip, stretch and axial shift that best overlay the RING PATTERN rp on rp_ref (normalised cross-correlation over
    the overlap). Returns (flip, shift, score, scale)."""
    best = (False, 0, -np.inf, 1.0)
    for flip in (False, True):
        for k in SCALES:
            for sh in range(-max_shift, max_shift + 1):
                q = _transform_1d(rp, flip, float(k), sh)
                ok = np.isfinite(q)
                if sh > 0:
                    ok[:sh] = False
                elif sh < 0:
                    ok[sh:] = False
                if ok.sum() < NS // 2:
                    continue
                x, y = q[ok] - q[ok].mean(), rp_ref[ok] - rp_ref[ok].mean()
                sc = float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))
                if sc > best[2]:
                    best = (flip, sh, sc, float(k))
    return best


def load_surface(run_dir):
    tex = cv2.imread(os.path.join(run_dir, "unrolled_full.png"))
    val = cv2.imread(os.path.join(run_dir, "unrolled_valid.png"), 0)
    if tex is None or val is None:
        return None
    return tex, val > 0


def _gauss(al):
    mu = np.zeros((NS, 4), np.float32); icov = np.zeros((NS, 4, 4), np.float32)
    for j in range(NS):
        X = np.concatenate([F[v[:, j], j] for F, v in al], 0)
        if len(X) < 20:
            mu[j] = 128.0; icov[j] = np.eye(4) / 400.0; continue
        mu[j] = X.mean(0)
        icov[j] = np.linalg.inv(np.cov(X.T) + np.diag([16.0, 4.0, 4.0, 16.0]))   # floor: noise / small lighting changes
    return mu, icov


def _align_set(feats, rp_ref=None):
    """Align every part's binned features on the ring pattern (to rp_ref, or to the best-covered part, refined to the
    mean pattern). Returns (aligned list, alignment parameters, reference pattern)."""
    rps = [ring_profile(F, v)[0] for F, v in feats]
    if rp_ref is None:
        rp_ref = rps[int(np.argmax([v.sum() for _, v in feats]))]
        for _ in range(2):
            pars = [align_rings(rp, rp_ref) for rp in rps]
            rp_ref = np.nanmedian([_transform_1d(rp, fl, k, sh) for rp, (fl, sh, _, k) in zip(rps, pars)], 0)
            rp_ref = np.nan_to_num(rp_ref, nan=0.0)
    pars = [align_rings(rp, rp_ref) for rp in rps]
    al = [apply_align(F, v, fl, sh, k) for (F, v), (fl, sh, _, k) in zip(feats, pars)]
    return al, pars, rp_ref


def fit_from(surfaces):
    """surfaces: list of (tex, valid). Returns model dict."""
    feats = [to_bins(features(t), band_valid(v)) for t, v in surfaces]
    al, pars, rp_ref = _align_set(feats)
    Pm = np.nanmedian(np.stack([profile(F, v) for F, v in al]), 0)
    mu, icov = _gauss(al)
    model = dict(mu=mu, icov=icov, profile=Pm, ring_profile_ref=rp_ref.astype(np.float32),
                 align_scores=np.array([p[2] for p in pars], np.float32),
                 align_scales=np.array([p[3] for p in pars], np.float32))
    # threshold from the good parts: leave-one-part-out scores (pooled 99.9th percentile)
    sc = []
    for i in range(len(surfaces)):
        rest = [al[k] for k in range(len(al)) if k != i]
        if len(rest) >= 2:
            mu_i, ic_i = _gauss(rest)
        else:
            mu_i, ic_i = mu, icov
        d2, v = score_aligned(al[i][0], al[i][1], dict(mu=mu_i, icov=ic_i))
        sc.append(d2[v])
    sc = np.concatenate(sc) if sc else np.array([20.0])
    model["tau"] = np.float32(max(np.percentile(sc, 99.9), 13.3))     # chi2(4) 99% = 13.3 as a floor
    model["tau_lo"] = np.float32(np.percentile(sc, 99.0))
    # ring bands: bins where the ALIGNED good parts are mostly ring-coloured (each part judged against its own bare-metal
    # level, so lighting differences between placements do not matter); consensus = median over the good parts
    rp_al = np.array([ring_profile(F, v)[0] for F, v in al])
    frac = np.median(rp_al, 0)
    ring = frac >= 0.35
    a_bin = mu[:, 1] - 128.0
    metal = float(np.median(a_bin))
    model["ring_profile_ref"] = frac.astype(np.float32)
    model["red_frac"] = frac.astype(np.float32)
    model["ring_bins"] = ring.astype(np.uint8)
    model["ring_a_thr"] = np.float32(metal + 3.0); model["metal_a"] = np.float32(metal)
    model["n_good"] = np.int32(len(surfaces))
    return model


def score_aligned(F, valid, model):
    D = F - model["mu"][None]
    d2 = np.einsum("pjk,jkl,pjl->pj", D, model["icov"], D)
    return np.where(valid, d2, 0.0), valid


def score(tex, valid, model, mm_per_bin, r_mm, min_area_mm2=4.0):
    """Anomaly map (at model bins), defect regions and ring-band check for one unrolled surface."""
    F, v = to_bins(features(tex), band_valid(valid))
    how = "profile"
    if "ring_profile_ref" in model and float(np.max(model["ring_profile_ref"])) > 0.2:
        flip, shift, sc, scale = align_rings(ring_profile(F, v)[0], model["ring_profile_ref"])
        cost, how = 1.0 - sc, "ring pattern"
        if sc < 0.3:                                     # weak ring evidence (rings missing?): fall back
            flip, shift, cost, scale = align(profile(F, v), model["profile"]); how = "profile"
    else:
        flip, shift, cost, scale = align(profile(F, v), model["profile"])
    F, v = apply_align(F, v, flip, shift, scale)
    own_thr = ring_profile(F, v)[1]                      # this part's own bare-metal level + 3
    d2, v = score_aligned(F, v, model)
    tau = float(model["tau"])
    hot = (d2 > tau) & v
    hot = cv2.morphologyEx(hot.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(hot, 8)
    nphi = d2.shape[0]; mm_per_row = 2 * np.pi * r_mm / nphi
    regions = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA] * mm_per_bin * mm_per_row
        if area < min_area_mm2:
            continue
        m = lab == k
        col = F[m][:, :3].mean(0)
        rust_frac = float(rustness(F[m][None])[0].mean())
        rows, cols = np.nonzero(m)
        regions.append(dict(s_mm=float(cols.mean() * mm_per_bin), s_bin=float(cols.mean()),
                            phi_deg=float(-180 + (rows.mean() + 0.5) * 360.0 / nphi),
                            area_mm2=float(area), max_score=float(d2[m].max()), mean_lab=col.tolist(),
                            kind="rust-like (orange-brown)" if rust_frac > 0.3 else "surface anomaly"))
    rings = []
    rb = model["ring_bins"].astype(bool)
    if rb.any():
        idx = np.flatnonzero(rb); groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        exp = model["red_frac"] if "red_frac" in model else np.full(NS, 0.6)
        for g in groups:
            vv = v[:, g]
            if vv.sum() < 20:
                rings.append(dict(bins=[int(g[0]), int(g[-1])], s_mm=float(g.mean() * mm_per_bin), present=None,
                                  red_frac=None)); continue
            fr = float((redness(F[:, g], own_thr) & vv).sum() / vv.sum())
            rings.append(dict(bins=[int(g[0]), int(g[-1])], s_mm=float(g.mean() * mm_per_bin), red_frac=fr,
                              expected=float(exp[g].mean()), present=bool(fr >= 0.5 * float(exp[g].mean()))))
    return dict(d2=d2, valid=v, flip=bool(flip), shift=int(shift), scale=float(scale), align=how,
                align_cost=float(cost), tau=tau, regions=regions, rings=rings,
                max_score=float(d2[v].max()) if v.any() else 0.0,
                p999=float(np.percentile(d2[v], 99.9)) if v.sum() > 50 else 0.0)


def heat(d2, valid, tau):
    z = np.clip(d2 / (2 * tau), 0, 1)
    h = cv2.applyColorMap((z * 255).astype(np.uint8), cv2.COLORMAP_JET)
    h[~valid] = (60, 60, 60)
    return h


def _good_runs():
    return [d for d in sorted(glob.glob(os.path.join("hw_logs", "autoinspect", "*", "pass_*")))
            if os.path.exists(os.path.join(d, "learn_good"))]


def cmd_fit(a):
    surfaces = [s for s in (load_surface(d) for d in _good_runs()) if s is not None]
    if len(surfaces) < 3:
        raise SystemExit("need >= 3 good-part passes recorded with  autoinspect.py run --learn-good  (found %d)" % len(surfaces))
    model = fit_from(surfaces)
    os.makedirs("deploy", exist_ok=True)
    np.savez(MODEL, **model)
    rb = np.flatnonzero(model["ring_bins"])
    groups = [g for g in np.split(rb, np.flatnonzero(np.diff(rb) > 1) + 1) if len(g)] if len(rb) else []
    print("axisymmetric model from %d good passes: tau %.1f (99.9%% of good, leave-one-out); ring-pattern alignment "
          "match %s (1 = perfect), stretch %s; learned bare-metal a* %+.1f, ring threshold a* > %+.1f; %d ring band(s) "
          "at bins %s -> %s"
          % (len(surfaces), float(model["tau"]), np.round(model["align_scores"], 2).tolist(),
             np.round(model["align_scales"], 2).tolist(), float(model["metal_a"]), float(model["ring_a_thr"]),
             len(groups), ["%d-%d" % (g[0], g[-1]) for g in groups], MODEL))


def cmd_profile(a):
    """Diagnostic (no motion): every good pass unrolled and aligned, plus the ring-colour fraction along the axis."""
    surfaces = [(d, load_surface(d)) for d in _good_runs()]
    surfaces = [(d, s_) for d, s_ in surfaces if s_ is not None]
    if not surfaces:
        raise SystemExit("no good passes found")
    feats = [to_bins(features(t), band_valid(v)) for _, (t, v) in surfaces]
    mdl = fit_from([s_ for _, s_ in surfaces]) if len(surfaces) >= 3 else None
    thr = float(mdl["ring_a_thr"]) if mdl is not None else None
    rp_ref = mdl["ring_profile_ref"] if mdl is not None else ring_profile(*feats[0])[0]
    if mdl is not None:
        print("learned: bare-metal a* %+.1f, ring threshold a* > %+.1f, ring bins %s"
              % (float(mdl["metal_a"]), thr, np.flatnonzero(mdl["ring_bins"]).tolist()))
    rows = []
    for (d, (t, v)), (F, vv) in zip(surfaces, feats):
        fl, sh, sc, kk = align_rings(ring_profile(F, vv)[0], rp_ref)
        F2, v2 = apply_align(F, vv, fl, sh, kk)
        fr = ring_profile(F2, v2)[0]
        band = np.abs(phi_rows(t.shape[0])) <= PHI_MAX
        tt = t[band][:, np.minimum((np.arange(NS) + 0.5) / NS * t.shape[1], t.shape[1] - 1).astype(int)]
        tt = tt[:, ::-1] if fl else tt
        tt = tt[:, np.maximum(_rescale_idx(kk), 0)]; tt = np.roll(tt, sh, 1)
        strip = cv2.resize(tt, (NS * 4, max(40, tt.shape[0])), interpolation=cv2.INTER_NEAREST)
        bar = np.full((40, NS * 4, 3), 255, np.uint8)
        for j in range(NS):
            hgt = int(36 * fr[j]); cv2.rectangle(bar, (4 * j, 38 - hgt), (4 * j + 3, 38), (40, 40, 180), -1)
        cv2.line(bar, (0, 38 - int(36 * 0.35)), (NS * 4, 38 - int(36 * 0.35)), (0, 160, 0), 1)
        cv2.putText(strip, os.path.basename(os.path.dirname(d)), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        rows += [strip, bar]
        print("%-10s valid px %6d, ring alignment flip %s stretch %.2f shift %+d (match %.2f), ring-colour bins (>=0.35): %s"
              % (os.path.basename(os.path.dirname(d)), int(vv.sum()), fl, kk, sh, sc, np.flatnonzero(fr >= 0.35).tolist()))
    os.makedirs("results_auto", exist_ok=True)
    cv2.imwrite(os.path.join("results_auto", "axisym_profile.png"), np.vstack(rows))
    print("saved results_auto/axisym_profile.png (each good pass unrolled over +-%d deg, aligned on its rings; red bars = "
          "fraction of ring-coloured pixels per axial bin, green line = 0.35)" % PHI_MAX)


def cmd_score(a):
    z = dict(np.load(MODEL))
    s = load_surface(a.run)
    meta = json.load(open(os.path.join(a.run, "pass.json")))
    r = score(s[0], s[1], z, meta["part"]["part"]["L"] / NS, meta["part"]["part"]["r"])
    print(json.dumps({k: v for k, v in r.items() if k not in ("d2", "valid")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "score", "profile"])
    ap.add_argument("--run")
    a = ap.parse_args()
    {"fit": cmd_fit, "score": cmd_score, "profile": cmd_profile}[a.cmd](a)


if __name__ == "__main__":
    main()
