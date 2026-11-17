#!/usr/bin/env python3
"""Part pose from the survey image: horizontal (lying, grooves visible) or vertical (end face, bore).

Four image features of the detected part, then a nearest-centroid classifier fitted on the library's
operator labels (leave-one-out accuracy reported, so the claim "pose is recognised automatically"
is measured, not assumed):
  aspect     long / short side of the part's rotated box   (lying part is elongated)
  fill       part area / rotated-box area                   (round end face ~0.79, profile ~0.9)
  core_dark  mean grey of the central 30 % / mean grey of the part   (dark bore -> low)
  coherence  gradient-orientation coherence inside the part (parallel grooves -> high)

  python3 pose.py fit            # fit on deploy/views/*/ (labels from collect_views.py) -> deploy/pose_clf.json
  python3 pose.py test img.jpg   # classify one image
"""
import glob, json, os, sys
import numpy as np
import cv2

FEATS = ["aspect", "fill", "core_dark", "coherence"]


def part_mask(img, G=None):
    """Binary mask of the part (largest dark compact blob, markers ignored, inside the drawn box)."""
    from perception import locate_part_scene, to_gray
    part = locate_part_scene(img, G)
    if part is None:
        return None, None
    g = cv2.GaussianBlur(to_gray(img), (5, 5), 0)
    x, y, w, h = part["bbox_px"]
    _, m = cv2.threshold(g, part["thresh"], 255, cv2.THRESH_BINARY_INV)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    keep = np.zeros_like(m); keep[y:y + h, x:x + w] = 255
    m = cv2.bitwise_and(m, keep)
    cs = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not cs:
        return None, part
    c = max(cs, key=cv2.contourArea)
    out = np.zeros_like(m); cv2.drawContours(out, [c], -1, 255, -1)
    return out, part


def features(img, G=None):
    from perception import to_gray
    m, part = part_mask(img, G)
    if m is None:
        return None
    g = to_gray(img).astype(np.float32)
    c = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    area = cv2.contourArea(c)
    aspect = max(rw, rh) / max(min(rw, rh), 1.0)
    fill = area / max(rw * rh, 1.0)
    r = 0.15 * min(rw, rh)
    core = np.zeros_like(m); cv2.circle(core, (int(cx), int(cy)), max(int(r), 3), 255, -1)
    core_dark = float(g[core > 0].mean() / max(g[m > 0].mean(), 1.0))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy) * (m > 0)
    th = np.arctan2(gy, gx)
    z = (mag * np.exp(2j * th)).sum() / max(mag.sum(), 1e-6)
    return dict(aspect=float(aspect), fill=float(fill), core_dark=core_dark, coherence=float(abs(z)))


class PoseClf:
    def __init__(self, d):
        self.mu, self.sd = np.array(d["mu"]), np.array(d["sd"])
        self.cent = {k: np.array(v) for k, v in d["centroids"].items()}

    def predict(self, f):
        x = (np.array([f[k] for k in FEATS]) - self.mu) / self.sd
        return min(self.cent, key=lambda k: np.linalg.norm(x - self.cent[k]))


def fit_centroids(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    return dict(mu=mu.tolist(), sd=sd.tolist(),
                centroids={c: Z[y == c].mean(0).tolist() for c in sorted(set(y))})


def load_clf(path="deploy/pose_clf.json"):
    return PoseClf(json.load(open(path))) if os.path.exists(path) else None


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("fit", "test"):
        raise SystemExit(__doc__)
    from perception import load_grounding
    G = load_grounding()
    if sys.argv[1] == "test":
        f = features(cv2.imread(sys.argv[2]), G)
        clf = load_clf()
        print(f, "->", clf.predict(f) if (clf and f) else "no classifier/part")
        return
    rows = []
    for d in sorted(glob.glob("deploy/views/*_*")):
        m = os.path.join(d, "meta.json")
        if not os.path.exists(m):
            continue
        f = features(cv2.imread(os.path.join(d, "survey.jpg")), G)
        if f:
            rows.append(dict(f, pose=json.load(open(m))["pose"], placement=os.path.basename(d)))
    if len(rows) < 4 or len(set(r["pose"] for r in rows)) < 2:
        raise SystemExit("need surveys of both poses in deploy/views (run collect_views.py)")
    X = np.array([[r[k] for k in FEATS] for r in rows]); y = np.array([r["pose"] for r in rows])
    hits = []
    for i in range(len(rows)):                                   # leave-one-out accuracy
        tr = np.arange(len(rows)) != i
        hits.append(PoseClf(fit_centroids(X[tr], y[tr])).predict(rows[i]) == y[i])
    d = fit_centroids(X, y); d["loo_accuracy"] = float(np.mean(hits)); d["n"] = len(rows)
    json.dump(d, open("deploy/pose_clf.json", "w"), indent=2)
    os.makedirs("results_views", exist_ok=True)
    import csv
    with open("results_views/pose_features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("pose classifier: leave-one-out accuracy %.1f%% on %d surveys (%s) -> deploy/pose_clf.json"
          % (100 * d["loo_accuracy"], len(rows), ", ".join("%s %d" % (c, (y == c).sum()) for c in sorted(set(y)))))


if __name__ == "__main__":
    main()
