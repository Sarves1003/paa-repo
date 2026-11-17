#!/usr/bin/env python3
"""Cut the PART out of every recorded image (survey + every view) for the ring / O-ring / rust classifier.

Run on the Jetson after build_view_dataset.py (it follows the placement order of data/view_library.npz):
  python3 build_task_dataset.py
Output: data/task_crops.npz  (packed for Colab by  python3 sync.py pack-for-colab)
        results_task/crop_examples.jpg  (check that every crop shows the whole part)
The crop of view k is the part box found in the survey (plus a margin: 10 mm lying, 25 mm standing, because the
side walls of a standing part lean outwards in off-centre views), mapped into view k with its measured
(tilted views: registration homography) or planned (straight views: servo-verified) survey->view mapping.
"""
import argparse, json, os
import numpy as np
import cv2
import task_model as TM
from views import ViewSet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="data/view_library.npz")
    ap.add_argument("--views", default="deploy/views")
    ap.add_argument("--out", default="data/task_crops.npz")
    a = ap.parse_args()
    z = np.load(a.library)
    meta = json.loads(str(z["meta"])); names = [str(n) for n in z["names"]]
    vs = ViewSet.from_meta(meta)
    ppm = float(meta["table_grid"]["box_px_per_mm"])
    box_px = np.asarray(meta["table_grid"].get("box_px", [[0, 0], [640, 0], [640, 480], [0, 480]]), float)
    whole = [float(box_px[:, 0].min()), float(box_px[:, 1].min()),
             float(np.ptp(box_px[:, 0])), float(np.ptp(box_px[:, 1]))]
    X, PL, KK, Y, M, FR = [], [], [], [], [], []
    Yp, Mp, conds, rust_f, poses, missing = [], [], [], [], [], []
    for i, nm in enumerate(names):
        d = os.path.join(a.views, nm)
        m = json.load(open(os.path.join(d, "meta.json")))
        if not m.get("condition"):
            missing.append(nm); continue
        y, mk = TM.labels(m["condition"], m.get("rust_facing", "none"))
        Yp.append(y); Mp.append(mk); conds.append(m["condition"]); rust_f.append(m.get("rust_facing", "none"))
        poses.append(m["pose"])
        bbox = m.get("part_bbox") or whole
        if not m.get("part_bbox"):
            print("  %s: part not found in the survey, whole box used (crops will be coarse)" % nm)
        jobs = [(vs.K, None, cv2.imread(os.path.join(d, "survey.jpg")), None)]
        for v in m["views"]:
            if v.get("ok"):
                H = np.asarray(v["H_meas"], float) if v.get("H_meas") is not None else None
                jobs.append((int(v["k"]), int(v["k"]), cv2.imread(os.path.join(d, "v%02d.jpg" % v["k"])), H))
        n_ok = 0
        for kk, k, img, H in jobs:
            c, frac = TM.crop_for_view(img, vs, k, bbox, m["pose"], ppm, H)
            if c is None:
                continue
            X.append(c); PL.append(i); KK.append(kk); Y.append(y); M.append(mk); FR.append(frac); n_ok += 1
        print("  %-16s %-26s %3d crops (%s)" % (nm, m["condition"], n_ok, TM.describe(y)))
    if missing:
        raise SystemExit("%d placement(s) without a condition: %s\nset them with  python3 part_labels.py set "
                         "--placement <name> --condition ...  and run again" % (len(missing), ", ".join(missing)))
    Yp = np.array(Yp); Mp = np.array(Mp)
    vary = [j for j in range(len(TM.ITEMS)) if len(set((Yp[:, j] * Mp[:, j] + 2 * (1 - Mp[:, j])).tolist()) - {2}) > 1]
    print("\nitems with both states in the library (these can be learned): %s" % [TM.ITEMS[j] for j in vary])
    for j in range(len(TM.ITEMS)):
        if j not in vary:
            print("  %s: only one state recorded -> not learnable; record placements with the other state" % TM.ITEMS[j])
    np.savez_compressed(a.out, X=np.array(X, np.uint8), placement=np.array(PL), k=np.array(KK), y=np.array(Y),
                        m=np.array(M), frac=np.array(FR), items=np.array(TM.ITEMS), learnable=np.array(vary),
                        Yp=Yp, Mp=Mp, conditions=np.array(conds), rust_facing=np.array(rust_f), poses=np.array(poses),
                        names=np.array(names), K=vs.K)
    os.makedirs("results_task", exist_ok=True)
    pick = np.linspace(0, len(X) - 1, min(24, len(X))).astype(int)
    tiles = [X[i].copy() for i in pick]
    for t, i in zip(tiles, pick):
        cv2.putText(t, "%s k%d" % (names[PL[i]][-3:], KK[i]), (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite("results_task/crop_examples.jpg", np.vstack([np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)]))
    mb = os.path.getsize(a.out) / 1e6
    print("%d crops of %d placements -> %s (%.1f MB); examples: results_task/crop_examples.jpg" % (len(X), len(Yp), a.out, mb))


if __name__ == "__main__":
    main()
