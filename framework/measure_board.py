#!/usr/bin/env python3
"""Use the drawn box of KNOWN size (default 100 x 100 mm) as the metric reference. No ruler needed.

From the reference image (capture_reference.py) it
  1. finds the drawn box (straight-line detection), or takes 4 clicked / typed corners,
  2. maps the image onto the box (homography, mm),
  3. computes the marker centre-to-centre distances BOARD_W_MM (id at top-left -> top-right)
     and BOARD_H_MM (top-left -> bottom-left),
  4. saves deploy/box.json: the box corners in marker-board coordinates, so that after
     `grounding.py calibrate` the BOX becomes the workspace limit (camera centre stays inside it).

  python3 measure_board.py                     # automatic box detection on deploy/reference/reference.jpg
  python3 measure_board.py --write             # also write BOARD_W_MM / BOARD_H_MM into config.py
  python3 measure_board.py --click             # click the 4 box corners (TL, TR, BR, BL) if detection fails
  python3 measure_board.py --points "129,141;372,142;370,373;130,373"   # corners typed in (pixels)
  options: --image other.jpg   --box-w 100 --box-h 100
Re-run it whenever a marker is moved.
"""
import argparse, json, os, re, sys
import numpy as np
import cv2
import config as C
import board as BD

REF = os.path.join(C.DEPLOY_DIR, "reference", "reference.jpg")


def _fit_line(segs):
    """Total-least-squares line through segment endpoints: returns (point, direction)."""
    P = np.array([[s[0], s[1]] for s in segs] + [[s[2], s[3]] for s in segs], float)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    return c, vt[0]


def _intersect(l1, l2):
    (p, d), (q, e) = l1, l2
    A = np.array([d, -e]).T
    t = np.linalg.solve(A, q - p)
    return p + t[0] * d


def find_box(img, markers=None):
    """Drawn box: two near-horizontal and two near-vertical long line clusters forming a near-square
    quadrilateral that contains the image centre and lies inside the marker rectangle."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 20, 60)
    segs = cv2.HoughLinesP(e, 1, np.pi / 360, 50, minLineLength=int(0.15 * min(H, W)), maxLineGap=12)
    if segs is None:
        return None
    hz, vt = [], []
    for s in segs[:, 0]:
        ang = np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) % 180
        (hz if (ang < 10 or ang > 170) else vt if 80 < ang < 100 else []).append(s)

    def clusters(S, axis):                       # group segments by their mean position
        S = sorted(S, key=lambda s: (s[axis] + s[axis + 2]) / 2)
        out = []
        for s in S:
            pos = (s[axis] + s[axis + 2]) / 2
            if out and abs(pos - out[-1]["pos"]) < 7:
                out[-1]["segs"].append(s); out[-1]["pos"] = np.mean([(t[axis] + t[axis + 2]) / 2 for t in out[-1]["segs"]])
            else:
                out.append(dict(pos=pos, segs=[s]))
        for c in out:
            c["len"] = sum(np.hypot(t[2] - t[0], t[3] - t[1]) for t in c["segs"])
        return [c for c in out if c["len"] > 0.25 * min(H, W)]

    H_c, V_c = clusters(hz, 1), clusters(vt, 0)
    mk = None
    if markers and len(markers) >= 4:
        mk = BD._order_tl_tr_br_bl(list(BD.centres(markers).values())[:4]).astype(np.float32)
    best, best_score = None, -1
    for i, t in enumerate(H_c):
        for b in H_c[i + 1:]:
            for j, l in enumerate(V_c):
                for r in V_c[j + 1:]:
                    w, h = r["pos"] - l["pos"], b["pos"] - t["pos"]
                    if w < 0.2 * W or h < 0.2 * H or not (0.7 < w / h < 1.43):
                        continue
                    lines = [_fit_line(c["segs"]) for c in (t, r, b, l)]
                    try:
                        q = np.array([_intersect(lines[0], lines[3]), _intersect(lines[0], lines[1]),
                                      _intersect(lines[2], lines[1]), _intersect(lines[2], lines[3])])
                    except np.linalg.LinAlgError:
                        continue
                    if cv2.pointPolygonTest(q.astype(np.float32), (W / 2.0, H / 2.0), False) < 0:
                        continue
                    if mk is not None and any(cv2.pointPolygonTest(mk, tuple(map(float, p)), False) < 0 for p in q):
                        continue
                    score = t["len"] + b["len"] + l["len"] + r["len"]
                    if score > best_score:
                        best, best_score = q, score
    return best


def click_corners(img):
    pts = []
    win = "click box corners: TL, TR, BR, BL  (Enter = done, r = restart)"
    def on(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((float(x), float(y)))
    cv2.namedWindow(win); cv2.setMouseCallback(win, on)
    while True:
        v = img.copy()
        for p in pts:
            cv2.circle(v, tuple(int(c) for c in p), 4, (0, 0, 255), -1)
        cv2.imshow(win, v)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) == 4:
            break
        if k == ord("r"):
            pts.clear()
    cv2.destroyAllWindows()
    return np.array(pts)


def write_config(w, h):
    s = open("config.py").read()
    for key, val in (("BOARD_W_MM", w), ("BOARD_H_MM", h)):
        s, n = re.subn(r"^%s\s*=.*$" % key, "%s = %.1f           # from measure_board.py (drawn box)" % (key, val),
                       s, flags=re.M)
        if n == 0:
            s += "\n%s = %.1f\n" % (key, val)
    open("config.py", "w").write(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=REF)
    ap.add_argument("--box-w", type=float, default=100.0, help="drawn box width (mm), left -> right")
    ap.add_argument("--box-h", type=float, default=100.0, help="drawn box height (mm), top -> bottom")
    ap.add_argument("--click", action="store_true")
    ap.add_argument("--points", default=None, help='"x,y;x,y;x,y;x,y" in TL, TR, BR, BL order')
    ap.add_argument("--write", action="store_true", help="write BOARD_W_MM / BOARD_H_MM into config.py")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    if img is None:
        sys.exit("cannot read %s (run capture_reference.py first)" % a.image)
    dets = BD.detect_markers(img)
    if len(dets) < 4:
        sys.exit("only %d/4 markers in %s; all 4 are needed (re-run capture_reference.py)" % (len(dets), a.image))

    if a.points:
        box = np.array([[float(v) for v in p.split(",")] for p in a.points.split(";") if p.strip()])
        how = "typed"
    elif a.click:
        box = click_corners(img); how = "clicked"
    else:
        box = find_box(img, dets); how = "detected"
        if box is None:
            sys.exit("box not found automatically: use --click or --points (TL, TR, BR, BL pixel corners)")
    box = BD._order_tl_tr_br_bl(box)

    # image -> box mm
    box_mm = np.array([[0, 0], [a.box_w, 0], [a.box_w, a.box_h], [0, a.box_h]], float)
    Hb, _ = cv2.findHomography(box.astype(np.float32), box_mm.astype(np.float32))
    cen = BD.centres(dets)
    keys = list(cen)
    q = BD._order_tl_tr_br_bl([cen[k] for k in keys])
    order = [keys[int(np.argmin([np.linalg.norm(cen[k] - p) for k in keys]))] for p in q]   # TL, TR, BR, BL
    m_mm = BD.apply_h(Hb, np.array([cen[k] for k in order]))
    W = float(np.mean([np.linalg.norm(m_mm[1] - m_mm[0]), np.linalg.norm(m_mm[2] - m_mm[3])]))
    Hh = float(np.mean([np.linalg.norm(m_mm[3] - m_mm[0]), np.linalg.norm(m_mm[2] - m_mm[1])]))
    # box corners in marker-board coordinates (TL marker = (0,0), TR = (W,0), BR = (W,H), BL = (0,H))
    Hmb, _ = cv2.findHomography(m_mm.astype(np.float32),
                                np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], np.float32))
    box_board = BD.apply_h(Hmb, box_mm)

    out = dict(box_w_mm=a.box_w, box_h_mm=a.box_h, box_px=box.round(1).tolist(), how=how,
               marker_order_tl_tr_br_bl=[str(k) for k in order], markers_mm_in_box=m_mm.round(2).tolist(),
               board_w_mm=round(W, 2), board_h_mm=round(Hh, 2), box_in_board_mm=box_board.round(2).tolist(),
               px_per_mm_approx=round(float(np.mean([np.linalg.norm(box[1] - box[0]) / a.box_w,
                                                      np.linalg.norm(box[3] - box[0]) / a.box_h])), 2),
               image=os.path.abspath(a.image))
    os.makedirs(C.DEPLOY_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(C.DEPLOY_DIR, "box.json"), "w"), indent=2)

    vis = img.copy()
    cv2.polylines(vis, [box.astype(np.int32)], True, (255, 0, 255), 2)
    for k, p in zip(order, [cen[k] for k in order]):
        cv2.circle(vis, tuple(int(v) for v in p), 6, (0, 200, 255), -1)
        cv2.putText(vis, str(k), (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.polylines(vis, [np.array([cen[k] for k in order], np.int32)], True, (0, 200, 0), 1)
    cv2.imwrite(os.path.join(C.DEPLOY_DIR, "reference", "box_overlay.jpg"), vis)

    side = [np.linalg.norm(box[(i + 1) % 4] - box[i]) for i in range(4)]
    print("box %s: corners %s px (sides %s px)" % (how, box.round(0).astype(int).tolist(), np.round(side, 0).astype(int).tolist()))
    print("markers TL, TR, BR, BL = ids %s" % [str(k) for k in order])
    print("BOARD_W_MM = %.1f   (top-left -> top-right marker centres)" % W)
    print("BOARD_H_MM = %.1f   (top-left -> bottom-left marker centres)" % Hh)
    print("scale about %.2f px/mm; saved deploy/box.json and deploy/reference/box_overlay.jpg" % out["px_per_mm_approx"])
    if a.write:
        write_config(W, Hh); print("written into config.py")
    else:
        print("add these two lines to config.py, or re-run with --write")
    print("after `grounding.py calibrate`, the workspace limit will be the drawn box")


if __name__ == "__main__":
    main()
