#!/usr/bin/env python3
"""Serial-number inspection: TEACH once, (optionally) SCAN for the best camera position, then INSPECT.

  python3 serial_inspect.py teach   --text "LS230 5/1 085 D25-1" [--roi X Y W H] [--image photo.jpg]
  python3 serial_inspect.py scan                      # robot tries small camera offsets, keeps the best pose
  python3 serial_inspect.py inspect                   # robot goes to the pose, captures, verifies -> PASS / FAIL
  python3 serial_inspect.py inspect --image test.jpg  # same check on a still image (no robot needed)

Teach saves, under <DEPLOY_DIR>/serial/: the golden frame, the preprocessed serial-number crop (template),
the marker positions, the expected text and thresholds. Inspect finds the serial region again (moved by the
marker shift), matches it against the template over a small angle sweep, optionally reads it with Tesseract
OCR (pip install pytesseract; apt install tesseract-ocr) and compares with the expected text.
Run auto_align.py and grounding.py calibrate first; teach at the final locked pose.
"""
import argparse, csv, difflib, json, os, pickle, re, time
import cv2
import numpy as np


# ---------------------------------------------------------------- paths
def _dir(kind):
    try:
        import config as C
        base = C.DEPLOY_DIR if kind == "deploy" else C.LOGS_DIR
    except Exception:
        base = "deploy" if kind == "deploy" else "hw_logs"
    d = os.path.join(base, "serial" if kind == "deploy" else "inspect")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- vision
def prep(img):
    """Illumination-invariant view of engraved text: remove the slow brightness, keep strokes, equalise."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = g.astype(np.float32)
    d = np.clip(g - cv2.GaussianBlur(g, (0, 0), 9), -40, 40)
    d = ((d + 40) * (255.0 / 80.0)).astype(np.uint8)
    return cv2.createCLAHE(2.0, (8, 8)).apply(d)


def marker_shift_px(img, markers_ref):
    """Shift (px) of the board between the taught frame and this one, tracked through the markers."""
    try:
        import board as BD
        sh, n = BD.marker_shift(markers_ref, BD.detect_markers(img))
        if sh is not None:
            return np.asarray(sh, float), int(n)
    except Exception:
        pass
    return np.zeros(2), 0


def find_template(search, tmpl, angles=range(-10, 11, 2)):
    """Best normalised correlation of tmpl inside search over a small rotation sweep."""
    best = (-1.0, (0, 0), 0)
    cx, cy = search.shape[1] / 2.0, search.shape[0] / 2.0
    for ang in angles:
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rot = cv2.warpAffine(search, M, (search.shape[1], search.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        res = cv2.matchTemplate(rot, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), loc, ang)
    return best


def rotate_bound(img, ang):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    return cv2.warpAffine(img, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)


def norm_txt(t):
    return re.sub(r"[^A-Z0-9]", "", t.upper())


def read_text(patch, expected, text_angle):
    """OCR the patch at 4 orientations x 2 polarities; return (best text, similarity to expected) or None."""
    try:
        import pytesseract
    except ImportError:
        return None
    exp = norm_txt(expected)
    best = ("", 0.0)
    for extra in (0, 90, 180, 270):
        im = rotate_bound(patch, text_angle + extra)
        im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for pol in (im, 255 - im):
            try:
                t = pytesseract.image_to_string(
                    pol, config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/ ")
            except Exception:
                return None
            r = difflib.SequenceMatcher(None, norm_txt(t), exp).ratio() if exp else 0.0
            if r > best[1]:
                best = (t.strip(), r)
    return best


def evaluate(img, T, with_ocr=True):
    """Find the taught serial region in img; return a result dict."""
    x, y, w, h = T["roi"]
    sh, n_mk = marker_shift_px(img, T["markers"])
    H, W = img.shape[:2]
    m = int(T.get("margin", 30))
    x0, y0 = int(round(x + sh[0])), int(round(y + sh[1]))
    sx0, sy0 = max(0, x0 - m), max(0, y0 - m)
    sx1, sy1 = min(W, x0 + w + m), min(H, y0 + h + m)
    if sx1 - sx0 < w or sy1 - sy0 < h:
        return dict(score=-1.0, ang=0, off=(0, 0), markers=n_mk, text=None, ratio=None, box=(x0, y0, w, h),
                    sharp=0.0, note="serial region is outside the image")
    search = prep(img[sy0:sy1, sx0:sx1])
    score, loc, ang = find_template(search, T["tmpl"])
    off = (loc[0] - (x0 - sx0), loc[1] - (y0 - sy0))        # residual shift after the marker correction
    bx, by = sx0 + loc[0], sy0 + loc[1]
    crop = img[by:by + h, bx:bx + w]
    sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if crop.size else 0.0
    ocr = read_text(prep(crop), T["expected"], T.get("text_angle", 0)) if (with_ocr and T["expected"] and crop.size) else None
    return dict(score=score, ang=ang, off=off, markers=n_mk, text=(ocr[0] if ocr else None),
                ratio=(ocr[1] if ocr else None), box=(bx, by, w, h), sharp=sharp, note="")


# ---------------------------------------------------------------- teach data
def load_teach():
    d = _dir("deploy")
    try:
        T = json.load(open(os.path.join(d, "teach.json")))
        T["tmpl"] = cv2.imread(os.path.join(d, "tmpl.png"), cv2.IMREAD_GRAYSCALE)
        T["markers"] = pickle.load(open(os.path.join(d, "markers.pkl"), "rb"))
    except Exception:
        raise SystemExit("nothing taught yet: run  python3 serial_inspect.py teach --text \"<serial>\"  first")
    return T


# ---------------------------------------------------------------- hardware
def open_hw(a, locked=False):
    import config as C, hardware as HW
    from hardware import make_arm, make_camera, wait_settled, load_locked_pose
    from kinematics import fk
    from fence import auto_fence
    from auto_align import LiveCam
    ip = os.path.join(C.DEPLOY_DIR, "inspect_pose.json")
    pose = json.load(open(ip)) if (os.path.exists(ip) and not locked) else load_locked_pose(a.dry_run)
    q = np.array(pose["angles_deg"], float)
    T0 = fk(np.deg2rad(q))
    HW.set_fence(auto_fence(), T0[:3, 3] * 1000)
    arm = make_arm(a.dry_run, q); arm.send_deg(q, 20); wait_settled(arm)
    cam = LiveCam(make_camera(a.dry_run, arm, a.ref_img), False)
    time.sleep(0.5)
    return arm, cam, q, T0[:3, :3]


def get_frame(a, locked=False):
    """Still image (--image) or a fresh camera frame at the taught pose. Returns (img, cleanup)."""
    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            raise SystemExit(f"cannot read {a.image}")
        return img, (lambda: None)
    from auto_align import fresh
    arm, cam, q, R = open_hw(a, locked)
    img = fresh(cam)

    def cleanup():
        cam.release(); arm.close()
    return img, cleanup


# ---------------------------------------------------------------- commands
def cmd_teach(a):
    img, done = get_frame(a, locked=True)
    done()
    if a.roi:
        x, y, w, h = a.roi
    else:
        if not os.environ.get("DISPLAY"):
            raise SystemExit("no display: give the serial-number box with --roi X Y W H (pixels)")
        x, y, w, h = map(int, cv2.selectROI("drag a box tightly around the serial number, then ENTER", img))
        cv2.destroyAllWindows()
    if w < 20 or h < 10:
        raise SystemExit("serial box too small")
    d = _dir("deploy")
    tmpl = prep(img[y:y + h, x:x + w])
    try:
        import board as BD
        markers = BD.detect_markers(img)
    except Exception:
        markers = {}
    cv2.imwrite(os.path.join(d, "golden.jpg"), img)
    cv2.imwrite(os.path.join(d, "tmpl.png"), tmpl)
    pickle.dump(markers, open(os.path.join(d, "markers.pkl"), "wb"))
    json.dump(dict(roi=[int(x), int(y), int(w), int(h)], expected=a.text, text_angle=a.text_angle,
                   min_score=a.min_score, ocr_min=a.ocr_min, margin=a.margin,
                   saved_at=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(d, "teach.json"), "w"), indent=2)
    vis = img.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(d, "teach_view.jpg"), vis)
    print(f"taught: box {x},{y} {w}x{h}px, expected '{a.text}', {len(markers)} marker(s) stored -> {d}")


def cmd_inspect(a):
    T = load_teach()
    img, done = get_frame(a)
    done()
    r = evaluate(img, T)
    ok_t = r["score"] >= T["min_score"]
    ok_o = True if r["ratio"] is None else r["ratio"] >= T["ocr_min"]
    verdict = "PASS" if (ok_t and ok_o) else "FAIL"
    msg = f"{verdict}: match {r['score']:.2f} (min {T['min_score']}), angle {r['ang']:+d} deg, residual shift {r['off']} px, markers {r['markers']}"
    if r["ratio"] is not None:
        msg += f", OCR '{r['text']}' similarity {r['ratio']:.2f} (min {T['ocr_min']})"
    elif T["expected"]:
        msg += ", OCR not available -> template check only"
    if r["note"]:
        msg += f" [{r['note']}]"
    print(msg)
    vis = img.copy()
    bx, by, w, h = r["box"]
    col = (0, 200, 0) if verdict == "PASS" else (0, 0, 255)
    cv2.rectangle(vis, (bx, by), (bx + w, by + h), col, 3)
    cv2.putText(vis, f"{verdict} {r['score']:.2f}", (max(bx, 5), max(by - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
    d = _dir("logs")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(d, f"inspect_{stamp}_{verdict}.jpg"), vis)
    new = not os.path.exists(os.path.join(d, "results.csv"))
    with open(os.path.join(d, "results.csv"), "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["time", "verdict", "score", "angle", "off_x", "off_y", "ocr_text", "ocr_ratio", "markers"])
        wr.writerow([stamp, verdict, f"{r['score']:.3f}", r["ang"], r["off"][0], r["off"][1], r["text"], r["ratio"], r["markers"]])
    print(f"saved annotated image + results.csv in {d}")
    raise SystemExit(0 if verdict == "PASS" else 1)


def cmd_scan(a):
    """Try small camera offsets around the locked pose; keep the one where the serial matches best and sharpest."""
    import config as C
    from auto_align import move_by, fresh
    from hardware import wait_settled
    T = load_teach()
    arm, cam, q0, R = open_hw(a, locked=True)
    s = a.scan_mm
    offsets = [(0, 0), (s, 0), (-s, 0), (0, s), (0, -s), (s, s), (-s, -s), (s, -s), (-s, s)]
    best = None
    try:
        for dx, dy in offsets:
            arm.send_deg(q0, 20); wait_settled(arm); time.sleep(0.3)
            qn = q0.copy() if (dx, dy) == (0, 0) else move_by(arm, q0, np.array([dx, dy], float), R, a.speed)
            if qn is None:
                print(f"  offset ({dx:+.0f},{dy:+.0f}) mm: move refused, skipped"); continue
            r = evaluate(fresh(cam), T, with_ocr=False)
            print(f"  offset ({dx:+.0f},{dy:+.0f}) mm: match {r['score']:.2f}, sharpness {r['sharp']:.0f}, markers {r['markers']}")
            key = (round(r["score"], 2), r["sharp"])
            if r["markers"] >= 2 and (best is None or key > best[0]):
                best = (key, qn.copy(), (dx, dy), r)
    finally:
        arm.send_deg(q0, 20); wait_settled(arm)
        cam.release(); arm.close()
    if best is None:
        raise SystemExit("no usable position (markers lost); check the view with live.html")
    json.dump(dict(angles_deg=[float(v) for v in best[1]], offset_mm=list(best[2]), match=best[3]["score"],
                   saved_at=time.strftime("%Y-%m-%d %H:%M:%S"), source="serial_inspect.py scan"),
              open(os.path.join(C.DEPLOY_DIR, "inspect_pose.json"), "w"), indent=2)
    print(f"best offset {best[2]} mm, match {best[3]['score']:.2f} -> saved inspect_pose.json (inspect uses it automatically)")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["teach", "scan", "inspect"])
    ap.add_argument("--text", default="", help="expected serial number (teach)")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="serial-number box in pixels (teach)")
    ap.add_argument("--text-angle", type=float, default=0.0, help="degrees to rotate the crop so the text reads horizontally (OCR)")
    ap.add_argument("--min-score", type=float, default=0.60, help="template match threshold")
    ap.add_argument("--ocr-min", type=float, default=0.80, help="OCR similarity threshold")
    ap.add_argument("--margin", type=int, default=30, help="search margin around the taught box, px")
    ap.add_argument("--scan-mm", type=float, default=5.0)
    ap.add_argument("--image", help="use a still image instead of the camera")
    ap.add_argument("--speed", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))
    a = ap.parse_args()
    {"teach": cmd_teach, "scan": cmd_scan, "inspect": cmd_inspect}[a.mode](a)


if __name__ == "__main__":
    main()
