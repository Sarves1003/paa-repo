#!/usr/bin/env python3
"""Capture a REFERENCE image at the robot's current position. The robot is NOT moved.

  python3 capture_reference.py                 # countdown 3 s, capture, detect boundary, save
  python3 capture_reference.py --wait 5        # longer countdown
  python3 capture_reference.py --save-pose     # also store the current pose as deploy/locked_pose.json
  python3 capture_reference.py --no-robot      # camera only (do not connect to the arm)
  python3 capture_reference.py --image x.jpg   # test the detection on an existing image

Boundary, in this order of preference:
  1. 4 corner markers visible   -> rectangle through the marker centres
  2. a drawn box (pen/tape) found -> that box
  3. 1-3 markers visible        -> region spanned by the visible markers (partial, warning)
Saves to deploy/reference/: reference.jpg (raw), reference_annotated.jpg, reference.json
"""
import argparse, json, os, shutil, sys, time
import numpy as np
import cv2
import config as C
import board as BD

OUT = os.path.join(C.DEPLOY_DIR, "reference")


def find_drawn_box(img):
    """Largest 4-cornered outline (pen or tape box) that does not touch the image border."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = gray.shape
    e = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    e = cv2.dilate(e, np.ones((5, 5), np.uint8), iterations=2)          # close small gaps in the line
    found = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cs = found[0] if len(found) == 2 else found[1]
    best = None
    for c in cs:
        a = cv2.contourArea(c)
        if a < 0.05 * H * W or a > 0.9 * H * W:
            continue
        poly = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        x, y, w, h = cv2.boundingRect(poly)
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue
        if best is None or a > best[0]:
            best = (a, poly.reshape(4, 2).astype(float))
    return None if best is None else BD._order_tl_tr_br_bl(best[1])


_CAP = [None]


def capture(n_frames=5):
    """Median of n frames; the camera stays open between calls (repeated captures are fast)."""
    from hardware import open_camera
    if _CAP[0] is not None:
        frames = []
        for _ in range(n_frames + 3):
            ok, f = _CAP[0].read()
            if ok and f is not None:
                frames.append(f.astype(np.float32))
            if len(frames) == n_frames:
                break
        if not frames:
            return capture.last
        return np.median(np.stack(frames), axis=0).astype(np.uint8)
    cap, info = open_camera()
    if cap is None:
        print("no working camera found. Tried:\n  " + "\n  ".join(info))
        sys.exit("check: ls /dev/video*   |   v4l2-ctl --list-devices   |   sudo fuser -v /dev/video*  "
                 "(another program may be holding the camera; close it or reboot)")
    for _ in range(15):                                              # let exposure settle
        cap.read()
    frames = []
    for _ in range(n_frames):
        ok, f = cap.read()
        if ok and f is not None:
            frames.append(f.astype(np.float32))
        time.sleep(0.05)
    _CAP[0] = cap                                                    # keep open for re-captures
    if not frames:
        sys.exit("no frame captured")
    capture.last = np.median(np.stack(frames), axis=0).astype(np.uint8)   # median of 5 frames: less noise
    return capture.last


def read_robot():
    try:
        from pymycobot import MyCobot
        mc = MyCobot(C.PORT, C.BAUD)                                 # no power_on, no motion
        time.sleep(0.3)
        q = c = None
        for _ in range(8):
            q = mc.get_angles(); c = mc.get_coords()
            if q and c and len(q) == 6 and len(c) == 6:
                break
            time.sleep(0.1)
        mc.close()
        return (list(map(float, q)) if q and len(q) == 6 else None,
                list(map(float, c)) if c and len(c) == 6 else None)
    except Exception as ex:
        print("robot not read (%s); continuing with the image only" % ex)
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=float, default=3.0, help="countdown before capture (s)")
    ap.add_argument("--save-pose", action="store_true", help="store the current pose as deploy/locked_pose.json")
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--image", default=None, help="use an existing image instead of the camera")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    q = coords = None
    if not a.no_robot and a.image is None:
        q, coords = read_robot()
        if q:
            print("robot joint angles (deg): %s" % np.round(q, 2).tolist())
            print("robot coords (mm, deg):   %s" % np.round(coords, 1).tolist() if coords else "coords unavailable")
            if os.path.exists(C.LOCKED_POSE_PATH):
                old = json.load(open(C.LOCKED_POSE_PATH))["angles_deg"]
                dmax = float(np.max(np.abs(np.array(q) - np.array(old))))
                print("difference to the saved locked pose: max %.2f deg %s" % (
                    dmax, "(same pose)" if dmax < 1.0 else "(DIFFERENT: use --save-pose to make this the locked pose)"))

    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            sys.exit("cannot read %s" % a.image)
    else:
        for s in range(int(round(a.wait)), 0, -1):
            print("capturing in %d s ... keep hands and objects out of view" % s); time.sleep(1.0)
        img = capture()
        dets = BD.detect_markers(img)
        tries = 1
        while len(dets) < 4 and tries < 6:               # a marker was missed: capture again and merge
            more = BD.detect_markers(capture(3))
            for k, c in more.items():
                dets.setdefault(k, c)
            tries += 1
        if tries > 1:
            print("markers merged over %d captures: %s" % (tries, sorted(map(str, dets))))
    h, w = img.shape[:2]
    cv2.imwrite(os.path.join(OUT, "reference.jpg"), img)

    # ---------------- boundary
    if a.image:
        dets = BD.detect_markers(img)
    box = find_drawn_box(img)
    how = "ArUco ids" if any(not str(k).startswith("c") for k in dets) else "dark squares"
    if len(dets) >= 4:
        cen = BD.centres(dets)
        boundary = BD._order_tl_tr_br_bl(list(cen.values())[:4]); source = "4 markers (%s)" % how
    elif box is not None:
        boundary = box; source = "drawn box"
    elif dets:
        pts = np.vstack(list(dets.values()))
        boundary = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(float)
        source = "%d marker(s) only: PARTIAL boundary" % len(dets)
    else:
        boundary, source = None, "none"

    vis = img.copy()
    for k, c in dets.items():
        cv2.polylines(vis, [c.astype(np.int32)], True, (0, 200, 255), 2)
        u = c.mean(0); cv2.putText(vis, str(k), (int(u[0]) + 8, int(u[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if box is not None:
        cv2.polylines(vis, [box.astype(np.int32)], True, (255, 0, 255), 1)
    if boundary is not None:
        cv2.polylines(vis, [boundary.astype(np.int32)], True, (0, 220, 0), 2)
    cv2.drawMarker(vis, (w // 2, h // 2), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)    # image centre
    cv2.putText(vis, "boundary: " + source, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUT, "reference_annotated.jpg"), vis)

    inside = None
    if boundary is not None and len(boundary) >= 3:
        inside = cv2.pointPolygonTest(boundary.astype(np.float32), (w / 2.0, h / 2.0), False) >= 0
    ref = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), image="reference.jpg", image_wh=[w, h],
               angles_deg=q, coords_mm=coords, boundary_source=source,
               boundary_px=None if boundary is None else boundary.round(1).tolist(),
               markers={str(k): dict(centre_px=v.mean(0).round(1).tolist(), corners_px=v.round(1).tolist())
                        for k, v in dets.items()},
               drawn_box_px=None if box is None else box.round(1).tolist(),
               image_centre_inside_boundary=inside, aruco_module=hasattr(cv2, "aruco"))
    json.dump(ref, open(os.path.join(OUT, "reference.json"), "w"), indent=2)

    if a.save_pose and q and coords:
        if os.path.exists(C.LOCKED_POSE_PATH):
            shutil.copy(C.LOCKED_POSE_PATH, C.LOCKED_POSE_PATH + ".bak_%s" % time.strftime("%Y%m%d_%H%M%S"))
        json.dump(dict(angles_deg=q, coords_mm=coords, saved_at=ref["created"], source="capture_reference.py"),
                  open(C.LOCKED_POSE_PATH, "w"), indent=2)
        print("saved current pose as %s (old one backed up)" % C.LOCKED_POSE_PATH)

    print("\nmarkers found: %d %s | drawn box: %s" % (len(dets), sorted(map(str, dets)), "yes" if box is not None else "no"))
    print("boundary: %s" % source)
    if inside is not None:
        print("image centre inside the boundary: %s" % ("yes" if inside else "NO - re-centre the robot"))
    if len(dets) < 4:
        print("tip: for calibration all 4 markers must be visible with white borders (not on tape)")
    print("saved %s/reference.jpg, reference_annotated.jpg, reference.json" % OUT)


if __name__ == "__main__":
    main()
