#!/usr/bin/env python3
"""Which markers does the camera see at the reference pose? (any number works; 2 diagonal is fine)

  python3 check_markers.py            # arm goes to the reference pose, captures, reports markers
  python3 check_markers.py --write    # also sets MARKER_IDS in config.py to the ids found
  python3 check_markers.py --image some.jpg   # check a saved image instead (no robot)
Output: hw_logs/markers_check.jpg (markers outlined with their ids, the spacing line between them).
"""
import argparse, os, re
import numpy as np
import cv2
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--image", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))
    a = ap.parse_args()
    import board as BD
    if a.image:
        img = cv2.imread(a.image)
    else:
        from hardware import make_arm, make_camera, wait_settled, load_locked_pose
        q = np.array(load_locked_pose(a.dry_run)["angles_deg"], float)
        arm = make_arm(a.dry_run, q); arm.send_deg(q, 15); wait_settled(arm)
        cam = make_camera(a.dry_run, arm, a.ref_img)
        try:
            img = cam.grab()
        finally:
            cam.release(); arm.close()
    # detect with every id allowed, so markers missing from MARKER_IDS are reported too
    saved = getattr(C, "MARKER_IDS", None)
    try:
        C.MARKER_IDS = list(range(50))
        dets = BD.detect_markers(img)
    finally:
        C.MARKER_IDS = saved
    vis = img.copy(); h, w = img.shape[:2]
    cen = {}
    for k, c in sorted(dets.items(), key=lambda kv: str(kv[0])):
        c = np.asarray(c, float); u = c.mean(0); cen[k] = u
        side = np.mean([np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)])
        edge = min(u[0], u[1], w - u[0], h - u[1]) - side / 2
        cv2.polylines(vis, [c.astype(np.int32)], True, (0, 200, 255), 2)
        cv2.putText(vis, str(k), (int(u[0]) + 10, int(u[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print("marker %s: centre (%.0f, %.0f) px, side %.0f px, %.0f px from the image edge%s"
              % (k, u[0], u[1], side, edge, "  <- close to the edge" if edge < 15 else ""))
    ks = list(cen)
    if len(ks) >= 2:
        a_, b_ = max(((x, y) for i, x in enumerate(ks) for y in ks[i + 1:]),
                     key=lambda p: np.linalg.norm(cen[p[0]] - cen[p[1]]))
        cv2.line(vis, tuple(cen[a_].astype(int)), tuple(cen[b_].astype(int)), (0, 255, 0), 1)
        print("longest marker spacing: %.0f px (markers %s-%s); scale and position are measured from it"
              % (np.linalg.norm(cen[a_] - cen[b_]), a_, b_))
    os.makedirs(C.LOGS_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(C.LOGS_DIR, "markers_check.jpg"), vis)
    print("%d marker(s) found: ids %s; config MARKER_IDS = %s; image hw_logs/markers_check.jpg"
          % (len(dets), sorted(map(str, dets)), saved))
    if not dets:
        raise SystemExit("no marker found: check lighting, focus and that the markers are flat with a white border")
    if a.write:
        ids = sorted(int(k) for k in dets if str(k).isdigit())
        s = open("config.py").read()
        s, n = re.subn(r"^MARKER_IDS\s*=.*$", "MARKER_IDS = %s   # found by check_markers.py" % ids, s, flags=re.M)
        if n == 0:
            s += "\nMARKER_IDS = %s\n" % ids
        open("config.py", "w").write(s)
        print("written MARKER_IDS = %s into config.py" % ids)


if __name__ == "__main__":
    main()
