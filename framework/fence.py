#!/usr/bin/env python3
"""Workspace limits (fence) WITHOUT markers. Every robot move is checked against it
(hardware.safe_move), and grid cells outside it are removed for every planner.

The fence is a convex polygon in mm, relative to the locked survey pose (x, y of the
camera/flange position). Three ways to define it:

  python3 fence.py teach           # release servos, move the camera by hand over each corner,
                                   # press Enter at each corner, 'd' + Enter when done (3-8 corners;
                                   # 2 corners = opposite corners of a rectangle)
  python3 fence.py click           # click the corners on the survey image (needs a display and
                                   # grounding.json); Enter = done, 'r' = restart
  python3 fence.py click --points "120,60;520,60;520,420;120,420"     # same, headless (pixels)
  python3 fence.py box --x -40 40 --y -40 40                           # typed limits in mm
Options: --margin-mm 5 (targets must stay this far inside), --dry-run
Output: deploy/fence.json and hw_logs/fence/fence_overlay.jpg (check it before running).
"""
import argparse, json, os, sys, time
import numpy as np
import config as C


# ------------------------------------------------------------------ geometry
def convex_hull(points):
    pts = sorted(set(map(tuple, np.round(np.asarray(points, float), 6))))
    if len(pts) <= 2:
        return np.array(pts)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])            # counter-clockwise


class Fence:
    """Convex polygon in workspace offsets (mm, relative to the survey pose)."""
    def __init__(self, polygon_mm, margin_mm, meta=None):
        self.poly = np.asarray(polygon_mm, float)
        self.margin = float(margin_mm)
        self.meta = meta or {}

    def signed_dist(self, d):
        P, d = self.poly, np.asarray(d, float)[:2]
        out = np.inf
        for k in range(len(P)):
            a, b = P[k], P[(k + 1) % len(P)]
            e = b - a
            n = np.array([-e[1], e[0]]) / np.linalg.norm(e)      # inward normal (CCW polygon)
            out = min(out, float((d - a) @ n))
        return out

    def contains(self, d):
        return self.signed_dist(d) >= self.margin

    def to_json(self):
        return dict(polygon_mm=self.poly.tolist(), margin_mm=self.margin, **self.meta)


def fence_from_points(points_mm, margin_mm, meta=None):
    pts = np.asarray(points_mm, float)
    if len(pts) < 2:
        raise ValueError("need at least 2 corner points")
    if len(pts) == 2:
        (x0, y0), (x1, y1) = pts
        poly = np.array([[min(x0, x1), min(y0, y1)], [max(x0, x1), min(y0, y1)],
                         [max(x0, x1), max(y0, y1)], [min(x0, x1), max(y0, y1)]])
    else:
        poly = convex_hull(pts)
        if len(poly) < 3:
            raise ValueError("corner points are collinear")
    return Fence(poly, margin_mm, dict(meta or {}, corners_mm=pts.tolist()))


def auto_fence(half_mm=C.FENCE_AUTO_HALF_MM, margin_mm=C.FENCE_MARGIN_MM):
    """Box of +/- half_mm around the locked pose: the default limit, no teaching required."""
    h = float(half_mm)
    return fence_from_points([[-h, -h], [h, h]], margin_mm, dict(mode="auto", half_mm=h))


def load_fence(path=C.FENCE_PATH, auto=True):
    """Priority: deploy/fence.json (custom) > marker rectangle from board grounding > automatic box."""
    if not os.path.exists(path):
        if not auto:
            return None
        if os.path.exists(C.GROUNDING_PATH):
            g = json.load(open(C.GROUNDING_PATH))
            if g.get("box_offsets_mm") and getattr(C, "FENCE_FROM_BOX", True):
                return fence_from_points(g["box_offsets_mm"], C.FENCE_MARGIN_MM, dict(mode="drawn box"))
            if g.get("mode") == "board" and "fence_offsets_mm" in g:
                return fence_from_points(g["fence_offsets_mm"], C.FENCE_MARGIN_MM, dict(mode="markers"))
        return auto_fence()
    j = json.load(open(path))
    return Fence(j["polygon_mm"], j["margin_mm"],
                 {k: v for k, v in j.items() if k not in ("polygon_mm", "margin_mm")})


# ------------------------------------------------------------------ definition modes
def teach(a, arm, p_lock):
    """Hand-guided corners: servos released, operator moves the camera, Enter records."""
    from kinematics import fk
    pts = []
    print("\nHOLD THE ARM: the servos will be released and the arm can drop.")
    input("Press Enter when you are holding the arm ...")
    if not a.dry_run:
        rel = getattr(arm.mc, "release_all_servos", None)
        if rel is None:
            sys.exit("this pymycobot version has no release_all_servos(); use `fence.py click` or `box`")
        rel(); time.sleep(0.5)
    scripted = iter([(-40, -40), (40, -40), (40, 40), (-40, 40)])   # dry-run corners (mm)
    try:
        while True:
            cmd = input(f"corner {len(pts) + 1}: move the camera over the corner, Enter = record, "
                        f"'d' = done: ").strip().lower() if not a.dry_run else ("d" if len(pts) == 4 else "")
            if cmd == "d":
                break
            if a.dry_run:
                d = np.array(next(scripted), float)
            else:
                q = arm.read_deg()
                d = fk(np.deg2rad(q))[:3, 3] * 1000 - p_lock
            pts.append(d[:2])
            print(f"   recorded ({d[0]:+.1f}, {d[1]:+.1f}) mm from the survey pose")
    finally:
        if not a.dry_run:
            arm.mc.power_on(); time.sleep(1.0)        # re-lock servos before any motion
    return np.array(pts)


def click(a, frame, G):
    import cv2
    from perception import px_to_offset
    to_mm = lambda u: px_to_offset(G, u)
    if a.points:
        px = [tuple(map(float, p.split(","))) for p in a.points.split(";") if p.strip()]
        return np.array([to_mm(u) for u in px]), px
    px = []
    win = "click fence corners: Enter = done, r = restart, q = quit"
    def on(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            px.append((float(x), float(y)))
    cv2.namedWindow(win); cv2.setMouseCallback(win, on)
    while True:
        img = frame.copy()
        for k, p in enumerate(px):
            cv2.circle(img, tuple(int(v) for v in p), 5, (0, 0, 255), -1)
            if k:
                cv2.line(img, tuple(int(v) for v in px[k - 1]), tuple(int(v) for v in p), (255, 120, 0), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(px) >= 2:
            break
        if k == ord("r"):
            px.clear()
        if k == ord("q"):
            sys.exit("cancelled")
    cv2.destroyAllWindows()
    return np.array([to_mm(u) for u in px]), px


def overlay(frame, G, fence, path):
    import cv2
    from perception import cell_offsets_mm
    img = frame.copy()
    from perception import offset_to_px
    to_px = lambda d: tuple(int(v) for v in offset_to_px(G, np.asarray(d, float)[:2]))
    cv2.polylines(img, [np.array([to_px(p) for p in fence.poly], np.int32)], True, (255, 120, 0), 2)
    for k, d in enumerate(cell_offsets_mm()):
        col = (0, 170, 0) if fence.contains(d) else (0, 0, 255)
        cv2.circle(img, to_px(d), 5, col, -1)
        cv2.putText(img, str(k), (to_px(d)[0] + 6, to_px(d)[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    cv2.circle(img, to_px((0, 0)), 8, (255, 0, 255), 2)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["teach", "click", "box", "show"])
    ap.add_argument("--x", type=float, nargs=2, default=None, help="box: x min max (mm)")
    ap.add_argument("--y", type=float, nargs=2, default=None, help="box: y min max (mm)")
    ap.add_argument("--points", default=None, help='click (headless): "u1,v1;u2,v2;..." in pixels')
    ap.add_argument("--margin-mm", type=float, default=C.FENCE_MARGIN_MM)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))
    a = ap.parse_args()

    from kinematics import fk
    from hardware import make_arm, make_camera, wait_settled, load_locked_pose
    from perception import cell_offsets_mm, load_grounding
    lock = load_locked_pose(a.dry_run)
    q_lock = np.array(lock["angles_deg"], float)
    p_lock = fk(np.deg2rad(q_lock))[:3, 3] * 1000
    G = load_grounding() if os.path.exists(C.GROUNDING_PATH) else None
    out = os.path.join(C.LOGS_DIR, "fence"); os.makedirs(out, exist_ok=True)

    if a.mode == "show":
        fence = load_fence()
        if fence is None:
            sys.exit("no deploy/fence.json yet")
    else:
        arm = make_arm(a.dry_run, q_lock); arm.send_deg(q_lock); wait_settled(arm)
        frame = None
        if G is not None:
            cam = make_camera(a.dry_run, arm, a.ref_img); frame = cam.grab(); cam.release()
        meta = dict(mode=a.mode, created=time.strftime("%Y-%m-%d %H:%M:%S"), grid_half_mm=C.GRID_HALF_MM)
        if a.mode == "box":
            if a.x is None or a.y is None:
                sys.exit("box mode needs --x MIN MAX --y MIN MAX (mm, relative to the survey pose)")
            pts = np.array([[a.x[0], a.y[0]], [a.x[1], a.y[1]]])
        elif a.mode == "click":
            if G is None or frame is None:
                sys.exit("click mode needs grounding.json: run `python3 grounding.py calibrate` first")
            pts, px = click(a, frame, G); meta["clicked_px"] = [list(p) for p in px]
        else:
            pts = teach(a, arm, p_lock)
            arm.send_deg(q_lock); wait_settled(arm)
        arm.close()
        fence = fence_from_points(pts, a.margin_mm, meta)
        os.makedirs(C.DEPLOY_DIR, exist_ok=True)
        inside = [k for k, d in enumerate(cell_offsets_mm()) if fence.contains(d)]
        fence.meta["cells_inside"] = inside
        json.dump(fence.to_json(), open(C.FENCE_PATH, "w"), indent=2)
        if frame is not None:
            overlay(frame, G, fence, os.path.join(out, "fence_overlay.jpg"))
    inside = [k for k, d in enumerate(cell_offsets_mm()) if fence.contains(d)]
    print(f"fence: {len(fence.poly)}-gon {np.round(fence.poly, 1).tolist()} mm, margin {fence.margin:.1f} mm")
    print(f"allowed grid cells {len(inside)}/{C.GRID_N ** 2}: {inside}")
    if not fence.contains((0, 0)):
        print("WARNING: the locked survey pose itself is outside the fence")
    if len(inside) < C.GRID_N ** 2:
        print("blocked cells are skipped by every method; enlarge the fence or reduce GRID_HALF_MM in config.py")
    if a.mode != "show":
        print(f"saved {C.FENCE_PATH}" + (f" and {out}/fence_overlay.jpg" if G is not None else ""))
        print("now rebuild the cost models: python3 cost_model.py build --ik hold  (and --ik position)")
    else:
        src = "file " + C.FENCE_PATH if os.path.exists(C.FENCE_PATH) else (
            "drawn box (box.json + grounding.json)" if fence.meta.get("mode") == "drawn box" else
            "marker rectangle (grounding.json)" if fence.meta.get("mode") == "markers" else
            "automatic +/-%.0f mm box (no drawn box found)" % C.FENCE_AUTO_HALF_MM)
        print("source of this limit: " + src)


if __name__ == "__main__":
    main()
