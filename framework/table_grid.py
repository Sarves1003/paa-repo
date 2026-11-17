#!/usr/bin/env python3
"""Table-anchored inspection grid.

Problem it fixes: grounding.json's matrix A (image shift per mm of robot motion) is sheared on this arm
(one robot axis ~2.2 px/mm, the other ~4 px/mm, 49 deg apart instead of 90), consistent with a small
uncommanded camera tilt during reach moves. A grid that is square in ROBOT coordinates therefore lands
as a slanted strip on the table and runs out of the drawn box (cells refused by the fence).

Fix: the grid is defined as a SQUARE ON THE TABLE inside the drawn 100 x 100 mm box (box homography),
each cell's survey-image point u_k is converted to the robot offset d_k that brings it under the image
centre using the measured A, and closed-loop servoing removes the remaining error at every cell.

  python3 table_grid.py                 # compute, check, overlay (no robot motion)
  python3 table_grid.py --install       # also make every script use it (patches perception.py, backup kept)
  python3 table_grid.py --uninstall     # back to the robot-space grid
  options: --half 30 (table mm from the grid centre to the outer cells), --center box|part,
           --box-w 100 --box-h 100, --image hw_logs/grounding/calib_survey.jpg
Outputs: deploy/table_grid.json, hw_logs/grounding/table_grid_overlay.jpg
Re-run after every `grounding.py calibrate` (A changes).
"""
import argparse, json, os, shutil, sys, time
import numpy as np
import cv2
import config as C

TG_PATH = os.path.join(C.DEPLOY_DIR, "table_grid.json")
MARK = "# >>> table_grid override (added by table_grid.py) >>>"
PATCH = MARK + '''
_cell_offsets_robot_grid = cell_offsets_mm
def cell_offsets_mm(*args, **kw):
    """Table-anchored grid from deploy/table_grid.json if present, else the original robot-space grid.
    A centre offset (part tracking) is added in robot mm, as before."""
    import json as _j, os as _o, numpy as _n, config as _C
    _p = _o.path.join(_C.DEPLOY_DIR, "table_grid.json")
    if not _o.path.exists(_p):
        return _cell_offsets_robot_grid(*args, **kw)
    base = _n.array(_j.load(open(_p))["offsets_mm"], float)
    c = args[0] if args else (next(iter(kw.values())) if kw else None)
    if c is not None:
        base = base + _n.asarray(c, float)[:2][None]
    return base
# <<< table_grid override <<<
'''


def order_tl_tr_br_bl(p):
    p = np.asarray(p, float)
    s, d = p.sum(1), p[:, 0] - p[:, 1]
    return np.array([p[np.argmin(s)], p[np.argmax(d)], p[np.argmax(s)], p[np.argmin(d)]])


def get_box(G, img):
    if G.get("box_px"):
        return order_tl_tr_br_bl(G["box_px"]), "grounding.json"
    try:
        import board as BD
        from measure_board import find_box
        dets = BD.detect_markers(img)
        b = find_box(img, dets if len(dets) >= 4 else None)
        if b is not None:
            return order_tl_tr_br_bl(b), "detected in the survey image"
    except Exception as e:
        print("box detection failed:", e)
    return None, None


def to_offsets(G, A, uc, U):
    """Robot offsets d with offset_to_px(G, d) == u. Sign convention taken from perception.offset_to_px."""
    from perception import offset_to_px
    for s in (+1.0, -1.0):                           # content shift = A d  ->  centre sees uc - A d
        D = np.linalg.solve(A, s * (uc[None] - U).T).T
        back = np.array([offset_to_px(G, d) for d in D])
        if np.abs(back - U).max() < 0.5:
            return D
    raise SystemExit("could not invert perception.offset_to_px with A (unexpected grounding format)")


def compute(a):
    from perception import load_grounding, locate_part
    from fence import load_fence
    if not os.path.exists(C.GROUNDING_PATH):
        sys.exit("no deploy/grounding.json: run `python3 grounding.py calibrate --mode track` first")
    G = load_grounding()
    A, uc = np.asarray(G["A"], float), np.asarray(G["uc"], float)
    img = cv2.imread(a.image)
    if img is None:
        sys.exit("cannot read %s" % a.image)
    box, src = get_box(G, img)
    if box is None:
        sys.exit("drawn box not found: run `python3 measure_board.py --click` or check the survey image")
    H, _ = cv2.findHomography(box.astype(np.float32),
                              np.array([[0, 0], [a.box_w, 0], [a.box_w, a.box_h], [0, a.box_h]], np.float32))
    Hi = np.linalg.inv(H)
    tab2px = lambda P: cv2.perspectiveTransform(np.asarray(P, np.float64).reshape(-1, 1, 2), Hi).reshape(-1, 2)
    px2tab = lambda P: cv2.perspectiveTransform(np.asarray(P, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)
    box_ppm = float(np.mean([np.linalg.norm(box[1] - box[0]) / a.box_w, np.linalg.norm(box[3] - box[0]) / a.box_h]))

    # grid centre on the table
    c_tab = np.array([a.box_w / 2.0, a.box_h / 2.0])
    if a.center == "part":
        part = locate_part(img)
        if part is not None:
            c_tab = px2tab([part["centroid_px"]])[0]
        else:
            print("part not found -> grid centred on the box")
    fence = load_fence()
    n = C.GRID_N
    half = float(a.half)
    while half >= 10.0:
        lo = np.array([half + a.inner_margin, half + a.inner_margin])
        hi = np.array([a.box_w, a.box_h]) - lo
        cc = np.clip(c_tab, lo, hi)
        g = np.linspace(-half, half, n)
        T = np.array([[cc[0] + g[j], cc[1] + g[i]] for i in range(n) for j in range(n)])   # k = i*n + j
        U = tab2px(T)
        D = to_offsets(G, A, uc, U)
        ok = np.array([fence is None or fence.contains(d) for d in D])
        if ok.all():
            break
        print(f"  half {half:.1f} mm: {int((~ok).sum())} cell(s) outside the robot limit, shrinking")
        half -= 2.5
    else:
        sys.exit("no grid fits inside the limit: re-run auto_align.py so the box is centred in the view")

    # how sheared is the arm's motion, compared with the box
    ppm_axes = np.linalg.norm(A, axis=0)
    ang = np.degrees(np.arccos(abs(A[:, 0] @ A[:, 1]) / np.prod(ppm_axes)))
    pitch = 2 * half / (n - 1)
    out = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), grounding_created=G.get("created"),
               box_px=box.round(1).tolist(), box_source=src, box_mm=[a.box_w, a.box_h], center=a.center,
               grid_center_table_mm=cc.round(2).tolist(), half_mm=half, pitch_mm=pitch, grid_n=n,
               table_mm=T.round(2).tolist(), image_px=U.round(1).tolist(), offsets_mm=D.round(3).tolist(),
               box_px_per_mm=round(box_ppm, 3), robot_axis_px_per_mm=ppm_axes.round(3).tolist(),
               robot_axis_angle_deg=round(float(ang), 1),
               note="square grid on the table inside the drawn box; offsets = robot moves from the survey pose")
    os.makedirs(C.DEPLOY_DIR, exist_ok=True)
    json.dump(out, open(TG_PATH, "w"), indent=2)

    vis = img.copy()
    cv2.polylines(vis, [box.astype(np.int32)], True, (255, 0, 255), 2)
    for k, u in enumerate(U):
        r = int(round(box_ppm * pitch / 2))
        p = tuple(int(v) for v in u)
        cv2.rectangle(vis, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), (0, 170, 0), 1)
        cv2.circle(vis, p, 3, (0, 170, 0), -1)
        cv2.putText(vis, str(k), (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 120, 0), 1)
    cv2.drawMarker(vis, tuple(int(v) for v in uc), (255, 120, 0), cv2.MARKER_CROSS, 18, 2)
    od = os.path.join(C.LOGS_DIR, "grounding"); os.makedirs(od, exist_ok=True)
    cv2.imwrite(os.path.join(od, "table_grid_overlay.jpg"), vis)

    print(f"drawn box ({src}): {box_ppm:.2f} px/mm on the table")
    print(f"robot axes in the image: {ppm_axes[0]:.2f} and {ppm_axes[1]:.2f} px/mm, {ang:.0f} deg apart "
          f"(a pure translation would give ~{box_ppm:.2f} px/mm on both, 90 deg apart)")
    print(f"table grid {n}x{n}: half {half:.1f} mm, pitch {pitch:.1f} mm, centre {cc.round(1).tolist()} mm in the box")
    print(f"robot offsets span x {D[:, 0].min():+.1f}..{D[:, 0].max():+.1f} mm, y {D[:, 1].min():+.1f}..{D[:, 1].max():+.1f} mm; "
          f"all {len(D)} cells inside the robot limit")
    print(f"saved {TG_PATH} and {od}/table_grid_overlay.jpg  (green squares must sit on the part, inside the box)")
    if abs(half - C.GRID_HALF_MM) > 0.1:
        print(f"NOTE: set GRID_HALF_MM = {half:.1f} in config.py so cell patch sizes match the table grid")
    return out


def install():
    src = open("perception.py").read()
    if MARK in src:
        print("perception.py already uses the table grid"); return
    if "def cell_offsets_mm" not in src:
        sys.exit("perception.py has no cell_offsets_mm(); send me perception.py")
    shutil.copy("perception.py", "perception.py.bak_%s" % time.strftime("%Y%m%d_%H%M%S"))
    open("perception.py", "a").write("\n\n" + PATCH)
    print("perception.py patched (backup kept): every script now uses the table grid")


def uninstall():
    src = open("perception.py").read()
    if MARK not in src:
        print("nothing to remove"); return
    open("perception.py", "w").write(src[:src.index(MARK)].rstrip() + "\n")
    print("removed the table-grid override from perception.py")


def check(tg):
    """Verify, with YOUR modules, that the cost model and the survey use the table grid."""
    import importlib, perception
    importlib.reload(perception)
    D = np.array(tg["offsets_mm"], float)
    ok = np.abs(np.asarray(perception.cell_offsets_mm(), float) - D).max() < 1e-6
    print("perception.cell_offsets_mm() -> table grid:", "OK" if ok else "NO")
    try:
        import cost_model as CM
        M = CM.build("hold", dry=True, save=False, center_mm=(0.0, 0.0))
        okc = np.abs(np.asarray(M["xy_mm"], float) - D).max() < 1e-3
        print("cost_model grid                   -> table grid:", "OK" if okc else
              "NO: cost_model.py builds its own grid; send me cost_model.py")
        from kinematics import fk
        from ik import ik_hold
        from hardware import load_locked_pose
        q0 = np.deg2rad(np.array(load_locked_pose(True)["angles_deg"], float)); T0 = fk(q0)
        for dz in [0.0] + [float(v) for v in getattr(C, "VIEW_LEVELS_DZ", ())]:
            n = sum(ik_hold(T0[:3, 3] + np.r_[d, dz] / 1000.0, q0, T0[:3, :3], iters=1500)[3] for d in D)
            print("grid cells reachable with the camera orientation held at %+.0f mm %s: %d/%d"
                  % (dz, "(survey height)" if dz == 0 else "(inspection views)", n, len(D)))
    except Exception as e:
        print("cost model check skipped:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=float, default=C.GRID_HALF_MM)
    ap.add_argument("--inner-margin", type=float, default=8.0, help="mm between outer cell centres and the box")
    ap.add_argument("--center", choices=["box", "part"], default="box")
    ap.add_argument("--box-w", type=float, default=100.0)
    ap.add_argument("--box-h", type=float, default=100.0)
    ap.add_argument("--image", default=os.path.join(C.LOGS_DIR, "grounding", "calib_survey.jpg"))
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    a = ap.parse_args()
    if a.uninstall:
        uninstall(); return
    tg = compute(a)
    if a.install:
        install()
    if MARK in open("perception.py").read():
        check(tg)
    else:
        print("not active yet: re-run with --install")


if __name__ == "__main__":
    main()
