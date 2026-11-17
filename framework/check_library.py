#!/usr/bin/env python3
"""No motion: check that every saved view in deploy/views/<pose>_NNN matches the robot pose it was planned for.

Per view it compares, from meta.json:
  - measured joints (q_meas) vs planned joints (q_plan)            -> max joint error (deg)
  - camera position from forward kinematics, measured vs planned   -> position error (mm)
  - height below the locked pose, measured vs the view's level     -> dz error (mm)
  - camera tilt vs the locked pose, measured vs the level's rpy    -> tilt error (deg)
  - image registration result saved at capture time                -> view error (mm), verified by
  - the image file exists and is not blank
and draws one contact sheet per placement (thumbnail + cell, level, errors; red frame = flagged).

  python3 check_library.py                    # all placements
  python3 check_library.py --placement horizontal_006
Outputs: results_task/library_check.csv, results_task/library_sheets/<placement>.jpg
"""
import argparse, csv, glob, json, os
import numpy as np
import cv2
import config as C
from kinematics import fk

VIEW_DIR = os.path.join(C.DEPLOY_DIR, "views")
OUT = "results_task"
LIM = dict(joint_deg=3.0, pos_mm=10.0, dz_mm=4.0, tilt_deg=3.0, view_mm=3.0)


def as_rad(q):
    """meta.json stores joints in degrees (arm.read_deg, lattice q_deg)."""
    return np.radians(np.asarray(q, float))


def pose(q):
    T = fk(as_rad(q))
    return T[:3, 3] * 1000.0, T[:3, :3]


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def expected_tilt(rpy):
    """Rotation angle of a level's roll/pitch/yaw (deg), same convention as refcover_updated.orient_R."""
    try:
        import refcover_updated as RC
        return rot_deg(RC.orient_R(rpy))
    except Exception:
        return float(np.linalg.norm(rpy))


def check_placement(d, sheet_dir):
    m = json.load(open(os.path.join(d, "meta.json")))
    name = os.path.basename(d)
    p0, R0 = pose(m["q_lock"])
    Qp = np.asarray(m["q_plan"], float)
    lv_dz = m.get("levels_dz", [0.0])
    lv_rpy = m.get("levels_rpy") or [[0, 0, 0]] * len(lv_dz)
    rows, thumbs = [], []
    for r in sorted(m["views"], key=lambda v: v["k"]):
        k, l = int(r["k"]), int(r["level"])
        row = dict(placement=name, condition=m.get("condition"), view=k, cell=int(r["cell"]), level=l,
                   level_dz=float(lv_dz[l]), level_rpy=str([round(float(x)) for x in lv_rpy[l]]), ok=int(r.get("ok", 0)),
                   refusal=r.get("refusal", ""))
        img_p = os.path.join(d, "v%02d.jpg" % k)
        img = cv2.imread(img_p) if row["ok"] else None
        flags = []
        if row["ok"]:
            if img is None:
                flags.append("image missing")
            elif float(img.std()) < 4.0:
                flags.append("image blank")
            if "q_meas" in r:
                pm, Rm = pose(r["q_meas"])
                pp, Rp = pose(Qp[k]) if np.any(Qp[k]) else (pm, Rm)
                jerr = float(np.max(np.abs(np.degrees(as_rad(r["q_meas"])) - np.degrees(as_rad(Qp[k]))))) if np.any(Qp[k]) else np.nan
                dz_meas = float(pm[2] - p0[2])
                tilt_meas = rot_deg(Rm @ R0.T)
                row.update(joint_err_deg=jerr, pos_err_mm=float(np.linalg.norm(pm - pp)),
                           dz_meas_mm=dz_meas, dz_err_mm=dz_meas - float(lv_dz[l]),
                           tilt_meas_deg=tilt_meas, tilt_err_deg=tilt_meas - expected_tilt(lv_rpy[l]),
                           xy_from_lock_mm=float(np.linalg.norm(pm[:2] - p0[:2])))
                if np.isfinite(jerr) and jerr > LIM["joint_deg"]:
                    flags.append("joints %.1f deg off plan" % jerr)
                if row["pos_err_mm"] > LIM["pos_mm"]:
                    flags.append("camera %.0f mm off plan" % row["pos_err_mm"])
                if abs(row["dz_err_mm"]) > LIM["dz_mm"]:
                    flags.append("height %+.0f mm vs level %+.0f" % (dz_meas, lv_dz[l]))
                if abs(row["tilt_err_deg"]) > LIM["tilt_deg"]:
                    flags.append("tilt %.1f deg vs level %.1f" % (tilt_meas, expected_tilt(lv_rpy[l])))
                if dz_meas < -10.5:
                    flags.append("BELOW the 10 mm limit (%.1f mm)" % dz_meas)
            else:
                flags.append("no q_meas saved")
            ve = r.get("view_err_mm", np.nan)
            row.update(view_err_mm=ve, verify=r.get("verify", "markers" if np.isfinite(ve if ve is not None else np.nan) else "open-loop"))
            if ve is not None and np.isfinite(ve) and ve > LIM["view_mm"]:
                flags.append("view error %.1f mm" % ve)
        row["flags"] = "; ".join(flags)
        rows.append(row)
        if img is not None:
            t = cv2.resize(img, (160, 120))
            col = (0, 0, 255) if flags else (0, 170, 0)
            cv2.rectangle(t, (0, 0), (159, 119), col, 3)
            cv2.putText(t, "v%d c%d L%d" % (k, row["cell"], l), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if "dz_meas_mm" in row:
                cv2.putText(t, "dz%+.0f tilt%.0f" % (row["dz_meas_mm"], row["tilt_meas_deg"]), (4, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            ve = row.get("view_err_mm")
            if ve is not None and np.isfinite(ve):
                cv2.putText(t, "err %.1fmm" % ve, (4, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            thumbs.append(t)
    if thumbs:
        nc = 10
        while len(thumbs) % nc:
            thumbs.append(np.full((120, 160, 3), 40, np.uint8))
        grid = np.vstack([np.hstack(thumbs[i:i + nc]) for i in range(0, len(thumbs), nc)])
        cv2.imwrite(os.path.join(sheet_dir, name + ".jpg"), grid)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="horizontal")
    ap.add_argument("--placement", default=None)
    a = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(VIEW_DIR, (a.placement or a.pose + "_*"))))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "meta.json"))]
    if not dirs:
        raise SystemExit("no placements with meta.json in %s" % VIEW_DIR)
    sheet_dir = os.path.join(OUT, "library_sheets"); os.makedirs(sheet_dir, exist_ok=True)
    allrows = []
    print("%-16s %-13s %5s %6s %7s %8s %7s %8s %8s %6s" % ("placement", "condition", "views", "flag", "joint", "pos_mm",
                                                          "dz_err", "tilt_err", "view_mm", "min_dz"))
    for d in dirs:
        rows = check_placement(d, sheet_dir); allrows += rows
        ok = [r for r in rows if r["ok"]]
        f = lambda key: np.array([r.get(key, np.nan) for r in ok], float)
        med = lambda x: np.nanmedian(np.abs(x)) if np.isfinite(x).any() else np.nan
        print("%-16s %-13s %2d/%-2d %6d %7.2f %8.2f %7.2f %8.2f %8.2f %6.1f" % (
            os.path.basename(d), rows[0]["condition"], len(ok), len(rows), sum(bool(r["flags"]) for r in ok),
            med(f("joint_err_deg")), med(f("pos_err_mm")), med(f("dz_err_mm")), med(f("tilt_err_deg")),
            med(f("view_err_mm")), np.nanmin(f("dz_meas_mm")) if np.isfinite(f("dz_meas_mm")).any() else np.nan))
    keys = sorted({k for r in allrows for k in r}, key=lambda k: list(allrows[0].keys()).index(k) if k in allrows[0] else 99)
    with open(os.path.join(OUT, "library_check.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(allrows)
    bad = [r for r in allrows if r["ok"] and r["flags"]]
    print("\n%d views checked, %d flagged (limits: joints %.0f deg, camera %.0f mm, height %.0f mm, tilt %.0f deg, view %.0f mm)"
          % (sum(r["ok"] for r in allrows), len(bad), LIM["joint_deg"], LIM["pos_mm"], LIM["dz_mm"], LIM["tilt_deg"], LIM["view_mm"]))
    for r in bad[:25]:
        print("  %s v%02d (cell %d, level %d): %s" % (r["placement"], r["view"], r["cell"], r["level"], r["flags"]))
    if len(bad) > 25:
        print("  ... %d more in %s/library_check.csv" % (len(bad) - 25, OUT))
    print("contact sheets: %s/<placement>.jpg (green = consistent, red = flagged)" % sheet_dir)


if __name__ == "__main__":
    main()
