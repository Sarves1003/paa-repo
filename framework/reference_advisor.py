#!/usr/bin/env python3
"""Where should the fixture box (and the taught reference pose) be so that the WHOLE part is reachable?

  python3 reference_advisor.py                 # evaluates shifts of the box + camera toward / beside the robot base
  python3 reference_advisor.py --half 60 --grid 5 --shifts 0 -20 -40 -60 -80 --lateral -30 0 30

Why: from the current reference the arm is nearly stretched (camera ~250 mm from the base, myCobot 280 reach ~280 mm),
so the reference is near-singular (low kappa) and every lattice point further out fails IK: only part of the part can
be covered. Moving the box and the camera TOGETHER keeps the survey image identical (same height, same orientation,
box centred), so only the arm's configuration changes.
For every candidate shift the tool solves the reference pose, its conditioning kappa and joint margin, and how many of
the grid views over the box (+-half mm, at dz 0 and the lowest descent) are reachable with kappa >= kappa_min.
Output: a table, the best shift, its joint angles (to move the arm there and re-teach), and results_views/reference_advisor.json.
After moving the box: re-teach the pose (live_align.py / auto_align.py), re-run grounding.py calibrate, table_grid.py
--install, measure_height.py --write, then refcover_updated.py plan.
"""
import argparse, json, os, time
import numpy as np
from kinematics import fk
from ik import ik_hold
import refcover_updated as RC


def evaluate(p_cam, R, q_seed, half, n, dzs, kmin, max_jump=60.0):
    q0, pe, re, ok = ik_hold(p_cam / 1000.0, q_seed, R, iters=1500)
    if not ok:
        return None
    k0, m0 = RC.kappa(q0), RC.limit_margin_deg(q0)
    xs = np.linspace(-half, half, n)
    tot = good = 0; kaps = []
    for dz in dzs:
        for gy in xs:
            qs = q0                                     # continuation along each row from the reference
            for gx in sorted(xs, key=abs):
                tot += 1
                q, pe, re, ok = ik_hold((p_cam + np.r_[gx, gy, dz]) / 1000.0, qs, R, iters=400)
                if not ok or np.degrees(np.max(np.abs(q - q0))) > max_jump:
                    continue
                k = RC.kappa(q); kaps.append(k)
                if k >= kmin and RC.limit_margin_deg(q) >= 5.0:
                    good += 1
                qs = q
    return dict(q_deg=np.degrees(q0).round(2).tolist(), kappa_ref=k0, margin_ref=m0, reachable_pct=100.0 * good / tot,
                kappa_min_grid=float(min(kaps)) if kaps else 0.0, n_grid=tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=float, default=55.0, help="half-size (mm) of the region the views must reach")
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--shifts", type=float, nargs="+", default=[0, -20, -40, -60, -80],
                    help="shift of box + camera along the base->camera direction (mm, negative = toward the base)")
    ap.add_argument("--lateral", type=float, nargs="+", default=[-30, 0, 30], help="sideways shifts (mm)")
    ap.add_argument("--max-descent", type=float, default=10.0)
    ap.add_argument("--kappa-min", type=float, default=0.04)
    a = ap.parse_args()
    q_ref, R_ref, p_ref = RC.load_ref()
    radial = p_ref[:2] / np.linalg.norm(p_ref[:2]); lat = np.array([-radial[1], radial[0]])
    print("current reference: camera %.0f mm from the base, height %.0f mm, kappa %.3f"
          % (np.linalg.norm(p_ref[:2]), p_ref[2], RC.kappa(q_ref)))
    rows, t0 = [], time.time()
    for s in a.shifts:
        for l in a.lateral:
            d = s * radial + l * lat
            p = p_ref + np.r_[d, 0.0]
            r = evaluate(p, R_ref, q_ref, a.half, a.grid, [0.0, -a.max_descent], a.kappa_min)
            row = dict(shift_radial_mm=s, shift_lateral_mm=l, move_box_mm=[round(float(x), 1) for x in d],
                       base_distance_mm=float(np.linalg.norm(p[:2])))
            row.update(r or dict(reachable_pct=0.0, kappa_ref=0.0, margin_ref=0.0, q_deg=None, kappa_min_grid=0.0))
            rows.append(row)
            print("  radial %+4.0f lateral %+4.0f mm (base distance %3.0f mm): kappa_ref %.3f, views reachable %5.1f%%  [%.0f s]"
                  % (s, l, row["base_distance_mm"], row["kappa_ref"], row["reachable_pct"], time.time() - t0), flush=True)
    best = max(rows, key=lambda r: (round(r["reachable_pct"]), r["kappa_ref"]))
    cur = next(r for r in rows if r["shift_radial_mm"] == 0 and r["shift_lateral_mm"] == 0) if 0 in a.shifts and 0 in a.lateral else None
    print("\ncurrent: %s" % ("%.0f%% of the views reachable, kappa_ref %.3f" % (cur["reachable_pct"], cur["kappa_ref"]) if cur else "n/a"))
    print("BEST: move the box (and the camera) by %s mm in the robot frame (%+.0f mm toward/away from the base, %+.0f mm "
          "sideways): %.0f%% of the views reachable, kappa_ref %.3f" % (best["move_box_mm"], best["shift_radial_mm"],
          best["shift_lateral_mm"], best["reachable_pct"], best["kappa_ref"]))
    print("joint angles of the new reference (deg): %s" % best["q_deg"])
    os.makedirs("results_views", exist_ok=True)
    json.dump(dict(rows=rows, best=best, half_mm=a.half, kappa_min=a.kappa_min), open("results_views/reference_advisor.json", "w"),
              indent=2)
    print("saved results_views/reference_advisor.json")


if __name__ == "__main__":
    main()
