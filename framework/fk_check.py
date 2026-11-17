#!/usr/bin/env python3
"""Which kinematic table matches the firmware? Run once before any experiment.

Moves through poses around the locked pose, reads get_angles() and get_coords(),
and reports |FK(q) - firmware position| for both tables in kinematics.py, plus the
firmware repeatability when returning to the locked pose. Set config.KIN_MODEL to
the table with the smaller error, then rebuild the cost models.

  python3 fk_check.py
"""
import csv, os, time
import numpy as np
import config as C
from kinematics import fk
from hardware import Arm, wait_settled, load_locked_pose
import hardware as HW
from fence import load_fence

lock = load_locked_pose()
q0 = np.array(lock["angles_deg"], float)
rng = np.random.default_rng(1)
HW.set_fence(load_fence(), fk(np.deg2rad(q0))[:3, 3] * 1000)
cand = [q0 + rng.uniform(-8, 8, 6) * np.array([1, 1, 1, 1, 0.5, 0]) for _ in range(40)]
cand = [q for q in cand if HW.in_fence_q(q)][:12]          # only poses inside the workspace fence
poses = [q0] + cand + [q0] * 4
arm = Arm(); rows = []
try:
    for k, q in enumerate(poses):
        arm.send_deg(q, C.MOVE_SPEED); _, qm = wait_settled(arm)
        fw = arm.read_coords()[:3]
        pa = fk(np.deg2rad(qm), "repo", z_offset_mm=0)[:3, 3] * 1000
        pb = fk(np.deg2rad(qm), "alt", z_offset_mm=C.KIN_Z_OFFSET_MM)[:3, 3] * 1000
        zoff = float(fw[2] - fk(np.deg2rad(qm), "alt", z_offset_mm=0)[2, 3] * 1000)
        rows.append(dict(k=k, fw_x=fw[0], fw_y=fw[1], fw_z=fw[2], repo_err=float(np.linalg.norm(pa - fw)),
                         alt_err=float(np.linalg.norm(pb - fw)), alt_z_offset=zoff, cmd_vs_read_deg=float(np.abs(qm - q).max())))
        print(f"pose {k:2d}: repo {rows[-1]['repo_err']:6.1f} mm | alt {rows[-1]['alt_err']:6.1f} mm | "
              f"max|q_cmd - q_read| {rows[-1]['cmd_vs_read_deg']:.2f} deg")
finally:
    arm.close()
os.makedirs(C.LOGS_DIR, exist_ok=True)
out = os.path.join(C.LOGS_DIR, f"fk_check_{time.strftime('%Y%m%d_%H%M%S')}.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ra, rb = np.array([r["repo_err"] for r in rows]), np.array([r["alt_err"] for r in rows])
rep = np.array([[r["fw_x"], r["fw_y"], r["fw_z"]] for r in rows[-4:]])
print(f"\nrepo table: mean {ra.mean():.1f} mm | alt table: mean {rb.mean():.1f} mm  -> set KIN_MODEL = "
      f"\"{'repo' if ra.mean() <= rb.mean() else 'alt'}\" in config.py")
zo = np.array([r["alt_z_offset"] for r in rows])
print(f"alt-table z offset (firmware - model): median {np.median(zo):.2f} mm, SD {zo.std():.2f} mm "
      f"-> set KIN_Z_OFFSET_MM = {np.median(zo):.1f} in config.py (current {C.KIN_Z_OFFSET_MM})")
print(f"firmware return-to-pose spread (4 returns): {np.ptp(rep, axis=0).round(2)} mm; saved {out}")
