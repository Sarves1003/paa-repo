#!/usr/bin/env python3
"""Collect grounded survey token sets of the real part for training and held-out testing.

For each placement the operator moves/rotates the part (or changes lighting), presses
Enter, and the arm captures one survey image at the locked pose. Tokens come from the
calibrated grounding (fixed world grid). Aim for 30+ placements; keep ~20% for testing.

  python3 collect_surveys.py --n 30 [--dry-run]
  -> deploy/surveys/survey_XXX.{jpg,npz}  and  data/surveys.npz (train) / data/surveys_test.npz
"""
import argparse, glob, os
import numpy as np
import cv2
import config as C
from hardware import make_arm, make_camera, wait_settled, load_locked_pose
from perception import grounded_survey, load_grounding, draw_survey, locate_part, locate_part_scene, track_grid_center
from fence import load_fence

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--test-frac", type=float, default=0.2)
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))
a = ap.parse_args()

G = load_grounding()
lock = load_locked_pose(a.dry_run)
q_lock = np.array(lock["angles_deg"], float)
arm = make_arm(a.dry_run, q_lock); arm.send_deg(q_lock); wait_settled(arm)
cam = make_camera(a.dry_run, arm, a.ref_img)
os.makedirs(C.SURVEY_DIR, exist_ok=True); os.makedirs("data", exist_ok=True)
start = len(glob.glob(os.path.join(C.SURVEY_DIR, "survey_*.npz")))
try:
    for k in range(start, start + a.n):
        if not a.dry_run:
            input(f"[{k}] place/rotate the part (vary position, angle, lighting), then press Enter ...")
        arm.send_deg(q_lock); wait_settled(arm)
        frame = cam.grab()
        part = locate_part_scene(frame, G) if C.TRACK_PART else None
        center, exact = track_grid_center(part, load_fence())
        if C.TRACK_PART and part is None:
            print("  part not found in this image -> grid at the locked pose (check lighting / placement)")
        sv = grounded_survey(frame, G, center=center)
        base = os.path.join(C.SURVEY_DIR, f"survey_{k:03d}")
        cv2.imwrite(base + ".jpg", frame); draw_survey(frame, sv, base + "_grid.jpg")
        np.savez(base + ".npz", static=sv.static, density=sv.density, observed=sv.observed, center_mm=center,
                 part_found=part is not None)
        print(f"  saved {base}.npz  grid centre ({center[0]:+.1f}, {center[1]:+.1f}) mm  "
              f"observed {sv.observed.sum()}/25  density max {np.nanmax(sv.density):.2f}")
finally:
    cam.release(); arm.close()

files = sorted(glob.glob(os.path.join(C.SURVEY_DIR, "survey_*.npz")))
S = np.stack([np.load(f)["static"] for f in files])
idx = np.random.default_rng(0).permutation(len(S))
n_test = max(1, int(round(a.test_frac * len(S)))) if len(S) > 4 else 0
np.savez(os.path.join("data", "surveys_test.npz"), static=S[idx[:n_test]], files=np.array(files)[idx[:n_test]])
np.savez(os.path.join("data", "surveys.npz"), static=S[idx[n_test:]], files=np.array(files)[idx[n_test:]])
print(f"{len(S)} surveys: {len(S) - n_test} train -> data/surveys.npz, {n_test} test -> data/surveys_test.npz")
