#!/usr/bin/env python3
"""Capture one survey image at the locked pose and save the grounded grid overlay.

  python3 validate_vision.py [--dry-run]
  -> hw_logs/validation/raw_camera.jpg, grounded_grid.jpg, edges.jpg
"""
import argparse, os
import cv2
import numpy as np
import config as C
from hardware import make_arm, make_camera, wait_settled, load_locked_pose
from perception import grounded_survey, load_grounding, draw_survey, edge_mask, to_gray, locate_part, locate_part_scene, track_grid_center
from fence import load_fence

ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg")); a = ap.parse_args()
lock = load_locked_pose(a.dry_run); q = np.array(lock["angles_deg"], float)
arm = make_arm(a.dry_run, q); arm.send_deg(q); wait_settled(arm)
cam = make_camera(a.dry_run, arm, a.ref_img)
frame = cam.grab(); cam.release(); arm.close()
out = os.path.join(C.LOGS_DIR, "validation"); os.makedirs(out, exist_ok=True)
cv2.imwrite(os.path.join(out, "raw_camera.jpg"), frame)
cv2.imwrite(os.path.join(out, "edges.jpg"), edge_mask(to_gray(frame)))
import board as BD
dets = BD.detect_markers(frame)
how = "ArUco ids" if any(not str(k).startswith("c") for k in dets) else "dark-square fallback"
vis_m = frame.copy()
for k, c in dets.items():
    u = c.mean(0)
    cv2.polylines(vis_m, [c.astype(np.int32)], True, (0, 200, 255), 2)
    cv2.putText(vis_m, str(k), (int(u[0]) + 12, int(u[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.imwrite(os.path.join(out, "markers.jpg"), vis_m)
print(f"corner markers found: {len(dets)}/4 ({how if dets else 'none'}); cv2.aruco available: "
      f"{hasattr(cv2, 'aruco')}  -> see markers.jpg")
if len(dets) < 4:
    print("  need all 4 at the survey pose: give every marker a white border, keep it off the tape and the part")
part = locate_part(frame)
if part:
    x, y, w, h = part["bbox_px"]
    vis = frame.copy(); cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out, "part_detection.jpg"), vis)
    print(f"part found: bbox {part['bbox_px']} px (see part_detection.jpg)")
else:
    print("part NOT found: check lighting/contrast, or set PART_THRESH in config.py")
if os.path.exists(C.GROUNDING_PATH):
    G = load_grounding()
    part = locate_part_scene(frame, G)
    center, exact = track_grid_center(part, load_fence())
    sv = grounded_survey(frame, G, center=center)
    draw_survey(frame, sv, os.path.join(out, "grounded_grid.jpg"))
    print(f"grid centre ({center[0]:+.1f}, {center[1]:+.1f}) mm from the locked pose"
          + ("" if exact else " (clamped to the workspace limit)"))
    print(f"observed cells {sv.observed.sum()}/25; densities:\n{np.round(sv.density.reshape(5, 5), 3)}")
else:
    print("no grounding.json yet -> run `python3 grounding.py calibrate`")
print(f"saved images in {out}")
