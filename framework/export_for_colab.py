#!/usr/bin/env python3
"""Pack ALL experiment data into one zip for plotting in Colab (colab_plots.ipynb).

  python3 export_for_colab.py                 # data + key images (mosaics, surveys, overlays, figures)
  python3 export_for_colab.py --images all    # also every view image (large)
  python3 export_for_colab.py --images none
Output: colab_data_<stamp>.zip with MANIFEST.json (what each file is). Before packing, every recorded
trajectory (samples.csv) gets a companion samples_fk.csv with the flange position (FK, mm) and the
conditioning kappa of the arm at each sample, so Colab needs no robot code for those plots.
"""
import argparse, glob, json, os, time, zipfile
import numpy as np

DESCR = {
    "commands.csv": "every command sent to the arm: time, commanded joints (deg), speed, calling script",
    "samples.csv": "joint angles polled while moving/settling (~20 Hz): trajectory of each command",
    "samples_fk.csv": "samples + flange x,y,z (mm, forward kinematics) + conditioning kappa",
    "moves.csv": "one row per move: duration, settling time, start/commanded/final joints, steady-state error, refusals",
    "status.csv": "servo status after each move when the firmware provides it (encoders, speeds, currents, ...)",
    "static.json": "recording metadata (kinematic model, versions)",
    "stream.csv": "PARALLEL tracking during the whole run (~15 Hz): joints, joint velocity/acceleration, flange pose (FK), Cartesian speed, conditioning, current inspection event",
    "events.csv": "inspection events with time and details (survey, AI decision + inference time, view reached, verified, captured)",
    "static_robot.json": "joint limits, FK vs firmware coordinates, encoder noise at the reference",
    "repeat.csv": "repeatability: returns to the reference (joint read-back, firmware coords, image shift in px/mm)",
    "backlash.csv": "approach from +/- per joint: read-back error and image shift",
    "views.csv": "per acquired view: planned/measured conditioning and scale, view error open-loop vs servoed",
    "run.json": "summary of one coverage run",
    "refcover_plan.json": "coverage plan: chosen views (relative offsets, heights, kappa) and the full route",
    "refcover_lattice.npz": "reachability lattice: joint solution, kappa, joint-limit margin for every relative pose",
    "refcover_plan_comparison.json": "proposed vs singularity-blind vs hand-programmed raster (plan level)",
    "refcover_sweep.json": "the same comparison over required resolutions",
    "refcover_summary.json": "statistics over all coverage runs",
    "validate_": "grounding validation per cell (open-loop vs servoed)",
    "episodes.csv": "view-selection runs: one row per method and placement",
    "view_library.npz": "recorded view library (qualities, footprints, time matrices) for RL",
}


def add_fk(path):
    """samples.csv -> samples_fk.csv with flange x,y,z (mm) and kappa. Standard csv + NumPy only (no pandas:
    the pandas on the Jetson needs a newer NumPy than JetPack's)."""
    import csv
    from kinematics import fk
    from refcover import kappa
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cols = list(rows[0].keys()) + ["x_mm", "y_mm", "z_mm", "kappa"]
    out = path.replace("samples.csv", "samples_fk.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            try:
                q = np.radians([float(r["q_%d" % i]) for i in range(1, 7)])
                p = fk(q)[:3, 3] * 1000; k = kappa(q)
                extra = [round(p[0], 3), round(p[1], 3), round(p[2], 3), round(k, 5)]
            except (ValueError, KeyError):
                extra = ["", "", "", ""]
            w.writerow([r[c] for c in rows[0].keys()] + extra)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", choices=["none", "key", "all"], default="key")
    a = ap.parse_args()
    import config as C
    n_fk = 0
    for p in glob.glob(os.path.join(C.LOGS_DIR, "**", "samples.csv"), recursive=True):
        fk_p = p.replace("samples.csv", "samples_fk.csv")
        if not os.path.exists(fk_p) or os.path.getmtime(fk_p) < os.path.getmtime(p):
            if add_fk(p):
                n_fk += 1
    files = set()
    pats = ["config.py", "kinematics.py", "ik.py", "views.py", "refcover.py", "colab_analysis.py",
            os.path.join(C.DEPLOY_DIR, "*.json"), os.path.join(C.DEPLOY_DIR, "*.npz"),
            os.path.join(C.DEPLOY_DIR, "views", "*", "meta.json"), os.path.join(C.DEPLOY_DIR, "views", "*", "defects.json"),
            "data/*.npz", "results_views/**/*.json", "results_views/**/*.csv", "results_views/*.txt",
            os.path.join(C.LOGS_DIR, "**", "*.csv"), os.path.join(C.LOGS_DIR, "**", "*.json"),
            os.path.join(C.LOGS_DIR, "**", "*.npy")]
    if a.images in ("key", "all"):
        pats += [os.path.join(C.LOGS_DIR, "**", "mosaic.png"), os.path.join(C.LOGS_DIR, "**", "survey.jpg"),
                 os.path.join(C.LOGS_DIR, "grounding", "*.jpg"), os.path.join(C.LOGS_DIR, "align", "*.jpg"),
                 "paper/figs/*.png", "paper/figs/*.pdf", os.path.join(C.DEPLOY_DIR, "reference", "*.jpg")]
    if a.images == "all":
        pats += [os.path.join(C.LOGS_DIR, "**", "*.jpg"), os.path.join(C.DEPLOY_DIR, "views", "*", "*.jpg")]
    for pat in pats:
        files |= {f for f in glob.glob(pat, recursive=True) if os.path.isfile(f)}
    files = sorted(f for f in files if "backup" not in f and "__pycache__" not in f)
    man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), n_files=len(files), files={})
    for f in files:
        base = os.path.basename(f)
        man["files"][f] = next((v for k, v in DESCR.items() if base.startswith(k) or base == k), "")
    out = "colab_data_%s.zip" % time.strftime("%Y%m%d_%H%M%S")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f)
        z.writestr("MANIFEST.json", json.dumps(man, indent=1))
    print("added FK/kappa to %d trajectory files; packed %d files (%.1f MB) -> %s"
          % (n_fk, len(files), os.path.getsize(out) / 1e6, out))
    print("upload it to Colab with colab_plots.ipynb")


if __name__ == "__main__":
    main()
