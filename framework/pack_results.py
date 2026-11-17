#!/usr/bin/env python3
"""Collect EVERY result needed for the paper figures into one zip for Colab (colab_final_results.ipynb).

  python3 pack_results.py            ->  results_pack_<stamp>.zip   (upload it to Colab)
Includes: Part-A coverage runs (hw_logs/refcover/*: views.csv, run.json, mosaics), SOGA runs (hw_logs/soga/*),
autonomous inspections (hw_logs/autoinspect/*: run.json, pass.json, unrolled surfaces, maps, survey and view images),
live RL runs (hw_logs/viewplan/*), and the results_* folders (soga, auto, task, views) plus deploy/*.json summaries.
"""
import glob, os, time, zipfile

PATTERNS = [
    "hw_logs/refcover/*/views.csv", "hw_logs/refcover/*/tilts.csv", "hw_logs/refcover/*/run.json",
    "hw_logs/refcover/*/mosaic*.png",
    "hw_logs/soga/*/*.json", "hw_logs/soga/*/*.png", "hw_logs/soga/*/*.jpg",
    "hw_logs/autoinspect/*/run.json", "hw_logs/autoinspect/*/pass_*/pass.json", "hw_logs/autoinspect/*/pass_*/*.png",
    "hw_logs/autoinspect/*/pass_*/*.jpg",
    "hw_logs/viewplan/*/*.csv", "hw_logs/viewplan/*/*.json",
    "results_soga/*", "results_auto/*", "results_task/*", "results_views/*.json", "results_views/*.txt",
    "results_views/*.csv", "deploy/refcover_plan.json", "deploy/lattice_views.json", "results_views/refcover_plan_comparison.json",
    "paper/figs/*.png",
]


def main():
    name = "results_pack_%s.zip" % time.strftime("%Y%m%d_%H%M%S")
    n, size = 0, 0
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
        seen = set()
        for p in PATTERNS:
            for f in sorted(glob.glob(p)):
                if f in seen or not os.path.isfile(f):
                    continue
                seen.add(f); z.write(f); n += 1; size += os.path.getsize(f)
    print("packed %d files (%.1f MB) -> %s   (upload to colab_final_results.ipynb)" % (n, size / 1e6, name))
    for k in ("refcover", "soga", "autoinspect", "viewplan"):
        print("  %-12s %d run folder(s)" % (k, len(glob.glob("hw_logs/%s/*/" % k))))


if __name__ == "__main__":
    main()
