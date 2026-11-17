#!/usr/bin/env bash
# Experiment runner (Jetson Nano). One phase per call; each phase prints what to check before the next.
#   bash run_all.sh e1        survey pose (your hand pose) + calibration + table grid
#   bash run_all.sh robot     robot characterization: step responses, repeatability, backlash (~15 min)
#   bash run_all.sh export    pack ALL data for Colab (colab_plots.ipynb)
#   bash run_all.sh cplan     REFERENCE COVERAGE: plan from your single taught pose (no motion), method comparison
#   bash run_all.sh crun <tag>  execute the coverage plan: verified views + mosaic (repeat per placement)
#   bash run_all.sh creport   coverage statistics over all runs
#   bash run_all.sh e2test    one slow test placement (watch the arm)
#   bash run_all.sh e2        view library: 20 horizontal + 20 vertical placements (you place the part)
#   bash run_all.sh e2defect  6 placements WITH defect marks (E6), then mark the defects
#   bash run_all.sh build     score the library, fit the pose classifier, pack for Colab
#   (Colab)                   colab_viewplan.ipynb -> to_jetson_*.zip
#   bash run_all.sh e4 <to_jetson zip>   install policies, live evaluation, 8 + 8 new placements
#   bash run_all.sh e5        on-board latency / memory
#   bash run_all.sh e6        label defect visibility (type tile numbers), analyse
#   bash run_all.sh paper     statistics, all figures, manuscript PDF
# Stop any script with Ctrl+C (never Ctrl+Z: it keeps the camera busy).
set -e
N_H=${N_H:-20}; N_V=${N_V:-20}; N_LIVE=${N_LIVE:-8}; BUDGET=${BUDGET:-35}

case "$1" in
e1)
  python3 auto_align.py --check
  python3 auto_align.py --accept            # your hand pose, unchanged, is the survey pose
  python3 grounding.py calibrate --mode track
  python3 table_grid.py --install
  echo "CHECK: hw_logs/grounding/table_grid_overlay.jpg (green squares on the part, inside the box) and that"
  echo "       table_grid.py reports 25/25 cells reachable at the inspection heights" ;;
robot)
  python3 characterize_robot.py all ;;
export)
  python3 export_for_colab.py ;;
cplan)
  python3 refcover.py plan --sweep
  python3 refcover.py plan
  echo "CHECK: paper/figs/fig_refcover_kappa.png (reachability/conditioning per height) and fig_refcover_plan.png" ;;
crun)
  python3 refcover.py run --tag "${2:-$(date +%Y%m%d_%H%M%S)}" --speed 10
  echo "CHECK: hw_logs/refcover/<tag>/mosaic.png" ;;
creport)
  python3 refcover.py report ;;
e2test)
  python3 collect_views.py --pose horizontal --n 1 --speed 10
  echo "CHECK: deploy/views/horizontal_000/*.jpg; delete that folder afterwards if the test was not clean" ;;
e2)
  python3 collect_views.py --pose horizontal --n "$N_H"
  python3 collect_views.py --pose vertical   --n "$N_V" ;;
e2defect)
  python3 collect_views.py --pose horizontal --n 3
  python3 collect_views.py --pose vertical   --n 3
  echo "NOW mark the defects of each of these placements in its survey.jpg:"
  echo "  python3 label_defects.py mark --placement <name> --points \"x,y;x,y\"   (or without --points to click)" ;;
build)
  python3 build_view_dataset.py
  python3 pose.py fit
  python3 sync.py pack-for-colab
  echo "UPLOAD the to_colab_*.zip and the framework zip to Colab; run colab_viewplan.ipynb" ;;
e4)
  [ -n "$2" ] && python3 sync.py unpack "$2"
  python3 run_view_policy.py --pose horizontal --trials "$N_LIVE" --budget-s "$BUDGET" --auto-pose
  python3 run_view_policy.py --pose vertical   --trials "$N_LIVE" --budget-s "$BUDGET" --auto-pose ;;
e5)
  sudo chmod a+r /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_* 2>/dev/null || true
  python3 jetson_check.py ;;
e6)
  python3 label_defects.py sheet
  python3 label_defects.py label
  python3 label_defects.py analyze ;;
paper)
  python3 analyze_views.py
  python3 plot_all.py
  python3 make_paper.py
  echo "paper/main.pdf: write every red [after results: ...] box from the numbers, then delete the box" ;;
*)
  sed -n '2,15p' "$0" ;;
esac
