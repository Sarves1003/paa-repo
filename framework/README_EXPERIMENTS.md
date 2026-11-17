# View-verified active inspection with a transformer RL policy (myCobot 280 JN + Jetson Nano 4 GB)

Copy every file of this folder into `mycobot_framework/` (next to your existing grounding.py, servo.py,
hardware.py, ...). Nothing of yours is replaced except that `table_grid.py --install` appends a small,
reversible override to perception.py (backup kept; `table_grid.py --uninstall` removes it).

Use the provided config.py. Before the first robot step, MEASURE two values with a ruler and set them:
```python
VIEW_CAM_HEIGHT_MM = ...    # camera lens -> table, arm at the locked (survey) pose
PART_MAX_HEIGHT_MM = ...    # height of the part standing vertically
```
Collection refuses to move without them (collision check: the lens must stay 25 mm above the part).

Your setup stays as it is: the survey image is taken at your hand-locked pose (whole box, all markers).
From that pose the arm cannot move outward at the same height, so every inspection view is taken lower
(VIEW_LEVELS_DZ = -60 / -80 mm), where all cells are reachable; the arm goes down and up along a vertical
path in short steps. The magnification of the lower views is measured from the marker size in every frame.

## Run order (Jetson unless marked Colab)
| Step | Command | Time | Send me / check |
|---|---|---|---|
| E1 survey pose + calibration | `bash run_all.sh e1` (`--accept` keeps your hand pose) | 15 min | table_grid output (must say 25/25 at both inspection heights) + `table_grid_overlay.jpg` |
| E2 test | `bash run_all.sh e2test` (speed 10) | 10 min | watch the descent and close-ups; check the printed scale (~1.3-1.6) and markers per view; then delete `deploy/views/horizontal_000` |
| E2 library | `bash run_all.sh e2` (20 + 20 placements) | ~3 h | vary position and rotation; never touch the box line |
| E6 defects | make 3-6 real marks, then `bash run_all.sh e2defect` and `label_defects.py mark ...` | 30 min | |
| build | `bash run_all.sh build` | 10 min | `results_views/library_stats.txt` (send it), pose classifier accuracy |
| E3 (Colab) | `colab_viewplan.ipynb` | 2-3 h | `results_views/offline_summary.txt`; parity < 1e-4 in every run |
| E4 live | `bash run_all.sh e4 to_jetson_<stamp>.zip` (8 + 8 NEW placements) | ~2 h | `results_views/live_tests.txt` |
| E5 | `bash run_all.sh e5` | 10 min | `hw_logs/jetson_check.json` |
| E6 labels | `bash run_all.sh e6` (type visible tile numbers) | 30 min | `results_views/defect_results.json` |
| paper | unzip `results_views.zip` from Colab, then `bash run_all.sh paper` | 2 min | `paper/main.pdf` (or upload `paper/` to Overleaf) |

If the pose classifier's leave-one-out accuracy is below ~90 %, remove `--auto-pose` in the e4 phase.

## What each file does
| File | Role |
|---|---|
| `table_grid.py` | square inspection grid on the table inside the drawn box, converted to robot moves with the measured image Jacobian |
| `views.py` | candidate views, measured image quality, coverage value, time model, environment, baselines, oracle |
| `collect_views.py` | E2: records every candidate view of every placement (servoed survey-height view + close-up) |
| `build_view_dataset.py` | scores the images, measures the close-up scale, builds `data/view_library.npz`, train/test split |
| `pose.py` | recognises horizontal / vertical from the survey image; leave-one-out accuracy |
| `train_views.py` | E3 (Colab): REINFORCE with multi-rollout baseline, transformer vs MLP, ablations, held-out test, NumPy export + parity |
| `run_view_policy.py` | E4: live paired evaluation of policy, ablation and baselines on new placements |
| `label_defects.py` | E6: defect marking, tile sheets, labels, visibility vs quality, defect capture per planner |
| `analyze_views.py` | statistics (Wilcoxon, Holm) offline and live |
| `plot_all.py` | all 9 paper figures -> `paper/figs/` |
| `make_paper.py` | every number of the manuscript as a LaTeX macro from your results; compiles `paper/main.pdf` |
| `paper/main.tex`, `refs.bib` | manuscript; numbers are macros, interpretations are red boxes to write after the results |

## Claim -> evidence
| Claim | Evidence |
|---|---|
| C1 open-loop views are unreliable on a low-cost arm; verification fixes it | fig1, fig2; macros `\RServo*`, `\RAxis*` |
| C2 RL trained on recorded real views, with an oracle | fig3, `library_stats.txt`, fig5/6 (% of oracle) |
| C3 attention helps (or not), pose and time inputs matter (or not) | `offline_summary.txt` attn-vs-mlp, no-pose, no-time lines; table_offline |
| C4 on hardware: live, two poses, real defects, Jetson timing | fig7, table_live, fig8, `\RInferMs` |

Report every outcome, including hypotheses that fail. Do not edit numbers into main.tex by hand: re-run
`make_paper.py` after new data.
