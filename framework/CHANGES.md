# Update 2 (after your first coverage runs)

Copy these 7 files into `mycobot_framework/` (overwrite). Nothing else changes.

| File | Change |
|---|---|
| `measure_height.py` | Also measures the DESCENT DRIFT (how far the view slides per mm the camera is lowered; your runs showed ~21-29 mm at 105 mm, i.e. a camera tilt of ~12 deg) and stores it in `deploy/grounding.json` with `--write` |
| `servo_scaled.py` | Feed-forward correction of the calibrated drift before the visual servo; a single marker's size is no longer used for magnification (it gave 1.17-1.40 instead of ~2.06); implausible magnifications are rejected; registration first on the table only (part masked: no parallax from the tall part); 10 correction steps instead of 6; logs 3 errors: as planned / after feed-forward / verified |
| `refcover.py` | Uses the feed-forward and the table mask; prints and logs the 3 errors; `report` without pandas |
| `collect_views.py`, `run_view_policy.py` | Same feed-forward and table mask (Part B) |
| `export_for_colab.py` | Without pandas (works with JetPack's NumPy 1.19.5) |
| `colab_analysis.py` | Coverage figure with the 3 error stages; verification-method counts |

## Then
    python3 measure_height.py --write            # H and the descent drift (~2 min, arm moves straight down and back)
    python3 refcover.py run --tag test2 --speed 8
    # if test2 looks good (verified error a few mm or less):
    python3 refcover.py run --tag horizontal_01   ... _05, vertical_01 ... _05
    python3 refcover.py report
    python3 export_for_colab.py --images all
Keep or move the runs made before this update: they had no feed-forward; for the paper use the new ones
(`mkdir hw_logs/refcover_old && mv hw_logs/refcover/* hw_logs/refcover_old/` before the new runs).
