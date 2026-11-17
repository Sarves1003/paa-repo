# View-consistent, kinematics-aware active inspection on the myCobot 280 JN

**Today's experiments: follow `EXPERIMENTS_TODAY.md` step by step.**

This folder replaces your project files with one coherent framework. File names match
your originals where a file was updated.

## What changed and why

| File | Status | What it does now |
|---|---|---|
| `config.py` | new | all settings in one place (copy your PORT/CAM values into it) |
| `kinematics.py` | updated | "alt" DH table + 8.2 mm z offset: matches your firmware at the locked pose (the original table was ~51 mm off) |
| `ik.py` | updated | position IK (0.5 mm tol) **and** `ik_hold`: position + orientation held at the survey pose |
| `hardware.py` | updated | Arm with settle detection, joint-step safety, firmware read-back; dry-run arm/camera |
| `perception.py` | updated | your `observe()` kept; **grounded survey tokens** on a fixed world grid; ROI density; shift estimation |
| `fence.py` | new | workspace limit: automatic +/-60 mm box around the locked pose (default, no teaching), or custom box / teach / click; every move is checked |
| `servo.py` | new | closed-loop visual servoing of every view (corrects the arm's ~1 deg joint errors) |
| `auto_align.py` | new | robot centres its survey view on the drawn box by itself and saves the pose |
| `board.py` | new | 4 corner markers (ArUco ids or plain dark squares): exact image->table mapping, single-marker pose, robot-motion check |
| `grounding.py` | new | **marker-free image-to-workspace calibration** + per-cell validation (mm error) |
| `cost_model.py` | new | per-cell joint targets, reachability, **joint-time cost matrix**, time-model fit from logs |
| `geo_attn.py` | rewritten | attention policy whose tokens include move time from the current pose; PyTorch + NumPy with parity test |
| `twin_env.py` | rewritten | training environment built from real surveys + the robot's own cost model |
| `baselines.py` | rewritten | boustrophedon, nearest neighbour (Cartesian / joint-time), 2-opt, greedy, budget local search, random |
| `train.py` | rewritten | policy-gradient training (shared K-sample baseline), validation vs planners, NumPy test |
| `collect_surveys.py` | new (replaces `collect_and_train.py`) | real survey dataset of your part (train/test split) |
| `run_inspection.py` | rewritten | paired, interleaved hardware trials of all methods, full logging |
| `fk_check.py` | new | which DH table matches the firmware |
| `validate_vision.py` | updated | grounded grid overlay of the survey image |
| `analyze_experiments.py` | new | every statistic for the paper from the hardware logs (run on a PC) |
| `sync.py` | new | packs robot data for Colab and installs trained policies on the Jetson (checksums, backups) |
| `colab_pipeline.ipynb` | new | Colab notebook: time-model fit, multi-seed training, parity, held-out test, analysis |
| `jetson_check.py` | new | verifies the Jetson Nano 4 GB and measures latency, memory and energy on the board |
| `train_live.py`, `train_metal.py`, `run_2026_experiments.py`, `collect_and_train.py`, `train.py` (old) | **retire** | synthetic-only training, energy from assumed power, uniform-attention "XAI"; do not use for results |


## Work split: robot on the Jetson, heavy computation in Colab

**Where does the RL learn?** In a simulation ("twin") built from your robot's own data, not on the live
arm and not on raw pixels. The policy's inputs are features computed from real survey images (edge
density and gradient direction per grounded cell) plus robot state (visited cells, measured move time
from the current pose). Its reward is move time (coverage) or collected inspection value within a
time budget, computed with the robot's identified joint-time cost model. Training on the live arm would
need ~768,000 episodes (3,000 iterations x 32 x 8), about 310 days at ~35 s per episode, so the robot
**collects data** and **evaluates** the trained policy.

| Step | Where | Command |
|---|---|---|
| Board check, FK check | Jetson | `python3 jetson_check.py`, `python3 fk_check.py` |
| Grounding (camera <-> workspace) | Jetson + robot | `python3 grounding.py calibrate`, `... validate` |
| Cost models | Jetson | `python3 cost_model.py build --ik hold` (and `--ik position`) |
| Timing runs (classical planners) | Jetson + robot | `python3 run_inspection.py --methods boustro,nn_time --trials 3` |
| Survey images of the part | Jetson + robot | `python3 collect_surveys.py --n 30` |
| Send data to Colab | Jetson | `python3 sync.py pack-for-colab` -> upload `to_colab_*.zip` |
| Time-model fit, training (3 seeds), parity, held-out test | **Colab** | `colab_pipeline.ipynb`, cells 1-7 |
| Send policies back | Colab -> Jetson | notebook cell 9 -> `python3 sync.py unpack to_jetson_*.zip` |
| Robot evaluation (>= 10 paired trials) | Jetson + robot | `python3 run_inspection.py ... --policy deploy/geo_attn_policy_coverage.npz` |
| Statistics and figures | **Colab** | pack again, notebook cell 8 |

The Jetson only ever runs NumPy/OpenCV; nothing it does needs more than ~100 MB of RAM.

## Platform: myCobot 280 JN = Jetson Nano **4 GB**

| Item | Value the code is built for |
|---|---|
| Compute | NVIDIA Jetson Nano 4 GB (4x Cortex-A57, 4 GB LPDDR4 shared CPU/GPU) |
| OS / Python | JetPack 4.6.x (L4T R32), Ubuntu 18.04, **Python 3.6** |
| On-robot dependencies | NumPy >= 1.17, OpenCV (JetPack's 4.1.1 is fine), pymycobot |
| Not needed on the robot | PyTorch, pandas, scipy, matplotlib (training/analysis run on a PC or Colab) |

All on-robot files pass a Python 3.6 grammar check, use no dataclasses or 3.7+ syntax, and
import pandas/scipy/torch only inside training or analysis code.

Setup on the Jetson (JetPack ships NumPy 1.13, which is too old):
```bash
sudo apt-get install -y python3-pip python3-opencv
pip3 install --user numpy==1.19.5 pymycobot
sudo nvpmodel -m 0 && sudo jetson_clocks      # 10 W MAXN mode; report the mode you use
python3 jetson_check.py                         # verifies 4 GB board + measures latency, memory, energy
```
`jetson_check.py` writes `hw_logs/jetson_check.json`: board model, RAM (~3.9 GB visible on the
4 GB Nano; a 2 GB board triggers a warning), power mode, decision latency, survey and IK
time, peak memory, and **measured** energy per decision from the Nano's INA3221 power
monitor. Use these numbers in the paper instead of assumed power values.

Expected resource use on the 4 GB Nano: the runtime process stays around 100 MB
(about 2% of RAM; 71 MB measured on a PC). The policy has 17,664 parameters (~70 kB as float32).

## Run order (each step is safe to dry-run first with `--dry-run`)

```bash
# 0. checks
python3 jetson_check.py                   # 4 GB board, versions, latency, memory, energy
python3 fk_check.py                       # set KIN_MODEL in config.py to the table it recommends
python3 validate_vision.py                # survey image at the locked pose

# 1. grounding (marker-free, no hand-eye matrix)
python3 grounding.py calibrate            # fits A (px/mm); all 25 cells must be inside the image
python3 grounding.py validate             # per-cell grounding error in mm  -> claim C1
python3 validate_vision.py                # check the grounded grid overlay

# 1b. workspace limit is automatic (+/-60 mm around the locked pose); the grid follows the part
python3 fence.py show                     # optional custom limit: fence.py box --x -40 40 --y -40 40

# 2. cost models for both IK modes (cells outside the fence are removed)
python3 cost_model.py build --ik hold
python3 cost_model.py build --ik position

# 3. calibration runs for the time model (classical planners only)
python3 run_inspection.py --task coverage --ik hold     --methods boustro,nn_time --trials 3
python3 run_inspection.py --task coverage --ik position --methods boustro,nn_time --trials 3
python3 cost_model.py fit "hw_logs/inspection/coverage_hold_*/log.csv"     --ik hold
python3 cost_model.py fit "hw_logs/inspection/coverage_position_*/log.csv" --ik position

# 4. real survey dataset (move/rotate the part, vary light), 30+ placements
python3 collect_surveys.py --n 30

# 5. train on PC/Colab (copy config.py, kinematics.py, ik.py, deploy/, data/ there)
for s in 0 1 2; do python3 train.py --task coverage --ik hold --seed $s --policy deploy/policy_cov_s$s.npz; done
for s in 0 1 2; do python3 train.py --task budget   --ik hold --seed $s --policy deploy/policy_bud_s$s.npz; done

# 6. hardware evaluation: >= 10 paired trials per condition
python3 run_inspection.py --task coverage --ik hold --methods policy,boustro,nn_cart,nn_time,twoopt \
        --trials 10 --policy deploy/policy_cov_s0.npz
python3 run_inspection.py --task budget --budget-s 15 --ik hold --methods policy,boustro,greedy,budget_ls \
        --trials 10 --policy deploy/policy_bud_s0.npz
python3 run_inspection.py --task coverage --ik position --methods boustro,nn_cart,twoopt --trials 10

# 7. statistics
python3 analyze_experiments.py "hw_logs/inspection/*/log.csv" --grounding "hw_logs/grounding/validate_*.csv"
```

Safety: speed 20; any move outside the workspace fence or with a joint change > 70 deg is refused
(hardware.safe_move, used by every script); the arm returns to the locked pose after every episode. Keep the workspace clear and a hand on the power switch.

## Claims and the evidence that decides them

Write each hypothesis down **before** running step 6, and report the outcome whatever it is.

| Claim | Hypothesis tested on hardware | Evidence file |
|---|---|---|
| C1 Marker-free grounding | grounding error per cell is small relative to the 17.5 mm pitch (e.g. median < 3 mm) | `grounding.txt`, in-run `ground_res_mm` |
| C2 View-consistent execution | orientation hold reduces view drift (logged up to 25 deg) and per-cell ROI-density variability vs position-only IK | `view_consistency.txt` |
| C3 Kinematics-aware cost | the identified time model predicts move time (R^2, MAE); joint-time 2-opt is faster than Cartesian boustrophedon under orientation hold (model predicts ~9%) | `time_model.txt`, `paired_tests.txt` |
| C4 Learned planner | policy is within X% of 2-opt on coverage time and/or competitive with greedy / local search on budgeted value; paired Wilcoxon, Holm-corrected | `paired_tests.txt`, `test_summary.json` per seed |
| C5 Edge deployment on Jetson Nano 4 GB | NumPy decision latency, peak memory and measured energy per decision on the board; parity with PyTorch | `jetson_check.json`, `infer_ms` in logs, `geo_attn.parity_test()` |

How to read the outcomes honestly:
* C1-C3 are framework claims and stand on their own if the measurements support them.
* C4 is the algorithmic claim. If the policy does not beat 2-opt/local search, report it as
  "matches/approaches classical planners", or as a negative result. Do not drop the baselines.
* Report mean ± SD across training seeds, never the best seed only.
* Dry-run numbers (synthetic camera, instant motion) are for testing code only and must not be reported.
