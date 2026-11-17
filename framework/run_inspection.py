#!/usr/bin/env python3
"""Run inspection episodes on the myCobot 280 JN and log everything needed for the paper.

Design (paired, interleaved):
  for each trial:  one survey image at the locked pose -> grounded tokens (fixed world grid)
                   every method runs on that SAME survey, in random order
  every move:      precomputed joint target from the cost model (same IK mode for all
                   methods), safety check, settle detection, measured move time,
                   encoder read-back, orientation drift, image at the cell, view-consistent
                   ROI edge density and grounding residual against the survey.

  python3 run_inspection.py --task coverage --ik hold --methods policy,boustro,nn_cart,nn_time,twoopt --trials 5
  python3 run_inspection.py --task budget --budget-s 15 --ik hold --methods policy,greedy,budget_ls,boustro --trials 5
  python3 run_inspection.py --task coverage --ik position --methods boustro,twoopt --trials 5
  python3 run_inspection.py --dry-run --trials 1          # no robot, synthetic camera
"""
import argparse, csv, json, os, time
import numpy as np
import cv2
import config as C
import cost_model as CM
import baselines as BL
from kinematics import fk
from ik import rot_angle_deg
import hardware as HW
import board as BD
from hardware import make_arm, make_camera, safe_move, wait_settled, load_locked_pose
from fence import load_fence
from perception import (grounded_survey, load_grounding, roi_edge_density, estimate_shift, patch,
                        locate_part, locate_part_scene, track_grid_center, offset_to_px,
                        to_gray, cell_offsets_mm, draw_survey)
from twin_env import build_obs, N
from geo_attn import forward_np, load_np

from servo import servo_to_view
SERVO = {"on": False, "m0": None}

FIELDS = ["trial", "method", "task", "ik_mode", "step", "cell", "moved", "planned_move_s", "move_s",
          "dq_max_deg", "infer_ms", "elapsed_s", "value", "tx_mm", "ty_mm", "tz_mm",
          "rb_x_mm", "rb_y_mm", "rb_z_mm", "rb_err_mm", "rb_z_err_mm", "view_drift_deg",
          "fw_x_mm", "fw_y_mm", "fw_z_mm", "roi_density", "survey_density", "ground_res_mm", "ground_peak",
          "in_fence", "refusal", "servo_iters", "servo_s", "view_err_open_mm", "view_err_mm",
          "q1", "q2", "q3", "q4", "q5", "q6", "img"]


class PolicyPlanner:
    """Step-by-step decisions with the deployed NumPy policy (same observation as training)."""
    def __init__(self, W, static, inst):
        self.W, self.static, self.inst = W, static, inst

    def next(self, visited, cur, t_used, v_sum, step):
        i = self.inst
        tok, ctx, mask = build_obs(self.static[None], i["value"][None], i["T"][None], i["reach"][None],
                                   visited[None], np.array([cur]), np.array([t_used]), np.array([v_sum]),
                                   np.array([i.get("budget", np.inf)]), np.array([i["task"] == "budget"]),
                                   np.array([step]))
        if mask.all():
            return None, 0.0
        t = time.perf_counter()
        lg = forward_np(self.W, tok, ctx, mask)
        return int(lg.argmax(1)[0]), (time.perf_counter() - t) * 1e3


def run_episode(arm, cam, w, trial, method, a, M, inst, sv, I0, G, q_lock, R_ref, outdir, W=None):
    T, Qd = M["T"], M["q_deg"]
    arm.send_deg(q_lock); wait_settled(arm)
    # every method starts at the same start cell (nearest the locked pose); this move is not timed
    moved0, _, _ = safe_move(arm, Qd[inst["start"]], a.speed)
    if not moved0:
        print(f"  cannot reach start cell {inst['start']} ({HW.LAST_REFUSAL}); episode skipped")
        return 0.0, 0.0, 0.0
    learned = method in ("policy", "policy_mlp")
    fixed = None if learned else BL.plan(method, inst, np.random.default_rng(1000 + trial))
    planner = PolicyPlanner(W[method], sv.static, inst) if learned else None
    visited = np.zeros(N, bool); cur, t_plan, v_sum, step = inst["start"], 0.0, 0.0, 0
    t_start, q_prev = time.time(), arm.read_deg()
    size = G["px_per_mm"] * 2 * C.GRID_HALF_MM / (C.GRID_N - 1) * 1.5 if G else None
    while True:
        if planner:
            cell, lat = planner.next(visited, cur, t_plan, v_sum, step)
        else:
            cell, lat = None, float("nan")
            while fixed:
                c = fixed.pop(0)
                if not visited[c] and inst["reach"][c] and (inst["task"] != "budget" or
                                                             t_plan + T[cur, c] <= inst["budget"] + 1e-9):
                    cell = c; break
        if cell is None:
            break
        visited[cell] = True
        row = dict(trial=trial, method=method, task=a.task, ik_mode=a.ik, step=step, cell=cell,
                   planned_move_s=float(T[cur, cell]), infer_ms=lat, value=float(inst["value"][cell]))
        tgt = M["p_lock_mm"] + np.r_[M["xy_mm"][cell], M["dz_mm"]]
        row.update(tx_mm=tgt[0], ty_mm=tgt[1], tz_mm=tgt[2])
        moved, mv_s, q_meas = safe_move(arm, Qd[cell], a.speed)
        row.update(moved=int(moved), in_fence=int(HW.in_fence_q(Qd[cell])), refusal=HW.LAST_REFUSAL)
        if moved:
            sv_out = None
            if SERVO["on"]:                      # closed-loop view correction (see servo.py)
                sv_out = servo_to_view(arm, cam, G["A"], G["uc"], SERVO["m0"], offset_to_px(G, M["xy_mm"][cell]),
                                       R_ref, safe_move, Qd[cell], speed=min(a.speed, 15))
                q_meas = arm.read_deg()
                row.update(servo_iters=sv_out["servo_iters"], servo_s=sv_out["servo_s"],
                           view_err_open_mm=sv_out["view_err_open_mm"], view_err_mm=sv_out["view_err_mm"])
            Tm = fk(np.deg2rad(q_meas)); p = Tm[:3, 3] * 1000
            fw = arm.read_coords()
            frame = sv_out["frame"] if sv_out is not None else cam.grab()
            img_name = f"t{trial}_{method}_s{step:02d}_c{cell:02d}.jpg"
            cv2.imwrite(os.path.join(outdir, "img", img_name), frame)
            gres, gpk = np.nan, np.nan
            if sv_out is not None and np.isfinite(sv_out["view_err_mm"]):
                gres, gpk = sv_out["view_err_mm"], float(sv_out["servo_markers"])
            elif G is not None and a.ik == "hold":
                if G.get("mode") == "board":       # markers measure where the camera centre really is
                    ck, _ = BD.camera_point(frame, G, M["xy_mm"][cell])
                    if ck is not None:
                        gres = float(np.linalg.norm(ck - (G["c0"] + G["M"] @ M["xy_mm"][cell]))); gpk = 1.0
                elif "markers0" in G:              # marker tracking: measured vs predicted shift
                    m0 = {int(k) if str(k).isdigit() else k: np.array(v) for k, v in G["markers0"].items()}
                    sh, nm = BD.marker_shift(m0, BD.detect_markers(frame))
                    if sh is not None:
                        gres = float(np.linalg.norm(np.linalg.solve(G["A"], sh - G["A"] @ M["xy_mm"][cell])))
                        gpk = float(nm)
                else:
                    P = patch(I0, offset_to_px(G, M["xy_mm"][cell]), size)
                    Q = patch(to_gray(frame), G["uc"], size)
                    if P is not None and Q is not None:
                        sh, gpk = estimate_shift(P, Q)
                        if gpk >= 0.3:
                            gres = float(np.linalg.norm(np.linalg.solve(G["A"], sh)))
            row.update(move_s=mv_s, dq_max_deg=float(np.max(np.abs(q_meas - q_prev))),
                       rb_x_mm=p[0], rb_y_mm=p[1], rb_z_mm=p[2], rb_err_mm=float(np.linalg.norm(p - tgt)),
                       rb_z_err_mm=float(p[2] - tgt[2]), view_drift_deg=rot_angle_deg(Tm[:3, :3], R_ref),
                       fw_x_mm=fw[0], fw_y_mm=fw[1], fw_z_mm=fw[2], roi_density=roi_edge_density(frame),
                       survey_density=float(sv.density[cell]), ground_res_mm=gres, ground_peak=gpk,
                       img=img_name, **{f"q{k + 1}": q_meas[k] for k in range(6)})
            q_prev = q_meas
            t_plan += T[cur, cell]; v_sum += inst["value"][cell]; cur = cell
        row["elapsed_s"] = time.time() - t_start
        w.writerow(row)
        print(f"  [{method:9s}] step {step:2d} cell {cell:2d} moved={row['moved']} "
              f"plan {row['planned_move_s']:.2f}s meas {row.get('move_s', float('nan')):.2f}s "
              f"drift {row.get('view_drift_deg', float('nan')):.1f}deg")
        step += 1
    return time.time() - t_start, t_plan, v_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["coverage", "budget"], default="coverage")
    ap.add_argument("--budget-s", type=float, default=15.0, help="planning budget (model seconds)")
    ap.add_argument("--ik", choices=["hold", "position"], default="hold")
    ap.add_argument("--methods", default="policy,boustro,nn_cart,nn_time,twoopt")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--speed", type=int, default=C.MOVE_SPEED)
    ap.add_argument("--policy", default=C.POLICY_PATH, help="transformer policy (method 'policy')")
    ap.add_argument("--policy-mlp", default=None, help="ablation weights (method 'policy_mlp')")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-prompt", action="store_true", help="do not pause between trials (part stays fixed)")
    ap.add_argument("--no-servo", action="store_true", help="open-loop views only (servoing is on by default "
                                                               "with --ik hold when markers were calibrated)")
    ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))
    a = ap.parse_args()

    lock = load_locked_pose(a.dry_run)
    q_lock = np.array(lock["angles_deg"], float)
    R_ref = fk(np.deg2rad(q_lock))[:3, :3]
    M = CM.build(a.ik, dry=a.dry_run, save=False, center_mm=(0.0, 0.0))    # timing params for meta.json
    G = load_grounding() if os.path.exists(C.GROUNDING_PATH) else None
    if G is None:
        raise SystemExit("run `python3 grounding.py calibrate` first (grounded tokens are required)")
    fence = load_fence()
    if fence is None and C.REQUIRE_FENCE:
        raise SystemExit("no workspace fence: run `python3 fence.py teach` (or click / box) first")
    p_lock_mm = fk(np.deg2rad(q_lock))[:3, 3] * 1000
    HW.set_fence(fence, p_lock_mm)
    SERVO["on"] = (a.ik == "hold" and not a.no_servo and "markers0" in G)
    if SERVO["on"]:
        SERVO["m0"] = {int(k) if str(k).isdigit() else k: np.array(v) for k, v in G["markers0"].items()}
    print("closed-loop view servoing: %s" % ("ON" if SERVO["on"] else "off"))
    methods = a.methods.split(",")
    W = {}
    for m, path in (("policy", a.policy), ("policy_mlp", a.policy_mlp)):
        if m not in methods:
            continue
        if not path or not os.path.exists(path):
            raise SystemExit(f"weights for '{m}' not found: {path} (train.py; use --policy / --policy-mlp)")
        W[m] = load_np(path)
    if "policy" in W:
        Wp = W["policy"]
        info = {k[5:]: str(Wp[k]) for k in Wp if k.startswith("info_")}
        if info.get("task") and info["task"] != a.task:
            print(f"WARNING: policy was trained for task={info['task']}, running task={a.task}")
        if info.get("ik") and info["ik"] != a.ik:
            print(f"WARNING: policy was trained with ik={info['ik']}, running ik={a.ik}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(C.LOGS_DIR, "inspection", f"{a.task}_{a.ik}_{stamp}")
    os.makedirs(os.path.join(outdir, "img"), exist_ok=True)
    json.dump(dict(vars(a), q_lock=q_lock.tolist(), t0=float(M["t0"]), omega=float(M["omega"]),
                   reach=int(M["reach"].sum()), grounding=G["created"], kin_model=C.KIN_MODEL,
                   fence=None if fence is None else fence.to_json()),
              open(os.path.join(outdir, "meta.json"), "w"), indent=2, default=str)
    arm = make_arm(a.dry_run, q_lock)
    arm.send_deg(q_lock); wait_settled(arm)
    cam = make_camera(a.dry_run, arm, a.ref_img)
    f = open(os.path.join(outdir, "log.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS, restval=""); w.writeheader()
    ep = open(os.path.join(outdir, "episodes.csv"), "w", newline="")
    we = csv.writer(ep); we.writerow(["trial", "method", "measured_s", "planned_s", "value", "n_cells"])
    tf = open(os.path.join(outdir, "tracking.csv"), "w", newline="")
    tw = csv.writer(tf); tw.writerow(["trial", "grid_cx_mm", "grid_cy_mm", "unclamped", "part_u_px", "part_v_px",
                                      "part_x_mm", "part_y_mm", "part_w_mm", "part_h_mm"])
    rng = np.random.default_rng(a.seed)
    try:
        for trial in range(a.trials):
            if not a.dry_run and not a.no_prompt and trial > 0:
                input(f"\nTrial {trial}: optionally re-place the part, then press Enter ...")
            arm.send_deg(q_lock); wait_settled(arm)
            if SERVO["on"]:                      # re-servo the survey view: same image frame as calibration
                r0 = servo_to_view(arm, cam, G["A"], G["uc"], SERVO["m0"], G["uc"], R_ref, safe_move, q_lock, speed=10)
                print(f"  survey view re-aligned: {r0['view_err_open_mm']:.2f} -> {r0['view_err_mm']:.2f} mm")
                frame0 = r0["frame"]
            else:
                frame0 = cam.grab()
            cv2.imwrite(os.path.join(outdir, f"survey_t{trial}.jpg"), frame0)
            part = locate_part_scene(frame0, G) if C.TRACK_PART else None
            center, exact = track_grid_center(part, fence)
            if C.TRACK_PART:
                print("  part " + ("NOT FOUND -> grid at the locked pose" if part is None else
                      "at (%+.1f, %+.1f) mm, size %.0f x %.0f mm -> grid centre (%+.1f, %+.1f)%s"
                      % (part["center_mm"][0], part["center_mm"][1], part["size_mm"][0], part["size_mm"][1],
                         center[0], center[1], "" if exact else " (clamped to the workspace limit)")))
            M = CM.build(a.ik, dry=a.dry_run, save=False, center_mm=center)   # this trial's grid
            np.savez(os.path.join(outdir, f"cost_model_t{trial}.npz"), **M)
            tw.writerow([trial, center[0], center[1], int(exact)] + (
                [part["centroid_px"][0], part["centroid_px"][1], part["center_mm"][0], part["center_mm"][1],
                 part["size_mm"][0], part["size_mm"][1]] if part else [""] * 6)); tf.flush()
            sv = grounded_survey(frame0, G, center=center)
            draw_survey(frame0, sv, os.path.join(outdir, f"survey_t{trial}_grid.jpg"))
            avail = M["reach"].copy()
            if a.task == "budget":
                avail &= sv.observed
            v = np.nan_to_num(sv.density) * avail
            inst = dict(T=M["T"], D_cart=M["D_cart_mm"], reach=avail, value=v / max(v.max(), 1e-6),
                        start=int(M["start"]), task=a.task, budget=a.budget_s)
            order = list(methods); rng.shuffle(order)
            print(f"\n=== trial {trial}: methods in order {order}; reachable {avail.sum()} cells")
            for m in order:
                meas, plan_s, val = run_episode(arm, cam, w, trial, m, a, M, inst, sv, to_gray(frame0), G,
                                                q_lock, R_ref, outdir, W)
                we.writerow([trial, m, round(meas, 2), round(plan_s, 2), round(val, 4), ""]); ep.flush(); f.flush()
                print(f"  -> {m}: measured {meas:.1f} s, planned {plan_s:.1f} s, value {val:.3f}")
    finally:
        f.close(); ep.close(); tf.close(); cam.release()
        arm.send_deg(q_lock); arm.close()
    print(f"\nsaved {outdir}")


if __name__ == "__main__":
    main()
