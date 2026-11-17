#!/usr/bin/env python3
"""EXPERIMENT E4: live evaluation on the robot, on NEW placements (never in the library).

Per placement: one servoed survey; every method then runs on that same placement, in random order,
under the same time budget (robot time model). Each captured image is scored immediately; the
improvement in inspection quality is the outcome. Paired by placement -> Wilcoxon in analyze_views.py.

  python3 run_view_policy.py --pose horizontal --trials 8 --budget-s 35 \
      --policy deploy/view_policy_attn.npz --policy-mlp deploy/view_policy_mlp.npz
  methods (default): policy,policy_mlp,greedy_pred,nn_time,raster_low

TASK mode (ring / O-ring / rust inspection, after task_colab.py in Colab):
  python3 run_view_policy.py --task --pose horizontal --trials 6 --budget-s 35 \
      --policy deploy/view_policy_attn_s0_task.npz --methods policy,greedy_pred,setcover,raster_low
  every captured image is classified on the Jetson (deploy/part_clf.npz); the planners maximise the evidence per
  item; at the end the part is ACCEPTED or REJECTED (which ring / O-ring / rust) and compared with the condition you
  type for the trial. --stop-conf 0.9 stops as soon as every item is decided with that confidence.
Output: hw_logs/viewplan/<pose>_<stamp>/ views.csv, episodes.csv, images
"""
import argparse, csv, json, os, time
import numpy as np
import cv2
import config as C
import hardware as HW
from hardware import make_arm, make_camera, safe_move, wait_settled, load_locked_pose
from kinematics import fk
from ik import ik_hold
from fence import load_fence
from perception import load_grounding, locate_part_scene
from views import ViewSet, BatchEnv, survey_features, view_quality, estimate_affine, time_matrix, choose
from collect_views import capture_tilted
from servo_scaled import servo_to_view_scaled
from geo_attn import load_np
from collect_views import plan_q, servo_view, survey_servo, descend, check_clearance, move_to


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", choices=["horizontal", "vertical"], required=True)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--auto-pose", action="store_true",
                    help="the policy gets the pose recognised by pose.py (deploy/pose_clf.json), not the operator label")
    ap.add_argument("--budget-s", type=float, default=35.0)
    ap.add_argument("--methods", default="policy,policy_mlp,greedy_pred,setcover,nn_time,raster_low")
    ap.add_argument("--policy", default="deploy/view_policy_attn.npz")
    ap.add_argument("--policy-mlp", default="deploy/view_policy_mlp.npz")
    ap.add_argument("--library", default="data/view_library.npz")
    ap.add_argument("--speed", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))
    ap.add_argument("--task", action="store_true", help="ring / O-ring / rust inspection with the part classifier")
    ap.add_argument("--classifier", default="deploy/part_clf.npz")
    ap.add_argument("--prior", default="deploy/task_prior.npz")
    ap.add_argument("--stop-conf", dest="stop_conf", type=float, default=0.0,
                    help="task mode: stop when every item is decided with this confidence (0 = use the whole budget)")
    ap.add_argument("--condition", default=None,
                    help="task mode: true condition of the part for ALL trials (otherwise asked at every trial)")
    a = ap.parse_args()
    TM = CLF = None
    if a.task:
        import task_model as TM
        for f in (a.classifier, a.prior):
            if not os.path.exists(f):
                raise SystemExit("task mode needs %s (from Colab: task_colab.py classify / library)" % f)
        CLF = TM.PartClassifier(a.classifier); BETA, LEARN = TM.load_prior(a.prior)
        print("task mode: items %s, classifier %s" % ([TM.ITEMS[j] for j in LEARN], a.classifier))

    z = np.load(a.library); meta = json.loads(str(z["meta"]))
    servo_mean = float(np.mean(z["servo_mean"]))
    G = load_grounding()
    levels = meta.get("levels_dz", (0.0, meta["dz"]))
    check_clearance(levels)
    vs = ViewSet.from_meta(meta, uc=G["uc"], tg=json.load(open(os.path.join(C.DEPLOY_DIR, "table_grid.json"))))
    px_per_mm = float(json.load(open(os.path.join(C.DEPLOY_DIR, "table_grid.json")))["box_px_per_mm"])
    import lattice_views as LVm
    LV = LVm.load_views() if meta.get("lattice") else None
    if LV is not None and "lattice_file" not in LV:
        raise SystemExit("deploy/lattice_views.json was built by an old lattice_views.py (no lattice file recorded; "
                         "heights %s mm). Delete it and rebuild:  python3 lattice_views.py <same options as the plan>"
                         % LV.get("levels_dz"))
    if LV is not None:
        print("candidate views from %s (lattice %s): heights %s mm" % ("deploy/lattice_views.json", LV["lattice_file"],
                                                                         LV["levels_dz"]))
    if meta.get("lattice") and LV is None:
        raise SystemExit("the library was collected in unified mode: deploy/lattice_views.json is needed")
    if LV is not None and len(LV.get("levels_dz", [])) != vs.L:
        raise SystemExit("deploy/lattice_views.json has %d candidate levels but the library %d: use the candidate file "
                         "the library was collected with" % (len(LV.get("levels_dz", [])), vs.L))
    W = {}
    methods = a.methods.split(",")
    for m, p in (("policy", a.policy), ("policy_mlp", a.policy_mlp)):
        if m in methods:
            if not os.path.exists(p):
                raise SystemExit("weights for %s not found: %s" % (m, p))
            W[m] = load_np(p)
    m0 = {int(k) if str(k).isdigit() else k: np.array(v) for k, v in G["markers0"].items()}
    lock = load_locked_pose(a.dry_run); q_lock = np.array(lock["angles_deg"], float)
    T0 = fk(np.deg2rad(q_lock)); R_ref, p_lock = T0[:3, :3], T0[:3, 3] * 1000
    HW.set_fence(load_fence(), p_lock)
    if LV is not None:                                   # unified: Part A candidates, safe-route times
        Qplan = np.array([v["q_deg"] if v["ok"] else [np.nan] * 6 for v in LV["views"]], float)
        okplan = np.array([v["ok"] for v in LV["views"]])
        T = np.asarray(LV["T"], float) + servo_mean
    else:
        Qplan, okplan = plan_q(vs, q_lock, R_ref, p_lock)
        qd = np.vstack([np.nan_to_num(Qplan), q_lock[None]])
        T = time_matrix(qd, np.full(vs.K, servo_mean), vs, meta["t0"], meta["omega"])

    out = os.path.join(C.LOGS_DIR, "viewplan", "%s_%s" % (a.pose, time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    json.dump(dict(vars(a), library_meta={k: v for k, v in meta.items() if k != "table_grid"}, servo_mean=servo_mean,
                   policy_info={m: {k[5:]: str(w[k]) for k in w if k.startswith("info_")} for m, w in W.items()}),
              open(os.path.join(out, "meta.json"), "w"), indent=2, default=str)
    fv = open(os.path.join(out, "views.csv"), "w", newline="")
    wv = csv.writer(fv); wv.writerow(["trial", "pose", "method", "step", "k", "cell", "level", "implicit", "planned_s",
                                      "wall_s", "view_err_mm", "patch_q_mean", "value", "infer_ms", "img"])
    fe = open(os.path.join(out, "episodes.csv"), "w", newline="")
    we = csv.writer(fe); we.writerow(["trial", "pose", "pose_auto", "pose_used", "method", "budget_s", "improvement", "value0", "value",
                                      "planned_s", "wall_s", "n_views", "condition", "truth", "verdict", "correct",
                                      "p_items", "min_conf"])
    import robot_trace as RT; RT.install(out)          # commands, joint trajectories, servo status
    arm = make_arm(a.dry_run, q_lock); arm.send_deg(q_lock); wait_settled(arm)
    cam = make_camera(a.dry_run, arm, a.ref_img)
    mover = LVm.LatticeMover(arm, LV, a.speed) if LV is not None else None
    survey_now = {}
    rng = np.random.default_rng(a.seed)
    lap_ref = meta["lap_ref"]
    import pose as PZ
    clf = PZ.load_clf()
    if a.auto_pose and clf is None:
        raise SystemExit("--auto-pose needs deploy/pose_clf.json: run  python3 pose.py fit")

    def capture(k, hi_imgs, at_hi):
        """Move to view k (servo for level 0, descend for close-ups) and return (image, view_err, aff)."""
        c = vs.cell[k]
        if mover is not None:                            # unified: lattice route + direct verification
            if not mover.goto_view(k):
                return None, np.nan, None
            if vs.rot[vs.level[k]]:                      # tilted view: planned pose + registration to the survey
                s = capture_tilted(arm, cam, vs, k, survey_now.get("img"), G, px_per_mm)
                H = s.get("H_meas")
                return s["frame"], s["view_err_mm"], (np.asarray(H, float) if H is not None else None)
            s = servo_view(arm, cam, G, m0, vs, k, R_ref, Qplan[k], min(a.speed, 12), survey_now.get("img"),
                           survey_now.get("mask"))
            return s["frame"], s["view_err_mm"], None
        if vs.level[k] == 0:
            moved, _, _ = move_to(arm, Qplan[k], R_ref, a.speed)
            if not moved:
                return None, np.nan, None
            s = servo_view(arm, cam, G, m0, vs, k, R_ref, Qplan[k], min(a.speed, 15), survey_now.get("img"),
                           survey_now.get("mask"))
            return s["frame"], s["view_err_mm"], None
        if not at_hi:
            img_h, _, _ = capture(vs.hi[k], hi_imgs, False)
            if img_h is None:
                return None, np.nan, None
            hi_imgs[c] = img_h
        img = descend(arm, cam, vs.dz[vs.level[k]] - vs.dz[0], R_ref, a.speed)
        if img is None:
            return None, np.nan, None
        aff = estimate_affine(hi_imgs[c], img) if c in hi_imgs else None
        move_to(arm, Qplan[vs.hi[k]], R_ref, a.speed)       # back up (part of the next transit)
        return img, np.nan, aff

    try:
        for trial in range(a.trials):
            if not a.dry_run:
                input("\n[trial %d] place the part %s at a NEW position/rotation inside the box, Enter ..."
                      % (trial, a.pose.upper()))
            cond = a.condition
            if a.task and cond is None:
                while True:
                    cond = "ok" if a.dry_run else input("  condition of this part (ok | no_ring_hex | no_ring_open | "
                                                        "no_rings | no_oring | rust, comma-combined): ").strip()
                    try:
                        TM.labels(cond); break
                    except ValueError as e:
                        print("  " + str(e))
            truth = TM.labels(cond)[0] if a.task else None
            order = list(methods); rng.shuffle(order)
            for method in order:
                if mover is not None:
                    mover.goto_reference(q_lock)
                else:
                    move_to(arm, q_lock, R_ref, 20)
                RT.event("survey", trial=trial, method=method)
                r0 = survey_servo(arm, cam, G, m0, R_ref, q_lock)
                survey = r0["frame"]; survey_now["img"] = survey
                from servo_scaled import part_mask
                survey_now["mask"] = part_mask(survey, G)
                cv2.imwrite(os.path.join(out, "img", "t%d_%s_survey.jpg" % (trial, method)), survey)
                pose_used, pose_auto = a.pose, ""
                if clf is not None:
                    fpz = PZ.features(survey, G)
                    pose_auto = clf.predict(fpz) if fpz else ""
                    if a.auto_pose and pose_auto:
                        pose_used = pose_auto
                sf = survey_features(survey, vs, lap_ref, lambda im: locate_part_scene(im, G))
                inst = {k: np.asarray(v)[None].astype(float) for k, v in sf.items()
                        if k in ("F", "qpred", "qs", "w", "fp_part", "fp_edge", "fp_glare")}
                inst.update(Q=np.zeros((1, vs.K, vs.P)), T=T[None], avail=okplan[None].copy(),
                            pose=np.array([1.0 if pose_used == "vertical" else 0.0]))
                probs = []
                if a.task:                               # task state: evidence per item instead of per patch
                    J = len(TM.ITEMS)
                    bbox = sf["bbox"] or [float(min(p[0] for p in vs.u)) - vs.half_px, float(min(p[1] for p in vs.u)) - vs.half_px,
                                          float(np.ptp(vs.u[:, 0])) + 2 * vs.half_px, float(np.ptp(vs.u[:, 1])) + 2 * vs.half_px]
                    c_s, _ = TM.crop_for_view(survey, vs, None, bbox, pose_used, px_per_mm)
                    p_s = CLF.prob(c_s) if c_s is not None else np.full(J, 0.5)
                    probs.append(p_s)
                    wj = np.zeros(J); wj[LEARN] = 1.0
                    Fk = (inst["fp_part"][0] > 0.02)[:, None] * np.ones((1, J))
                    Xf = TM.prior_features(vs, sf["w"], sf["fp_part"], sf["fp_edge"], sf["fp_glare"], inst["pose"][0])
                    inst.update(Q=np.zeros((1, vs.K, J)), qs=TM.confidence(p_s)[None], w=wj[None], F=Fk[None],
                                qpred=TM.prior_predict(BETA, Xf)[None])
                env = BatchEnv(inst, vs, a.budget_s)
                hi_imgs, t0, step, here = {}, time.time(), 0, None
                while not env.done().all():
                    RT.event("deciding", method=method, step=step)
                    tc = time.perf_counter()
                    k = int(choose(env, method, rng, W.get(method))[0])
                    ms = (time.perf_counter() - tc) * 1e3
                    RT.event("decision", method=method, step=step, view=k, infer_ms=round(ms, 3))
                    if k < 0:
                        break
                    h = vs.hi[k]
                    todo = ([h] if vs.via_hi and vs.level[k] > 0 and not env.visited[0, h] else []) + [k]
                    planned = env.i["T"][0, env.cur[0], k]
                    for kk in todo:
                        tw = time.time()
                        RT.event("moving_to_view", method=method, view=kk)
                        img, verr, aff = capture(kk, hi_imgs, here == vs.cell[kk])
                        RT.event("captured", method=method, view=kk, view_err_mm=verr)
                        here = vs.cell[kk] if img is not None else None
                        if img is None:
                            env.i["avail"][0, kk] = False
                            continue
                        if vs.level[kk] == 0 and vs.via_hi:
                            hi_imgs[vs.cell[kk]] = img
                        if a.task:
                            c_k, _ = TM.crop_for_view(img, vs, kk, bbox, pose_used, px_per_mm, aff)
                            if c_k is not None:
                                p_k = CLF.prob(c_k); probs.append(p_k)
                                q = TM.confidence(p_k); vis = np.ones(len(q), bool)
                            else:
                                q = np.zeros(len(TM.ITEMS)); vis = np.zeros(len(q), bool)
                        else:
                            q, vis = view_quality(img, kk, vs, lap_ref, aff)
                        env.i["Q"][0, kk] = q
                        name = "t%d_%s_s%02d_v%02d.jpg" % (trial, method, step, kk)
                        cv2.imwrite(os.path.join(out, "img", name), img)
                        wv.writerow([trial, a.pose, method, step, kk, vs.cell[kk], vs.level[kk], int(kk != k),
                                     planned if kk == k else "", time.time() - tw, verr,
                                     float(q[vis].mean()) if vis.any() else "", "", ms if kk == k else "", name])
                    if not env.i["avail"][0, k]:
                        continue
                    env.step(np.array([k])); step += 1
                    print("  [%-11s] step %d view %2d (cell %2d, level %d) value %.3f, planned %.1f/%.0f s"
                          % (method, step, k, vs.cell[k], vs.level[k], env.value()[0], env.t_used[0], a.budget_s))
                    if a.task and a.stop_conf > 0:
                        pf, _ = TM.fuse(np.array(probs)[:, LEARN])
                        if TM.confidence(pf).min() >= a.stop_conf:
                            print("  [%-11s] every item decided with confidence >= %.2f: stop" % (method, a.stop_conf))
                            break
                RT.event("episode_end", method=method, trial=trial, improvement=float(env.improvement()[0]))
                extra = ["", "", "", "", "", ""]
                if a.task:
                    pf, _ = TM.fuse(np.array(probs)[:, LEARN])
                    dec = TM.GOOD.copy(); dec[LEARN] = (pf > 0.5).astype(int)
                    tr_ = truth.copy(); tr_[[j for j in range(len(TM.ITEMS)) if j not in LEARN]] = TM.GOOD[
                        [j for j in range(len(TM.ITEMS)) if j not in LEARN]]
                    correct = bool(np.all(dec == tr_))
                    extra = [cond, TM.describe(tr_), TM.describe(dec), int(correct),
                             " ".join("%s=%.3f" % (TM.ITEMS[j], p) for j, p in zip(LEARN, pf)),
                             float(TM.confidence(pf).min())]
                    print("  -> %s verdict: %s | truth: %s | %s" % (method, TM.describe(dec), TM.describe(tr_),
                                                                   "CORRECT" if correct else "WRONG"))
                we.writerow([trial, a.pose, pose_auto, pose_used, method, a.budget_s, env.improvement()[0], env.v0[0], env.value()[0],
                             env.t_used[0], time.time() - t0, step] + extra); fe.flush(); fv.flush()
                print("  -> %s: improvement %.4f in %d views (%.0f s wall)" % (method, env.improvement()[0], step,
                                                                              time.time() - t0))
    finally:
        fv.close(); fe.close()
        try:
            if mover is not None:
                mover.goto_reference(q_lock)
            else:
                move_to(arm, q_lock, R_ref, 15)
        finally:
            cam.release(); arm.close()
    print("saved", out)


if __name__ == "__main__":
    main()
