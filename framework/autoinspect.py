#!/usr/bin/env python3
"""AUTONOMOUS INSPECTION from the taught reference pose: the robot finds the part, traces it with the SOGA arc
(coverage-complete, glare-complementary, certified-safe views), judges every image with the trained classifier
(rings / O-ring / rust), checks the WHOLE unrolled surface against the axisymmetric normality model (any surface
issue, also defects never seen in training), and autonomously LOOKS AGAIN, closer and more frontally, wherever the
evidence is uncertain, until every item and every suspicious region is decided or the budget is spent.

  python3 autoinspect.py run --tag good_01 --condition ok --learn-good $OPTS        # good parts: model data
  python3 axisym.py fit                                                             # after >= 3 good runs
  python3 autoinspect.py run --tag t_01 --condition no_ring_open $OPTS              # test parts (truth typed)
  python3 autoinspect.py run --tag t_02 --condition rust --rust-facing side --rolls 2 $OPTS   # 2 passes, rolled
  python3 autoinspect.py rescore                    # re-score every saved run with the current models (no robot)
  python3 autoinspect.py report                     # accuracy, confusion, AUROC, time, coverage -> results_auto/
Every pass is saved in hw_logs/autoinspect/<tag>/pass_<k>/ : survey, views, unrolled surface, coverage map,
anomaly heat map, defect overlay, pass.json. The run verdict is in hw_logs/autoinspect/<tag>/run.json.

Decision rule (all parts recorded separately so each source can be evaluated on its own):
  classifier  item j defective if the fused probability (most confident image) is on the defect side of 0.5
  surface     any connected region with Mahalanobis score above tau (learned from good parts) and area >= 4 mm2,
              or a learned ring band that is mostly not brown
  verdict     REJECT if either source finds a defect, else ACCEPT
"""
import argparse, glob, json, os, time
import numpy as np
import cv2
import config as C
import refcover_updated as RC
import soga as SG
import task_model as TM
import axisym as AX

ROOT = os.path.join(C.LOGS_DIR, "autoinspect")
OUT = "results_auto"


# ------------------------------------------------------------------ evidence from one image
def part_corners(part, pad=10.0):
    a, b = part.a, part.b
    out = []
    for sa in (-1, 1):
        for sb in (-1, 1):
            for z in (0.0, 2 * part.r):
                out.append(np.r_[part.c, 0.0] + sa * (part.L / 2 + pad) * a + sb * (part.r + pad) * b + np.r_[0, 0, z])
    return np.array(out)


def crop_part(geo, part, shot):
    u, vz = SG.project(geo, part_corners(part), shot["C"], shot["Rt"])
    if not np.all(np.isfinite(u)):
        return None
    x0, y0 = u.min(0); x1, y1 = u.max(0)
    crop, frac = TM.part_crop(shot["img"], np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    return crop


class Judge:
    """Classifier (deploy/part_clf.npz) + axisymmetric surface model (deploy/axisym_model.npz); either may be absent."""
    def __init__(self):
        self.clf = TM.PartClassifier() if os.path.exists("deploy/part_clf.npz") else None
        self.ax = dict(np.load(AX.MODEL)) if os.path.exists(AX.MODEL) else None
        print("judge: classifier %s, surface model %s" % ("yes" if self.clf else "NO (run Colab task_colab.py classify)",
                                                         ("yes (%d good parts, tau %.1f)" % (int(self.ax["n_good"]), float(self.ax["tau"])))
                                                         if self.ax is not None else "NO (record good parts, axisym.py fit)"))

    def probs(self, geo, part, shot):
        if self.clf is None:
            return None
        c = crop_part(geo, part, shot)
        return None if c is None else self.clf.prob(c).tolist()


def items_from(probs_list):
    P = [p for p in probs_list if p is not None]
    if not P:
        return None
    pf, who = TM.fuse(P)
    y = (pf > 0.5).astype(int)
    return dict(p=pf.tolist(), y=y.tolist(), conf=TM.confidence(pf).tolist(), decided_by=who.tolist(),
                verdict=TM.describe(y))


# ------------------------------------------------------------------ autonomous re-look planning
def relook_view(part, geo, s_mm, phi_deg, dz_close, arc_max):
    """Closer, frontal view of surface point (s, phi): camera on the surface normal side, re-aimed by a rotation
    about the part axis, at the lowest allowed height."""
    al = float(np.clip(phi_deg, -arc_max, arc_max)); A = np.radians(al)
    n = np.sin(np.radians(phi_deg)) * part.b + np.cos(np.radians(phi_deg)) * SG.EZ
    X = part.c3 + s_mm * part.a + part.r * n
    dirc = np.sin(A) * part.b + np.cos(A) * SG.EZ
    d = (geo.H + dz_close - X[2]) / max(dirc[2], 0.2)
    return dict(name="relook_s%+.0f_p%+.0f" % (s_mm, phi_deg), alpha=al, C=X + d * dirc, Rt=SG.rot(part.a, A),
                target=[float(s_mm), float(phi_deg)])


def uncertainty_weights(PH, good_upper_valid, d2=None, tau=None, tau_lo=None):
    """Weight of every chart patch for the next view: unseen/glare-only patches of the upper surface 1, borderline
    anomaly scores (between the 99 % and 99.9 % good levels) 3."""
    up = np.abs(PH) <= np.pi / 2
    w = (up & ~good_upper_valid).astype(float)
    if d2 is not None and tau is not None:
        w += 3.0 * ((d2 > tau_lo) & (d2 <= tau))
    return w


# ------------------------------------------------------------------ one pass (one placement / roll of the part)
def run_pass(a, k, arm, cam, reach, judge, out_root, safe_move, wait_settled, HW):
    from servo import servo_to_view
    geo = reach.geo
    out = os.path.join(out_root, "pass_%d" % k); os.makedirs(out, exist_ok=True)
    if a.learn_good:
        open(os.path.join(out, "learn_good"), "w").write("good part pass\n")
    t0 = time.time()
    safe_move(arm, np.degrees(reach.q_ref), a.speed); wait_settled(arm)
    m0 = {int(kk) if str(kk).isdigit() else kk: np.array(v) for kk, v in geo.G["markers0"].items()}
    sv = servo_to_view(arm, cam, geo.A, geo.uc, m0, geo.uc, reach.R_ref, safe_move, np.degrees(reach.q_ref), speed=10)
    survey = sv["frame"]; cv2.imwrite(os.path.join(out, "survey.jpg"), survey)
    # 1) geometry, lights and highlight width from the survey
    part, alpha_L, w_h, info = SG.estimate(geo, survey, axis_deg=a.axis_deg, part_r=a.part_r, part_L=a.part_L)
    SG.draw_estimate(geo, survey, part, os.path.join(out, "part_estimate.jpg"))
    rho_min = a.rho_min or 0.9 * geo.f / geo.H
    start = np.r_[0.0, 0.0, geo.H]
    chart = part.chart(); PH = chart[1]
    print("[pass %d] part r %.1f mm, L %.1f mm (%s; image estimate r %.1f, L %.1f), axis (%.2f, %.2f) [%s]; lights %s deg, highlight widths %s deg"
          % (k, part.r, part.L, info["size_from"], info["r_from_image"], info["L_from_image"], *part.a[:2], info["axis_from"], [round(v) for v in info["lights_alpha_deg"]],
             [round(v) for v in info["highlight_width_deg"]]))
    # 2) SOGA coverage set (certified)
    cands, why = SG.feasible(reach, SG.arc_views(part, geo, a.dz, np.arange(-a.arc_max, a.arc_max + 1e-6, a.arc_step)))
    if not cands:
        raise SystemExit("no certified arc view (reasons %s): check the reference pose / fence" % why)
    M = SG.model_masks(geo, part, cands, alpha_L, w_h, a.psi_max, rho_min, chart)
    plan, cov_pred = SG.select_soga(geo, part, cands, M, PH, start, a.n_max)
    print("[pass %d] SOGA: %d views at alpha %s deg (certified arc %+.0f..%+.0f deg); predicted glare-free upper %.1f%%"
          % (k, len(plan), [round(v["alpha"]) for v in plan], min(c["alpha"] for c in cands),
             max(c["alpha"] for c in cands), 100 * cov_pred))
    shots = [dict(name="survey", C=start, Rt=np.eye(3), img=survey, stage="survey")]
    probs = [judge.probs(geo, part, shots[0])]
    log = []
    q_prev = reach.q_ref.copy()

    def capture(v, stage):
        nonlocal q_prev
        q, reason, way = reach.solve(v["C"], v["Rt"], seed=q_prev)
        if q is None and not np.allclose(q_prev, reach.q_ref):
            # not reachable in one hop from the previous view: return to the reference, continue from there
            safe_move(arm, np.degrees(reach.q_ref), a.speed); wait_settled(arm); q_prev = reach.q_ref.copy()
            q, reason, way = reach.solve(v["C"], v["Rt"], seed=q_prev)
        if q is None:
            print("    %-22s refused: %s" % (v["name"], reason)); return False
        ts = time.time()
        for w in way[1::2] + [way[-1]]:
            if not safe_move(arm, np.degrees(w), a.speed)[0]:
                print("    %-22s stopped by the hardware guard: %s" % (v["name"], HW.LAST_REFUSAL)); return False
        wait_settled(arm)
        oa = v["Rt"][:, 2]; xy_t = v["C"][:2] - v["C"][2] / oa[2] * oa[:2]
        r = RC.capture_registered(cam, arm, survey, np.asarray(geo.o2p(xy_t), float), geo,
                                  geo.scale(v["C"][2] - geo.H), speed=a.speed)
        Cm, Rm = SG.measured_pose(arm, reach, xy_t)
        if abs(part.alpha(Cm) - np.radians(v["alpha"])) > np.radians(a.max_alpha_err) or r["view_err_mm"] > 10.0:
            # the image registration pulled the camera away from the planned view (bad matches): do not trust it,
            # go back to the certified planned pose and take the image there
            print("    %-22s registration rejected (alpha %+.1f vs planned %+.1f deg, %.1f mm): planned pose used"
                  % (v["name"], np.degrees(part.alpha(Cm)), v["alpha"], r["view_err_mm"]))
            safe_move(arm, np.degrees(q), a.speed); wait_settled(arm)
            for _ in range(2):
                cam.grab()
            r = dict(r, frame=cam.grab(), verify="planned pose (registration rejected)", view_err_mm=float("nan"))
            Cm, Rm = np.asarray(v["C"], float), np.asarray(v["Rt"], float)
        nm = "%02d_%s.jpg" % (len(shots), v["name"]); cv2.imwrite(os.path.join(out, nm), r["frame"])
        sh = dict(name=v["name"], C=Cm, Rt=Rm, img=r["frame"], stage=stage); shots.append(sh)
        p = judge.probs(geo, part, sh); probs.append(p)
        q_prev = np.radians(arm.read_deg())
        log.append(dict(view=v["name"], stage=stage, C=np.asarray(Cm, float).tolist(), Rt=np.asarray(Rm, float).tolist(),
                        alpha=float(np.degrees(part.alpha(Cm))), alpha_plan=v["alpha"], img=nm, aim_open_mm=r["view_err_open_mm"],
                        aim_mm=r["view_err_mm"], verify=r["verify"], s=time.time() - ts, probs=p,
                        target=v.get("target")))
        print("    %-22s alpha %+5.1f deg, aim %.1f -> %.1f mm, %.1f s%s" % (v["name"], np.degrees(part.alpha(Cm)), r["view_err_open_mm"],
              r["view_err_mm"], time.time() - ts, "" if p is None else ", P(ring_hex,ring_open,oring,rust)=%s"
              % [round(x, 2) for x in p]))
        return True

    for v in plan:
        capture(v, "coverage")
    used = {v["name"] for v in plan}

    def evaluate():
        tex, lab, good, seen, st, PHu = SG.unwrap(geo, part, shots, a.psi_max, rho_min)
        sc = None
        if judge.ax is not None:
            sc = AX.score(tex, good, judge.ax, part.L / AX.NS, part.r)
        return tex, lab, good, seen, st, PHu, sc

    tex, lab, good, seen, st, PHu, sc = evaluate()
    # 3) AI re-look loop: uncertain items (classifier) or suspicious regions (surface model) -> closer views
    for it in range(a.relooks):
        items = items_from(probs)
        unsure_items = [] if items is None else [TM.ITEMS[j] for j in range(len(TM.ITEMS)) if items["conf"][j] < a.stop_conf]
        regions = [] if sc is None else sc["regions"]
        if not unsure_items and not regions and (st["glarefree_upper"] >= a.cov_goal):
            break
        cand_views = []
        for rg in regions[:3]:                           # verify every suspected defect from closer and more frontally
            s_bin = rg["s_mm"] / (part.L / AX.NS); cc = (AX.NS - 1) / 2.0
            s_b = cc + ((s_bin - sc["shift"]) - cc) / sc.get("scale", 1.0); s_b = (AX.NS - 1 - s_b) if sc["flip"] else s_b
            s_mm = (s_b + 0.5) / AX.NS * part.L - part.L / 2
            v = relook_view(part, geo, s_mm, rg["phi_deg"], -a.max_descent, a.arc_max)
            if v["name"] not in used:
                cand_views.append(v)
        cand_views += [c for c in cands if c["name"] not in used]
        if not cand_views:
            break
        # choose by uncertainty-weighted new glare-free coverage per second of motion (chart of this pass)
        good_c = np.zeros(PH.shape, bool)
        Ms = SG.model_masks(geo, part, [dict(C=s["C"], Rt=s["Rt"]) for s in shots], alpha_L, w_h, a.psi_max, rho_min, chart)
        for m_ in Ms:
            good_c |= m_[1]
        wgt = uncertainty_weights(PH, good_c)
        if unsure_items:
            wgt += 0.5 * (np.abs(PH) <= np.pi / 2)     # any new frontal evidence helps an undecided item
        Mc = SG.model_masks(geo, part, cand_views, alpha_L, w_h, a.psi_max, rho_min, chart)
        cur = shots[-1]["C"]
        score_v = []
        for v, m_ in zip(cand_views, Mc):
            gain = float((wgt * (m_[1] & ~good_c)).sum()) + (50.0 if v["name"].startswith("relook") else 0.0)
            tm = np.linalg.norm(np.asarray(v["C"]) - np.asarray(cur)) / 25.0 + 3.0
            score_v.append(gain / tm)
        order = np.argsort(score_v)[::-1]
        done = False
        for i in order[:4]:
            if score_v[i] <= 0:
                break
            used.add(cand_views[i]["name"])
            if capture(cand_views[i], "relook"):
                done = True; break
        if not done:
            break
        tex, lab, good, seen, st, PHu, sc = evaluate()
    safe_move(arm, np.degrees(reach.q_ref), a.speed)
    wall = time.time() - t0
    # 4) outputs
    up = np.abs(PHu[:, 0]) <= np.pi / 2
    SG.save_unwrap(out, tex, lab, good, seen, part, PHu, [s["name"] for s in shots])
    cv2.imwrite(os.path.join(out, "unrolled_full.png"), tex)
    cv2.imwrite(os.path.join(out, "unrolled_valid.png"), (good * 255).astype(np.uint8))
    anomaly = None
    if sc is not None:
        h = AX.heat(sc["d2"], sc["valid"], sc["tau"])
        cv2.imwrite(os.path.join(out, "anomaly_heat.png"), cv2.resize(h, (h.shape[1] * 4, h.shape[0] * 2),
                                                                     interpolation=cv2.INTER_NEAREST))
        anomaly = {kk: vv for kk, vv in sc.items() if kk not in ("d2", "valid")}
        draw_defects(out, tex, sc, part)
    items = items_from(probs)
    S = dict(pass_k=k, part=info, lights_deg=info["lights_alpha_deg"], n_views=len(log), wall_s=wall,
             path_mm=SG.path_mm([dict(C=s["C"]) for s in shots[1:]], np.r_[0.0, 0.0, geo.H]),
             coverage=st, coverage_predicted=cov_pred, items=items, anomaly=anomaly, views=log,
             learn_good=bool(a.learn_good), rho_min=rho_min, survey_probs=probs[0])
    json.dump(S, open(os.path.join(out, "pass.json"), "w"), indent=1, default=float)
    print("[pass %d] %d views, %.0f s; glare-free coverage of the upper surface %.1f%% (seen %.1f%%)%s%s"
          % (k, len(log), wall, 100 * st["glarefree_upper"], 100 * st["seen_upper"],
             "" if items is None else "; classifier: %s" % items["verdict"],
             "" if anomaly is None else "; surface: %d region(s), rings %s" % (len(anomaly["regions"]),
                                                                              [r_["present"] for r_ in anomaly["rings"]])))
    return S


def draw_defects(out, tex, sc, part):
    t = cv2.resize(tex, (tex.shape[1] * 3, tex.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    sx = tex.shape[1] / float(AX.NS)
    for r in sc["regions"]:
        s_bin = r["s_mm"] / (part.L / AX.NS); cc = (AX.NS - 1) / 2.0
        s_b = cc + ((s_bin - sc["shift"]) - cc) / sc.get("scale", 1.0); s_b = (AX.NS - 1 - s_b) if sc["flip"] else s_b
        x = int(s_b * sx * 3); y = int((r["phi_deg"] + 180) / 360.0 * tex.shape[0] * 3)
        rad = int(max(8, 3 * np.sqrt(r["area_mm2"]) / (part.L / tex.shape[1])))
        cv2.circle(t, (x, y), rad, (0, 0, 255), 2)
        cv2.putText(t, r["kind"].split(" ")[0], (x + rad + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out, "defects_overlay.png"), t)


# ------------------------------------------------------------------ run verdict
def decide(passes, judge_has_clf=True):
    items = None
    allp = []
    for S in passes:
        if S.get("items"):
            allp.append(np.asarray(S["items"]["p"]))
    if allp:
        pf, _ = TM.fuse(allp); y = (pf > 0.5).astype(int)
        items = dict(p=pf.tolist(), y=y.tolist(), verdict=TM.describe(y))
    regions = [dict(r, pass_k=S["pass_k"]) for S in passes if S.get("anomaly") for r in S["anomaly"]["regions"]]
    ring_missing = [S["pass_k"] for S in passes if S.get("anomaly")
                    for r in S["anomaly"]["rings"] if r.get("present") is False]
    clf_reject = items is not None and TM.is_defective(items["y"])
    surf_reject = bool(regions) or bool(ring_missing)
    reasons = []
    if clf_reject:
        reasons.append(items["verdict"].replace("REJECT: ", "classifier: "))
    for r in regions:
        reasons.append("surface: %s at s %.0f mm, phi %+.0f deg, %.0f mm2 (pass %d)" % (r["kind"], r["s_mm"], r["phi_deg"],
                                                                                    r["area_mm2"], r["pass_k"]))
    if ring_missing:
        reasons.append("surface: a learned ring band is not brown (pass %s)" % ring_missing)
    max_score = max([S["anomaly"]["max_score"] / max(S["anomaly"]["tau"], 1e-6) for S in passes if S.get("anomaly")] or [np.nan])
    return dict(verdict="REJECT" if (clf_reject or surf_reject) else "ACCEPT", reasons=reasons, items=items,
                classifier_reject=clf_reject if items is not None else None,
                surface_reject=surf_reject if any(S.get("anomaly") for S in passes) else None,
                anomaly_max_over_tau=float(max_score))


def cmd_run(a):
    import hardware as HW
    from hardware import make_arm, make_camera, safe_move, wait_settled
    from fence import auto_fence
    truth_y, truth_m = TM.labels(a.condition, a.rust_facing)
    reach = SG.Reach(a); judge = Judge()
    tag = a.tag or time.strftime("auto_%Y%m%d_%H%M%S")
    out = os.path.join(ROOT, tag); os.makedirs(out, exist_ok=True)
    if not a.no_trace:                                   # robot motion log: commands, ~25 Hz joints, servo status
        import robot_trace as RT; RT.install(os.path.join(out, "trace"), stream_hz=25.0, status_hz=5.0)
    HW.set_fence(auto_fence(), reach.p_ref)
    arm = make_arm(a.dry_run, np.degrees(reach.q_ref)); arm.send_deg(np.degrees(reach.q_ref), 15); wait_settled(arm)
    cam = make_camera(a.dry_run, arm, a.ref_img)
    passes = []
    for k in range(a.rolls):
        if not a.no_trace:
            RT.event("pass_%d" % k)
        if k > 0 and not a.dry_run:
            input("\n[pass %d] ROLL the part about its axis by about %d deg (keep it inside the box), hands out, Enter "
                  % (k, int(360 / a.rolls)))
        passes.append(run_pass(a, k, arm, cam, reach, judge, out, safe_move, wait_settled, HW))
    D = decide(passes)
    R = dict(tag=tag, condition=a.condition, rust_facing=a.rust_facing, truth=TM.describe(truth_y),
             truth_y=truth_y.tolist(), truth_defective=TM.is_defective(truth_y), learn_good=bool(a.learn_good),
             passes=len(passes), wall_s=float(sum(p["wall_s"] for p in passes)),
             n_views=int(sum(p["n_views"] for p in passes)),
             glarefree_upper=[p["coverage"]["glarefree_upper"] for p in passes], **D)
    R["correct"] = (R["verdict"] == "REJECT") == R["truth_defective"]
    json.dump(R, open(os.path.join(out, "run.json"), "w"), indent=1, default=float)
    print("\n%s  |  truth: %s  |  %s" % (R["verdict"], R["truth"], "CORRECT" if R["correct"] else "WRONG"))
    for r in R["reasons"]:
        print("   - " + r)
    print("saved %s" % out)


def cmd_reunwrap(a):
    """No motion: re-estimate the part from each saved survey and rebuild the unrolled surfaces from the saved images
    and camera poses (after a change to the geometry code). Passes recorded before poses were stored are listed."""
    reach = SG.Reach(a); geo = reach.geo; old = []
    for pj in sorted(glob.glob(os.path.join(ROOT, "*", "pass_*", "pass.json"))):
        pd = os.path.dirname(pj); S = json.load(open(pj))
        if any("C" not in v for v in S["views"]):
            old.append(os.path.relpath(pd, ROOT)); continue
        sv = cv2.imread(os.path.join(pd, "survey.jpg"))
        part, alpha_L, w_h, info = SG.estimate(geo, sv, axis_deg=a.axis_deg, part_r=a.part_r, part_L=a.part_L)
        SG.draw_estimate(geo, sv, part, os.path.join(pd, "part_estimate.jpg"))
        shots = [dict(name="survey", C=np.r_[0.0, 0.0, geo.H], Rt=np.eye(3), img=sv)]
        for v in S["views"]:
            shots.append(dict(name=v["view"], C=np.asarray(v["C"], float), Rt=np.asarray(v["Rt"], float),
                              img=cv2.imread(os.path.join(pd, v["img"]))))
        rho_min = S.get("rho_min") or 0.9 * geo.f / geo.H
        tex, lab, good, seen, st, PHu = SG.unwrap(geo, part, shots, a.psi_max, rho_min)
        SG.save_unwrap(pd, tex, lab, good, seen, part, PHu, [x["name"] for x in shots])
        cv2.imwrite(os.path.join(pd, "unrolled_full.png"), tex)
        cv2.imwrite(os.path.join(pd, "unrolled_valid.png"), (good * 255).astype(np.uint8))
        S.update(part=info, coverage=st); json.dump(S, open(pj, "w"), indent=1, default=float)
        print("%-24s axis [%s] r %.1f L %.1f, glare-free upper %.1f%%" % (os.path.relpath(pd, ROOT), info["axis_from"],
                                                                         part.r, part.L, 100 * st["glarefree_upper"]))
    if old:
        print("recorded before camera poses were saved (re-record these runs): %s" % ", ".join(old))


def cmd_rescore(a):
    judge = Judge()
    geo = RC.Geo()
    for rd in sorted(glob.glob(os.path.join(ROOT, "*", "run.json"))):
        R = json.load(open(rd)); d = os.path.dirname(rd); passes = []
        for pj in sorted(glob.glob(os.path.join(d, "pass_*", "pass.json"))):
            S = json.load(open(pj)); pd = os.path.dirname(pj)
            if judge.ax is not None:
                s = AX.load_surface(pd)
                if s is not None:
                    P = S["part"]["part"]
                    sc = AX.score(s[0], s[1], judge.ax, P["L"] / AX.NS, P["r"])
                    S["anomaly"] = {kk: vv for kk, vv in sc.items() if kk not in ("d2", "valid")}
                    h = AX.heat(sc["d2"], sc["valid"], sc["tau"])
                    cv2.imwrite(os.path.join(pd, "anomaly_heat.png"), cv2.resize(h, (h.shape[1] * 4, h.shape[0] * 2),
                                                                                interpolation=cv2.INTER_NEAREST))
            if judge.clf is not None:                    # re-classify every saved image with the current classifier
                part = SG.Part(S["part"]["part"]["c"], S["part"]["part"]["a"], S["part"]["part"]["r"], S["part"]["part"]["L"])
                pr = []
                sv = cv2.imread(os.path.join(pd, "survey.jpg"))
                pr.append(judge.probs(geo, part, dict(C=np.r_[0.0, 0.0, geo.H], Rt=np.eye(3), img=sv)))
                pr += [v.get("probs") for v in S["views"]]       # view probabilities as logged during the run
                S["items"] = items_from(pr)
            json.dump(S, open(pj, "w"), indent=1, default=float); passes.append(S)
        D = decide(passes)
        R.update(D); R["correct"] = (R["verdict"] == "REJECT") == R["truth_defective"]
        json.dump(R, open(rd, "w"), indent=1, default=float)
        print("%-14s %-7s truth %-34s %s" % (R["tag"], R["verdict"], R["truth"], "ok" if R["correct"] else "WRONG"))


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def cmd_report(a):
    import csv
    rows = []
    for rd in sorted(glob.glob(os.path.join(ROOT, "*", "run.json"))):
        R = json.load(open(rd))
        if R.get("learn_good") and not a.include_good:
            continue
        rows.append(R)
    if not rows:
        raise SystemExit("no test runs in %s (good-part runs are excluded; --include-good to list them)" % ROOT)
    os.makedirs(OUT, exist_ok=True)
    keys = ["tag", "condition", "rust_facing", "truth", "verdict", "correct", "classifier_reject", "surface_reject",
            "anomaly_max_over_tau", "passes", "n_views", "wall_s"]
    with open(os.path.join(OUT, "auto_runs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys + ["glarefree_upper_mean", "reasons"])
        for R in rows:
            w.writerow([R.get(k) for k in keys] + [np.mean(R["glarefree_upper"]), " | ".join(R["reasons"])])
    def rate(sel, key):
        v = [R for R in rows if sel(R) and R.get(key) is not None]
        return (float(np.mean([(R[key] if key == "correct" else (R[key] == R["truth_defective"])) for R in v])) if v else None), len(v)
    tp = sum(R["verdict"] == "REJECT" and R["truth_defective"] for R in rows)
    tn = sum(R["verdict"] == "ACCEPT" and not R["truth_defective"] for R in rows)
    fp = sum(R["verdict"] == "REJECT" and not R["truth_defective"] for R in rows)
    fn = sum(R["verdict"] == "ACCEPT" and R["truth_defective"] for R in rows)
    by_cond = {}
    for R in rows:
        by_cond.setdefault(R["condition"], []).append(R["correct"])
    S = dict(n=len(rows), accuracy=rate(lambda R: True, "correct")[0], confusion=dict(TP=tp, TN=tn, FP=fp, FN=fn),
             recall_defective=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
             classifier_only_accuracy=rate(lambda R: True, "classifier_reject")[0],
             surface_only_accuracy=rate(lambda R: True, "surface_reject")[0],
             anomaly_auroc=auc([R["anomaly_max_over_tau"] for R in rows if R["truth_defective"] and np.isfinite(R["anomaly_max_over_tau"])],
                               [R["anomaly_max_over_tau"] for R in rows if not R["truth_defective"] and np.isfinite(R["anomaly_max_over_tau"])]),
             accuracy_by_condition={k: float(np.mean(v)) for k, v in by_cond.items()},
             n_by_condition={k: len(v) for k, v in by_cond.items()},
             time_s_median=float(np.median([R["wall_s"] for R in rows])),
             views_median=float(np.median([R["n_views"] for R in rows])),
             glarefree_upper_median=float(np.median([np.mean(R["glarefree_upper"]) for R in rows])))
    json.dump(S, open(os.path.join(OUT, "auto_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float))
    print("saved %s/auto_runs.csv and auto_summary.json" % OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "rescore", "report", "reunwrap"])
    RC.add_planner_args(ap)
    ap.add_argument("--condition", default="ok", help="TRUE state of the part: ok | no_ring_hex | no_ring_open | "
                                                       "no_rings | no_oring | rust (comma-combined)")
    ap.add_argument("--rust-facing", dest="rust_facing", choices=["none", "up", "side", "down"], default="none")
    ap.add_argument("--learn-good", dest="learn_good", action="store_true", help="this is a GOOD part: use it to fit the surface model")
    ap.add_argument("--no-trace", dest="no_trace", action="store_true", help="do not record the robot motion log")
    ap.add_argument("--rolls", type=int, default=1, help="passes; you roll the part between them (2-3 for 360 deg)")
    ap.add_argument("--relooks", type=int, default=3, help="most autonomous extra views per pass")
    ap.add_argument("--stop-conf", dest="stop_conf", type=float, default=0.9)
    ap.add_argument("--cov-goal", dest="cov_goal", type=float, default=0.6, help="glare-free upper coverage goal")
    ap.add_argument("--dz", type=float, default=-10.0)
    ap.add_argument("--arc-max", dest="arc_max", type=float, default=30.0)
    ap.add_argument("--arc-step", dest="arc_step", type=float, default=2.0)
    ap.add_argument("--n-max", dest="n_max", type=int, default=4)
    ap.add_argument("--psi-max", dest="psi_max", type=float, default=60.0)
    ap.add_argument("--rho-min", dest="rho_min", type=float, default=0.0)
    ap.add_argument("--axis-deg", dest="axis_deg", type=float, default=None)
    ap.add_argument("--part-r", dest="part_r", type=float, default=None, help="part radius from a ruler (mm)")
    ap.add_argument("--part-L", dest="part_L", type=float, default=None, help="part length from a ruler (mm)")
    ap.add_argument("--max-alpha-err", dest="max_alpha_err", type=float, default=6.0,
                    help="reject the registration correction if it moves the view angle more than this (deg)")
    ap.add_argument("--include-good", dest="include_good", action="store_true")
    ap.add_argument("--inspect-max-descent", dest="inspect_max_descent", type=float, default=10.0,
                    help="lowest the camera may go below the reference in this script (mm); caps --max-descent")
    a = ap.parse_args()
    a.max_descent = min(float(a.max_descent), float(a.inspect_max_descent))   # opts.sh says 30 for the lattice files
    if a.dz < -a.max_descent:
        a.dz = -a.max_descent
    if a.cmd == "run":
        TM.parse_condition(a.condition)
        if "rust" in a.condition and a.rust_facing == "none":
            raise SystemExit("--condition contains rust: give --rust-facing up, side or down")
    {"run": cmd_run, "rescore": cmd_rescore, "report": cmd_report, "reunwrap": cmd_reunwrap}[a.cmd](a)


if __name__ == "__main__":
    main()
