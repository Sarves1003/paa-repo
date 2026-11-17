#!/usr/bin/env python3
"""Unified framework (A + B): the learned view selection acts INSIDE the safe set computed by Part A.

  python3 refcover_updated.py plan --region part      # Part A first (builds the reachability lattice)
  python3 lattice_views.py                            # build the candidate views for learning
  python3 lattice_views.py --levels 0 -10             # choose the inspection heights (mm below the reference)
  python3 lattice_views.py --orient-views --pitch-list -10 10    # + TILTED candidates the policy can choose
Use the SAME movement options as for the plan (--max-descent, --dz-step, --step, --roll-list, --pitch-list,
--yaw-list, ...): they select the lattice file written by refcover_updated.py.
Output: deploy/lattice_views.json, used automatically by collect_views.py and run_view_policy.py.
Tilted candidates (--orient-views + the rotation lists used for the plan): every requested rotation becomes an extra
candidate level at the heights given by --orient-levels (default: the lowest inspection height). Each tilted view is
aimed at its cell (the camera shifts sideways so the optical axis meets the same table point), gets a planned image
mapping (3x3 homography, pinhole model) and is verified after capture by registration to the survey. The policy then
chooses among straight AND tilted views: where to look and from which angle.

What it builds
  candidates  every table-grid cell (25) at the chosen inspection heights (default: the highest height at which all
              cells are reachable and well-conditioned, and the lowest allowed one). With --orient-views every
              requested camera rotation (roll / pitch / yaw) is added as further candidate "levels". Each candidate's
              joint solution is solved from its nearest lattice node, so all candidates lie on the reference's joint
              branch; kappa and the predicted number of markers in view are stored.
  T           (K+1) x K matrix of SAFE motion times: lattice route (Dijkstra, singularity-penalised) between the
              candidates, timed with the robot's time model; row K = start at the reference. This is the cost the
              RL policy and every baseline are charged.
  LatticeMover executes those routes on the robot (small steps only), so the policy's choices are always
              carried out along the safe routes of Part A.
"""
import argparse, json, os, time
import numpy as np
import config as C
from kinematics import fk
from ik import ik_hold
try:
    import refcover_updated as RC
except ImportError:                                  # the updated planner saved under the old name
    import refcover as RC

OUT = os.path.join(C.DEPLOY_DIR, "lattice_views.json")


def load_lattice(path):
    if not os.path.exists(path):
        raise SystemExit("no reachability lattice %s: run  python3 refcover_updated.py plan  with the same movement "
                         "options first (Part A)" % path)
    z = np.load(path, allow_pickle=True)
    return {k: (z[k].item() if z[k].shape == () else z[k]) for k in z.files if k != "key"}


def route_time(L, path, prm):
    return float(sum(prm["t0"] + np.degrees(np.max(np.abs(L["q"][a] - L["q"][b]))) / prm["omega"]
                     for a, b in zip(path[:-1], path[1:])))


def move_time(qa, qb, prm):
    return prm["t0"] + np.degrees(np.max(np.abs(np.asarray(qa) - np.asarray(qb)))) / prm["omega"]


def build(a):
    geo = RC.Geo(); lat_file = RC.lattice_path(a); L = load_lattice(lat_file)
    q_ref, R_ref, p_ref = RC.load_ref()
    tg = json.load(open(os.path.join(C.DEPLOY_DIR, "table_grid.json")))
    U = np.asarray(tg["image_px"], float); OFF = np.asarray(tg["offsets_mm"], float); P = len(U)
    t0, om = RC.time_params(); prm = dict(t0=t0, omega=om, lam=a.lam)
    dist0, _ = RC.dijkstra(L, L["start"], prm)
    reach0 = np.array([n in dist0 for n in range(len(L["ok"]))])
    orients = RC.orientation_list(a)
    xs = np.asarray(L["xs"], float); max_px = geo.rho_ref * float(xs[1] - xs[0])      # one lattice step in px

    def nearest(u, dz, o):
        sel = np.flatnonzero((np.abs(L["dz"] - dz) < 1e-6) & (L["orient"] == o) & L["ok"] & reach0)
        if not len(sel):
            return None
        d = np.linalg.norm(L["U"][sel] - u, axis=1)
        return int(sel[np.argmin(d)]) if d.min() < 2.5 * max_px else None

    heights = sorted(set(L["dz"][L["orient"] == 0].tolist()), reverse=True)
    full = [dz for dz in heights if all(nearest(U[c], dz, 0) is not None for c in range(P))]
    if a.levels:
        hs = [float(v) for v in a.levels]
    else:
        if not heights:
            raise SystemExit("no height at which every grid cell is reachable: re-run refcover_updated.py plan (see its figure)")
        hs = [heights[0], heights[-1]] if len(heights) > 1 else [heights[0]]   # unreachable cells -> ok = False
    if len(hs) < 2 and not a.single_level:
        raise SystemExit("only one inspection height (%s mm) in lattice %s: the learning pipeline needs two. Plan with "
                         "a larger --max-descent (e.g. 45) and pass the SAME options here, or --single-level."
                         % (hs, lat_file))
    levels = [(0, dz) for dz in hs]                                     # level l = (orientation, height)
    if a.orient_views:
        if len(orients) == 1:
            raise SystemExit("--orient-views needs the rotation lists the plan was made with, e.g. --pitch-list -10 10")
        ohs = [float(v) for v in a.orient_levels] if a.orient_levels else [min(hs)]
        for h in ohs:
            if not any(abs(h - x) < 1e-6 for x in heights):
                raise SystemExit("--orient-levels %g mm is not a lattice height %s" % (h, heights))
        levels += [(o, dz) for o in range(1, len(orients)) for dz in ohs]
    print("inspection heights: %s mm (heights where every cell is reachable: %s); orientations used: %s"
          % ([int(v) for v in hs], [int(v) for v in full], sorted(set(orients[o] for o, _ in levels))))

    def level_h(o, dz):
        """Planned homography: cell-relative survey pixels (uc + d) -> image of a view aimed at that cell."""
        import cv2
        Rt = RC.orient_R(orients[o]); shift = RC.tilt_shift(geo.H, dz, Rt)
        d = np.array([[x, y] for x in (-120.0, 0.0, 120.0) for y in (-90.0, 0.0, 90.0)])
        Pw = geo.px2off(geo.uc[None] + d) - geo.px2off(geo.uc[None])
        X = geo.project(Pw, shift, dz, Rt)
        Hm, _ = cv2.findHomography((geo.uc[None] + d).astype(np.float64), X.astype(np.float64), 0)
        return Hm / Hm[2, 2]
    levels_H = [level_h(o, dz).tolist() for o, dz in levels]

    views = []
    for l, (o, dz) in enumerate(levels):
        Rt = RC.orient_R(orients[o]); shift = RC.tilt_shift(geo.H, dz, Rt)
        s = geo.scale(dz) * (1.0 if o == 0 else float(Rt[2, 2]))
        for c in range(P):
            n = nearest(U[c], dz, o)
            if n is None:
                views.append(dict(k=l * P + c, cell=c, level=l, ok=False)); continue
            d_off = OFF[c] + shift                                         # re-aimed optical axis for a rotated camera
            tgt = (p_ref + np.r_[d_off, dz]) / 1000.0
            q, pe, re, good = ik_hold(tgt, L["q"][n], Rt @ R_ref, iters=1200)
            good = good and np.degrees(np.max(np.abs(q - L["q"][n]))) < 15.0
            views.append(dict(k=l * P + c, cell=c, level=l, ok=bool(good), node=n, dz=dz, orient=int(o),
                              rpy=[float(x) for x in orients[o]], offset_mm=[float(x) for x in d_off],
                              q_deg=np.degrees(q).tolist() if good else None,
                              kappa=RC.kappa(q) if good else None, scale=s, u_target=U[c].tolist(),
                              n_markers_pred=int(geo.markers_visible(U[c], geo.scale(dz), d_off, dz, Rt))))
    K = len(views)
    nodes = [v.get("node") for v in views]
    T = np.full((K + 1, K), 1e3)
    t_start = time.time()
    srcs = sorted(set(n for v, n in zip(views, nodes) if v["ok"])) + [L["start"]]
    paths = {}
    for src in srcs:
        dist, prev = RC.dijkstra(L, src, prm)
        paths[src] = prev
    for i in range(K + 1):
        if i < K and not views[i]["ok"]:
            continue
        a_node = L["start"] if i == K else nodes[i]
        qa = q_ref if i == K else np.radians(views[i]["q_deg"])
        for j in range(K):
            if not views[j]["ok"]:
                continue
            b_node = nodes[j]
            path = [a_node] if a_node == b_node else RC.path_to(paths[a_node], a_node, b_node)
            if path is None:
                continue
            qb = np.radians(views[j]["q_deg"])
            T[i, j] = (move_time(qa, L["q"][a_node], prm) * (len(path) > 1) + route_time(L, path, prm)
                       + move_time(L["q"][b_node], qb, prm))
    ok = np.array([v["ok"] for v in views])
    nm = np.array([v.get("n_markers_pred", 0) for v in views])
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), levels_dz=[dz for _, dz in levels],
                   levels_rpy=[[float(x) for x in orients[o]] for o, _ in levels], levels_H=levels_H,
                   scales=[geo.scale(dz) * (1.0 if o == 0 else float(RC.orient_R(orients[o])[2, 2])) for o, dz in levels],
                   P=P, K=K, start_node=int(L["start"]), lattice_file=lat_file, views=views, T=T.tolist(),
                   time_model=prm, via_hi=False), open(OUT, "w"))
    print("candidates: %d of %d reachable on the reference's joint branch; marker in view predicted for %d, "
          "registration for %d" % (ok.sum(), K, (ok & (nm > 0)).sum(), (ok & (nm == 0)).sum()))
    for l, (o, dz) in enumerate(levels):
        n_ok = sum(v["ok"] for v in views if v["level"] == l)
        print("  level %d: dz %+d mm, rpy %s deg: %d of %d cells reachable" % (l, dz, orients[o], n_ok, P))
    print("safe route times between candidates: median %.1f s (from the reference %.1f s); built in %.0f s"
          % (np.median(T[:K][np.ix_(ok, ok)]), np.median(T[K][ok]), time.time() - t_start))
    print("lattice used:", lat_file)
    print("saved", OUT)


def load_views():
    return json.load(open(OUT)) if os.path.exists(OUT) else None


class LatticeMover:
    """Moves the arm between candidate views (and the reference) along the lattice routes of Part A."""

    def __init__(self, arm, LV, speed=12):
        import hardware as HW
        self.HW, self.arm, self.LV, self.speed = HW, arm, LV, speed
        self.L = load_lattice(LV["lattice_file"])           # the lattice this candidate set was built from
        t0, om = RC.time_params(); self.prm = dict(t0=t0, omega=om, lam=LV["time_model"].get("lam", 2.0))
        self.node = int(LV["start_node"]); self.cache = {}

    def _route(self, a, b):
        if a not in self.cache:
            self.cache[a] = RC.dijkstra(self.L, a, self.prm)[1]
        return [a] if a == b else RC.path_to(self.cache[a], a, b)

    def _follow(self, path):
        for n in path[1:]:
            ok, _, _ = self.HW.safe_move(self.arm, np.degrees(self.L["q"][n]), self.speed)
            if not ok:
                return False
            self.node = n
        return True

    def goto_view(self, k):
        v = self.LV["views"][k]
        path = self._route(self.node, v["node"])
        if path is None:
            self.HW.LAST_REFUSAL = "no lattice route"; return False
        if len(path) > 1 and not self.HW.safe_move(self.arm, np.degrees(self.L["q"][self.node]), self.speed)[0]:
            return False
        if not self._follow(path):
            return False
        return self.HW.safe_move(self.arm, np.asarray(v["q_deg"], float), self.speed)[0]

    def goto_reference(self, q_ref_deg):
        path = self._route(self.node, int(self.LV["start_node"]))
        if path is not None:
            self.HW.safe_move(self.arm, np.degrees(self.L["q"][self.node]), self.speed)
            self._follow(path)
        return self.HW.safe_move(self.arm, np.asarray(q_ref_deg, float), self.speed)[0]


def main():
    ap = argparse.ArgumentParser()
    RC.add_planner_args(ap)                              # same movement options as the plan -> same lattice file
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="inspection heights (mm below the reference), e.g. 0 -10")
    ap.add_argument("--orient-views", dest="orient_views", action="store_true",
                    help="also create TILTED candidates for every rotation given by --roll/pitch/yaw-list")
    ap.add_argument("--orient-levels", dest="orient_levels", type=float, nargs="+", default=None,
                    help="heights (mm) of the tilted candidates; default: the lowest inspection height")
    ap.add_argument("--single-level", dest="single_level", action="store_true",
                    help="allow a candidate set with one height only")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
