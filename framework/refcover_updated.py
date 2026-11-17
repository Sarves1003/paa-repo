#!/usr/bin/env python3
"""Single-reference coverage: teach ONE pose, the robot plans and executes complete, verified coverage of the box.

  python3 refcover.py plan                      # no motion: lattice, coverage set, path, comparison, figures
  python3 refcover.py plan --region part        # cover only the part's bounding box (from the survey)
  python3 refcover.py run  [--tag horizontal_01]  # execute the plan: verified views + mosaic + log
  python3 refcover.py report                    # statistics over all runs -> results_views/refcover_summary.json
Options (plan): --step 10 (lattice spacing in table mm), --max-descent 10 (camera never goes more than this many mm
                below the reference), --dz-step 5, --yaw-list (optional extra camera rotations about the vertical, deg),
                --roll-list / --pitch-list / --yaw-list (optional camera rotations in deg, one axis at a time, e.g.
                --roll-list -6 6 --pitch-list -6 6 --yaw-list -8 8; limits --max-tilt 10, --max-yaw 15),
                --mag 1.6 / --res <px/mm> (required resolution), --kappa-min 0.04, --lam 2.0
--central 0.25 (a table point counts as covered only inside this central fraction of an image: smaller -> more
sideways views that trace the part), --tilt-captures (at every coverage view also capture the requested rotations)
Movement choices made by the planner: sideways (x, y) positions at every height, heights from the reference down to
--max-descent only (never up, never deeper), and, only if given, small camera rotations (roll, pitch, yaw about the
world axes, applied to the reference orientation; the optical axis is re-aimed at the same table point). Without
rotation options the camera keeps its reference orientation and stays within 10 mm of the reference height.
Rotated views are planned with a pinhole model (focal length from the measured camera height and the image Jacobian)
and are captured at the planned pose (the marker servo assumes the reference orientation).

Method (all relative to the reference pose q_ref taught with live_align.py):
 1. View space. A view is a camera displacement v = (dx, dy, dz) from the reference with the camera orientation
    held. Its target is defined in the REFERENCE image: the table point t_v that must appear at the image centre.
    Table <-> reference image <-> robot offset use the drawn box (homography) and the measured image Jacobian A.
 2. Branch-continuous reachability lattice. Views on a lattice over the box and several heights are solved by
    breadth-first continuation from the reference: each node's IK is warm-started from an already solved
    neighbour and accepted only if the joint change to it is small, so all solutions stay on one joint branch.
    Each node gets the conditioning kappa(v) = sigma_min / sigma_max of the 6x6 Jacobian (position scaled by
    L = 0.1 m), and a joint-limit margin. Nodes with kappa < kappa_min are excluded (near-singular).
 3. Coverage. Point p of the region is covered by view v if it projects into the central 80 % of v's image and
    v's resolution rho(v) = rho_ref * s(v) >= rho_min, s(v) = H / (H + dz). Choose the view set S by greedy
    weighted set cover: argmax_v |new points covered by v| * kappa(v)^alpha / c(v)   (ln-n approximation).
 4. Path. Edge cost between neighbouring lattice nodes: t_ij + lam * (1/kappa_i + 1/kappa_j) / 2,
    t_ij = t0 + max|dq| / omega. Views are connected by Dijkstra shortest paths through the lattice (never
    through excluded nodes); the visiting order is nearest neighbour + 2-opt, starting and ending at q_ref.
 5. Verified execution. At each view the arm servos until the reference-defined target is centred
    (scale-aware, scale measured from the markers), then captures. All views are mapped back into table
    coordinates and blended into one mosaic of the box at the highest planned resolution.
"""
import argparse, glob, heapq, json, os, time
import numpy as np
import cv2
import config as C
from kinematics import fk
from ik import ik_hold

PLAN = os.path.join(C.DEPLOY_DIR, "refcover_plan.json")


def Rx(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def Ry(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def orient_R(rpy):
    """World-frame rotation (roll about x, pitch about y, yaw about z, deg) applied to the reference orientation."""
    return Rz(rpy[2]) @ Ry(rpy[1]) @ Rx(rpy[0])


def orientation_list(a):
    """Reference orientation first, then every requested single-axis rotation (roll, pitch, yaw)."""
    out = [(0.0, 0.0, 0.0)]
    for ax, vals, lim, nm in ((0, a.roll_list, a.max_tilt, "roll"), (1, a.pitch_list, a.max_tilt, "pitch"),
                              (2, a.yaw_list, a.max_yaw, "yaw")):
        for v in sorted(set(round(float(x), 3) for x in (vals or []))):
            if v == 0.0:
                continue
            if abs(v) > lim:
                raise SystemExit("%s %.1f deg exceeds its limit of %.1f deg (--max-tilt / --max-yaw)" % (nm, v, lim))
            r = [0.0, 0.0, 0.0]; r[ax] = v; out.append(tuple(r))
    return out


def tilt_shift(H, dz, Rt):
    """Extra horizontal camera offset (mm) that keeps the re-aimed optical axis on the same table point."""
    return (H + dz) / Rt[2, 2] * Rt[:2, 2]


def _tag(a):
    t = "d%d" % int(round(float(a.max_descent)))
    for nm, vals in (("r", a.roll_list), ("p", a.pitch_list), ("y", a.yaw_list)):
        vs = sorted(set(float(v) for v in (vals or []) if float(v) != 0.0))
        if vs:
            t += "_%s%s" % (nm, "_".join("%g" % v for v in vs))
    return t


def plan_path(a):
    """Plans for different movement choices are saved under different names, so that exploring never overwrites the
    plan that 'run' executes with the same options. deploy/refcover_plan.json and deploy/refcover_lattice.npz are
    only MIRRORS of the latest plan, kept for export_for_colab.py / colab_analysis.py; 'run' never reads them."""
    return os.path.join(C.DEPLOY_DIR, "refcover_plan_%s.json" % _tag(a))


def lattice_path(a):
    return os.path.join(C.DEPLOY_DIR, "refcover_lattice_%s.npz" % _tag(a))


OUTF = os.path.join("paper", "figs")
LIM = np.radians(168.0)


# ------------------------------------------------------------------ kinematic quality
def kappa(q, L=0.1, eps=1e-6):
    """Inverse condition number of the 6x6 geometric Jacobian (0 = singular, 1 = isotropic)."""
    T = fk(q); p, R = T[:3, 3], T[:3, :3]
    J = np.zeros((6, 6))
    for i in range(6):
        q2 = np.array(q, float); q2[i] += eps
        T2 = fk(q2); dR = T2[:3, :3] @ R.T
        J[:3, i] = (T2[:3, 3] - p) / eps / L
        J[3:, i] = 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / eps
    s = np.linalg.svd(J, compute_uv=False)
    return float(s[-1] / s[0])


def limit_margin_deg(q):
    return float(np.degrees(np.min(LIM - np.abs(q))))


# ------------------------------------------------------------------ geometry: table <-> reference image <-> robot
class Geo:
    def __init__(self):
        from perception import load_grounding
        self.G = load_grounding()
        tg = json.load(open(os.path.join(C.DEPLOY_DIR, "table_grid.json")))
        self.box_mm = np.array(tg["box_mm"], float)
        box = np.asarray(tg["box_px"], np.float32)
        self.Hb, _ = cv2.findHomography(box, np.array([[0, 0], [self.box_mm[0], 0], self.box_mm,
                                                       [0, self.box_mm[1]]], np.float32))     # px -> table mm
        self.Hi = np.linalg.inv(self.Hb)
        self.A = np.asarray(self.G["A"], float); self.uc = np.asarray(self.G["uc"], float)
        self.rho_ref = float(tg["box_px_per_mm"])
        self.H = float(getattr(C, "VIEW_CAM_HEIGHT_MM", 0) or 0)
        if self.H <= 0:
            raise SystemExit("set VIEW_CAM_HEIGHT_MM in config.py (ruler: camera lens -> table at the reference pose)")
        from perception import offset_to_px
        self.o2p = lambda d: offset_to_px(self.G, d)
        self.markers0 = [np.asarray(c, float) for c in self.G.get("markers0", {}).values()]   # reference corners (px)
        # sign convention of the offset <-> pixel map (decided once from a few probe pixels)
        pr = self.uc[None] + np.array([[80.0, 0.0], [0.0, 80.0], [-80.0, -60.0]])
        D1 = np.linalg.solve(self.A, (self.uc[None] - pr).T).T
        self._flip = bool(np.abs(np.array([self.o2p(d) for d in D1]) - pr).max() > 0.5)
        # pinhole model for rotated views: robot offset W (mm, base frame) of a table point relative to the camera ->
        # image offset  M W.  M is measured (the image Jacobian); f = H * (mean singular value of M) is the focal
        # length in px, S the orthogonal part of M (image axes in the robot frame, including a possible mirror).
        probes = self.uc[None] + np.array([[100.0, 0.0], [-100.0, 0.0], [0.0, 100.0], [0.0, -100.0]])
        Wp = self.px2off(probes)
        X, *_ = np.linalg.lstsq(Wp, probes - self.uc[None], rcond=None)
        M = X.T
        u_, sv_, vt_ = np.linalg.svd(M)
        self.S = u_ @ vt_; self.f = float(self.H * np.mean(sv_))

    def project(self, Pw, d, dz, Rt):
        """Image pixels of table points (given by their robot offsets Pw, mm, relative to the reference camera
        position) seen by a camera displaced horizontally by d, lowered by -dz, and rotated by Rt (world frame)."""
        Pw = np.atleast_2d(Pw)
        w = np.c_[Pw - np.asarray(d, float)[None], np.full(len(Pw), -(self.H + dz))]
        v = w @ Rt                                           # rows: (Rt^T w)^T, the vector in the rotated camera
        vx = v[:, :2] @ self.S.T; vz = -v[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.uc[None] + self.f * vx / np.where(vz > 1e-6, vz, np.nan)[:, None]

    def markers_visible(self, u_view, s, d=None, dz=0.0, Rt=None, margin=6.0):
        """How many reference markers are fully inside the image of a view. Reference orientation: same model as the
        scale-aware servo, x = uc + s (m0 - u_view); rotated view (Rt given): pinhole projection."""
        W, Hh = 2 * self.uc; n = 0
        rotated = Rt is not None and not np.allclose(Rt, np.eye(3))
        for c in self.markers0:
            if rotated:
                X = self.project(self.px2off(np.asarray(c, float)), d, dz, Rt)
            else:
                X = self.uc + s * (np.asarray(c, float) - u_view)
            n += bool(np.all((X[:, 0] >= margin) & (X[:, 0] <= W - margin) & (X[:, 1] >= margin) & (X[:, 1] <= Hh - margin)))
        return n

    def t2px(self, T):
        return cv2.perspectiveTransform(np.asarray(T, np.float64).reshape(-1, 1, 2), self.Hi).reshape(-1, 2)

    def px2t(self, U):
        return cv2.perspectiveTransform(np.asarray(U, np.float64).reshape(-1, 1, 2), self.Hb).reshape(-1, 2)

    def px2off(self, U):
        """Robot offset that brings survey pixel u under the image centre (inverse of perception.offset_to_px)."""
        U = np.atleast_2d(np.asarray(U, float))
        if self._flip:
            return np.linalg.solve(self.A, (U - self.uc[None]).T).T
        return np.linalg.solve(self.A, (self.uc[None] - U).T).T

    def scale(self, dz):
        return self.H / max(self.H + dz, 1.0)


def node_markers(geo, L, n):
    """Markers predicted fully inside the image of lattice node n."""
    k = n // (int(L["nx"]) * int(L["ny"]))
    if L["orient"][n] == 0:
        return geo.markers_visible(L["U"][n], geo.scale(L["dz"][n]))
    return geo.markers_visible(L["U"][n], geo.scale(L["dz"][n]), L["D"][n], L["dz"][n], L["Rt_layers"][k])


# ------------------------------------------------------------------ 1-2: branch-continuous lattice
def layer_neighbours(k, nzh, nz):
    """Layers are (orientation) x (height), k = o * nzh + zi. A move changes the height by one step, or switches
    between the reference orientation (o = 0) and one rotated orientation at the same height."""
    o, zi = divmod(k, nzh)
    out = []
    if zi > 0:
        out.append(k - 1)
    if zi < nzh - 1:
        out.append(k + 1)
    if o == 0:
        out += [oo * nzh + zi for oo in range(1, nz // nzh)]
    else:
        out.append(zi)
    return out


def build_lattice(geo, q_ref, R_ref, p_ref, a):
    part_h = float(getattr(C, "PART_MAX_HEIGHT_MM", 60.0)); clear = float(getattr(C, "VIEW_MIN_CLEARANCE_MM", 25.0))
    # camera never goes lower than --max-descent below the reference (and never into the part clearance)
    dz_min = -min(geo.H - part_h - clear, float(a.max_descent))
    dzs = [0.0]
    while dzs[-1] - a.dz_step >= dz_min - 1e-9 and len(dzs) < 10:
        dzs.append(dzs[-1] - a.dz_step)
    orients = orientation_list(a)
    nzh = len(dzs)
    layers = [(o, dz) for o in range(len(orients)) for dz in dzs]       # layer k = orientation index * nzh + height index
    ldz = np.array([l[1] for l in layers]); lor = np.array([l[0] for l in layers])
    lrpy = np.array([orients[l[0]] for l in layers], float)
    Rt_layers = np.array([orient_R(orients[l[0]]) for l in layers])
    Rl = [Rt @ R_ref for Rt in Rt_layers]                                # camera orientation of every layer
    pad = a.step
    t0 = geo.px2t(geo.uc[None])[0]                     # table point under the image centre at the reference
    def axis(c, lo, hi):                               # lattice through t0 exactly, covering [lo, hi]
        k0, k1 = int(np.floor((lo - c) / a.step)), int(np.ceil((hi - c) / a.step))
        return c + a.step * np.arange(k0, k1 + 1)
    xs = axis(t0[0], -pad, geo.box_mm[0] + pad); ys = axis(t0[1], -pad, geo.box_mm[1] + pad)
    T = np.array([[x, y] for y in ys for x in xs]); U = geo.t2px(T); D0 = geo.px2off(U)
    nx, ny, nz = len(xs), len(ys), len(layers)
    idx = lambda i, j, k: (k * ny + j) * nx + i
    N = nx * ny * nz
    q = np.full((N, 6), np.nan); ok = np.zeros(N, bool); kap = np.zeros(N); marg = np.zeros(N)
    # start: the reference itself (offset exactly 0, reference height, reference orientation)
    i0 = int(np.argmin(np.abs(xs - t0[0]))); j0 = int(np.argmin(np.abs(ys - t0[1])))
    start = idx(i0, j0, 0)
    D0[j0 * nx + i0] = 0.0
    # horizontal camera offset of every node: brings the target table point onto the optical axis (for a rotated
    # camera the axis is re-aimed, which needs an extra shift)
    Dall = np.vstack([D0 + tilt_shift(geo.H, ldz[k], Rt_layers[k])[None] for k in range(nz)])
    seeds = {start: np.asarray(q_ref, float)}
    queue = [start]; queued = {start}; tries = np.zeros(N, int)
    t_b = time.time(); done = 0; next_report = time.time() + 20.0
    print("building the reachability lattice: up to %d relative poses (%d heights x %d orientations; progress every 20 s) ..."
          % (N, nzh, len(orients)), flush=True)
    while queue:
        n = queue.pop(0); queued.discard(n); done += 1
        if time.time() >= next_report:
            el = time.time() - t_b; solved = int(np.isfinite(q[:, 0]).sum())
            rate = done / max(el, 1e-6); left = max(N - done, len(queue))
            print("  lattice: %d poses processed, %d reachable, %d queued | %.0f s elapsed, about %.0f min left (upper bound)"
                  % (done, solved, len(queue), el, left / max(rate, 1e-6) / 60), flush=True)
            next_report = time.time() + 20.0
        k, r = divmod(n, nx * ny); j, i = divmod(r, nx)
        tgt = (p_ref + np.r_[Dall[n], ldz[k]]) / 1000.0          # p_ref and offsets in mm -> m
        tries[n] += 1
        qs, pe, re, good = ik_hold(tgt, seeds[n], Rl[k])
        if not good:
            qs, pe, re, good = ik_hold(tgt, seeds[n], Rl[k], iters=800)
        if not (good and np.max(np.abs(np.degrees(qs - seeds[n]))) < a.max_jump):
            continue                                   # may be retried from another solved neighbour
        q[n] = qs; kap[n] = kappa(qs); marg[n] = limit_margin_deg(qs)
        ok[n] = kap[n] >= a.kappa_min and marg[n] >= 5.0 and np.all(np.abs(Dall[n]) < C.FENCE_AUTO_HALF_MM - 5)
        nbrs = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < nx and 0 <= jj < ny:
                nbrs.append(idx(ii, jj, k))
        nbrs += [idx(i, j, kk) for kk in layer_neighbours(k, nzh, nz)]
        for m in nbrs:
            if not np.isfinite(q[m, 0]) and m not in queued and tries[m] < 3:
                seeds[m] = q[n]; queue.append(m); queued.add(m)
    nodes = dict(xs=xs, ys=ys, dzs=ldz, layer_orient=lor, layer_rpy=lrpy, Rt_layers=Rt_layers, nzh=nzh,
                 nx=nx, ny=ny, nz=nz, q=q, ok=ok, kappa=kap, margin=marg,
                 T=np.tile(T, (nz, 1)), U=np.tile(U, (nz, 1)), D=Dall,
                 dz=np.repeat(ldz, nx * ny), orient=np.repeat(lor, nx * ny), rpy=np.repeat(lrpy, nx * ny, axis=0),
                 start=start, solved=np.isfinite(q[:, 0]))
    return nodes


def neighbours(n, L, diag=True):
    nx, ny, nz, nzh = int(L["nx"]), int(L["ny"]), int(L["nz"]), int(L["nzh"])
    k, r = divmod(n, nx * ny); j, i = divmod(r, nx)
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for di, dj in steps:
        ii, jj = i + di, j + dj
        if 0 <= ii < nx and 0 <= jj < ny:
            yield (k * ny + jj) * nx + ii
    for kk in layer_neighbours(k, nzh, nz):
        yield (kk * ny + j) * nx + i


def edge_cost(L, a, b, prm):
    dq = np.degrees(np.max(np.abs(L["q"][a] - L["q"][b])))
    return prm["t0"] + dq / prm["omega"] + prm["lam"] * 0.5 * (1 / max(L["kappa"][a], 1e-3) + 1 / max(L["kappa"][b], 1e-3))


def dijkstra(L, src, prm):
    dist = {src: 0.0}; prev = {}; pq = [(0.0, src)]
    while pq:
        d, n = heapq.heappop(pq)
        if d > dist.get(n, np.inf):
            continue
        for m in neighbours(n, L):
            if not L["ok"][m] and m != L["start"]:
                continue
            nd = d + edge_cost(L, n, m, prm)
            if nd < dist.get(m, np.inf):
                dist[m] = nd; prev[m] = n; heapq.heappush(pq, (nd, m))
    return dist, prev


def path_to(prev, src, dst):
    out = [dst]
    while out[-1] != src:
        if out[-1] not in prev:
            return None
        out.append(prev[out[-1]])
    return out[::-1]


# ------------------------------------------------------------------ 3: coverage
def region_points(geo, a, survey=None):
    if a.region == "part" and survey is not None:
        from perception import locate_part_scene
        part = locate_part_scene(survey, geo.G)
        if part is not None:
            x, y, w, h = part["bbox_px"]
            T = geo.px2t(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float))
            lo, hi = np.maximum(T.min(0), 0), np.minimum(T.max(0), geo.box_mm)
            g = [np.arange(lo[i], hi[i] + 1e-6, a.pt_step) for i in range(2)]
            return np.array([[x_, y_] for y_ in g[1] for x_ in g[0]]), "part bounding box"
    g = [np.arange(a.pt_step / 2, geo.box_mm[i], a.pt_step) for i in range(2)]
    return np.array([[x_, y_] for y_ in g[1] for x_ in g[0]]), "drawn box"


def coverage_matrix(geo, L, P, rho_min, frac=0.8, need_markers=1):
    """Cov[r, p]: candidate view r covers point p at >= rho_min AND can be verified (>= need_markers markers
    predicted fully inside its image, so the servo can check the view). Rotated views use the pinhole model."""
    Up = geo.t2px(P); Pw = None
    W, Hh = 2 * geo.uc
    nxy = int(L["nx"]) * int(L["ny"])
    cand = np.flatnonzero(L["ok"])
    Cov = np.zeros((len(cand), len(P)), bool)
    for r, n in enumerate(cand):
        Rt = L["Rt_layers"][n // nxy]; ref = L["orient"][n] == 0
        s = geo.scale(L["dz"][n])
        if geo.rho_ref * s * (1.0 if ref else float(Rt[2, 2])) < rho_min:
            continue
        if need_markers > 0 and node_markers(geo, L, n) < need_markers:
            continue
        if ref:
            X = geo.uc + s * (Up - L["U"][n])
        else:
            if Pw is None:
                Pw = geo.px2off(Up)
            X = geo.project(Pw, L["D"][n], L["dz"][n], Rt)
        Cov[r] = (np.abs(X[:, 0] - geo.uc[0]) <= frac * W / 2) & (np.abs(X[:, 1] - geo.uc[1]) <= frac * Hh / 2)
    return cand, Cov


def greedy_cover(L, cand, Cov, cost_from_start, alpha):
    need = Cov.any(0).copy(); covered = np.zeros(Cov.shape[1], bool); S = []
    while (need & ~covered).any():
        gain = (Cov & (need & ~covered)[None]).sum(1).astype(float)
        score = gain * L["kappa"][cand] ** alpha / (1.0 + cost_from_start[cand]) * L["verify_w"][cand]
        r = int(np.argmax(score))
        if gain[r] <= 0:
            break
        S.append(int(cand[r])); covered |= Cov[r]
    return S, need, covered


# ------------------------------------------------------------------ 4: order
def order_views(S, start, Dm):
    nodes = [start] + S
    idx = {n: i for i, n in enumerate(nodes)}
    rem = set(S); seq = [start]
    while rem:
        nxt = min(rem, key=lambda n: Dm[idx[seq[-1]], idx[n]]); seq.append(nxt); rem.discard(nxt)
    seq.append(start)
    cost = lambda s: sum(Dm[idx[x], idx[y]] for x, y in zip(s[:-1], s[1:]))
    improved = True
    while improved:
        improved = False
        for i in range(1, len(seq) - 2):
            for j in range(i + 1, len(seq) - 1):
                new = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                if cost(new) < cost(seq) - 1e-9:
                    seq, improved = new, True
    return seq, cost(seq)


def plan_variant(geo, L, P, a, prm, rho_min, restrict=None, kappa_floor=None, lam=None, label=""):
    """Plan with optional restrictions (for the ablations). Returns a summary dict (+ full plan)."""
    L2 = dict(L); L2["ok"] = L["ok"].copy()
    if kappa_floor is not None:                       # singularity-blind: accept every solved node
        L2["ok"] = L["solved"] & (L["kappa"] >= kappa_floor)
    if restrict is not None:
        L2["ok"] &= restrict
    p2 = dict(prm, lam=prm["lam"] if lam is None else lam)
    dist0, prev0 = dijkstra(L2, L2["start"], p2)
    c0 = np.array([dist0.get(n, np.inf) for n in range(len(L2["ok"]))])
    L2["ok"] &= np.isfinite(c0)
    cand, Cov = coverage_matrix(geo, L2, P, rho_min, frac=a.central, need_markers=a.need_markers)
    if not len(cand):
        return dict(label=label, coverage_pct=0.0, n_views=0, path_cost=np.nan, min_kappa=np.nan), None
    S, need, covered = greedy_cover(L2, cand, Cov, np.nan_to_num(c0, posinf=1e3), a.alpha)
    row = {int(n): r for r, n in enumerate(cand)}
    S = prune(S, [Cov[row[n]] for n in S])
    covered = np.any([Cov[row[n]] for n in S], axis=0)
    nodes = [L2["start"]] + S
    Dm = np.zeros((len(nodes), len(nodes))); paths = {}
    for i, n in enumerate(nodes):
        dist, prev = dijkstra(L2, n, p2)
        for j, m in enumerate(nodes):
            Dm[i, j] = dist.get(m, 1e6); paths[(n, m)] = path_to(prev, n, m) if n != m else [n]
    seq, cost = order_views(S, L2["start"], Dm)
    route = [seq[0]]
    for x, y in zip(seq[:-1], seq[1:]):
        route += (paths[(x, y)] or [x, y])[1:]
    rs = route_stats([L["q"][n] for n in route])
    s = dict(label=label, coverage_pct=100.0 * covered.mean(), n_views=len(S), path_cost=float(cost),
             min_kappa=rs["min_kappa_path"], max_cmd_joint_change_deg=rs["max_cmd_joint_change_deg"],
             n_cmds_over_safety=rs["n_cmds_over_safety"], n_waypoints=len(route),
             est_time_s=float(sum(prm["t0"] + np.degrees(np.max(np.abs(L["q"][x] - L["q"][y]))) / prm["omega"]
                                  for x, y in zip(route[:-1], route[1:]))))
    return s, dict(views=[n for n in seq[1:-1]], route=route, covered=covered.tolist())


def segment_stats(qa, qb, n=8):
    """A joint-space command qa -> qb as the controller executes it: (max joint change deg, min kappa inside)."""
    ks = [kappa(qa + (qb - qa) * i / (n + 1)) for i in range(1, n + 1)]
    return float(np.degrees(np.max(np.abs(qb - qa)))), float(min(ks))


def route_stats(Q, skip_ref=True):
    """Over consecutive commands: largest single joint change and lowest conditioning away from the reference."""
    jumps, kmin = [], np.inf
    for a_, b_ in zip(Q[:-1], Q[1:]):
        j, k = segment_stats(a_, b_); jumps.append(j)
        kmin = min(kmin, k)
    inner = [kappa(q) for q in Q[1:-1]] if skip_ref else [kappa(q) for q in Q]
    return dict(max_cmd_joint_change_deg=float(max(jumps)) if jumps else 0.0,
                min_kappa_path=float(min([kmin] + inner)) if inner or np.isfinite(kmin) else float("nan"),
                n_cmds_over_safety=int(sum(j > C.MAX_JOINT_STEP_DEG for j in jumps)))


def prune(S, Cov_rows):
    """Remove views whose covered points are all covered by the other selected views (largest-first)."""
    S = list(S); keep = list(range(len(S)))
    for i in sorted(range(len(S)), key=lambda i: Cov_rows[i].sum()):
        others = [j for j in keep if j != i]
        if others and not (Cov_rows[i] & ~np.any([Cov_rows[j] for j in others], axis=0)).any():
            keep = others
    return [S[i] for i in keep]


def raster_baseline(geo, L, P, rho_min, prm, frac=0.8):
    """What one would program by hand: a regular raster of views at ONE height giving rho_min, spaced by 80 %
    of the footprint, each sent straight to IK from the reference (no lattice, no conditioning check)."""
    levels = sorted(set(L["dz"][L["solved"] & (L["orient"] == 0)]), reverse=True)
    dz = next((d for d in levels if geo.rho_ref * geo.scale(d) >= rho_min), levels[-1])
    s = geo.scale(dz); W, Hh = 2 * geo.uc
    fw, fh = frac * W / (geo.rho_ref * s), frac * Hh / (geo.rho_ref * s)        # footprint in table mm (approx.)
    nx = max(1, int(np.ceil(geo.box_mm[0] / fw))); ny = max(1, int(np.ceil(geo.box_mm[1] / fh)))
    cx = (np.arange(nx) + 0.5) * geo.box_mm[0] / nx; cy = (np.arange(ny) + 0.5) * geo.box_mm[1] / ny
    centres = np.array([[x, y] for y in cy for x in cx])
    q_ref, R_ref, p_ref = load_ref()
    covered = np.zeros(len(P), bool); kaps, bad, Qr, nver = [], 0, [], 0
    Up = geo.t2px(P)
    order = [r * nx + (c if r % 2 == 0 else nx - 1 - c) for r in range(ny) for c in range(nx)]
    centres = centres[order]
    for t in centres:
        u = geo.t2px([t])[0]; d = geo.px2off(u)[0]
        q, pe, re, ok = ik_hold((p_ref + np.r_[d, dz]) / 1000.0, q_ref, R_ref, iters=1200)
        if not ok:
            bad += 1; continue
        kaps.append(kappa(q)); Qr.append(q); nver += int(geo.markers_visible(u, s) >= 1)
        X = geo.uc + s * (Up - u)
        covered |= (np.abs(X[:, 0] - geo.uc[0]) <= frac * W / 2) & (np.abs(X[:, 1] - geo.uc[1]) <= frac * Hh / 2)
    rs = route_stats([q_ref] + Qr + [q_ref]) if Qr else dict(max_cmd_joint_change_deg=np.nan, min_kappa_path=np.nan,
                                                              n_cmds_over_safety=0)
    return dict(label="baseline: hand-programmed raster (one height, direct moves)", coverage_pct=100 * covered.mean(),
                n_views=len(kaps), n_raster=len(centres), n_unreachable=bad, dz=float(dz),
                min_kappa=rs["min_kappa_path"], max_cmd_joint_change_deg=rs["max_cmd_joint_change_deg"],
                n_cmds_over_safety=rs["n_cmds_over_safety"], n_verifiable=nver)


# ------------------------------------------------------------------ commands
def load_ref():
    from hardware import load_locked_pose
    q = np.radians(np.array(load_locked_pose()["angles_deg"], float))
    T = fk(q)
    return q, T[:3, :3], T[:3, 3] * 1000


def time_params():
    p = os.path.join(C.DEPLOY_DIR, "time_model_hold.json")
    if os.path.exists(p):
        j = json.load(open(p)); return float(j["t0"]), float(j["omega"])
    return 0.6, 30.0


def cmd_plan(a):
    geo = Geo()
    q_ref, R_ref, p_ref = load_ref()
    survey = cv2.imread(os.path.join(C.LOGS_DIR, "grounding", "calib_survey.jpg"))
    t = time.time()
    key = dict(step=a.step, dz_step=a.dz_step, H=geo.H, kmin=a.kappa_min, q=np.round(np.degrees(q_ref), 2).tolist(),
               g=str(geo.G.get("created")), box=np.round(geo.box_mm, 1).tolist(),
               maxdesc=float(a.max_descent), orients=[list(o) for o in orientation_list(a)], maxjump=float(a.max_jump))
    cache = lattice_path(a)
    L = None
    if os.path.exists(cache) and not a.rebuild:
        z = np.load(cache, allow_pickle=True)
        if json.loads(str(z["key"])) == key:
            L = {k: (z[k].item() if z[k].shape == () else z[k]) for k in z.files if k != "key"}
            print("lattice loaded from cache (use --rebuild after changing the setup)")
    if L is None:
        L = build_lattice(geo, q_ref, R_ref, p_ref, a)
        np.savez(cache, key=np.array(json.dumps(key)), **L)
    t0, om = time_params()
    prm = dict(t0=t0, omega=om, lam=a.lam)
    P, region = region_points(geo, a, survey)
    rho_max = geo.rho_ref * max(geo.scale(dz) for dz in L["dz"][L["ok"]])
    rho_min = a.res or min(a.mag * geo.rho_ref, 0.95 * rho_max)   # default: --mag x survey, capped by what is reachable
    print("movement choices: sideways (x, y) at every height; heights %s mm (max descent %.0f mm); camera rotations "
          "(roll, pitch, yaw) %s deg" % (sorted(set(int(round(v)) for v in L["dzs"]), reverse=True), a.max_descent,
                                          [tuple(float(x) for x in o) for o in orientation_list(a)]))
    print("lattice: %d nodes (%d x %d sideways x %d layers), solved %d, usable %d (kappa >= %.2f) in %.0f s"
          % (len(L["ok"]), L["nx"], L["ny"], L["nz"], L["solved"].sum(), L["ok"].sum(),
             a.kappa_min, time.time() - t))
    print("reference conditioning kappa = %.3f; region: %s, %d points; required resolution %.2f px/mm (survey %.2f)"
          % (kappa(q_ref), region, len(P), rho_min, geo.rho_ref))
    nm = len(geo.markers0)
    nxy_ = int(L["nx"]) * int(L["ny"])
    usable = [n for n in np.flatnonzero(L["ok"]) if geo.rho_ref * geo.scale(L["dz"][n])
              * (1.0 if L["orient"][n] == 0 else float(L["Rt_layers"][n // nxy_][2, 2])) >= rho_min]
    ver = sum(node_markers(geo, L, n) >= 1 for n in usable)
    print("markers in the reference image: %d; candidate views at this resolution: %d, with a marker in view: %d, "
          "verified by registration: %d%s" % (nm, len(usable), ver, len(usable) - ver,
                                             "" if a.need_markers == 0 else " (excluded: --need-markers %d)" % a.need_markers))
    if nm == 0 and a.need_markers > 0:
        raise SystemExit("no markers stored in grounding.json: run  python3 grounding.py calibrate --mode track  again")
    # preference (not a requirement): views with a marker in sight are verified by the markers, the others by
    # image registration to the reference survey; markers are preferred when coverage is otherwise equal
    L["verify_w"] = np.array([1.5 if node_markers(geo, L, n) >= 1 else 1.0 for n in range(len(L["dz"]))])
    variants = [("proposed (conditioning-aware, all heights)", {}),
                ("ablation: singularity-blind (kappa ignored, lam = 0)", dict(kappa_floor=1e-6, lam=0.0))]
    if a.sweep:
        sweep = []
        for f in (1.2, 1.4, 1.6, 1.8, 2.0):
            print("sweep: planning all methods for resolution %.1f x survey ..." % f, flush=True)
            r = f * geo.rho_ref
            if r > rho_max:
                continue
            for name, kw in variants:
                s_, _ = plan_variant(geo, L, P, a, prm, r, label=name, **kw); s_["rho_factor"] = f; sweep.append(s_)
            rb_ = raster_baseline(geo, L, P, r, prm, a.central); rb_["rho_factor"] = f; sweep.append(rb_)
            for x in sweep[-3:]:
                print("  res x%.1f  %-58s cov %5.1f%% views %2d  min kappa %.3f  largest cmd %3.0f deg  over safety %d"
                      % (f, x["label"][:58], x["coverage_pct"], x["n_views"], x["min_kappa"],
                         x["max_cmd_joint_change_deg"], x["n_cmds_over_safety"]))
        os.makedirs("results_views", exist_ok=True)
        sw = "results_views/refcover_sweep_%s.json" % _tag(a)
        json.dump(sweep, open(sw, "w"), indent=2)
        json.dump(sweep, open("results_views/refcover_sweep.json", "w"), indent=2)
        print("saved %s (mirror: results_views/refcover_sweep.json)" % sw)
    rows, best = [], None
    for name, kw in variants:
        s, plan = plan_variant(geo, L, P, a, prm, rho_min, label=name, **kw)
        rows.append(s)
        if best is None:
            best = plan
        print("  %-54s coverage %5.1f%%  views %2d  min kappa on path %.3f  largest command %3.0f deg  over safety %d"
              % (name, s["coverage_pct"], s["n_views"], s.get("min_kappa", np.nan),
                 s.get("max_cmd_joint_change_deg", np.nan), s.get("n_cmds_over_safety", 0)))
    rb = raster_baseline(geo, L, P, rho_min, prm, a.central)
    rows.append(rb)
    print("  %-54s coverage %5.1f%%  views %2d  min kappa on path %.3f  largest command %3.0f deg  over safety %d"
          % (rb["label"], rb["coverage_pct"], rb["n_views"], rb["min_kappa"], rb["max_cmd_joint_change_deg"],
             rb["n_cmds_over_safety"]))
    if best is None or not best["views"]:
        raise SystemExit("no coverage plan found: check VIEW_CAM_HEIGHT_MM, the drawn box and grounding")
    nxy = int(L["nx"]) * int(L["ny"]); nzh = int(L["nzh"]); n_or = int(L["nz"]) // nzh

    def tilts_of(n):
        """Rotated captures at the same table cell and height as coverage view n (other orientation layers)."""
        if not a.tilt_captures or L["orient"][n] != 0 or n_or == 1:
            return []
        k, r = divmod(int(n), nxy); zi = k % nzh; out_ = []
        for o in range(1, n_or):
            m = (o * nzh + zi) * nxy + r
            ok_ = bool(L["ok"][m]                                  # reachable, conditioned, joint margin, INSIDE the fence
                       and np.degrees(np.max(np.abs(L["q"][m] - L["q"][n]))) < a.max_jump)
            out_.append(dict(node=int(m), rpy=[float(x) for x in L["rpy"][m]], ok=ok_,
                             q_deg=np.degrees(L["q"][m]).tolist() if ok_ else None, offset_mm=L["D"][m].tolist()))
        return out_
    out = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), region=region, rho_min=rho_min, kappa_min=a.kappa_min,
               lam=a.lam, step=a.step, dz_step=a.dz_step, comparison=rows, start=int(L["start"]),
               max_descent=float(a.max_descent), orientations=[list(o) for o in orientation_list(a)],
               views=[dict(node=int(n), q_deg=np.degrees(L["q"][n]).tolist(), dz=float(L["dz"][n]),
                           rpy=[float(x) for x in L["rpy"][n]], orient=int(L["orient"][n]),
                           u_target=L["U"][n].tolist(), t_mm=L["T"][n].tolist(), offset_mm=L["D"][n].tolist(),
                           kappa=float(L["kappa"][n]), scale=geo.scale(L["dz"][n]),
                           markers_expected=node_markers(geo, L, n), tilts=tilts_of(n))
                      for n in best["views"]],
               route=[dict(node=int(n), q_deg=np.degrees(L["q"][n]).tolist(), kappa=float(L["kappa"][n]),
                           dz=float(L["dz"][n]), rpy=[float(x) for x in L["rpy"][n]], view=int(n) in set(best["views"]))
                      for n in best["route"]],
               points_mm=P.tolist(), covered=best["covered"], central=float(a.central))
    vz = [v["dz"] for v in out["views"]]
    print("coverage plan: %d view(s) at heights %s mm, sideways offsets up to %.0f mm; tilted captures reachable: %d of %d"
          % (len(out["views"]), sorted(set(int(round(x)) for x in vz), reverse=True),
             max(float(np.hypot(*v["offset_mm"])) for v in out["views"]),
             sum(t["ok"] for v in out["views"] for t in v["tilts"]), sum(len(v["tilts"]) for v in out["views"])))
    for i, v in enumerate(out["views"]):
        print("  view %d: offset (%+.1f, %+.1f) mm, dz %+d mm, rpy %s; tilts %s" % (
            i, v["offset_mm"][0], v["offset_mm"][1], v["dz"], v["rpy"],
            [tuple(t["rpy"]) for t in v["tilts"] if t["ok"]] or "none"))
    os.makedirs(C.DEPLOY_DIR, exist_ok=True); json.dump(out, open(plan_path(a), "w"), indent=1)
    os.makedirs("results_views", exist_ok=True)
    json.dump(dict(rows=rows, n_nodes=int(len(L["ok"])), n_usable=int(L["ok"].sum()), kappa_ref=kappa(q_ref),
                   heights=L["dzs"].tolist()), open("results_views/refcover_plan_comparison.json", "w"), indent=2)
    import shutil as _sh
    _sh.copyfile("results_views/refcover_plan_comparison.json", "results_views/refcover_plan_comparison_%s.json" % _tag(a))
    figures(geo, L, out, survey)
    import shutil
    shutil.copyfile(plan_path(a), PLAN)                                   # mirrors for the analysis / export tools
    shutil.copyfile(cache, os.path.join(C.DEPLOY_DIR, "refcover_lattice.npz"))
    print("saved %s (mirror: %s), results_views/refcover_plan_comparison.json, figures in %s" % (plan_path(a), PLAN, OUTF))


def figures(geo, L, plan, survey):
    """Plan overlay (OpenCV, always). Conditioning maps need matplotlib: skipped with a note if it cannot be
    imported (e.g. JetPack NumPy too old for the installed matplotlib); colab_analysis.py draws them in Colab."""
    os.makedirs(OUTF, exist_ok=True)
    if survey is not None:
        v = survey.copy(); W, Hh = 2 * geo.uc
        for i, vw in enumerate(plan["views"]):
            s = vw["scale"]; c = np.array(vw["u_target"])
            half = float(plan.get("central", 0.8)) * np.array([W, Hh]) / 2
            col = (int(255 * (1 - min(vw["kappa"] / 0.15, 1))), 160, int(255 * min(vw["kappa"] / 0.15, 1)))
            if vw.get("orient", 0) == 0:
                cv2.rectangle(v, tuple((c - half / s).astype(int)), tuple((c + half / s).astype(int)), col, 1)
            else:                                    # rotated view: mark the aimed table point and the rotation
                cv2.circle(v, tuple(c.astype(int)), 12, col, 2)
                cv2.putText(v, "rpy %g,%g,%g" % tuple(vw["rpy"]), (int(c[0]) + 14, int(c[1]) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            cv2.putText(v, str(i), tuple(c.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(OUTF, "fig_refcover_plan.png"), v)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print("note: conditioning maps skipped on this computer (%s); they are drawn in Colab by colab_analysis.py"
              % str(e).split(";")[0])
        return
    nz = L["nz"]; cols = min(nz, 4); rows = int(np.ceil(nz / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.1 * rows), squeeze=False)
    for k in range(nz):
        ax = axs[k // cols][k % cols]
        sl = slice(k * L["nx"] * L["ny"], (k + 1) * L["nx"] * L["ny"])
        K = np.where(L["solved"][sl], L["kappa"][sl], np.nan).reshape(L["ny"], L["nx"])
        im = ax.imshow(K, origin="upper", vmin=0, vmax=0.15, cmap="viridis",
                       extent=[L["xs"][0], L["xs"][-1], L["ys"][-1], L["ys"][0]])
        ax.set_title("dz %+d, rpy %g,%g,%g" % ((L["dzs"][k],) + tuple(L["layer_rpy"][k])), fontsize=6)
        ax.tick_params(labelsize=6)
        ax.add_patch(plt.Rectangle((0, 0), *geo.box_mm, fill=False, ec="w", lw=0.8))
    for k in range(nz, rows * cols):
        axs[k // cols][k % cols].axis("off")
    fig.colorbar(im, ax=axs, shrink=0.8, label="conditioning kappa")
    fig.savefig(os.path.join(OUTF, "fig_refcover_kappa.pdf")); fig.savefig(os.path.join(OUTF, "fig_refcover_kappa.png"), dpi=200)
    plt.close(fig)


def capture_registered(cam, arm, survey, u_target, geo, scale_plan, speed=8):
    """Tilted view: capture, register to the survey and CENTRE the aimed point by moving the camera with its
    orientation held (registration servo), then report the remaining error."""
    from hardware import wait_settled, safe_move
    from views import registration_servo, table_mask
    t0 = time.time()
    mask = table_mask(survey, geo.G)

    def grab():
        wait_settled(arm); time.sleep(0.4)
        for _ in range(2):
            cam.grab()
        return cam.grab()

    def move_by(dd):
        q = np.radians(arm.read_deg()); T = fk(q)
        p = T[:3, 3] * 1000 + np.r_[np.asarray(dd, float), 0.0]
        q2, pe, re, ok = ik_hold(p / 1000.0, q, T[:3, :3], iters=800)
        return bool(ok) and safe_move(arm, np.degrees(q2), speed, max_step=12.0)[0]

    r = registration_servo(grab, move_by, survey, u_target, lambda u: geo.px2off(np.asarray(u, float)[None])[0],
                           geo.uc, geo.rho_ref, mask=mask)
    return dict(frame=r["frame"], scale=float(r["scale"]) if np.isfinite(r["scale"]) else float(scale_plan),
                servo_markers=0, view_err_raw_mm=r["err_first_mm"], ff_mm=float("nan"),
                view_err_open_mm=r["err_first_mm"], view_err_mm=r["err_mm"], servo_iters=r["iters"],
                servo_s=time.time() - t0, verify=r["verify"],
                H_meas=r["H"].tolist() if r["H"] is not None else None, reg_inliers=r["inliers"])


def cmd_run(a):
    import hardware as HW
    from hardware import make_arm, make_camera, safe_move, wait_settled
    from fence import auto_fence
    from servo_scaled import servo_to_view_scaled
    import board as BD
    pp = plan_path(a)
    if not os.path.exists(pp):
        raise SystemExit("no plan %s: run  python3 refcover.py plan <same options>  first" % pp)
    plan = json.load(open(pp)); geo = Geo()
    lowest = min(v["dz"] for v in plan["views"] + plan["route"])
    if lowest < -float(a.max_descent) - 1e-6:             # never execute a plan that descends more than allowed
        raise SystemExit("plan %s goes %.0f mm below the reference but --max-descent is %.0f: refusing to move. "
                         "Re-plan with the same --max-descent." % (pp, -lowest, a.max_descent))
    rp = [v.get("rpy", [0.0, 0.0, 0.0]) for v in plan["views"] + plan["route"]] + \
         [t["rpy"] for v in plan["views"] for t in v.get("tilts", []) if t["ok"]]
    tilt = max(max(abs(r[0]), abs(r[1])) for r in rp); yawm = max(abs(r[2]) for r in rp)
    if tilt > a.max_tilt + 1e-6 or yawm > a.max_yaw + 1e-6:
        raise SystemExit("plan %s tilts the camera by %.1f deg / yaws it by %.1f deg but --max-tilt is %.1f and --max-yaw "
                         "%.1f: refusing to move" % (pp, tilt, yawm, a.max_tilt, a.max_yaw))
    print("executing %s: %d view(s), lowest %.0f mm below the reference, largest roll/pitch %.1f deg, largest yaw %.1f deg"
          % (pp, len(plan["views"]), -min(lowest, 0.0), tilt, yawm))
    q_ref, R_ref, p_ref = load_ref()
    HW.set_fence(auto_fence(), p_ref)
    m0 = {int(k) if str(k).isdigit() else k: np.array(v) for k, v in geo.G["markers0"].items()}
    tag = a.tag or time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(C.LOGS_DIR, "refcover", tag); os.makedirs(out, exist_ok=True)
    import robot_trace as RT; RT.install(out)          # commands, joint trajectories, servo status
    arm = make_arm(a.dry_run, np.degrees(q_ref)); arm.send_deg(np.degrees(q_ref), 15); wait_settled(arm)
    cam = make_camera(a.dry_run, arm, a.ref_img)
    rows, imgs, tilt_rows = [], [], []
    t_all = time.time()
    try:
        from servo import servo_to_view
        RT.event("survey")
        sv = servo_to_view(arm, cam, geo.A, geo.uc, m0, geo.uc, R_ref, safe_move, np.degrees(q_ref), speed=10)
        from servo_scaled import part_mask
        s_mask = part_mask(sv["frame"], geo.G)
        drift = np.asarray(geo.G.get("descent_drift_px_per_mm", [0.0, 0.0]), float)
        if not np.any(drift):
            print("note: no descent drift calibrated (run  python3 measure_height.py --write); no feed-forward")
        RT.event("route")
        cv2.imwrite(os.path.join(out, "survey.jpg"), sv["frame"])
        views = {v["node"]: v for v in plan["views"]}
        q_cmd = np.degrees(q_ref); n_view = 0
        for r_ in plan["route"][1:]:
            q_t = np.array(r_["q_deg"], float)
            moved, mv_s, qm = safe_move(arm, q_t, a.speed)
            if not moved:
                print("  refused at node %d: %s" % (r_["node"], HW.LAST_REFUSAL)); break
            if r_["node"] in views and r_["view"]:
                v = views.pop(r_["node"])
                RT.event("verifying_view", view=n_view, node=v["node"])
                if v.get("orient", 0) == 0:
                    res = servo_to_view_scaled(arm, cam, geo.A, geo.uc, m0, np.array(v["u_target"]), R_ref, safe_move,
                                               q_t, v["scale"], speed=min(a.speed, 12), survey=sv["frame"],
                                               survey_mask=s_mask, ff_px=drift * (-v["dz"]))
                else:
                    # rotated view (roll/pitch/yaw): the marker/registration servo assumes the reference orientation, so it is not
                    # used here; the arm captures at the planned, joint-space pose (no verified error available)
                    res = capture_registered(cam, arm, sv["frame"], np.array(v["u_target"]), geo, v["scale"])
                img = res["frame"]; qn = arm.read_deg()
                name = "view_%02d.jpg" % n_view; cv2.imwrite(os.path.join(out, name), img)
                rows.append(dict(view=n_view, node=v["node"], dz=v["dz"], roll=v["rpy"][0], pitch=v["rpy"][1], yaw=v["rpy"][2],
                                 markers_expected=v.get("markers_expected"),
                                 kappa_plan=v["kappa"], kappa_meas=kappa(np.radians(qn)),
                                 scale_plan=v["scale"], scale_meas=res["scale"], markers=res["servo_markers"],
                                 view_err_raw_mm=res["view_err_raw_mm"], ff_mm=res["ff_mm"],
                                 view_err_open_mm=res["view_err_open_mm"], view_err_mm=res["view_err_mm"],
                                 servo_iters=res["servo_iters"], servo_s=res["servo_s"], img=name, verify=res["verify"],
                                 u_target=v["u_target"]))
                imgs.append((img, v, res))
                RT.event("captured", view=n_view, verify=res["verify"], view_err_mm=res["view_err_mm"])
                RT.event("route")
                print("  view %2d (dz %+d mm, rpy %g,%g,%g deg, kappa %.3f): view error as planned %.1f -> after feed-forward %.1f -> "
                      "verified %.1f mm, scale %.2f, by %s (%d marker(s))"
                      % (n_view, v["dz"], v["rpy"][0], v["rpy"][1], v["rpy"][2], v["kappa"], res["view_err_raw_mm"], res["view_err_open_mm"], res["view_err_mm"],
                         res["scale"], res["verify"], res["servo_markers"]))
                for ti, tv in enumerate(v.get("tilts", [])):
                    if not tv["ok"]:
                        continue
                    if not safe_move(arm, np.array(tv["q_deg"], float), a.speed)[0]:
                        print("    tilt %s refused: %s" % (tv["rpy"], HW.LAST_REFUSAL)); continue
                    rt = capture_registered(cam, arm, sv["frame"], np.array(v["u_target"]), geo, v["scale"])
                    tname = "view_%02d_tilt%d.jpg" % (n_view, ti); cv2.imwrite(os.path.join(out, tname), rt["frame"])
                    tilt_rows.append(dict(view=n_view, tilt=ti, roll=tv["rpy"][0], pitch=tv["rpy"][1], yaw=tv["rpy"][2],
                                          verify=rt["verify"], view_err_open_mm=rt["view_err_open_mm"],
                                          view_err_mm=rt["view_err_mm"], corrections=rt["servo_iters"], scale=rt["scale"],
                                          reg_inliers=rt["reg_inliers"], img=tname))
                    if rt.get("H_meas") is not None:
                        imgs.append((rt["frame"], dict(v, orient=1, rpy=tv["rpy"]), rt))
                    print("    tilt rpy %s: %s, aimed point %.1f -> %.1f mm from the image centre (%d correction(s))"
                          % (tv["rpy"], rt["verify"], rt["view_err_open_mm"], rt["view_err_mm"], rt["servo_iters"]))
                    safe_move(arm, q_t, a.speed)              # back to the coverage view before moving on
                n_view += 1
            q_cmd = q_t
    finally:
        try:
            for r_ in plan["route"][::-1]:                 # return along the planned route if interrupted
                if np.max(np.abs(arm.read_deg() - np.degrees(q_ref))) < 1.0:
                    break
                safe_move(arm, np.array(r_["q_deg"], float), a.speed)
            safe_move(arm, np.degrees(q_ref), a.speed)
        finally:
            cam.release(); arm.close()
    import csv
    if rows:
        with open(os.path.join(out, "views.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if tilt_rows:
        with open(os.path.join(out, "tilts.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tilt_rows[0])); w.writeheader(); w.writerows(tilt_rows)
    mosaic(geo, plan, imgs, out)
    json.dump(dict(tag=tag, n_views=len(rows), n_tilts=len(tilt_rows), n_planned=len(plan["views"]), wall_s=time.time() - t_all,
                   plan_created=plan["created"]), open(os.path.join(out, "run.json"), "w"), indent=2)
    print("run saved in", out)


def mosaic(geo, plan, imgs, out):
    """Blend all verified views into one image of the box in table coordinates (and a resolution map)."""
    if not imgs:
        return
    good = [x for x in imgs if np.isfinite(x[2].get("view_err_mm", np.nan)) and x[2]["view_err_mm"] <= 5.0]
    if len(good) < len(imgs):
        print("mosaic: %d of %d views verified to <= 5 mm are blended (unverified views would ghost the mosaic)"
              % (len(good), len(imgs)))
    imgs = good or imgs[:1]
    rho = max(v["scale"] for _, v, _ in imgs) * geo.rho_ref
    Wm, Hm = int(geo.box_mm[0] * rho), int(geo.box_mm[1] * rho)
    gx, gy = np.meshgrid((np.arange(Wm) + 0.5) / rho, (np.arange(Hm) + 0.5) / rho)
    Up = geo.t2px(np.c_[gx.ravel(), gy.ravel()])
    # No averaging: the box is mapped on the TABLE plane, but the part stands ~35-40 mm above it, so the same groove
    # appears at different table coordinates in each view (parallax). Averaging those views gives the ghosted, repeated
    # grooves. Each mosaic pixel is therefore taken from ONE real view: the one that sees it closest to its image centre
    # at the highest scale (winner-take-all). Seams remain where the source view changes; no detail is invented.
    best = np.full(Hm * Wm, -1.0); acc = np.zeros((Hm * Wm, 3)); src = np.full(Hm * Wm, -1); res_map = np.zeros(Hm * Wm)
    W, Hh = 2 * geo.uc
    Pw_g = None
    for vi, (img, v, r) in enumerate(imgs):
        s = r["scale"] if np.isfinite(r["scale"]) else v["scale"]
        if r.get("H_meas") is not None:              # measured survey -> view mapping (tilted views)
            Hmat = np.asarray(r["H_meas"], float); Xh = np.c_[Up, np.ones(len(Up))] @ Hmat.T
            X = Xh[:, :2] / Xh[:, 2:3]
        elif v.get("orient", 0) == 0:
            X = geo.uc + s * (Up - np.array(v["u_target"]))
        else:                                        # rotated view: pinhole model at the planned pose
            if Pw_g is None:
                Pw_g = geo.px2off(Up)
            X = np.nan_to_num(geo.project(Pw_g, np.asarray(v["offset_mm"], float), v["dz"], orient_R(v["rpy"])), nan=-10.0)
        inside = (X[:, 0] >= 1) & (X[:, 0] < W - 2) & (X[:, 1] >= 1) & (X[:, 1] < Hh - 2)
        mx = X[:, 0].reshape(Hm, Wm).astype(np.float32); my = X[:, 1].reshape(Hm, Wm).astype(np.float32)
        smp = cv2.remap(img, mx, my, cv2.INTER_LINEAR).reshape(-1, 3).astype(float)
        dist = np.hypot((X[:, 0] - geo.uc[0]) / (W / 2), (X[:, 1] - geo.uc[1]) / (Hh / 2))
        wgt = inside * np.exp(-2.0 * dist ** 2) * s
        take = wgt > best
        acc[take] = smp[take]; best[take] = wgt[take]; src[take] = vi
        res_map = np.maximum(res_map, inside * s * geo.rho_ref)
    mos = acc.reshape(Hm, Wm, 3).astype(np.uint8)
    cv2.imwrite(os.path.join(out, "mosaic.png"), mos)
    pal = np.array([[230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66], [0, 114, 178], [213, 94, 0],
                    [204, 121, 167], [120, 120, 120]], np.uint8)[:, ::-1]
    lab = np.where(src[:, None] >= 0, pal[np.maximum(src, 0) % len(pal)], 0).reshape(Hm, Wm, 3)
    over = cv2.addWeighted(mos, 0.6, lab.astype(np.uint8), 0.4, 0)
    for vi in range(len(imgs)):
        ys, xs = np.nonzero(src.reshape(Hm, Wm) == vi)
        if len(xs):
            cv2.putText(over, "v%d" % vi, (int(xs.mean()) - 8, int(ys.mean()) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out, "mosaic_views.png"), over)     # which real view supplies each region
    np.save(os.path.join(out, "resolution_map.npy"), res_map.reshape(Hm, Wm))
    print("mosaic %dx%d px of the %gx%g mm box at %.2f px/mm; area imaged at >= required resolution: %.1f%%"
          % (Wm, Hm, geo.box_mm[0], geo.box_mm[1], rho, 100 * np.mean(res_map >= 0.95 * plan["rho_min"])))


def cmd_report(a):
    """Statistics over all coverage runs. Standard csv + NumPy only (works on the Jetson); scipy optional."""
    import csv
    runs = sorted(glob.glob(os.path.join(C.LOGS_DIR, "refcover", "*", "views.csv")))
    if not runs:
        raise SystemExit("no runs yet")
    V = []
    for f in runs:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                r["run"] = os.path.basename(os.path.dirname(f)); V.append(r)
    num = lambda k: np.array([float(r[k]) if r.get(k, "") not in ("", "nan") else np.nan for r in V])
    raw, op, cl = num("view_err_raw_mm"), num("view_err_open_mm"), num("view_err_mm")
    st = lambda x: dict(n=int(np.isfinite(x).sum()), median=float(np.nanmedian(x)) if np.isfinite(x).any() else None,
                        p95=float(np.nanpercentile(x, 95)) if np.isfinite(x).any() else None)
    pp_ = plan_path(a) if os.path.exists(plan_path(a)) else PLAN
    rho_min = json.load(open(pp_))["rho_min"] if os.path.exists(pp_) else None
    cov = [100 * np.mean(np.load(os.path.join(os.path.dirname(f), "resolution_map.npy")) >= 0.95 * rho_min)
           for f in runs if rho_min and os.path.exists(os.path.join(os.path.dirname(f), "resolution_map.npy"))]
    meth = {}
    for r in V:
        meth[r.get("verify", "?")] = meth.get(r.get("verify", "?"), 0) + 1
    S = dict(n_runs=len(runs), n_views=len(V), view_err_as_planned_mm=st(raw), view_err_after_feedforward_mm=st(op),
             view_err_verified_mm=st(cl), verified_by=meth, verified_pct=100.0 * float(np.mean(np.isfinite(cl))),
             kappa_min_reached=float(np.nanmin(num("kappa_meas"))), coverage_achieved_pct=cov,
             scale_meas_median=float(np.nanmedian(num("scale_meas"))))
    try:
        from scipy.stats import wilcoxon
        ok = np.isfinite(op) & np.isfinite(cl)
        if ok.sum() > 5:
            S["wilcoxon_open_vs_verified_p"] = float(wilcoxon(op[ok], cl[ok]).pvalue)
    except Exception:
        pass
    os.makedirs("results_views", exist_ok=True)
    json.dump(S, open("results_views/refcover_summary.json", "w"), indent=2)
    print(json.dumps(S, indent=2))


def add_planner_args(ap):
    """Options shared by refcover.py and lattice_views.py (the lattice file is found from them)."""
    ap.add_argument("--region", choices=["box", "part"], default="box")
    ap.add_argument("--step", type=float, default=10.0, help="lattice spacing on the table (mm)")
    ap.add_argument("--max-descent", dest="max_descent", type=float,
                    default=float(getattr(C, "VIEW_MAX_DESCENT_MM", 10.0)),
                    help="most the camera may go BELOW the reference height (mm); 0 = stay at the reference height")
    ap.add_argument("--dz-step", type=float, default=5.0, help="spacing of the allowed heights below the reference (mm)")
    ap.add_argument("--roll-list", dest="roll_list", type=float, nargs="*", default=[],
                    help="camera roll choices about the world x axis (deg), e.g. --roll-list -6 6")
    ap.add_argument("--pitch-list", dest="pitch_list", type=float, nargs="*", default=[],
                    help="camera pitch choices about the world y axis (deg), e.g. --pitch-list -6 6")
    ap.add_argument("--yaw-list", dest="yaw_list", type=float, nargs="*", default=[],
                    help="camera yaw choices about the vertical axis (deg), e.g. --yaw-list -8 8")
    ap.add_argument("--max-tilt", dest="max_tilt", type=float, default=10.0, help="hard limit for roll and pitch (deg)")
    ap.add_argument("--central", type=float, default=float(getattr(C, "VIEW_CENTRAL_FRAC", 0.25)),
                    help="central image fraction that counts as covered (0.8 = almost whole image -> 1 view; 0.25 -> "
                         "several sideways views that trace the part)")
    ap.add_argument("--tilt-captures", dest="tilt_captures", action="store_true",
                    help="at every coverage view also capture the rotations of --roll/pitch/yaw-list (same cell)")
    ap.add_argument("--max-yaw", dest="max_yaw", type=float, default=15.0, help="hard limit for yaw (deg)")
    ap.add_argument("--pt-step", type=float, default=4.0, help="coverage point spacing (mm)")
    ap.add_argument("--res", type=float, default=None,
                    help="required resolution px/mm (default: --mag x survey, capped by what is reachable)")
    ap.add_argument("--mag", type=float, default=float(getattr(C, "VIEW_DEFAULT_MAG", 1.6)),
                    help="default required magnification vs the survey (capped by what --max-descent allows)")
    ap.add_argument("--kappa-min", type=float, default=0.04)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--need-markers", type=int, default=0,
                    help="views must show at least this many markers; 0 (default) = views without a marker are "
                         "verified by image registration to the reference survey")
    ap.add_argument("--max-jump", type=float, default=25.0, help="max joint change between lattice neighbours (deg)")
    ap.add_argument("--speed", type=int, default=12)
    ap.add_argument("--sweep", action="store_true", help="plan: compare methods over required resolutions 1.2-2.0 x survey")
    ap.add_argument("--rebuild", action="store_true", help="recompute the lattice (after changing pose, heights or calibration)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref-img", default=os.path.join("data", "raw_camera.jpg"))


def cmd_mosaic(a):
    """Rebuild mosaic.png / mosaic_views.png of an existing run from its saved views (no robot motion)."""
    import csv, ast
    if not a.tag:
        raise SystemExit("give the run:  python3 refcover_updated.py mosaic --tag cover_01")
    out = os.path.join(C.LOGS_DIR, "refcover", a.tag)
    rows = list(csv.DictReader(open(os.path.join(out, "views.csv"))))
    geo = Geo(); imgs = []
    for r in rows:
        img = cv2.imread(os.path.join(out, r["img"]))
        rpy = [float(r["roll"]), float(r["pitch"]), float(r["yaw"])]
        if img is None or any(abs(x) > 1e-6 for x in rpy):
            continue                                   # straight views only (tilted views are not table-plane mosaics)
        sc = float(r["scale_meas"]) if r["scale_meas"] not in ("", "nan") else float(r["scale_plan"])
        v = dict(scale=float(r["scale_plan"]), u_target=list(ast.literal_eval(r["u_target"])), orient=0,
                 dz=float(r["dz"]), rpy=rpy)
        imgs.append((img, v, dict(scale=sc, view_err_mm=float(r["view_err_mm"]) if r["view_err_mm"] else np.nan)))
    rho_min = geo.rho_ref
    try:
        rho_min = json.load(open(plan_path(a)))["rho_min"]
    except Exception:
        pass
    mosaic(geo, dict(rho_min=rho_min), imgs, out)
    print("wrote %s/mosaic.png and mosaic_views.png from %d straight views" % (out, len(imgs)))


def cmd_balance(a):
    """Why does the robot use one side? For every straight sideways pose of the cached lattice: the base-joint (J1)
    angle relative to the reference, and whether it is usable or which test removed it (no robot motion)."""
    geo = Geo(); q_ref, R_ref, p_ref = load_ref()
    cache = lattice_path(a)
    if not os.path.exists(cache):
        raise SystemExit("no lattice %s: run  python3 refcover_updated.py plan <same options>  first" % cache)
    z = np.load(cache, allow_pickle=True)
    L = {k: (z[k].item() if z[k].shape == () else z[k]) for k in z.files if k != "key"}
    plan = json.load(open(plan_path(a))) if os.path.exists(plan_path(a)) else {"views": []}
    nx, ny = int(L["nx"]), int(L["ny"]); nxy = nx * ny
    fence = C.FENCE_AUTO_HALF_MM - 5
    t_c = geo.box_mm / 2.0
    t0 = geo.px2t(geo.uc[None])[0]
    print("reference: J1 = %.1f deg; image centre is %.1f mm, %.1f mm from the box centre (table x, y)"
          % (np.degrees(q_ref[0]), *(t0 - t_c)))
    print("camera offset needed to centre the box centre: %s mm (fence: +/-%.0f mm per axis)"
          % (np.round(geo.px2off(geo.t2px(t_c[None]))[0], 1).tolist(), fence))
    views = {int(v["node"]) for v in plan["views"]}
    figs = []
    for k in range(int(L["nz"])):
        lo = L.get("layer_orient", np.zeros(int(L["nz"]), int))
        if int(lo[k]) != 0:
            continue
        dz = float(L["dz"][k * nxy]); ids = np.arange(k * nxy, (k + 1) * nxy)
        why = []
        for n in ids:
            if not L["solved"][n]:
                why.append("no IK / jump")
            elif np.any(np.abs(L["D"][n]) >= fence):
                why.append("fence")
            elif L["kappa"][n] < a.kappa_min:
                why.append("kappa")
            elif L["margin"][n] < 5.0:
                why.append("joint limit")
            else:
                why.append("usable")
        why = np.array(why)
        dj1 = np.degrees(L["q"][ids, 0] - q_ref[0])
        print("\nheight %+.0f mm: %d sideways poses" % (dz, len(ids)))
        for side, m in (("J1 > ref (+)", dj1 > 0.5), ("J1 ~ ref", np.abs(dj1) <= 0.5), ("J1 < ref (-)", dj1 < -0.5)):
            print("  %-13s solved %3d: %s" % (side, m.sum(), ", ".join("%s %d" % (r, np.sum(why[m] == r))
                  for r in ("usable", "fence", "kappa", "joint limit"))))
        solved = np.isfinite(dj1)
        if solved.any():
            print("  J1 range reached: %+.1f .. %+.1f deg; unsolved poses (IK failed / joint jump): %d"
                  % (np.nanmin(dj1), np.nanmax(dj1), np.sum(~L["solved"][ids])))
        chosen = [n for n in ids if int(n) in views]
        if chosen:
            print("  planned views here: J1 %s deg" % ", ".join("%+.1f" % np.degrees(L["q"][n, 0] - q_ref[0]) for n in chosen))
        # figure: one cell per sideways pose, coloured by status, labelled with the J1 change
        cs = 46; img = np.full((ny * cs + 40, nx * cs + 20, 3), 255, np.uint8)
        col = {"usable": (120, 200, 120), "fence": (80, 160, 240), "kappa": (200, 120, 220),
               "joint limit": (60, 60, 220), "no IK / jump": (190, 190, 190)}
        for jj in range(ny):
            for ii in range(nx):
                n = ids[jj * nx + ii]; x0, y0 = 10 + ii * cs, 30 + jj * cs
                cv2.rectangle(img, (x0, y0), (x0 + cs - 2, y0 + cs - 2), col[why[jj * nx + ii]], -1)
                if np.isfinite(dj1[jj * nx + ii]):
                    cv2.putText(img, "%+.0f" % dj1[jj * nx + ii], (x0 + 4, y0 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                (0, 0, 0), 1, cv2.LINE_AA)
                if int(n) in views:
                    cv2.rectangle(img, (x0, y0), (x0 + cs - 2, y0 + cs - 2), (0, 0, 0), 3)
                if int(n) == int(L["start"]) or (n - k * nxy) == int(L["start"]) % nxy:
                    cv2.circle(img, (x0 + cs // 2, y0 + 8), 4, (0, 0, 0), -1)
        cv2.putText(img, "dz %+.0f mm  green usable, orange fence, purple kappa, red joint limit, grey no IK; "
                    "number = J1 change (deg); black box = planned view; dot = reference"
                    % dz, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1, cv2.LINE_AA)
        os.makedirs(OUTF, exist_ok=True)
        f = os.path.join(OUTF, "fig_balance_dz%+d.png" % int(round(dz))); cv2.imwrite(f, img); figs.append(f)
    print("\nfigures: %s  (cells are laid out like the table in the survey image)" % ", ".join(figs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "report", "mosaic", "balance"])
    add_planner_args(ap)
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "report": cmd_report, "mosaic": cmd_mosaic, "balance": cmd_balance}[a.cmd](a)


if __name__ == "__main__":
    main()
