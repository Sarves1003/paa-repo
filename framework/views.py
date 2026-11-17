"""Active view selection for inspection: shared definitions (robot, Colab and analysis all use this file).

Candidate views  k = level * P + cell     (P = 25 table-grid cells from deploy/table_grid.json)
  level 0 : camera above the cell at VIEW_LEVELS_DZ[0] below the survey pose, centred by closed-loop servoing
  level 1 : same cell, camera lowered further to VIEW_LEVELS_DZ[1] (close-up); reached through level 0
  The survey image is taken at the locked pose (dz = 0); its scale is 1, view scales are measured.
Patches  p = the same 25 table cells (the part surface to be inspected)

Inspection quality of patch p in an image (label-free, measured on the real image):
  q = resolution x sharpness x (1 - glare)
    resolution : image scale relative to the closest level (1.0 for the close-up)
    sharpness  : log variance of the Laplacian, normalised by a library-wide reference
    glare      : fraction of saturated pixels (specular reflections on machined metal)
Value of a set of views S (weighted max-coverage, submodular -> views interact, which is why
a policy that attends between candidate views is tested against one that does not):
  V(S) = sum_p w_p * max(q_survey[p], max_{v in S} q[v, p]) / sum_p w_p,   w_p = part fraction of patch p
Reward = V(S) - V(survey only), under a time budget from the robot's own time model.

Tilted candidate views (unified mode with  lattice_views.py --orient-views): a level may also carry a camera
rotation (roll, pitch, yaw about the world axes). Such a level has a planned 3x3 homography (pinhole model, from
lattice_views.py) that maps cell-relative survey pixels into its image; after capture the real mapping is MEASURED by
registering the view to the survey (register_to_survey) and used for scoring. The policy sees the rotation of every
candidate (3 extra token features), so it can learn where a tilt pays off (e.g. less glare on machined metal) from
the real images, which the survey cannot predict.
"""
import numpy as np

T_SCALE = 3.0           # s, normalises times in the tokens
G_SCALE = 5.0           # scales the predicted-gain token
TOK_DIM, CTX_DIM = 14, 8      # tokens 0-10 as before; 11-13 = roll, pitch, yaw / ROT_SCALE
ROT_SCALE = 15.0      # deg
GLARE_LEVEL = 245


# ------------------------------------------------------------------ image measurements
def patch_window(center, half, shape, min_inside=0.9):
    h, w = shape[:2]
    x0, x1 = center[0] - half, center[0] + half
    y0, y1 = center[1] - half, center[1] + half
    ix0, ix1, iy0, iy1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    if (ix1 - ix0) * (iy1 - iy0) < min_inside * (x1 - x0) * (y1 - y0):
        return None
    return int(round(iy0)), int(round(iy1)), int(round(ix0)), int(round(ix1))


def patch_stats(gray, center, half):
    """(log Laplacian variance, glare fraction, edge density) of a square patch, or None if not visible."""
    import cv2
    win = patch_window(center, half, gray.shape)
    if win is None:
        return None
    y0, y1, x0, x1 = win
    p = gray[y0:y1, x0:x1]
    if p.size < 16:
        return None
    lap = float(np.log1p(cv2.Laplacian(p, cv2.CV_64F).var()))
    glare = float((p >= GLARE_LEVEL).mean())
    edge = float((cv2.Canny(p, 50, 150) > 0).mean())
    return lap, glare, edge


def quality(lap, glare, res, lap_ref):
    return float(np.clip(lap / max(lap_ref, 1e-6), 0, 1) * (1.0 - glare) * res)


def to_gray(img):
    import cv2
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ------------------------------------------------------------------ geometry
def _apply_h(H, u):
    u = np.asarray(u, float)
    x = H @ np.r_[u, 1.0]
    return x[:2] / x[2]


class ViewSet:
    """Candidate views and patches from table_grid.json (survey-image pixels of the 25 cells)."""
    def __init__(self, tg, levels_dz=(-60.0, -80.0), scales=None, uc=(320.0, 240.0), via_hi=True,
                 level_H=None, level_rpy=None):
        """via_hi=True: close-ups are reached from (and mapped through) the level-0 view of the same cell;
        via_hi=False (lattice mode): every view is reached on its own lattice route and verified directly.
        level_H / level_rpy (lattice mode): planned homography and camera rotation of every level (tilted views)."""
        self.via_hi = bool(via_hi)
        self.P = len(tg["image_px"])
        self.n = int(round(np.sqrt(self.P)))
        self.u = np.asarray(tg["image_px"], float)                       # (P,2) survey pixels
        self.off = np.asarray(tg["offsets_mm"], float)                   # (P,2) robot offsets
        self.tab = np.asarray(tg["table_mm"], float)
        self.half_px = float(tg["box_px_per_mm"]) * float(tg["pitch_mm"]) / 2.0
        self.dz = np.asarray(levels_dz, float)
        self.L = len(self.dz)
        self.K = self.L * self.P
        self.uc = np.asarray(uc, float)
        self.scales = np.ones(self.L) if scales is None else np.asarray(scales, float)
        self.rpy = np.zeros((self.L, 3)) if level_rpy is None else np.asarray(level_rpy, float).reshape(self.L, 3)
        self.rot = np.abs(self.rpy).max(1) > 1e-9                         # (L,) level is a tilted/rotated view
        self.Hl = None if level_H is None else [np.asarray(h, float) for h in level_H]
        if self.rot.any() and self.Hl is None:
            raise ValueError("rotated levels need their planned homographies (level_H)")
        if self.rot.all():
            raise ValueError("at least one level must keep the reference orientation")
        st = np.flatnonzero(~self.rot)
        self.lv_high = int(st[np.argmax(self.dz[st])]); self.lv_low = int(st[np.argmin(self.dz[st])])
        c = self.tab.mean(0); r = np.abs(self.tab - c).max()
        self.xy = np.tile((self.tab - c) / max(r, 1e-6), (self.L, 1))   # (K,2) in [-1,1]
        self.level = np.repeat(np.arange(self.L), self.P)
        self.cell = np.tile(np.arange(self.P), self.L)
        self.hi = self.cell.copy()                                        # level-0 view of the same cell
        self.res = self.scales / self.scales.max()                        # resolution factor per level
        self.res_survey = 1.0 / self.scales.max()                         # survey image (scale 1)
        self.rot_tok = np.repeat(self.rpy / ROT_SCALE, self.P, axis=0)    # (K,3) token features

    @classmethod
    def from_meta(cls, meta, uc=None, tg=None):
        """ViewSet of a view library (data/view_library.npz meta) or of a placement meta.json."""
        return cls(tg if tg is not None else meta["table_grid"], meta.get("levels_dz", (0.0, meta.get("dz", 0.0))),
                   scales=meta.get("scales"), uc=uc if uc is not None else meta["uc"], via_hi=meta.get("via_hi", True),
                   level_H=meta.get("levels_H"), level_rpy=meta.get("levels_rpy"))

    def point_px(self, k, u, aff=None):
        """Pixel of survey point u in the image of view k.
        aff 3x3: MEASURED survey -> view homography (register_to_survey); aff 2x3: measured level0 -> close-up map
        (old two-level mode); otherwise the planned model of the level."""
        u = np.asarray(u, float)
        if aff is not None and np.shape(aff) == (3, 3):
            return _apply_h(np.asarray(aff, float), u)
        c, l = self.cell[k], self.level[k]
        u0 = self.uc + (u - self.u[c])                                    # cell-relative, survey scale
        if self.rot[l]:
            return _apply_h(self.Hl[l], u0)
        if l == 0:
            return self.uc + self.scales[0] * (u0 - self.uc)
        if aff is not None and self.via_hi:                               # measured level0 -> close-up map
            return aff[:, :2] @ (self.uc + self.scales[0] * (u0 - self.uc)) + aff[:, 2]
        return self.uc + self.scales[l] * (u0 - self.uc)

    def view_px(self, k, p, aff=None):
        """Pixel of patch p in the image of view k."""
        return self.point_px(k, self.u[p], aff)

    def half(self, k):
        return self.half_px * self.scales[self.level[k]]

    def footprint(self, shape, k, aff=None):
        h, w = shape[:2]
        out = np.zeros(self.P, bool)
        for p in range(self.P):
            out[p] = patch_window(self.view_px(k, p, aff), self.half(k), (h, w)) is not None
        return out


_TABLE_MASK = {}


def table_mask(survey, G=None, pad=30):
    """255 = table, 0 = the part (its detected box + pad). The part stands up to ~60 mm above the table, so in a
    tilted or shifted view its features move differently from the table (parallax): matching them would pull the
    table homography off. Cached per survey image."""
    key = id(survey)
    if key not in _TABLE_MASK:
        m = None
        try:
            from servo_scaled import part_mask
            m = part_mask(survey, G, pad=pad)
        except Exception:
            m = None
        if len(_TABLE_MASK) > 4:
            _TABLE_MASK.clear()
        _TABLE_MASK[key] = m
    return _TABLE_MASK[key]


def register_to_survey(img, survey, min_inliers=25, mask=None):
    """MEASURED homography survey -> view (ORB + RANSAC on the table plane: box line, markers, sheet texture), or
    None. Used to verify and score views the marker servo cannot handle (tilted views). mask (survey pixels,
    255 = use): keep survey features on the TABLE only (see table_mask)."""
    import cv2
    a, b = to_gray(survey), to_gray(img)
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(a, mask); kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < min_inliers:
        return None, len(good)
    pa = np.float32([ka[x.queryIdx].pt for x in good]); pb = np.float32([kb[x.trainIdx].pt for x in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    n = int(inl.sum()) if inl is not None else 0
    if H is None or n < min_inliers:
        return None, n
    det = abs(np.linalg.det(H[:2, :2] / H[2, 2]))
    if not (0.3 < det < 9.0):                                            # implausible scale: reject
        return None, n
    return H, n


def registration_servo(grab, move_by, survey, u_target, px2off, uc, px_per_mm, iters=3, tol_mm=1.5,
                       max_step_mm=20.0, mask=None):
    """Closed-loop centring of a view that the marker servo cannot handle (tilted camera): register the image to the
    survey, find the survey point now at the image centre (u_c = H^-1 uc), move the camera (orientation held) by the
    robot offset between u_c and the aimed survey point, repeat. Pure function: grab() -> image,
    move_by(d_mm) -> bool, px2off(u) -> robot offset (mm) that centres survey pixel u.
    Returns dict(frame, H, inliers, err_first_mm, err_mm, scale, iters, verify)."""
    out = dict(frame=None, H=None, inliers=0, err_first_mm=float("nan"), err_mm=float("nan"), scale=float("nan"),
               iters=0, verify="none")
    u_target = np.asarray(u_target, float)
    for it in range(iters + 1):
        img = grab(); out["frame"] = img
        H, n = register_to_survey(img, survey, mask=mask)
        out["inliers"] = n
        if H is None:
            out["verify"] = "none (registration failed, %d matches)" % n
            return out
        err, sc = registered_view_check(H, u_target, uc, px_per_mm)
        out.update(H=H, err_mm=err, scale=sc, iters=it, verify="registration servo (tilted)")
        if it == 0:
            out["err_first_mm"] = err
        if err <= tol_mm or it == iters:
            return out
        u_c = _apply_h(np.linalg.inv(H), uc)                         # survey point currently on the optical axis
        dd = np.asarray(px2off(u_target), float) - np.asarray(px2off(u_c), float)
        nrm = float(np.linalg.norm(dd))
        if nrm > max_step_mm:
            dd = dd * max_step_mm / nrm
        if not move_by(dd):
            out["verify"] = "registration (tilted), correction refused"
            return out
    return out


def registered_view_check(H, u_target, uc, px_per_mm):
    """(view error in mm, local scale) of a view registered to the survey: where the aimed survey pixel landed."""
    x = _apply_h(H, u_target)
    eps = 2.0
    dx = (_apply_h(H, np.asarray(u_target) + [eps, 0]) - x) / eps
    dy = (_apply_h(H, np.asarray(u_target) + [0, eps]) - x) / eps
    s = float(np.sqrt(abs(dx[0] * dy[1] - dx[1] * dy[0])))
    err_mm = float(np.linalg.norm(x - np.asarray(uc, float)) / max(s * px_per_mm, 1e-6))
    return err_mm, s


def part_weights(survey, vs, locate_part=None):
    """Fraction of each patch covered by the part (bbox of the detected part; whole box if not found)."""
    bbox = None
    if locate_part is not None:
        try:
            part = locate_part(survey)
            bbox = part["bbox_px"] if part else None
        except Exception:
            bbox = None
    w = np.ones(vs.P)
    if bbox is not None:
        x, y, bw, bh = [float(v) for v in bbox]
        for p in range(vs.P):
            ux, uy = vs.u[p]; h = vs.half_px
            ox = max(0.0, min(ux + h, x + bw) - max(ux - h, x))
            oy = max(0.0, min(uy + h, y + bh) - max(uy - h, y))
            w[p] = ox * oy / (4 * h * h)
    return w, bbox


def survey_features(survey, vs, lap_ref, locate_part=None):
    """Everything the planner may know BEFORE moving: per-patch survey stats and predicted view quality."""
    g = to_gray(survey)
    P, K = vs.P, vs.K
    lap, glare, edge = np.zeros(P), np.zeros(P), np.zeros(P)
    for p in range(P):
        s = patch_stats(g, vs.u[p], vs.half_px)
        if s is not None:
            lap[p], glare[p], edge[p] = s
    res_survey = vs.res_survey
    qs = np.array([quality(lap[p], glare[p], res_survey, lap_ref) for p in range(P)])
    w, bbox = part_weights(survey, vs, locate_part)
    F = np.stack([vs.footprint(g.shape, k) for k in range(K)])           # (K,P) predicted visibility
    qpred = np.zeros((K, P))
    for k in range(K):
        qpred[k] = np.clip(lap / max(lap_ref, 1e-6), 0, 1) * (1 - glare) * vs.res[vs.level[k]] * F[k]
    Wt = max(w.sum(), 1e-6)
    fp_part = (F * w[None]).sum(1) / Wt
    den = np.maximum((F * w[None]).sum(1), 1e-6)
    fp_edge = (F * (w * edge)[None]).sum(1) / den
    fp_glare = (F * (w * glare)[None]).sum(1) / den
    return dict(qs=qs, w=w, F=F.astype(np.float32), qpred=qpred, fp_part=fp_part, fp_edge=fp_edge,
                fp_glare=fp_glare, lap_s=lap, glare_s=glare, edge_s=edge, bbox=bbox)


def view_quality(img, k, vs, lap_ref, aff=None):
    """Measured quality of every patch in the image actually taken at view k (0 where not visible)."""
    g = to_gray(img)
    q = np.zeros(vs.P); vis = np.zeros(vs.P, bool)
    for p in range(vs.P):
        s = patch_stats(g, vs.view_px(k, p, aff), vs.half(k))
        if s is not None:
            q[p] = quality(s[0], s[1], vs.res[vs.level[k]], lap_ref); vis[p] = True
    return q, vis


def estimate_affine(img_hi, img_lo, min_inliers=15):
    """2x3 map from level-0 view pixels to close-up pixels (ORB + RANSAC similarity), or None."""
    import cv2
    a, b = to_gray(img_hi), to_gray(img_lo)
    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(a, None); kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < min_inliers:
        return None
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inl is None or inl.sum() < min_inliers:
        return None
    return M


# ------------------------------------------------------------------ time model
def levels_from_config():
    import config as C
    return tuple(float(v) for v in getattr(C, "VIEW_LEVELS_DZ", (-60.0, -80.0)))


def scale_guess(levels, cam_height_mm):
    """Magnification of each level relative to the survey image (pinhole, camera looking down)."""
    return [cam_height_mm / max(cam_height_mm + dz, 1.0) for dz in levels]


def time_matrix(qdeg, servo_s, vs, t0=0.5, omega=30.0):
    """qdeg (K+1,6): joint angles of every view, index K = survey pose. servo_s (K): servo time per view.
    T[i, j] = time to go from node i and capture view j (close-ups go through their level-0 view)."""
    K = vs.K
    dq = np.abs(qdeg[:, None, :] - qdeg[None, :, :]).max(-1)
    mv = t0 + dq / omega
    T = np.zeros((K + 1, K))
    for j in range(K):
        h = vs.hi[j]
        if vs.level[j] == 0:
            T[:, j] = mv[:, j] + servo_s[j]
        else:
            T[:, j] = mv[:, h] + servo_s[h] + mv[h, j]
            T[h, j] = mv[h, j]
    return T


# ------------------------------------------------------------------ batched environment
class BatchEnv:
    """Batch of placements. inst: dict of stacked arrays (B, ...). Q may be filled progressively (live)."""
    def __init__(self, inst, vs, budget):
        self.i, self.vs = inst, vs
        B = inst["w"].shape[0]
        self.B, self.K = B, vs.K
        self.budget = np.asarray(budget, float) * np.ones(B)
        self.best = inst["qs"].copy()
        self.visited = np.zeros((B, vs.K), bool)
        self.cur = np.full(B, vs.K)
        self.t_used = np.zeros(B)
        self.step_i = np.zeros(B)
        self.W = np.maximum(inst["w"].sum(1), 1e-6)
        self.v0 = self.value()

    def value(self):
        return (self.i["w"] * self.best).sum(1) / self.W

    def gain(self, q):
        return (np.maximum(q - self.best[:, None, :], 0) * self.i["w"][:, None, :]).sum(2) / self.W[:, None]

    def t_next(self):
        return self.i["T"][np.arange(self.B), self.cur]

    def mask(self):
        return (self.visited | ~self.i["avail"] |
                (self.t_used[:, None] + self.t_next() > self.budget[:, None] + 1e-9))

    def obs(self):
        vs, i = self.vs, self.i
        B, K = self.B, self.K
        m = self.mask()
        pg = self.gain(i["qpred"] * i["F"])
        hv = self.visited[:, vs.hi]
        tok = np.concatenate([
            np.broadcast_to(vs.xy[None], (B, K, 2)), np.broadcast_to(vs.level[None, :, None], (B, K, 1)) / max(vs.L - 1, 1),
            i["fp_part"][..., None], i["fp_edge"][..., None], i["fp_glare"][..., None],
            (G_SCALE * pg)[..., None], i["avail"][..., None], self.visited[..., None],
            (self.t_next() / T_SCALE)[..., None], hv[..., None],
            np.broadcast_to(vs.rot_tok[None], (B, K, 3))], -1).astype(np.float32)
        cur_ok = self.cur < K
        cx = np.where(cur_ok, vs.xy[np.minimum(self.cur, K - 1), 0], 0.0)
        cy = np.where(cur_ok, vs.xy[np.minimum(self.cur, K - 1), 1], 0.0)
        cl = np.where(cur_ok, vs.level[np.minimum(self.cur, K - 1)], 0.0)
        left = (self.budget - self.t_used) / np.maximum(self.budget, 1e-6)
        ctx = np.stack([left, G_SCALE * (self.value() - self.v0), self.step_i / K, i["pose"],
                        cx, cy, cl, self.visited.sum(1) / K], 1).astype(np.float32)
        return tok, ctx, m

    def done(self):
        return self.mask().all(1)

    def step(self, a):
        live = ~self.done()
        for b in np.flatnonzero(live):
            k = int(a[b]); h = self.vs.hi[k]
            self.t_used[b] += self.i["T"][b, self.cur[b], k]
            if self.vs.via_hi and self.vs.level[k] > 0 and not self.visited[b, h]:
                self.visited[b, h] = True
                self.best[b] = np.maximum(self.best[b], self.i["Q"][b, h])
            self.visited[b, k] = True
            self.best[b] = np.maximum(self.best[b], self.i["Q"][b, k])
            self.cur[b] = k; self.step_i[b] += 1
        return live

    def improvement(self):
        return self.value() - self.v0


# ------------------------------------------------------------------ planners
def snake(vs, level):
    n = vs.n
    out = []
    for r in range(n):
        cols = range(n) if r % 2 == 0 else range(n - 1, -1, -1)
        out += [level * vs.P + r * n + c for c in cols]
    return out


def choose(env, method, rng, W=None):
    """Next view for every row (B,), by method name. -1 where nothing is allowed."""
    m = env.mask(); B, K = m.shape
    a = np.full(B, -1)
    if method in ("policy", "policy_mlp"):
        from geo_attn import forward_np
        tok, ctx, mm = env.obs()
        lg = forward_np(W, tok, ctx, mm)
        a = np.where(m.all(1), -1, np.argmax(lg, 1))
        return a
    T = env.t_next()
    if method == "greedy_pred":
        score = env.gain(env.i["qpred"] * env.i["F"]) / np.maximum(T, 1e-3)
    elif method == "oracle":
        score = env.gain(env.i["Q"]) / np.maximum(T, 1e-3)
    elif method == "nn_time":
        score = np.where(env.i["fp_part"] > 0.02, -T, -1e6 - T)          # nearest view that sees the part
    elif method == "setcover":
        # classical geometric coverage (Part A): resolution gained on not-yet-covered part area per second
        R = env.vs.res[env.vs.level][None, :, None] * env.i["F"]                       # (B,K,P)
        cov = np.maximum(env.vs.res_survey, (R * env.visited[:, :, None]).max(1))      # (B,P)
        g = (np.maximum(R - cov[:, None, :], 0) * env.i["w"][:, None, :]).sum(2) / env.W[:, None]
        score = g / np.maximum(T, 1e-3)
    elif method == "random":
        score = rng.random(m.shape)
    elif method in ("raster_high", "raster_low"):
        order = snake(env.vs, env.vs.lv_high if method == "raster_high" else env.vs.lv_low)
        rank = np.full(K, -1e9); rank[order] = -np.arange(len(order))
        score = np.where(env.i["fp_part"] > 0.02, rank[None], rank[None] - 1e6)
    else:
        raise ValueError(method)
    score = np.where(m, -np.inf, score)
    a = np.where(m.all(1), -1, np.argmax(score, 1))
    return a


def rollout(inst, vs, budget, method, rng=None, W=None):
    env = BatchEnv(inst, vs, budget)
    rng = rng or np.random.default_rng(0)
    seqs = [[] for _ in range(env.B)]
    while not env.done().all():
        a = choose(env, method, rng, W)
        for b in np.flatnonzero(a >= 0):
            seqs[b].append(int(a[b]))
        env.step(np.maximum(a, 0))
    return env.improvement(), env.t_used, seqs


def stack(insts, keys=("Q", "F", "qpred", "qs", "w", "T", "avail", "fp_part", "fp_edge", "fp_glare", "pose")):
    return {k: np.stack([np.asarray(d[k]) for d in insts]).astype(bool if k == "avail" else float) for k in keys}
