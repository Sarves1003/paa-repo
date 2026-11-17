#!/usr/bin/env python3
"""SOGA: Symmetry-Orthogonal, Glare-complementary Arc. A robot-motion strategy for inspecting shiny
axisymmetric parts (the ringed cylinder) with a wrist camera and very small robot motion.

Geometry it rests on (part lying on the table, axis a, radius r, centre c at height r):
  P1  For a camera whose field of view contains the part, which surface patches can be seen, and at what angle,
      depends only on the camera POSITION in the plane perpendicular to the axis: the angle
      alpha = atan2((C - c).b, (C - c).z), with b = a x z. Moving along the axis, or rotating the camera, adds
      nothing to the circumferential coverage; isotropic xy motion wastes about half of the motion budget.
  P2  With a distant light, the specular highlight on the cylinder sits at the surface angle where the normal
      bisects the directions to the camera and the light: phi_h = (alpha_v + alpha_L) / 2. Moving the camera
      by d_alpha moves the highlight by d_alpha / 2.
  P3  Two views whose highlights are at least w_h apart (w_h = angular width of the highlight) never hide the
      same patch: |alpha_1 - alpha_2| >= 2 w_h makes the pair "glare-complementary".
So SOGA (i) finds the part axis, radius, light direction and highlight width from the ONE survey image, (ii) puts
every view on an arc in the plane perpendicular to the axis (all motion spent where it changes what is seen),
(iii) chooses the fewest arc positions that see every upper-surface patch at the required resolution and incidence
WITHOUT glare in at least one view, inside the certified safe set (IK continuation, kappa, joint margin, fence),
and (iv) traces them as one monotone sweep. The result is checked on the real images: every patch of the
unrolled part surface is sampled from the captured images, and glare is measured there, not modelled.

  python3 soga.py analyze $OPTS                     # no motion: SOGA vs the current plans, 6 part orientations
  python3 soga.py run --tag soga_01 $OPTS           # robot: survey -> SOGA views -> unrolled surface + coverage
  python3 soga.py run --tag base_01 --method pitch $OPTS   # same, with the current fixed pitch +-10 deg views
  python3 soga.py report                            # all runs: measured glare-free coverage, motion, time
Outputs: results_soga/*.json, *.png ; hw_logs/soga/<tag>/
"""
import argparse, glob, itertools, json, os, time
import numpy as np
import cv2
import config as C
import refcover_updated as RC
from kinematics import fk
from ik import ik_hold

OUT = "results_soga"
GLARE_V = 240                         # value channel at/above which a pixel is treated as specular glare
EZ = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------ small geometry helpers
def rot(axis, ang):
    """Rodrigues rotation (axis unit 3-vector, angle rad)."""
    k = np.asarray(axis, float); k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


class Part:
    """Lying cylinder: centre c (robot xy, mm, relative to the reference camera nadir), unit axis a (xy), radius r,
    length L. Surface chart: s along the axis (-L/2..L/2), phi around it (0 = top, + towards b = a x z)."""
    def __init__(self, c, a, r, L):
        self.c = np.asarray(c, float); a = np.asarray(a, float)
        self.a = np.r_[a / np.linalg.norm(a), 0.0]
        self.b = np.cross(self.a, EZ)
        self.r, self.L = float(r), float(L)
        self.c3 = np.r_[self.c, self.r]

    def rotated(self, beta_deg):
        th = np.radians(beta_deg)
        a = np.array([np.cos(th), np.sin(th)])
        return Part(self.c, a, self.r, self.L)

    def chart(self, ns=36, nphi=120):
        s = (np.arange(ns) + 0.5) / ns * self.L - self.L / 2
        phi = np.radians(-180 + (np.arange(nphi) + 0.5) * 360.0 / nphi)
        S, PH = np.meshgrid(s, phi)                                         # (nphi, ns)
        n = np.sin(PH)[..., None] * self.b + np.cos(PH)[..., None] * EZ
        X = self.c3 + S[..., None] * self.a + self.r * n
        return S, PH, X, n

    def alpha(self, Cam):
        """Angle of a camera position around the axis (0 = straight above the axis)."""
        d = np.asarray(Cam, float) - self.c3
        return float(np.arctan2(d @ self.b, d[2]))

    def to_json(self):
        return dict(c=self.c.tolist(), a=self.a[:2].tolist(), r=self.r, L=self.L)


def project(geo, X, Cam, Rt):
    """Pixels of 3-D points X (mm; xy = robot offset from the reference nadir, z above the table) in a camera at Cam
    (same frame) rotated by Rt (world frame, as refcover_updated.orient_R). Same model as Geo.project."""
    w = X.reshape(-1, 3) - np.asarray(Cam, float)[None]
    v = w @ Rt
    vx = v[:, :2] @ geo.S.T; vz = -v[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = geo.uc[None] + geo.f * vx / np.where(vz > 1e-6, vz, np.nan)[:, None]
    return u.reshape(X.shape[:-1] + (2,)), vz.reshape(X.shape[:-1])


def backproject(geo, U, z):
    """Inverse of project() for the SURVEY camera (reference pose): pixels -> xy (mm, robot frame relative to the
    reference nadir) of points at height z above the table. Same pinhole model as project(), so the part estimate,
    the planned views and the unrolled surface all use one consistent geometry."""
    U = np.atleast_2d(np.asarray(U, float))
    return ((U - geo.uc[None]) @ geo.S) * (geo.H - z) / geo.f


def visibility(geo, part, Cam, Rt, psi_max, rho_min, margin=12, chart=None):
    """Per surface patch: seen (in image, facing the camera, incidence <= psi_max, resolution >= rho_min),
    incidence angle, pixel coordinates."""
    S, PH, X, n = chart or part.chart()
    u, vz = project(geo, X, Cam, Rt)
    to = np.asarray(Cam, float) - X
    dist = np.linalg.norm(to, axis=-1)
    cosp = (n * to).sum(-1) / dist
    res = geo.f * cosp / dist                                               # px per mm on the surface
    W, Hh = 2 * geo.uc
    inside = (u[..., 0] >= margin) & (u[..., 0] < W - margin) & (u[..., 1] >= margin) & (u[..., 1] < Hh - margin)
    seen = inside & (vz > 0) & (cosp >= np.cos(np.radians(psi_max))) & (res >= rho_min)
    return seen, cosp, u


def highlight_phi(part, Cam, alpha_L, iters=6):
    """Predicted highlight angle (P2), solved by fixed-point iteration (the direction to the camera depends on the
    highlight point itself)."""
    phi = 0.5 * (part.alpha(Cam) + alpha_L)
    for _ in range(iters):
        X = part.c3 + part.r * (np.sin(phi) * part.b + np.cos(phi) * EZ)
        d = np.asarray(Cam, float) - X
        av = np.arctan2(d @ part.b, d[2])
        phi = 0.5 * (av + alpha_L)
    return float(phi)


# ------------------------------------------------------------------ 1) part, light and highlight from ONE survey image
def estimate(geo, survey, pose="horizontal", axis_deg=None, part_r=None, part_L=None):
    """Axis, radius, length, centre (robot mm), light angle alpha_L and highlight width w_h from the survey."""
    from perception import locate_part_scene
    H = geo.H
    det = locate_part_scene(survey, geo.G)
    if det is None:
        raise SystemExit("no part found in the survey image")
    x, y, w, h = [int(v) for v in det["bbox_px"]]; pad = 15
    x0, y0 = max(0, x - pad), max(0, y - pad); x1, y1 = min(survey.shape[1], x + w + pad), min(survey.shape[0], y + h + pad)
    roi = survey[y0:y1, x0:x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m |= ((hsv[..., 2] >= GLARE_V) & (cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0)).astype(np.uint8) * 255   # glare is part of the part
    # an opening with a 7 px kernel removes the drawn box line and other thin strokes (about 3 px wide) even where
    # they touch the part, so they cannot join its silhouette and inflate the radius
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in cs if min(cv2.minAreaRect(c)[1]) >= 10]          # drop thin strokes (lines, edges of the sheet)
    if not cs:
        raise SystemExit("part segmentation failed (only thin strokes found)")
    if not cs:
        raise SystemExit("part segmentation failed")
    # shiny metal splits into dark and bright bands: keep every blob inside the detected part box and take the
    # convex hull (a cylinder seen from above is convex), so bright bands do not cut the silhouette
    bx0, by0, bx1, by1 = x - x0 - 3, y - y0 - 3, x - x0 + w + 3, y - y0 + h + 3
    keep = [c for c in cs if cv2.contourArea(c) > 0.002 * w * h and
            (lambda r_: r_[0] >= bx0 and r_[1] >= by0 and r_[0] + r_[2] <= bx1 and r_[1] + r_[3] <= by1)(cv2.boundingRect(c))]
    if not keep:
        keep = [max(cs, key=cv2.contourArea)]
    cnt = cv2.convexHull(np.vstack(keep))
    mask = np.zeros_like(m); cv2.drawContours(mask, [cnt], -1, 255, -1)
    pts = cnt.reshape(-1, 2).astype(float) + [x0, y0]
    mom = cv2.moments(mask); cu = np.array([mom["m10"] / mom["m00"] + x0, mom["m01"] / mom["m00"] + y0])
    # axis direction in the image. 1st cue: specular highlights on a cylinder are stripes PARALLEL to the axis
    # (principal direction of the saturated pixels). 2nd cue: the shading bands are parallel to the axis, so the
    # dominant intensity gradient is PERPENDICULAR to it (minor eigenvector of the gradient structure tensor).
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    yy, xx = np.nonzero((hsv[..., 2] >= GLARE_V) & inner)
    ax_img, how, coherence = None, "", 0.0
    # 1st cue: COLOUR bands. Rings, seals and coatings run AROUND the axis, so the edges between colours (a*, the
    # red-green channel, blind to glare and shading) are perpendicular to the axis: the dominant colour gradient
    # points along it
    lab_ = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    ga = cv2.Sobel(lab_[..., 1], cv2.CV_32F, 1, 0, ksize=5); gb = cv2.Sobel(lab_[..., 1], cv2.CV_32F, 0, 1, ksize=5)
    Ja = np.array([[np.sum(ga[inner] ** 2), np.sum(ga[inner] * gb[inner])],
                   [np.sum(ga[inner] * gb[inner]), np.sum(gb[inner] ** 2)]])
    eva, eveca = np.linalg.eigh(Ja)
    coh_a = float((eva[1] - eva[0]) / max(eva[1] + eva[0], 1e-9))
    energy_a = float(np.sqrt(eva[1] / max(inner.sum(), 1)))
    if coh_a >= 0.3 and energy_a >= 8.0:
        ax_img, how, coherence = eveca[:, 1], "colour bands", coh_a
    if ax_img is None and len(xx) >= 40:
        P0 = np.c_[xx, yy].astype(float); P0 -= P0.mean(0)
        ev, evec = np.linalg.eigh(P0.T @ P0)
        if ev[1] > 4 * max(ev[0], 1e-9):
            ax_img, how, coherence = evec[:, 1], "highlight stripes", float(1 - ev[0] / ev[1])
    if ax_img is None:
        gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        J = np.array([[np.sum(gx[inner] ** 2), np.sum(gx[inner] * gy[inner])],
                      [np.sum(gx[inner] * gy[inner]), np.sum(gy[inner] ** 2)]])
        ev, evec = np.linalg.eigh(J)
        ax_img, how, coherence = evec[:, 0], "shading bands", float((ev[1] - ev[0]) / max(ev[1] + ev[0], 1e-9))
    (rc, (rw, rh), rang) = cv2.minAreaRect(cnt.astype(np.float32))
    if axis_deg is not None:
        ax_img, how = np.array([np.cos(np.radians(axis_deg)), np.sin(np.radians(axis_deg))]), "given (--axis-deg)"
    pe_img = np.array([-ax_img[1], ax_img[0]])
    # pixels -> robot frame with the same pinhole model used for planning and unwrapping (backproject); the
    # silhouette edges and the axis lie at about the height of the axis (z = r), found by a short fixed point
    if pose != "horizontal":
        raise SystemExit("SOGA is implemented for the lying (horizontal) part")
    tw, tl = (pts - cu) @ pe_img, (pts - cu) @ ax_img
    C0 = np.array([0.0, 0.0, H])
    a_rob = backproject(geo, (cu + 40 * ax_img)[None], 30.0)[0] - backproject(geo, cu[None], 30.0)[0]
    a_rob /= np.linalg.norm(a_rob); a3 = np.r_[a_rob, 0.0]; b3 = np.cross(a3, EZ)
    def ray(u):                                           # world direction of the survey ray through pixel u
        d = np.r_[backproject(geo, np.asarray(u, float)[None], H - 1.0)[0], -1.0]
        return d / np.linalg.norm(d)
    # cross-section: the two silhouette rays are tangent to the circle (centre at height r); the centre lies on their
    # bisector at distance D = r / sin(delta/2) and at height r  =>  r = H / (1 + cos(beta) / sin(delta/2))
    angs = []
    for u in (cu + tw.min() * pe_img, cu + tw.max() * pe_img):
        d = ray(u); d = d - (d @ a3) * a3
        angs.append(np.arctan2(d @ b3, -d[2]))
    beta, delta = 0.5 * (angs[0] + angs[1]), abs(angs[1] - angs[0])
    r = H / (1.0 + np.cos(beta) / np.sin(delta / 2)); D = r / np.sin(delta / 2)
    s_c = backproject(geo, cu[None], r)[0] @ a_rob
    c = s_c * a_rob + D * np.sin(beta) * b3[:2]
    # length: the end silhouettes are the top rims (z = 2r), except an end facing the nadir from outside the part,
    # whose visible extreme is its bottom rim (z = 0)
    L = float(np.linalg.norm(backproject(geo, np.array([cu + tl.min() * ax_img, cu + tl.max() * ax_img]), r) @ a_rob))
    for _ in range(3):
        s_n = -(c @ a_rob)                               # nadir position along the axis, relative to the centre
        ends = []
        for u in (cu + tl.min() * ax_img, cu + tl.max() * ax_img):
            s2 = backproject(geo, u[None], 2 * r)[0] @ a_rob - c @ a_rob
            s0 = backproject(geo, u[None], 0.0)[0] @ a_rob - c @ a_rob
            z0_side = abs(s_n) > L / 2 and np.sign(s2) == np.sign(s_n)
            ends.append(s0 if z0_side else s2)
        L = float(abs(ends[1] - ends[0]))
        c = c + 0.5 * (ends[0] + ends[1]) * a_rob        # centre between the two ends
    r_img, L_img = float(r), float(L)
    if part_r:
        r = float(part_r)                                   # ruler value given: the image estimate is only reported
    if part_L:
        L = float(part_L)
    part = Part(c, a_rob, r, L)
    # lights: highlight pixels on the part -> surface angle phi of each (inverse projection on the cylinder);
    # the phi histogram is split into lobes (one per light / lamp tube); lobe i: alpha_L,i = 2 phi_i - alpha_v (P2)
    hl = (hsv[..., 2] >= GLARE_V) & inner
    yy, xx = np.nonzero(hl)
    lobes, n_hl = [], int(len(xx))
    if n_hl >= 20:
        U = np.c_[xx + x0, yy + y0].astype(float)
        phis = np.radians(np.linspace(-89, 89, 357))
        Xs = part.c3[None] + part.r * (np.sin(phis)[:, None] * part.b + np.cos(phis)[:, None] * EZ)
        u_s, _ = project(geo, Xs, np.array([0.0, 0.0, H]), np.eye(3))
        u_c, _ = project(geo, part.c3[None], np.array([0.0, 0.0, H]), np.eye(3))
        u_pred = (u_s - u_c) @ pe_img                     # perpendicular image coordinate of every surface angle
        u_obs = (U - u_c) @ pe_img
        ph = np.degrees(phis[np.argmin(np.abs(u_obs[:, None] - u_pred[None]), 1)])
        hist, edges = np.histogram(ph, bins=np.arange(-90, 92, 2))
        on = hist >= max(3, 0.03 * n_hl)
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                sel = (ph >= edges[i]) & (ph < edges[j + 1])
                phi_c = np.radians(np.average(ph[sel]))
                w = np.radians(max(4.0, edges[j + 1] - edges[i]))
                av = part.alpha(np.array([0.0, 0.0, H]))
                lobes.append((2 * phi_c - av, w, int(sel.sum())))
                i = j + 1
            else:
                i += 1
    if not lobes:
        lobes = [(0.0, np.radians(10.0), 0)]
    alpha_L = [l[0] for l in lobes]; w_h = [l[1] for l in lobes]
    info = dict(part=part.to_json(), r_from_image=r_img, L_from_image=L_img,
                size_from="ruler (--part-r/--part-L)" if (part_r or part_L) else "image", axis_from=how, axis_coherence=coherence, rect_deg=float(rang),
                rect_wh=[float(rw), float(rh)], lights_alpha_deg=[float(np.degrees(v)) for v in alpha_L],
                highlight_width_deg=[float(np.degrees(v)) for v in w_h], highlight_px=n_hl,
                centroid_px=cu.tolist(), axis_img=ax_img.tolist())
    return part, alpha_L, w_h, info


def draw_estimate(geo, survey, part, path):
    """Survey image with the estimated cylinder drawn on it (outline at the axis height, end circles, axis): a quick
    visual check that radius, length, centre and axis are right before trusting the plan."""
    img = survey.copy(); C0 = np.array([0.0, 0.0, geo.H]); I3 = np.eye(3)
    s = np.linspace(-part.L / 2, part.L / 2, 40)
    for sgn in (-1, 1):                                       # silhouette lines (sides of the cylinder)
        X = part.c3[None] + s[:, None] * part.a + sgn * part.r * part.b
        u, _ = project(geo, X, C0, I3)
        cv2.polylines(img, [np.nan_to_num(u).astype(np.int32)], False, (0, 255, 0), 1, cv2.LINE_AA)
    ph = np.linspace(0, 2 * np.pi, 60)
    for e in (-1, 1):                                         # end circles
        X = part.c3[None] + e * part.L / 2 * part.a + part.r * (np.sin(ph)[:, None] * part.b + np.cos(ph)[:, None] * EZ)
        u, _ = project(geo, X, C0, I3)
        cv2.polylines(img, [np.nan_to_num(u).astype(np.int32)], True, (0, 200, 255), 1, cv2.LINE_AA)
    X = part.c3[None] + np.array([-part.L / 2, part.L / 2])[:, None] * part.a + np.r_[0, 0, part.r]
    u, _ = project(geo, X, C0, I3)
    cv2.line(img, tuple(int(v) for v in u[0]), tuple(int(v) for v in u[1]), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "r %.1f mm  L %.1f mm (green: sides, orange: ends, magenta: axis)" % (part.r, part.L), (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


# ------------------------------------------------------------------ 2) certified reachability of a camera pose
class Reach:
    def __init__(self, a):
        self.q_ref, self.R_ref, self.p_ref = RC.load_ref()
        self.geo = RC.Geo(); self.a = a
        # planned views stay 8 mm further inside than the fence margin, so the closed-loop correction after the
        # move (typically 2-9 mm on this arm) still has room and is not refused at the fence
        self.fence = float(C.FENCE_AUTO_HALF_MM) - 5.0 - 8.0

    def solve(self, Cam, Rt, seed=None, steps=6):
        """IK continuation from the reference (or seed) to camera pose (Cam, Rt), in small interpolated steps so the
        solution stays on the reference joint branch. Returns (q or None, reason, waypoints)."""
        H = self.geo.H
        d = np.asarray(Cam[:2], float); dz = float(Cam[2] - H)
        if np.any(np.abs(d) >= self.fence):
            return None, "fence", []
        if dz < -self.a.max_descent - 1e-6:
            return None, "descent", []
        q = np.asarray(self.q_ref if seed is None else seed, float)
        T0 = fk(q); p0 = T0[:3, 3] * 1000; R0 = T0[:3, :3]
        p1 = self.p_ref + np.r_[d, dz]; R1 = Rt @ self.R_ref
        way = []
        # interpolate position linearly and orientation by the relative rotation's axis-angle
        dR = R1 @ R0.T; ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        ax = ax / np.linalg.norm(ax) if np.linalg.norm(ax) > 1e-9 else EZ
        # small continuation steps (<= 2 deg of rotation, <= 8 mm of travel each) keep the IK on one joint branch
        steps = int(max(steps, np.ceil(np.degrees(ang) / 2.0), np.ceil(np.linalg.norm(p1 - p0) / 8.0)))
        for i in range(1, steps + 1):
            t = i / steps
            Ri = rot(ax, t * ang) @ R0 if ang > 1e-9 else R0
            qn, pe, re, ok = ik_hold((p0 + t * (p1 - p0)) / 1000.0, q, Ri, iters=800)
            if not ok:
                return None, "ik", way
            if np.degrees(np.max(np.abs(qn - q))) > self.a.max_jump:
                return None, "branch jump", way
            q = qn; way.append(q.copy())
        if RC.kappa(q) < min(self.a.kappa_min, 0.9 * RC.kappa(self.q_ref)):
            return None, "kappa", way
        if RC.limit_margin_deg(q) < 5.0:
            return None, "joint limit", way
        return q, "ok", way


# ------------------------------------------------------------------ 3) candidate views of each strategy
def arc_views(part, geo, dz0, alphas_deg, aim=True):
    """SOGA candidates: camera on the arc perpendicular to the axis through the part centre, at the inspection
    height (level arc: never lower than dz0), optionally re-aimed at the axis (rotation about the part axis)."""
    out = []
    h = geo.H + dz0 - part.r                                 # height of the camera above the axis
    for al in alphas_deg:
        A = np.radians(al)
        Cam = part.c3 + h * np.tan(A) * part.b + h * EZ
        Rt = rot(part.a, A) if aim else np.eye(3)
        # rot(a, A) maps z to cos A z + sin A (a x z) = cos A z + sin A b: the optical axis (-Rt z) points at the axis
        out.append(dict(name="arc%+.0f" % al, alpha=float(al), C=Cam, Rt=Rt))
    return out


def pitch_views(part, geo, dz0, tilts=(-10.0, 0.0, 10.0)):
    """Current strategy: fixed pitch about the robot y axis, re-aimed at the part centre (refcover tilt captures)."""
    out = []
    for t in tilts:
        Rt = RC.orient_R((0.0, t, 0.0))
        h = geo.H + dz0 - part.r
        Cam = part.c3 + h / Rt[2, 2] * Rt[:, 2]
        out.append(dict(name="pitch%+.0f" % t, alpha=float(np.degrees(part.alpha(Cam))), C=Cam, Rt=Rt))
    return out


def ring_views(part, geo, dz0, radius_mm, n=4):
    """Direction-agnostic orbit: n straight views on a circle around the part centre (+ the centre)."""
    out = [dict(name="ring_c", C=np.r_[part.c, geo.H + dz0], Rt=np.eye(3))]
    for i in range(n):
        th = 2 * np.pi * i / n
        Cam = np.r_[part.c + radius_mm * np.array([np.cos(th), np.sin(th)]), geo.H + dz0]
        out.append(dict(name="ring%d" % i, C=Cam, Rt=np.eye(3)))
    for v in out:
        v["alpha"] = float(np.degrees(part.alpha(v["C"])))
    return out


def plan_views(geo, dz_min=None):
    """The Part-A coverage plan (deploy/refcover_plan.json): straight views at their offsets; heights below dz_min
    (the inspection limit) are raised to it, so every strategy is compared at the same allowed descent."""
    p = os.path.join(C.DEPLOY_DIR, "refcover_plan.json")
    if not os.path.exists(p):
        return []
    P = json.load(open(p))
    return [dict(name="plan%d" % i, C=np.r_[np.asarray(v["offset_mm"], float),
                                            geo.H + (float(v["dz"]) if dz_min is None else max(float(v["dz"]), dz_min))],
                 Rt=RC.orient_R(v.get("rpy", [0, 0, 0]))) for i, v in enumerate(P["views"])]


# ------------------------------------------------------------------ 4) model coverage and the SOGA selection
def model_masks(geo, part, views, alpha_L, w_h, psi_max, rho_min, chart):
    """Per view: patches seen, and seen without (predicted) glare."""
    S, PH, X, n = chart
    out = []
    for v in views:
        seen, cosp, _ = visibility(geo, part, v["C"], v["Rt"], psi_max, rho_min, chart=chart)
        glare = np.zeros(PH.shape, bool)
        for aL, wh in zip(np.atleast_1d(alpha_L), np.atleast_1d(w_h)):
            ph = highlight_phi(part, v["C"], aL)
            glare |= np.abs(np.angle(np.exp(1j * (PH - ph)))) < wh / 2
        out.append((seen, seen & ~glare))
    return out


def coverage(masks, PH):
    up = np.abs(PH) <= np.pi / 2
    if not masks:
        return dict(seen_upper=0.0, glarefree_upper=0.0, glarefree_all=0.0)
    seen = np.any([m[0] for m in masks], 0); good = np.any([m[1] for m in masks], 0)
    return dict(seen_upper=float(seen[up].mean()), glarefree_upper=float(good[up].mean()),
                glarefree_all=float(good.mean()))


def path_mm(views, start):
    pts = [np.asarray(start, float)] + [np.asarray(v["C"], float) for v in views] + [np.asarray(start, float)]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])))


def select_soga(geo, part, cands, masks, PH, start, n_max=4):
    """Fewest arc views that maximise glare-free upper coverage (exhaustive over the small candidate set);
    ties broken by the shortest closed path. Views are then ordered as one monotone sweep in alpha."""
    up = np.abs(PH) <= np.pi / 2
    G = np.array([m[1][up] for m in masks])
    best, best_key = None, None
    for k in range(1, min(n_max, len(cands)) + 1):
        for comb in itertools.combinations(range(len(cands)), k):
            cov = G[list(comb)].any(0).mean()
            sel = sorted(comb, key=lambda i: cands[i]["alpha"])
            key = (round(cov, 3), -k, -path_mm([cands[i] for i in sel], start))
            if best_key is None or key > best_key:
                best_key, best = key, sel
    return [cands[i] for i in best], float(best_key[0])


def feasible(reach, views):
    """Certify each view (IK continuation, kappa, margin, fence, descent)."""
    ok, why = [], {}
    for v in views:
        q, reason, _ = reach.solve(v["C"], v["Rt"])
        why[v["name"]] = reason
        if q is not None:
            v["q"] = q; ok.append(v)
    return ok, why


# ------------------------------------------------------------------ analyze (no motion)
def cmd_analyze(a):
    reach = Reach(a); geo = reach.geo
    sv_path = a.survey or os.path.join(C.LOGS_DIR, "grounding", "calib_survey.jpg")
    if a.sim_r:
        part0 = Part([a.sim_cx, a.sim_cy], [1.0, 0.0], a.sim_r, a.sim_L); alpha_L, w_h = [np.radians(a.sim_light)], [np.radians(a.sim_wh)]
        info = dict(source="simulated part")
    else:
        survey = cv2.imread(sv_path)
        if survey is None:
            raise SystemExit("no survey image %s (give --survey or the --sim-* options)" % sv_path)
        part0, alpha_L, w_h, info = estimate(geo, survey, axis_deg=a.axis_deg, part_r=a.part_r, part_L=a.part_L)
        info["source"] = sv_path
    start = np.r_[0.0, 0.0, geo.H]
    rho_min = a.rho_min or 0.9 * geo.f / geo.H               # 90 % of the survey's resolution on the table
    print("part: r %.1f mm, L %.1f mm, centre (%.1f, %.1f) mm, axis (%.2f, %.2f) [%s]; lights at %s deg, highlight widths %s deg; "
          "required %.2f px/mm, incidence <= %.0f deg" % (part0.r, part0.L, *part0.c, *part0.a[:2], info.get("axis_from", "-"),
          [round(float(np.degrees(v))) for v in np.atleast_1d(alpha_L)], [round(float(np.degrees(v))) for v in np.atleast_1d(w_h)],
          rho_min, a.psi_max))
    os.makedirs(OUT, exist_ok=True)
    rows = []
    betas = [None] if a.no_rotate else [None, 0, 30, 60, 90, 120, 150]
    for beta in betas:
        part = part0 if beta is None else part0.rotated(beta + 0.0)
        tag = "as placed" if beta is None else "axis %3d deg" % beta
        chart = part.chart()
        PH = chart[1]
        res = {}
        # SOGA: certified arc candidates -> exhaustive minimal selection
        cands, why = feasible(reach, arc_views(part, geo, a.dz, np.arange(-a.arc_max, a.arc_max + 1e-6, a.arc_step)))
        if cands:
            M = model_masks(geo, part, cands, alpha_L, w_h, a.psi_max, rho_min, chart)
            sel, cov = select_soga(geo, part, cands, M, PH, start, a.n_max)
            Ms = model_masks(geo, part, sel, alpha_L, w_h, a.psi_max, rho_min, chart)
            res["SOGA"] = dict(coverage(Ms, PH), n=len(sel), path_mm=path_mm(sel, start),
                               alphas=[v["alpha"] for v in sel], arc_reach=[float(c["alpha"]) for c in cands])
        else:
            res["SOGA"] = dict(coverage([], PH), n=0, path_mm=0.0, alphas=[], arc_reach=[], refused=why)
        for nm, vs in (("survey only", [dict(name="survey", C=start, Rt=np.eye(3))]),
                       ("pitch +-10 (current)", pitch_views(part, geo, a.dz)),
                       ("isotropic ring", ring_views(part, geo, a.dz, a.ring_mm)),
                       ("Part-A plan", plan_views(geo, a.dz))):
            ok, why = feasible(reach, vs)
            Mv = model_masks(geo, part, ok, alpha_L, w_h, a.psi_max, rho_min, chart)
            res[nm] = dict(coverage(Mv, PH), n=len(ok), path_mm=path_mm(ok, start), refused=sum(r != "ok" for r in why.values()))
        rows.append(dict(placement=tag, beta=beta, **{k: v for k, v in res.items()}))
        print("\n%s" % tag)
        for k, v in res.items():
            print("  %-22s views %d  path %6.0f mm  seen upper %5.1f%%  GLARE-FREE upper %5.1f%%%s"
                  % (k, v["n"], v["path_mm"], 100 * v["seen_upper"], 100 * v["glarefree_upper"],
                     ("  arc alphas %s" % [round(x) for x in v["alphas"]]) if k == "SOGA" else ""))
    json.dump(dict(info=info, lights_alpha_deg=np.degrees(np.atleast_1d(alpha_L)).tolist(),
                   highlight_width_deg=np.degrees(np.atleast_1d(w_h)).tolist(),
                   rho_min=rho_min, psi_max=a.psi_max, dz=a.dz, rows=rows),
              open(os.path.join(OUT, "soga_analyze.json"), "w"), indent=1, default=float)
    draw_bars(rows)
    print("\nsaved %s/soga_analyze.json and fig_soga_orientations.png" % OUT)


def draw_bars(rows):
    meths = [k for k in rows[0] if k not in ("placement", "beta")]
    cols = [(60, 140, 30), (150, 150, 150), (200, 120, 40), (40, 120, 220), (160, 60, 160)]
    Wd, Hd, m = 900, 420, 60
    img = np.full((Hd, Wd, 3), 255, np.uint8)
    n = len(rows); gw = (Wd - 2 * m) / n; bw = gw * 0.8 / len(meths)
    for y in range(0, 101, 20):
        yy = int(Hd - m - (Hd - 2 * m) * y / 100)
        cv2.line(img, (m, yy), (Wd - m, yy), (225, 225, 225), 1)
        cv2.putText(img, "%d" % y, (m - 30, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    for i, r in enumerate(rows):
        for j, k in enumerate(meths):
            v = 100 * r[k]["glarefree_upper"]
            x0 = int(m + i * gw + gw * 0.1 + j * bw)
            cv2.rectangle(img, (x0, int(Hd - m - (Hd - 2 * m) * v / 100)), (int(x0 + bw - 2), Hd - m), cols[j % 5], -1)
        cv2.putText(img, r["placement"].replace("axis ", ""), (int(m + i * gw + 5), Hd - m + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    for j, k in enumerate(meths):
        cv2.rectangle(img, (m + j * 160, 12), (m + j * 160 + 12, 24), cols[j % 5], -1)
        cv2.putText(img, k, (m + j * 160 + 16, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "glare-free coverage of the upper surface (%) vs part orientation", (m, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, "fig_soga_orientations.png"), img)


# ------------------------------------------------------------------ run (robot)
def measured_pose(arm, reach, u_target_xy):
    """Camera pose after capture: orientation from the measured joints, position corrected so that the optical axis
    passes through the verified table point (the registration servo put it at the image centre)."""
    q = np.radians(arm.read_deg()); T = fk(q)
    Rt = T[:3, :3] @ reach.R_ref.T
    zc = reach.geo.H + (T[2, 3] * 1000 - reach.p_ref[2])
    oa = Rt[:, 2]                                         # -optical axis
    Cam = np.r_[np.asarray(u_target_xy, float), 0.0] + zc / oa[2] * oa
    return Cam, Rt


def _profile_shift(ref, refv, x, xv, max_shift, axis):
    """Shift (in cells along `axis`) that best matches profile of x to ref (zero-mean, over cells valid in both)."""
    def prof(F, V):                     # mean feature per cell along `axis`, over the other axis
        num = (F * V[..., None]).sum(1 - axis)
        den = V.sum(1 - axis)
        return num / np.maximum(den, 1)[:, None], den > 3
    pr, mr = prof(ref, refv); px, mx = prof(x, xv)
    best, bs = np.inf, 0
    n = len(pr)
    for d in range(-max_shift, max_shift + 1):
        q = np.roll(px, d, 0); qm = np.roll(mx, d, 0)
        if d > 0:
            qm[:d] = False
        elif d < 0:
            qm[d:] = False
        ok = mr & qm
        if ok.sum() < 0.3 * n:
            continue
        a_, b_ = pr[ok] - pr[ok].mean(0), q[ok] - q[ok].mean(0)
        c = float(np.mean((a_ - b_) ** 2))
        if c < best:
            best, bs = c, d
    return bs


def unwrap(geo, part, shots, psi_max, rho_min, mm_px=0.5, align_mm=(12.0, 10.0)):
    """Unrolled surface (s along the axis, arc length r*phi around it) sampled from the real images: every patch
    takes the view that sees it most frontally without glare. Each view is first ALIGNED to the survey by its image
    content (shift along the axis and around it that best matches the survey's unrolled colour/brightness profiles),
    so the result does not depend on the arm reaching its pose exactly (open-loop error 3-8 mm on this arm).
    Returns texture, label map, glare-free mask, seen mask, stats (incl. the applied shifts), phi grid."""
    ns = int(part.L / mm_px); nphi = int(2 * np.pi * part.r / mm_px)
    chart = part.chart(ns, nphi); S, PH, X, n = chart
    samples = []
    for i, sh in enumerate(shots):
        seen, cosp, u = visibility(geo, part, sh["C"], sh["Rt"], psi_max, rho_min, chart=chart)
        mx = u[..., 0].astype(np.float32); my = u[..., 1].astype(np.float32)
        smp = cv2.remap(sh["img"], np.nan_to_num(mx, nan=-1), np.nan_to_num(my, nan=-1), cv2.INTER_LINEAR)
        V = cv2.cvtColor(smp, cv2.COLOR_BGR2HSV)[..., 2]
        gl = seen & (V >= GLARE_V)
        samples.append([smp, seen, gl, seen & ~gl, cosp])
    # content alignment of every view to the reference shot (the survey, or the view that sees most)
    ref = 0 if samples[0][3].sum() > 200 else int(np.argmax([x[3].sum() for x in samples]))
    feat = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[..., :2] * np.array([1.0, 3.0], np.float32)
    Fr, Vr = feat(samples[ref][0]), samples[ref][3]
    shifts = []
    for i, x in enumerate(samples):
        if i == ref:
            shifts.append((0, 0)); continue
        Fx = feat(x[0])
        dc = _profile_shift(Fr, Vr, Fx, x[3], int(align_mm[0] / mm_px), axis=1)     # along the axis (columns)
        Fx2 = np.roll(Fx, dc, 1); V2 = np.roll(x[3], dc, 1)
        dr = _profile_shift(Fr, Vr, Fx2, V2, int(align_mm[1] / mm_px), axis=0)      # around the axis (rows)
        for k in range(5):
            x[k] = np.roll(np.roll(x[k], dc, 1), dr, 0)
        for k in (1, 2, 3):                                                          # invalidate wrapped columns
            if dc > 0:
                x[k][:, :dc] = False
            elif dc < 0:
                x[k][:, dc:] = False
        shifts.append((dc, dr))
    best = np.full(PH.shape, -1.0); tex = np.zeros(PH.shape + (3,), np.uint8)
    lab = np.full(PH.shape, -1); seen_any = np.zeros(PH.shape, bool)
    for i, (smp, seen, gl, good, cosp) in enumerate(samples):
        seen_any |= seen
        sc = np.where(good, cosp, -1.0)
        take = sc > best
        tex[take] = smp[take]; best[take] = sc[take]; lab[take] = i
    up = np.abs(PH) <= np.pi / 2
    st = dict(seen_upper=float(seen_any[up].mean()), glarefree_upper=float((best > -1)[up].mean()),
              glare_only_upper=float((seen_any & (best <= -1))[up].mean()),
              align_shift_mm=[[round(dc * mm_px, 1), round(dr * mm_px, 1)] for dc, dr in shifts])
    return tex, lab, best > -1, seen_any, st, PH


def cmd_run(a):
    import hardware as HW
    from hardware import make_arm, make_camera, safe_move, wait_settled
    from fence import auto_fence
    from servo import servo_to_view
    reach = Reach(a); geo = reach.geo
    tag = a.tag or time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(C.LOGS_DIR, "soga", tag); os.makedirs(out, exist_ok=True)
    HW.set_fence(auto_fence(), reach.p_ref)
    arm = make_arm(a.dry_run, np.degrees(reach.q_ref)); arm.send_deg(np.degrees(reach.q_ref), 15); wait_settled(arm)
    cam = make_camera(a.dry_run, arm, a.ref_img)
    m0 = {int(k) if str(k).isdigit() else k: np.array(v) for k, v in geo.G["markers0"].items()}
    sv = servo_to_view(arm, cam, geo.A, geo.uc, m0, geo.uc, reach.R_ref, safe_move, np.degrees(reach.q_ref), speed=10)
    survey = sv["frame"]; cv2.imwrite(os.path.join(out, "survey.jpg"), survey)
    part, alpha_L, w_h, info = estimate(geo, survey, axis_deg=a.axis_deg, part_r=a.part_r, part_L=a.part_L)
    draw_estimate(geo, survey, part, os.path.join(out, "part_estimate.jpg"))
    rho_min = a.rho_min or 0.9 * geo.f / geo.H
    print("part: r %.1f mm, L %.1f mm (%s; image estimate r %.1f, L %.1f), axis (%.2f, %.2f) [%s]; lights %s deg, "
          "highlight widths %s deg -> check %s/part_estimate.jpg"
          % (part.r, part.L, info["size_from"], info["r_from_image"], info["L_from_image"], *part.a[:2], info["axis_from"],
             [round(v) for v in info["lights_alpha_deg"]], [round(v) for v in info["highlight_width_deg"]], out))
    chart = part.chart(); PH = chart[1]; start = np.r_[0.0, 0.0, geo.H]
    if a.method == "soga":
        cands, why = feasible(reach, arc_views(part, geo, a.dz, np.arange(-a.arc_max, a.arc_max + 1e-6, a.arc_step)))
        if not cands:
            raise SystemExit("no certified arc view: %s" % why)
        M = model_masks(geo, part, cands, alpha_L, w_h, a.psi_max, rho_min, chart)
        views, cov = select_soga(geo, part, cands, M, PH, start, a.n_max)
        print("SOGA: %d views at alpha %s deg (arc reachable %+.0f..%+.0f), predicted glare-free upper coverage %.1f%%"
              % (len(views), [round(v["alpha"]) for v in views], min(c["alpha"] for c in cands),
                 max(c["alpha"] for c in cands), 100 * cov))
    elif a.method == "pitch":
        views, why = feasible(reach, pitch_views(part, geo, a.dz))
    elif a.method == "ring":
        views, why = feasible(reach, ring_views(part, geo, a.dz, a.ring_mm))
    else:
        views, why = feasible(reach, plan_views(geo, a.dz))
    shots = [dict(name="survey", C=start, Rt=np.eye(3), img=survey)]
    log = []; t0 = time.time(); q_prev = reach.q_ref.copy()
    for v in views:
        q, reason, way = reach.solve(v["C"], v["Rt"], seed=q_prev)
        if q is None and not np.allclose(q_prev, reach.q_ref):
            # not reachable in one hop from the previous view: go back to the reference (every view is certified from
            # there) and continue from it
            safe_move(arm, np.degrees(reach.q_ref), a.speed); wait_settled(arm); q_prev = reach.q_ref.copy()
            q, reason, way = reach.solve(v["C"], v["Rt"], seed=q_prev)
        if q is None:
            print("  %s refused: %s" % (v["name"], reason)); continue
        ts = time.time(); moved = True
        for w in way[1::2] + [way[-1]]:                    # small joint steps along the certified continuation
            moved, _, _ = safe_move(arm, np.degrees(w), a.speed)
            if not moved:
                break
        if not moved:
            print("  %s refused by the hardware guard: %s" % (v["name"], HW.LAST_REFUSAL)); continue
        wait_settled(arm)
        # the table point on the planned optical axis is the registration target
        oa = v["Rt"][:, 2]; xy_t = v["C"][:2] - v["C"][2] / oa[2] * oa[:2]
        u_t = np.asarray(geo.o2p(xy_t), float)
        r = RC.capture_registered(cam, arm, survey, u_t, geo, geo.scale(v["C"][2] - geo.H), speed=a.speed)
        Cm, Rm = measured_pose(arm, reach, xy_t)
        if abs(np.degrees(part.alpha(Cm)) - v.get("alpha", np.degrees(part.alpha(v["C"])))) > 6.0 or r["view_err_mm"] > 10.0:
            print("  %-8s registration rejected (%.1f mm): planned, certified pose used" % (v["name"], r["view_err_mm"]))
            safe_move(arm, np.degrees(q), a.speed); wait_settled(arm)
            for _ in range(2):
                cam.grab()
            r = dict(r, frame=cam.grab(), verify="planned pose (registration rejected)", view_err_mm=float("nan"))
            Cm, Rm = np.asarray(v["C"], float), np.asarray(v["Rt"], float)
        name = "%s.jpg" % v["name"]; cv2.imwrite(os.path.join(out, name), r["frame"])
        shots.append(dict(name=v["name"], C=Cm, Rt=Rm, img=r["frame"]))
        q_prev = np.radians(arm.read_deg())
        log.append(dict(view=v["name"], alpha_plan=v.get("alpha"), alpha_meas=float(np.degrees(part.alpha(Cm))), img=name,
                        aim_open_mm=r["view_err_open_mm"], aim_mm=r["view_err_mm"], verify=r["verify"],
                        move_s=time.time() - ts, C_plan=np.asarray(v["C"]).tolist(), C_meas=Cm.tolist()))
        print("  %-8s alpha %+5.1f deg: aim %.1f -> %.1f mm (%s), %.1f s" % (v["name"], np.degrees(part.alpha(Cm)), r["view_err_open_mm"],
                                                                         r["view_err_mm"], r["verify"], time.time() - ts))
    safe_move(arm, np.degrees(reach.q_ref), a.speed)
    wall = time.time() - t0
    tex, lab, good, seen, st, PHu = unwrap(geo, part, shots, a.psi_max, rho_min)
    save_unwrap(out, tex, lab, good, seen, part, PHu, [s["name"] for s in shots])
    S = dict(tag=tag, method=a.method, part=info, n_views=len(log), wall_s=wall,
             path_mm=path_mm([dict(C=s["C"]) for s in shots[1:]], start), measured=st, views=log,
             rho_min=rho_min, psi_max=a.psi_max)
    json.dump(S, open(os.path.join(out, "soga_run.json"), "w"), indent=1, default=float)
    print("measured on the real images: seen %.1f%%, GLARE-FREE %.1f%% of the upper surface (glare-only %.1f%%); "
          "%d views, %.0f mm camera path, %.0f s -> %s" % (100 * st["seen_upper"], 100 * st["glarefree_upper"],
          100 * st["glare_only_upper"], len(log), S["path_mm"], wall, out))


def save_unwrap(out, tex, lab, good, seen, part, PH, names):
    up = np.abs(PH[:, 0]) <= np.pi / 2
    t = tex[up][::-1]                                     # rows: phi from +90 (top of image) to -90
    cv2.imwrite(os.path.join(out, "unrolled_surface.png"), cv2.resize(t, (t.shape[1] * 2, t.shape[0] * 2),
                                                                      interpolation=cv2.INTER_NEAREST))
    pal = np.array([[230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66], [0, 114, 178], [213, 94, 0],
                    [204, 121, 167], [120, 120, 120]], np.uint8)[:, ::-1]
    cmap = np.full(tex.shape, 235, np.uint8)
    cmap[seen & ~good] = (0, 0, 220)                      # seen only under glare (red)
    for i in range(len(names)):
        cmap[(lab == i) & good] = pal[i % len(pal)]
    c = cmap[up][::-1]
    c = cv2.resize(c, (c.shape[1] * 2, c.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    y = 14
    for i, nm in enumerate(names):
        cv2.putText(c, nm, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, tuple(int(v) for v in pal[i % len(pal)]) if i else (0, 0, 0),
                    1, cv2.LINE_AA); y += 12
    cv2.imwrite(os.path.join(out, "coverage_map.png"), c)


def cmd_estimate(a):
    """No motion: estimate the part from a saved survey image and draw it (check before trusting a plan)."""
    geo = RC.Geo()
    sv = a.survey or os.path.join(C.LOGS_DIR, "grounding", "calib_survey.jpg")
    img = cv2.imread(sv)
    if img is None:
        raise SystemExit("no image %s (give --survey hw_logs/soga/<run>/survey.jpg)" % sv)
    part, alpha_L, w_h, info = estimate(geo, img, axis_deg=a.axis_deg, part_r=a.part_r, part_L=a.part_L)
    os.makedirs(OUT, exist_ok=True); out = os.path.join(OUT, "part_estimate.jpg")
    draw_estimate(geo, img, part, out)
    print("image estimate: r %.1f mm, L %.1f mm, axis (%.2f, %.2f) [%s], centre (%.1f, %.1f) mm; used: r %.1f, L %.1f (%s)"
          % (info["r_from_image"], info["L_from_image"], *part.a[:2], info["axis_from"], *part.c, part.r, part.L,
             info["size_from"]))
    print("lights %s deg, highlight widths %s deg; drawing: %s" % ([round(v) for v in info["lights_alpha_deg"]],
                                                                [round(v) for v in info["highlight_width_deg"]], out))


def cmd_report(a):
    rows = []
    for f in sorted(glob.glob(os.path.join(C.LOGS_DIR, "soga", "*", "soga_run.json"))):
        S = json.load(open(f))
        rows.append((S["tag"], S["method"], S["n_views"], S["path_mm"], S["wall_s"], 100 * S["measured"]["seen_upper"],
                     100 * S["measured"]["glarefree_upper"],
                     np.nanmedian([v["aim_mm"] for v in S["views"]] + [np.nan]) if S["views"] else np.nan))
    print("%-12s %-7s %5s %8s %7s %8s %11s %8s" % ("run", "method", "views", "path_mm", "time_s", "seen_%",
                                                  "glarefree_%", "aim_mm"))
    for r in rows:
        print("%-12s %-7s %5d %8.0f %7.0f %8.1f %11.1f %8.2f" % r)
    os.makedirs(OUT, exist_ok=True)
    json.dump([dict(zip(["run", "method", "views", "path_mm", "time_s", "seen_upper_pct", "glarefree_upper_pct",
                         "aim_median_mm"], r)) for r in rows], open(os.path.join(OUT, "soga_runs.json"), "w"), indent=1,
              default=float)
    meths = sorted(set(r[1] for r in rows))
    for m in meths:
        g = [r[6] for r in rows if r[1] == m]; p = [r[3] for r in rows if r[1] == m]
        print("  %-7s n=%d  glare-free %.1f +- %.1f %%   path %.0f mm" % (m, len(g), np.mean(g), np.std(g), np.mean(p)))
    if "soga" in meths and len(meths) > 1:
        try:
            from scipy.stats import mannwhitneyu
            for m in meths:
                if m != "soga":
                    s = [r[6] for r in rows if r[1] == "soga"]; b = [r[6] for r in rows if r[1] == m]
                    if len(s) >= 3 and len(b) >= 3:
                        print("  SOGA vs %-7s Mann-Whitney p = %.4f" % (m, mannwhitneyu(s, b, alternative="greater").pvalue))
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["analyze", "run", "report", "estimate"])
    RC.add_planner_args(ap)                              # same $OPTS as refcover_updated.py (max-descent, kappa, ...)
    ap.add_argument("--method", choices=["soga", "pitch", "ring", "plan"], default="soga")
    ap.add_argument("--dz", type=float, default=-10.0, help="inspection height (mm below the reference)")
    ap.add_argument("--arc-max", dest="arc_max", type=float, default=30.0, help="largest arc angle tried (deg)")
    ap.add_argument("--arc-step", dest="arc_step", type=float, default=2.0)
    ap.add_argument("--n-max", dest="n_max", type=int, default=4, help="most SOGA views")
    ap.add_argument("--psi-max", dest="psi_max", type=float, default=60.0, help="largest usable incidence angle (deg)")
    ap.add_argument("--rho-min", dest="rho_min", type=float, default=0.0, help="px/mm on the surface (0 = 90%% of survey)")
    ap.add_argument("--ring-mm", dest="ring_mm", type=float, default=40.0, help="radius of the isotropic baseline")
    ap.add_argument("--axis-deg", dest="axis_deg", type=float, default=None, help="override the image axis direction")
    ap.add_argument("--part-r", dest="part_r", type=float, default=None, help="part radius from a ruler (mm), overrides the image estimate")
    ap.add_argument("--part-L", dest="part_L", type=float, default=None, help="part length from a ruler (mm), overrides the image estimate")
    ap.add_argument("--survey", default=None)
    ap.add_argument("--no-rotate", dest="no_rotate", action="store_true", help="analyze only the part as placed")
    for k, d in (("--sim-r", 0.0), ("--sim-L", 70.0), ("--sim-cx", 0.0), ("--sim-cy", 0.0), ("--sim-light", 0.0),
                 ("--sim-wh", 10.0)):
        ap.add_argument(k, dest=k[2:].replace("-", "_"), type=float, default=d)
    ap.add_argument("--inspect-max-descent", dest="inspect_max_descent", type=float, default=10.0,
                    help="lowest the camera may go below the reference in this script (mm); caps --max-descent")
    a = ap.parse_args()
    a.max_descent = min(float(a.max_descent), float(a.inspect_max_descent))   # opts.sh says 30 for the lattice files
    if a.dz < -a.max_descent:
        a.dz = -a.max_descent
    {"analyze": cmd_analyze, "run": cmd_run, "report": cmd_report, "estimate": cmd_estimate}[a.cmd](a)


if __name__ == "__main__":
    main()
