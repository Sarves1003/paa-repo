#!/usr/bin/env python3
"""Journal figures of the framework, drawn from the robot's OWN images and data (vector PDF + 600-dpi PNG).

  python3 paper_figures.py                 # in the mycobot_framework folder -> paper/figs/fig_overview.*, fig_policy.*
  python3 paper_figures.py --views a.jpg b.jpg c.jpg d.jpg   # choose the view images shown in panels (c) and (d)

Fig. 1 (fig_overview): one taught pose -> safe, verifiable view set -> learned view selection -> verified capture,
        with the offline RL training on recorded real views. Every image, the lattice and the numbers come from the
        robot's files (survey, recorded views, deploy/refcover_lattice*.npz, measured view errors).
Fig. 2 (fig_policy): the view-selection policy with tensor shapes, masking and the REINFORCE objective.
Sized for a double-column figure (183 mm); fonts 6.5-8 pt as journals require; colours remain distinguishable in greyscale.
"""
import argparse, glob, json, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "mathtext.fontset": "dejavusans",
                     "axes.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42})
INK = "#1F2A36"; GREY = "#8C96A0"; LIGHT = "#EEF1F4"
COL = dict(a="#2F6FAD", b="#D9822B", c="#6A4BA8", d="#2E8B57", off="#5A6570")
MM = 1 / 25.4


# ------------------------------------------------------------------ data
def first(*pats):
    for p in pats:
        fs = sorted(glob.glob(p))
        if fs:
            return fs[-1]
    return None


def load_assets(views=None, lattice=None):
    A = {}
    lib = [d for d in sorted(glob.glob("deploy/views/*_*")) if os.path.exists(os.path.join(d, "meta.json"))]
    A["survey"] = first(*(os.path.join(d, "survey.jpg") for d in lib[::-1])) or first("hw_logs/refcover/*/survey.jpg")
    if views:
        A["views"] = list(views)
    elif lib:
        d = lib[-1]; m = json.load(open(os.path.join(d, "meta.json")))
        ok = [v for v in m["views"] if v.get("ok") and os.path.exists(os.path.join(d, "v%02d.jpg" % v["k"]))]
        ok = sorted(ok, key=lambda v: (v["level"], v["cell"]))
        pick = [ok[i] for i in np.linspace(0, len(ok) - 1, min(4, len(ok))).astype(int)] if ok else []
        A["views"] = [os.path.join(d, "v%02d.jpg" % v["k"]) for v in pick]
    else:
        A["views"] = sorted(glob.glob("hw_logs/refcover/*/view_00.jpg"))[-4:]
    A["lattice"] = lattice or first("deploy/refcover_lattice.npz", "deploy/refcover_lattice_*.npz")
    A["tg"] = json.load(open("deploy/table_grid.json")) if os.path.exists("deploy/table_grid.json") else None
    # measured view errors (open loop -> verified) of every recorded / coverage view
    o, c = [], []
    for f in glob.glob("deploy/views/*/meta.json"):
        for v in json.load(open(f)).get("views", []):
            if v.get("ok"):
                o.append(v.get("view_err_open_mm", np.nan)); c.append(v.get("view_err_mm", np.nan))
    import csv
    for f in ([] if o else glob.glob("hw_logs/refcover/*/views.csv")):   # library views preferred (the safe set)
        for r in csv.DictReader(open(f)):
            try:
                o.append(float(r["view_err_open_mm"])); c.append(float(r["view_err_mm"]))
            except (KeyError, ValueError):
                pass
    o, c = np.array(o, float), np.array(c, float); m = np.isfinite(o) & np.isfinite(c)
    A["err"] = (float(np.median(o[m])), float(np.median(c[m])), int(m.sum())) if m.sum() else None
    return A


def rgb(path, w=None):
    im = cv2.imread(path)
    if im is None:
        im = np.full((480, 640, 3), 230, np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def patch_quality(img, nx=8, ny=6, glare_level=245):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); h, w = g.shape
    Q = np.zeros((ny, nx)); E = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            p = g[j * h // ny:(j + 1) * h // ny, i * w // nx:(i + 1) * w // nx]
            lap = np.log1p(cv2.Laplacian(p, cv2.CV_64F).var()); gl = (p >= glare_level).mean()
            Q[j, i] = lap * (1 - gl); E[j, i] = (cv2.Canny(p, 50, 150) > 0).mean()
    Q = Q / max(np.percentile(Q, 95), 1e-6)
    return np.clip(Q, 0, 1), E


# ------------------------------------------------------------------ drawing helpers
def box(ax, x, y, w, h, fc, ec, lw=0.6, r=0.012, z=1, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r, fc=fc, ec=ec, lw=lw, zorder=z,
                       alpha=alpha, transform=ax.transAxes)
    ax.add_patch(p); return p


def arrow(fig, p0, p1, color=INK, lw=0.9, style="-|>", ls="-", rad=0.0, ms=7):
    a = FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                        linestyle=ls, connectionstyle="arc3,rad=%g" % rad, zorder=10)
    fig.add_artist(a)


def label(ax, s, color, x=0.0, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=color, va="bottom", ha="left")


def img_ax(fig, rect, im, title=None, color=INK):
    ax = fig.add_axes(rect); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(color); s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=6.5, color=color, pad=2)
    return ax


# ------------------------------------------------------------------ layout check
def check_overlaps(fig, name):
    """Report pairs of text items whose bounding boxes overlap (a figure for a journal must have none)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    T = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
    bb = [t.get_window_extent(r).expanded(1.0, 1.0) for t in T]
    bad = []
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            if bb[i].overlaps(bb[j]) and bb[i].width * bb[i].height > 0 and bb[j].width * bb[j].height > 0:
                inter = (min(bb[i].x1, bb[j].x1) - max(bb[i].x0, bb[j].x0)) * (min(bb[i].y1, bb[j].y1) - max(bb[i].y0, bb[j].y0))
                if inter > 4:
                    bad.append((T[i].get_text()[:30], T[j].get_text()[:30]))
    W, H = fig.get_size_inches() * fig.dpi
    clip = [t.get_text()[:30] for t, b in zip(T, bb) if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1]
    print("  %s: %d overlapping text pairs, %d texts outside the figure" % (name, len(bad), len(clip)))
    for a, b in bad[:12]:
        print("     overlap: %r / %r" % (a, b))
    for c in clip[:6]:
        print("     outside: %r" % c)
    return len(bad) + len(clip)


def tx(ax, x, y, s, size=6.0, color=INK, ha="left", va="top", weight="normal", **k):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, linespacing=1.3,
                   transform=ax.transAxes, **k)


def iax(fig, x, y, w, im, color=INK, lw=0.6):
    """Image axes with the image's own aspect: width w (figure fraction) -> height from the aspect."""
    Wf, Hf = fig.get_size_inches()
    h = w * Wf / Hf * im.shape[0] / im.shape[1]
    a = fig.add_axes([x, y, w, h]); a.imshow(im); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(color); sp.set_linewidth(lw)
    return a, h


# ------------------------------------------------------------------ Fig. 1: overview
def fig_overview(A, out):
    W, H = 183 * MM, 108 * MM
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    box(bg, 0.006, 0.355, 0.988, 0.635, "#FFFFFF", "#C9D1D9", lw=0.6, r=0.008)
    tx(bg, 0.016, 0.978, "ONLINE  |  Jetson Nano 4 GB + myCobot 280 with wrist camera, for every new part placement",
       5.8, GREY, weight="bold")
    box(bg, 0.006, 0.010, 0.988, 0.325, LIGHT, "#C9D1D9", lw=0.6, r=0.008)
    tx(bg, 0.984, 0.322, "OFFLINE  |  Colab: learning on recorded real views", 5.8, GREY, weight="bold",
       ha="right")

    sv = rgb(A["survey"]) if A["survey"] else np.full((480, 640, 3), 230, np.uint8)
    y0, wimg = 0.600, 0.205
    X = dict(a=0.020, b=0.262, c=0.520, d=0.780)

    # (a)
    ax, hh = iax(fig, X["a"], y0, wimg, sv, COL["a"])
    ytit = y0 + hh + 0.014
    tx(bg, X["a"], ytit, "(a)  One taught pose", 7.2, COL["a"], weight="bold", va="bottom")
    if A["tg"]:
        bx = np.array(A["tg"]["box_px"] + [A["tg"]["box_px"][0]]); ax.plot(bx[:, 0], bx[:, 1], color=COL["a"], lw=0.8)
    ax.text(0.03, 0.05, r"survey $I_0$ at $q_\mathrm{ref}$", transform=ax.transAxes, fontsize=5.8, color="w",
            bbox=dict(fc=COL["a"], ec="none", pad=1.0, alpha=0.9))
    tx(bg, X["a"], y0 - 0.018, "taught once by hand; the survey\nshows the whole fixture box\nand the fiducial markers", 5.8)

    # (b)
    tx(bg, X["b"], ytit, "(b)  Safe, verifiable views " + r"$\mathcal{V}$", 7.2, COL["b"], weight="bold", va="bottom")
    ax, _ = iax(fig, X["b"], y0, wimg, (0.5 * sv + 0.5 * 255).astype(np.uint8), COL["b"])
    nl = nu = 0; hs = [0]
    if A["lattice"]:
        z = np.load(A["lattice"], allow_pickle=True)
        U, kap, ok, dz = z["U"], z["kappa"], z["ok"].astype(bool), z["dz"]
        orient = z["orient"] if "orient" in z.files else np.zeros(len(dz))
        top = (np.abs(dz - dz.max()) < 1e-6) & (orient == 0)
        ax.scatter(U[top & ~ok, 0], U[top & ~ok, 1], s=3, marker="x", lw=0.45, color="#B03A2E", zorder=3)
        sc = ax.scatter(U[top & ok, 0], U[top & ok, 1], s=6, c=kap[top & ok], cmap="viridis", vmin=0.03, vmax=0.13,
                        edgecolors="k", linewidths=0.15, zorder=4)
        if A["tg"]:
            bx = np.array(A["tg"]["box_px"] + [A["tg"]["box_px"][0]]); ax.plot(bx[:, 0], bx[:, 1], color=INK, lw=0.5, ls=":")
        ax.set_xlim(0, sv.shape[1]); ax.set_ylim(sv.shape[0], 0)
        cax = fig.add_axes([X["b"] + wimg + 0.005, y0 + 0.004, 0.006, hh * 0.36])
        cb = fig.colorbar(sc, cax=cax, ticks=[0.05, 0.10]); cb.ax.tick_params(labelsize=4.8, width=0.4, length=1.5, pad=1)
        cb.outline.set_linewidth(0.4); cb.ax.set_title(r"$\kappa$", fontsize=6, pad=2)
        nl, nu = int(len(ok)), int(ok.sum()); hs = sorted(set(np.round(dz).astype(int)), reverse=True)
    leg = [Line2D([], [], ls="", marker="o", ms=2.6, mfc="#3B9C74", mec="k", mew=0.2, label="usable"),
           Line2D([], [], ls="", marker="x", ms=2.6, color="#B03A2E", mew=0.5, label="unreachable / singular")]
    lg = ax.legend(handles=leg, loc="lower left", fontsize=4.8, frameon=True, framealpha=0.9, borderpad=0.25,
                   handletextpad=0.15, labelspacing=0.15, borderaxespad=0.3); lg.get_frame().set_linewidth(0.3)
    tx(bg, X["b"], y0 - 0.018, "reachable on one joint branch,\n" + r"$\kappa\geq\kappa_\mathrm{min}$, markers or registration;" +
       "\n%d poses, %d usable, dz %d to %d mm" % (nl, nu, hs[0], hs[-1]), 5.8)

    # (c) learned view selection, drawn in figure coordinates
    xc = X["c"]
    tx(bg, xc, ytit, "(c)  Learned view selection", 7.2, COL["c"], weight="bold", va="bottom")
    vims = [rgb(p) for p in A["views"][:3]] or [sv] * 3
    tw = 0.062; ys = []
    for i, im in enumerate(vims):
        yy = y0 + hh - (i + 1) * 0.088 + 0.006
        a_, th = iax(fig, xc, yy, tw, im, COL["c"] if i == 1 else GREY, 1.1 if i == 1 else 0.4)
        ys.append(yy + th / 2)
        a_.text(0.04, 0.06, ["view 1", "view k", "view K"][i], transform=a_.transAxes, fontsize=4.6, color="w",
                bbox=dict(fc=COL["c"] if i == 1 else GREY, ec="none", pad=0.6, alpha=0.9))
    tb = (xc + 0.082, y0 + 0.050, 0.078, hh - 0.10)               # transformer box
    box(bg, *tb, "#F1ECFA", COL["c"], lw=0.8, r=0.006)
    tx(bg, tb[0] + tb[2] / 2, tb[1] + tb[3] - 0.012, "Transformer", 5.6, COL["c"], ha="center", weight="bold")
    tx(bg, tb[0] + tb[2] / 2, tb[1] + tb[3] / 2 + 0.028, r"$\pi_\theta(a_t\,|\,s_t)$", 7.5, ha="center")
    tx(bg, tb[0] + tb[2] / 2, tb[1] + 0.052, "attention\nacross views", 5.2, ha="center")
    for yy in ys:
        arrow(fig, (xc + tw + 0.002, yy), (tb[0] - 0.002, tb[1] + tb[3] / 2), GREY, lw=0.5, ms=5)
    mb = (tb[0] + tb[2] + 0.012, tb[1] + tb[3] / 2 - 0.045, 0.046, 0.090)
    box(bg, *mb, "#FFFFFF", COL["b"], lw=0.7, r=0.005)
    tx(bg, mb[0] + mb[2] / 2, mb[1] + mb[3] - 0.012, "mask", 5.8, COL["b"], ha="center", weight="bold")
    tx(bg, mb[0] + mb[2] / 2, mb[1] + 0.040, r"$a_t\in\mathcal{V}$", 6.2, ha="center")
    arrow(fig, (tb[0] + tb[2] + 0.002, tb[1] + tb[3] / 2), (mb[0] - 0.002, tb[1] + tb[3] / 2), INK, lw=0.6, ms=5)
    tx(bg, xc, y0 - 0.018, r"state $s_t$: survey" + "\nfeatures, views\ntaken, value, time\nleft, part pose", 5.6)

    # (d) verified capture
    xd = X["d"]
    tx(bg, xd, ytit, "(d)  Verified capture", 7.2, COL["d"], weight="bold", va="bottom")
    vd = rgb(A["views"][3]) if len(A["views"]) > 3 else vims[-1]
    ax, _ = iax(fig, xd, y0, wimg, vd, COL["d"])
    Q, E = patch_quality(vd); hq, wq = vd.shape[:2]
    part = E > np.percentile(E, 55)
    Qm = np.ma.masked_where(~part, Q)
    ax.imshow(Qm, extent=[0, wq, hq, 0], cmap="magma", alpha=0.5, vmin=0, vmax=1, interpolation="nearest")
    for j in range(Q.shape[0]):
        for i in range(Q.shape[1]):
            if part[j, i]:
                ax.add_patch(Rectangle((i * wq / Q.shape[1], j * hq / Q.shape[0]), wq / Q.shape[1], hq / Q.shape[0],
                                       fill=False, ec="w", lw=0.25))
    ax.plot([wq / 2 - 16, wq / 2 + 16], [hq / 2] * 2, color="#00E5FF", lw=0.8)
    ax.plot([wq / 2] * 2, [hq / 2 - 16, hq / 2 + 16], color="#00E5FF", lw=0.8)
    if A["err"]:
        o, c, n = A["err"]
        ax.text(0.97, 0.05, "error %.1f$\\rightarrow$%.1f mm (n = %d)" % (o, c, n), transform=ax.transAxes, fontsize=5.0,
                color="w", ha="right", bbox=dict(fc=COL["d"], ec="none", pad=1.0, alpha=0.92))
    tx(bg, xd, y0 - 0.018, "servoed until the target is\ncentred; patch quality " + r"$q$ =" +
       "\nresolution " + r"$\times$ sharpness $\times$" + "\n(1 " + r"$-$ glare)", 5.8)

    # online arrows
    ym = y0 + hh / 2
    arrow(fig, (X["a"] + wimg + 0.004, ym), (X["b"] - 0.004, ym), COL["a"])
    arrow(fig, (X["b"] + wimg + 0.036, ym), (xc - 0.004, ym), COL["b"])
    arrow(fig, (mb[0] + mb[2] + 0.003, ym), (xd - 0.004, ym), COL["c"])
    tx(bg, (mb[0] + mb[2] + xd) / 2, ym + 0.012, "route", 5.0, ha="center", va="bottom")
    # feedback (d) -> state, under the panels
    yf = 0.430
    arrow(fig, (xd + wimg * 0.5, yf), (xc + 0.15, yf), COL["d"], ls="--", lw=0.8)
    tx(bg, (xc + 0.15 + xd + wimg * 0.5) / 2, yf + 0.006, r"reward $r_t=V(S\cup\{a_t\})-V(S)$; update $s_{t+1}$",
       5.4, COL["d"], ha="center", va="bottom")

    # offline band: boxes aligned so the policy sits under the transformer
    yo, ho, wbx = 0.055, 0.180, 0.150
    cx_pol = tb[0] + tb[2] / 2
    xs = [cx_pol - wbx / 2 - 3 * (wbx + 0.028) + k * (wbx + 0.028) for k in range(4)]
    xs = [max(0.02, x) for x in xs]
    gap = (cx_pol - wbx / 2 - 0.02 - 3 * wbx) / 3
    xs = [0.02 + k * (wbx + gap) for k in range(3)] + [cx_pol - wbx / 2]
    items = [("View library " + r"$\mathcal{D}$", "every safe candidate\nview, recorded for\nN placements", COL["d"]),
             ("Replay environment", "real recorded images,\nexact oracle per\nplacement", COL["b"]),
             ("REINFORCE", "8 rollouts per\nplacement, mean-\nreward baseline", COL["c"]),
             ("Policy " + r"$\pi_\theta$", "NumPy export to\nthe Jetson, parity\n< $10^{-4}$", COL["c"])]
    for (t, sub, ccol), x in zip(items, xs):
        box(bg, x, yo, wbx, ho, "#FFFFFF", ccol, lw=0.7, r=0.008)
        tx(bg, x + wbx / 2, yo + ho - 0.025, t, 6.0, ccol, ha="center", weight="bold")
        tx(bg, x + wbx / 2, yo + 0.030, sub, 5.5, ha="center", va="bottom")
    for k in range(3):
        arrow(fig, (xs[k] + wbx + 0.003, yo + ho / 2), (xs[k + 1] - 0.003, yo + ho / 2), GREY)
    xe = xs[3] + wbx + 0.025; we = 0.975 - xe
    box(bg, xe, yo, we, ho, "#FFFFFF", GREY, lw=0.6, r=0.008)
    tx(bg, xe + we / 2, yo + ho - 0.025, "Held-out test", 6.0, INK, ha="center", weight="bold")
    tx(bg, xe + we / 2, yo + 0.022, "vs MLP (no attention),\ngreedy on survey, set\ncover, raster, random,\noracle",
       5.3, ha="center", va="bottom")
    arrow(fig, (xs[3] + wbx + 0.003, yo + ho / 2), (xe - 0.003, yo + ho / 2), GREY)
    arrow(fig, (cx_pol, yo + ho + 0.003), (cx_pol, tb[1] - 0.003), COL["c"], lw=1.0)
    tx(bg, cx_pol - 0.006, 0.262, r"weights $\theta$", 5.6, COL["c"], va="center", ha="right")
    arrow(fig, (X["b"] + wimg * 0.5, y0 - 0.115), (xs[0] + wbx * 0.5, yo + ho + 0.003), COL["b"], ls="--", lw=0.8, rad=0.25)
    tx(bg, X["b"] + wimg * 0.40, 0.392, r"candidates $\mathcal{V}$", 5.6, COL["b"], va="center", ha="right")
    check_overlaps(fig, "fig_overview")
    save(fig, out, "fig_overview")


# ------------------------------------------------------------------ Fig. 2: policy network
def fig_policy(out, tok_dim=14, ctx_dim=8, d=48, layers=2, ff=96):
    W, H = 183 * MM, 76 * MM
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def blk(x, y, w, h, title, col, fc="#FFFFFF", lw=0.8):
        box(ax, x, y, w, h, fc, col, lw=lw, r=0.008)
        tx(ax, x + w / 2, y + h - 0.03, title, 6.4, col, ha="center", weight="bold")

    def arr(p0, p1, col=INK, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=col, linestyle=ls,
                                     connectionstyle="arc3,rad=%g" % rad, transform=ax.transAxes))
    # token matrix
    gx, gy, gw, gh = 0.025, 0.47, 0.135, 0.34
    tx(ax, gx + gw / 2, 0.955, "Candidate-view tokens", 6.4, COL["d"], ha="center", weight="bold")
    tx(ax, gx + gw / 2, 0.895, r"$X\in\mathbb{R}^{K\times %d}$, one row per view" % tok_dim, 5.8, ha="center")
    for i in range(6):
        for j in range(tok_dim):
            c = "#F7E3CF" if j >= tok_dim - 3 else ("#E3F1E1" if j < 3 else "#F2F4F6")
            ax.add_patch(Rectangle((gx + j * gw / tok_dim, gy + gh - (i + 1) * gh / 6.5), gw / tok_dim * 0.92,
                                   gh / 6.5 * 0.85, fc=c, ec="#B7C0C8", lw=0.25, transform=ax.transAxes))
    tx(ax, gx + gw / 2, gy + 0.02, r"$\vdots$", 6.5, GREY, ha="center")
    tx(ax, gx, 0.43, "position, level, part share,\nedges, glare, predicted gain,\navailable, visited, time to\n"
       "reach, roll, pitch, yaw", 5.2)
    tx(ax, gx + gw / 2, 0.205, r"context $c\in\mathbb{R}^{%d}$" % ctx_dim, 6.0, COL["a"], ha="center", weight="bold")
    for j in range(ctx_dim):
        ax.add_patch(Rectangle((gx + 0.025 + j * 0.085 / ctx_dim, 0.105), 0.085 / ctx_dim * 0.9, 0.05, fc="#E8EEF6",
                               ec="#B7C0C8", lw=0.25, transform=ax.transAxes))
    tx(ax, gx + gw / 2, 0.090, "budget, value, step,\npose, current view", 5.0, ha="center")
    # embedding
    blk(0.195, 0.44, 0.10, 0.26, "Embedding", COL["a"])
    tx(ax, 0.245, 0.60, r"$W_xX,\;W_cc$", 6.2, ha="center")
    tx(ax, 0.245, 0.525, r"$(K{+}1)\times %d$" % d, 5.8, ha="center")
    arr((gx + gw + 0.004, 0.64), (0.193, 0.60)); arr((gx + gw / 2 + 0.05, 0.13), (0.193, 0.48), rad=0.25)
    # encoder
    ex, ey, ew, eh = 0.335, 0.335, 0.205, 0.565
    box(ax, ex + 0.009, ey - 0.022, ew, eh, "#F1ECFA", COL["c"], lw=0.5, r=0.008, alpha=0.8)
    blk(ex, ey, ew, eh, r"Encoder layer $\times$%d" % layers, COL["c"])
    subs = [("Self-attention over all views", r"$\mathrm{softmax}(QK^{\top}/\sqrt{d})\,V$", 0.105),
            ("Add & LayerNorm", None, 0.055), ("Feed-forward", "%d hidden units, GELU" % ff, 0.105),
            ("Add & LayerNorm", None, 0.055)]
    yy = ey + eh - 0.085
    for t, sub, hsub in subs:
        yy -= hsub
        box(ax, ex + 0.012, yy, ew - 0.024, hsub - 0.012, "#F7F4FC" if sub else "#F2F4F6", "#B9A9DA", lw=0.45, r=0.005)
        tx(ax, ex + ew / 2, yy + hsub - 0.020, t, 5.6, ha="center", weight="bold")
        if sub:
            tx(ax, ex + ew / 2, yy + hsub - 0.058, sub, 5.4, ha="center")
        yy -= 0.012
    tx(ax, ex + ew / 2, ey - 0.045, "d = %d, single head, about %dk parameters" % (d, (layers * (4 * d * d + 2 * d * ff)
       + (tok_dim + ctx_dim + 2 * d) * d) // 1000), 5.2, GREY, ha="center")
    arr((0.297, 0.57), (ex - 0.003, 0.57))
    # pointer decoder
    blk(0.585, 0.44, 0.135, 0.26, "Pointer decoder", COL["c"])
    tx(ax, 0.6525, 0.60, r"$\ell_k=10\,\tanh(q_c^{\top}k_k/\sqrt{d})$", 5.8, ha="center")
    tx(ax, 0.6525, 0.53, "query: context token\nkeys: view tokens", 5.0, ha="center")
    arr((ex + ew + 0.003, 0.57), (0.583, 0.57))
    blk(0.755, 0.47, 0.085, 0.20, "Mask", COL["b"])
    tx(ax, 0.7975, 0.585, "unsafe, visited,\nover budget", 5.0, ha="center")
    arr((0.722, 0.57), (0.753, 0.57))
    blk(0.875, 0.47, 0.10, 0.20, r"$a_t$", COL["d"])
    tx(ax, 0.925, 0.585, "softmax over\nthe views in " + r"$\mathcal{V}$", 5.0, ha="center")
    arr((0.842, 0.57), (0.873, 0.57))
    # training
    box(ax, 0.585, 0.025, 0.39, 0.355, LIGHT, "#C9D1D9", lw=0.5, r=0.008)
    tx(ax, 0.60, 0.355, "Training on recorded real views (REINFORCE)", 6.2, COL["c"], weight="bold")
    tx(ax, 0.60, 0.285, r"$\nabla_\theta J=\mathbb{E}[(R-\bar R)\,\nabla_\theta\log\pi_\theta(a_{1:T}\,|\,s)]$", 6.2)
    tx(ax, 0.60, 0.215, r"$R=V(S_T)-V(\emptyset)$ on the real images of the placement", 5.6)
    tx(ax, 0.60, 0.160, r"$\bar R$: mean of 8 rollouts of the same placement;" + "\nbudgets 20-50 % of the full inspection time;\n"
       "ablations: MLP without attention, no pose, no time, no tilt", 5.2)
    check_overlaps(fig, "fig_policy")
    save(fig, out, "fig_policy")


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=600)
    fig.savefig(os.path.join(out, name + ".svg"))
    plt.close(fig)
    print("  %s/%s.{pdf,png,svg}" % (out, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="*", default=None, help="view images for panels (c) and (d)")
    ap.add_argument("--out", default=os.path.join("paper", "figs"))
    ap.add_argument("--lattice", default=None, help="lattice for panel (b), e.g. deploy/refcover_lattice_d30_p-10_10.npz")
    a = ap.parse_args()
    A = load_assets(a.views, a.lattice)
    print("survey: %s | views: %s | lattice: %s | view errors: %s" % (A["survey"], A["views"], A["lattice"], A["err"]))
    fig_overview(A, a.out)
    fig_policy(a.out)


if __name__ == "__main__":
    main()
