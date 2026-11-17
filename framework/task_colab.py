#!/usr/bin/env python3
"""Colab: the inspection TASK on top of the real view library (ring / O-ring / rust).

  python3 task_colab.py classify     # part-state classifier: 5-fold CV grouped by placement (out-of-fold
                                     #   probabilities for every recorded image) + final model -> deploy/part_clf.npz
  python3 task_colab.py library      # task evidence of every view -> data/view_library_task.npz + deploy/task_prior.npz
  python3 train_views.py --data data/view_library_task.npz --arch attn --seed 0     # RL on the TASK reward
  python3 task_colab.py evaluate     # part verdicts per planner and time budget on held-out placements

Task evidence of view k for item j = P(item j judged correctly | the image of view k), from the OUT-OF-FOLD
classifier (never trained on that placement). The RL reward is the gain in the best evidence per item per second
of safe robot motion; the verdict fuses, per item, the most confident image taken.
Outputs: results_task/*.json, *.png
"""
import argparse, json, os, time
import numpy as np
import task_model as TM

OUT = "results_task"
FRACS = [0.2, 0.35, 0.5]


# ------------------------------------------------------------------ classifier
def make_model(J):
    import torch.nn as nn
    def blk(i, o, s):
        return [nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(*(blk(3, 16, 2) + blk(16, 32, 2) + blk(32, 64, 2) + blk(64, 96, 2) + blk(96, 128, 1)))
            self.drop = nn.Dropout(0.3); self.fc = nn.Linear(128, J)
        def forward(self, x):
            return self.fc(self.drop(self.f(x).mean((2, 3))))
    return Net()


def to_tensor(X, dev):
    import torch
    return torch.tensor(X, dtype=torch.float32, device=dev).permute(0, 3, 1, 2) / 255.0 - 0.5


def augment(x, g):
    import torch
    B = x.shape[0]
    f = torch.rand(B, 1, 1, 1, device=x.device) < 0.5; x = torch.where(f, x.flip(3), x)
    f = torch.rand(B, 1, 1, 1, device=x.device) < 0.5; x = torch.where(f, x.flip(2), x)
    x = x * torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.75, 1.25) + \
        torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
    x = x + 0.03 * torch.randn(B, 3, 1, 1, device=x.device)                 # colour cast
    s = int(g.integers(0, 9)); x = torch.roll(x, (s - 4, int(g.integers(-4, 5))), (2, 3))
    return x


def train_net(X, Y, Mk, epochs, seed, dev, Xv=None):
    import torch, torch.nn as nn
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    net = make_model(Y.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * max(1, int(np.ceil(len(X) / 64))))
    pos = (Y * Mk).sum(0); neg = ((1 - Y) * Mk).sum(0)
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 0.2, 5.0), dtype=torch.float32, device=dev)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    Xt = to_tensor(X, dev); Yt = torch.tensor(Y, dtype=torch.float32, device=dev); Mt = torch.tensor(Mk, dtype=torch.float32, device=dev)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(X), device=dev)
        for b in range(0, len(X), 64):
            idx = perm[b:b + 64]
            loss = (bce(net(augment(Xt[idx], g)), Yt[idx]) * Mt[idx]).sum() / Mt[idx].sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    return net


def predict(net, X, dev):
    import torch
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 256):
            out.append(torch.sigmoid(net(to_tensor(X[b:b + 256], dev))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, net.fc.out_features))


def folds_by_placement(conds, poses, n=5, seed=0):
    """Placements -> fold id; every (pose, condition) group is spread over the folds."""
    rng = np.random.default_rng(seed); f = np.zeros(len(conds), int)
    for key in sorted(set(zip(poses, conds))):
        idx = [i for i in range(len(conds)) if (poses[i], conds[i]) == key]; rng.shuffle(idx)
        start = int(rng.integers(n))
        for r, i in enumerate(idx):
            f[i] = (start + r) % n
    return f


def export_np(net, path, items, temp):
    """Fold BatchNorm into the convolutions and save NumPy weights for task_model.PartClassifier."""
    import torch.nn as nn
    W, convs, layers = {}, [], list(net.f)
    i = 0
    while i < len(layers):
        c, bn = layers[i], layers[i + 1]
        w = c.weight.detach().cpu().numpy(); s = c.stride[0]
        g = bn.weight.detach().cpu().numpy(); be = bn.bias.detach().cpu().numpy()
        mu = bn.running_mean.cpu().numpy(); var = bn.running_var.cpu().numpy()
        k = g / np.sqrt(var + bn.eps)
        convs.append((w * k[:, None, None, None], be - mu * k, s)); i += 3
    for n, (w, b, s) in enumerate(convs):
        W["c%d_w" % n], W["c%d_b" % n], W["c%d_s" % n] = w.astype(np.float32), b.astype(np.float32), np.array(s)
    W.update(n_conv=np.array(len(convs)), fc_w=net.fc.weight.detach().cpu().numpy(), fc_b=net.fc.bias.detach().cpu().numpy(),
             items=np.array(items), temperature=np.asarray(temp, np.float32))
    np.savez(path, **W)


# ------------------------------------------------------------------ vision transformer (DINOv2 ViT-S/14, frozen) + linear head
VIT_SIZE = 112                                       # 8 x 8 patches of 14 px


def load_vit(dev):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact float32 for parity
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def _vit_input(X, dev):
    import torch
    x = torch.tensor(np.ascontiguousarray(X[..., ::-1]), dtype=torch.float32, device=dev).permute(0, 3, 1, 2) / 255.0
    if x.shape[-1] != VIT_SIZE:
        x = torch.nn.functional.interpolate(x, size=(VIT_SIZE, VIT_SIZE), mode="area")
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None]
    return (x - mean) / std


def vit_features(m, X, dev, flip=None):
    """[CLS, mean patch token] after the final norm, (N, 768). flip: None | 'h' | 'v' (augmentation views)."""
    import torch
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            x = _vit_input(X[b:b + 128], dev)
            if flip == "h":
                x = x.flip(3)
            elif flip == "v":
                x = x.flip(2)
            f = m.forward_features(x)
            out.append(torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1).cpu().numpy())
    return np.concatenate(out)


def train_head(F, Y, Mk, seed, dev, epochs=400):
    """Linear head on frozen transformer features, masked BCE with class balancing (multi-label: 4 items)."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    head = nn.Linear(F.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=5e-2)
    pos = (Y * Mk).sum(0); neg = ((1 - Y) * Mk).sum(0)
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 0.2, 5.0), dtype=torch.float32, device=dev)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    Ft = torch.tensor(F, dtype=torch.float32, device=dev); Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    Mt = torch.tensor(Mk, dtype=torch.float32, device=dev)
    mu, sd = Ft.mean(0, keepdim=True), Ft.std(0, keepdim=True) + 1e-6
    for ep in range(epochs):
        x = (Ft - mu) / sd
        x = nn.functional.dropout(x, 0.2, training=True)
        loss = (bce(head(x), Yt) * Mt).sum() / Mt.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    # fold the feature standardisation into the head: w' = w / sd, b' = b - w' mu
    with torch.no_grad():
        w = head.weight / sd; b = head.bias - (w * mu).sum(1)
    return w.cpu().numpy(), b.cpu().numpy()


def head_prob(w, b, F):
    return 1.0 / (1.0 + np.exp(-(F @ w.T + b)))


def export_vit(m, w, b, path, items, temp, dev):
    """Save the transformer weights (+ the positional embedding interpolated to 8 x 8 patches) for the NumPy
    ViTClassifier on the Jetson."""
    import torch
    W = {}
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    import re
    prefixes = sorted({k[:-len("norm1.weight")] for k in sd if k.endswith("norm1.weight")},
                      key=lambda q: [int(t) for t in re.findall(r"\d+", q)])   # 'blocks.i.' (or chunked 'blocks.c.i.')
    with torch.no_grad():
        x = m.patch_embed(torch.zeros(1, 3, VIT_SIZE, VIT_SIZE, device=dev))
        x = torch.cat((m.cls_token.expand(1, -1, -1), x), 1)
        pe = m.interpolate_pos_encoding(x, VIT_SIZE, VIT_SIZE).cpu().numpy().astype(np.float32)
    W.update(arch=np.array("vit"), items=np.array(items), img_size=np.array(VIT_SIZE), patch=np.array(14),
             heads=np.array(m.num_heads), depth=np.array(len(prefixes)),
             img_mean=np.array([0.485, 0.456, 0.406], np.float32), img_std=np.array([0.229, 0.224, 0.225], np.float32),
             patch_w=sd["patch_embed.proj.weight"], patch_b=sd["patch_embed.proj.bias"], cls_token=sd["cls_token"],
             pos_embed=pe, norm_w=sd["norm.weight"], norm_b=sd["norm.bias"],
             head_w=np.asarray(w, np.float32), head_b=np.asarray(b, np.float32), temperature=np.asarray(temp, np.float32))
    for i, q in enumerate(prefixes):
        for k, n in (("n1_w", "norm1.weight"), ("n1_b", "norm1.bias"), ("qkv_w", "attn.qkv.weight"), ("qkv_b", "attn.qkv.bias"),
                     ("proj_w", "attn.proj.weight"), ("proj_b", "attn.proj.bias"), ("ls1", "ls1.gamma"),
                     ("n2_w", "norm2.weight"), ("n2_b", "norm2.bias"), ("fc1_w", "mlp.fc1.weight"), ("fc1_b", "mlp.fc1.bias"),
                     ("fc2_w", "mlp.fc2.weight"), ("fc2_b", "mlp.fc2.bias"), ("ls2", "ls2.gamma")):
            W["b%d_%s" % (i, k)] = sd[q + n]
    np.savez(path, **W)


def auc(score, y):
    y = np.asarray(y); pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_temperature(p, y, m):
    """Per-item temperature on out-of-fold logits (calibration: confidence = probability of being right)."""
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    out = []
    for j in range(p.shape[1]):
        sel = m[:, j] > 0
        best, bt = 1e9, 1.0
        for t in np.exp(np.linspace(np.log(0.3), np.log(5), 60)):
            q = 1 / (1 + np.exp(-lg[sel, j] / t))
            nll = -np.mean(y[sel, j] * np.log(q + 1e-9) + (1 - y[sel, j]) * np.log(1 - q + 1e-9)) if sel.any() else 0
            if nll < best:
                best, bt = nll, t
        out.append(bt)
    return np.array(out)


def cmd_classify(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(a.crops)
    X, pl, k, y, m = z["X"], z["placement"], z["k"], z["y"].astype(np.float32), z["m"].astype(np.float32)
    learn = [int(j) for j in z["learnable"]]
    if not learn:
        raise SystemExit("no item has both states in the library: record defective AND good placements")
    m = m.copy(); m[:, [j for j in range(m.shape[1]) if j not in learn]] = 0
    conds, poses = [str(c) for c in z["conditions"]], [str(p) for p in z["poses"]]
    K = int(z["K"]); n_pl = len(conds)
    fold_pl = folds_by_placement(conds, poses, a.folds, a.seed); fold = fold_pl[pl]
    P = np.full(y.shape, np.nan, np.float32)
    t0 = time.time()
    print("classifier: %d crops, %d placements, %d folds, items %s, device %s" % (len(X), n_pl, a.folds,
                                                                                 [TM.ITEMS[j] for j in learn], dev))
    if a.backbone == "vit":
        vit = load_vit(dev)
        F0 = vit_features(vit, X, dev); Fh = vit_features(vit, X, dev, "h"); Fv = vit_features(vit, X, dev, "v")
        print("  transformer features: %s per crop (DINOv2 ViT-S/14, frozen), %.1f min" % (F0.shape[1], (time.time() - t0) / 60))
    for f in range(a.folds):
        tr, te = fold != f, fold == f
        if not te.any():
            continue
        if a.backbone == "vit":
            hw, hb = train_head(np.vstack([F0[tr], Fh[tr], Fv[tr]]), np.vstack([y[tr]] * 3), np.vstack([m[tr]] * 3),
                                a.seed + f, dev)
            P[te] = head_prob(hw, hb, F0[te])
        else:
            net = train_net(X[tr], y[tr], m[tr], a.epochs, a.seed + f, dev)
            P[te] = predict(net, X[te], dev)
        print("  fold %d: train %d, test %d crops (%d placements), %.1f min" % (f, tr.sum(), te.sum(),
                                                                             len(set(pl[te].tolist())), (time.time() - t0) / 60))
    T = fit_temperature(P, y, m)
    lg = np.log(np.clip(P, 1e-6, 1 - 1e-6) / np.clip(1 - P, 1e-6, 1)); Pc = 1 / (1 + np.exp(-lg / T[None]))
    os.makedirs(OUT, exist_ok=True); os.makedirs("deploy", exist_ok=True)
    sfx = a.suffix
    np.savez(os.path.join("data", "task_oof%s.npz" % sfx), P=Pc, fold=fold, temperature=T)
    # ---- evidence: per image type, per item; part verdicts from survey only vs all images
    R = dict(backbone=a.backbone, n_crops=int(len(X)), n_placements=n_pl, folds=a.folds, epochs=a.epochs,
             temperature=T.tolist(), items=[TM.ITEMS[j] for j in learn])
    meta = json.loads(str(np.load(a.library)["meta"])) if os.path.exists(a.library) else None
    rot = None
    if meta is not None:
        from views import ViewSet
        vs = ViewSet.from_meta(meta); rot = vs.rot[vs.level]
    kind = np.where(k == K, "survey", "view")
    if rot is not None:
        kind = np.where(k == K, "survey", np.where(rot[np.minimum(k, K - 1)], "tilted", "straight"))
    for j in learn:
        sel = m[:, j] > 0
        r = dict(auc_all=auc(Pc[sel, j], y[sel, j]), acc_all=float(((Pc[sel, j] > 0.5) == (y[sel, j] > 0.5)).mean()))
        for kd in ("survey", "straight", "tilted", "view"):
            s2 = sel & (kind == kd)
            if s2.sum() > 3:
                r["auc_" + kd] = auc(Pc[s2, j], y[s2, j]); r["acc_" + kd] = float(((Pc[s2, j] > 0.5) == (y[s2, j] > 0.5)).mean())
                r["n_" + kd] = int(s2.sum())
        for pz in ("horizontal", "vertical"):
            s2 = sel & (np.array(poses)[pl] == pz)
            if s2.sum() > 3:
                r["auc_" + pz] = auc(Pc[s2, j], y[s2, j])
        R[TM.ITEMS[j]] = r
    Yp, Mp = z["Yp"], z["Mp"]
    for name, use in (("survey_only", lambda i: (pl == i) & (k == K)), ("all_images", lambda i: pl == i)):
        corr, rec = [], []
        for i in range(n_pl):
            s = use(i)
            if not s.any():
                continue
            pf, _ = TM.fuse(Pc[s][:, learn])
            dec = (pf > 0.5).astype(int); truth = Yp[i][learn]; mk = Mp[i][learn] > 0
            corr.append(bool(np.all((dec == truth) | ~mk)))
            rec += [bool(dec[jj] == truth[jj]) for jj in range(len(learn)) if mk[jj] and truth[jj] != TM.GOOD[learn[jj]]]
        R["part_verdict_" + name] = dict(accuracy=float(np.mean(corr)) if corr else None,
                                         defect_recall=float(np.mean(rec)) if rec else None, n_defects=len(rec))
    # ---- final model on all placements -> Jetson
    path = os.path.join("deploy", "part_clf%s.npz" % sfx)
    if a.backbone == "vit":
        hw, hb = train_head(np.vstack([F0, Fh, Fv]), np.vstack([y] * 3), np.vstack([m] * 3), a.seed + 100, dev)
        export_vit(vit, hw, hb, path, TM.ITEMS, T, dev)
        pt = head_prob(hw, hb, F0[:8])
        nsel = 8
    else:
        net = train_net(X, y, m, a.epochs, a.seed + 100, dev)
        export_np(net, path, TM.ITEMS, T)
        pt = predict(net, X[:16], dev)
        nsel = 16
    clf = TM.PartClassifier(path)
    t1 = time.time()
    pn = np.array([1 / (1 + np.exp(-clf.logits(x))) for x in X[:nsel]])
    R["numpy_ms_per_crop_colab_cpu"] = 1000 * (time.time() - t1) / nsel
    R["numpy_parity_max_abs"] = float(np.abs(pt - pn).max())
    json.dump(R, open(os.path.join(OUT, "classifier_cv%s.json" % sfx), "w"), indent=2)
    print(json.dumps(R, indent=2))
    print("PyTorch vs NumPy (Jetson) max probability difference: %.2e (must be < 1e-3)" % R["numpy_parity_max_abs"])
    print("saved %s, data/task_oof%s.npz, %s/classifier_cv%s.json" % (path, sfx, OUT, sfx))


# ------------------------------------------------------------------ task library (RL environment)
def ridge(X, t, lam=1.0):
    A = X.T @ X + lam * np.eye(X.shape[1]); A[0, 0] -= lam
    return np.linalg.solve(A, X.T @ t)


def cmd_library(a):
    from views import ViewSet
    z = np.load(a.library); meta = json.loads(str(z["meta"])); vs = ViewSet.from_meta(meta)
    c = np.load(a.crops); o = np.load(os.path.join("data", "task_oof.npz"))
    names = [str(n) for n in z["names"]]
    if [str(n) for n in c["names"]] != names:
        raise SystemExit("task_crops.npz was built from a different library: run build_task_dataset.py again")
    B, K, J = len(names), vs.K, len(TM.ITEMS)
    learn = [int(j) for j in c["learnable"]]
    P = np.full((B, K + 1, J), np.nan)
    P[c["placement"], c["k"]] = o["P"]
    Y = c["Yp"].astype(float); Mk = c["Mp"].astype(float)
    Mk[:, [j for j in range(J) if j not in learn]] = 0
    E = np.where(np.isfinite(P), TM.correctness(np.nan_to_num(P, nan=0.5), Y[:, None, :]), 0.0)
    E_views, E_survey = E[:, :K], np.where(np.isfinite(P[:, K]), E[:, K], 0.5)
    seen = np.isfinite(P[:, :K, 0])                           # part crop available in this recorded view
    avail = z["avail"] & seen
    # prior: expected evidence of every view from survey-only features (ridge on logits, training placements)
    feats = np.stack([TM.prior_features(vs, z["w"][b], z["fp_part"][b], z["fp_edge"][b], z["fp_glare"][b], z["pose"][b])
                      for b in range(B)])                     # (B,K,F)
    tr = np.flatnonzero(z["split"] == 0)
    beta = np.zeros((J, feats.shape[2]))
    for j in learn:
        rows = [(b, kk) for b in tr for kk in range(K) if avail[b, kk] and Mk[b, j] > 0]
        if not rows:
            continue
        Xf = np.array([feats[b, kk] for b, kk in rows]); t = np.array([E_views[b, kk, j] for b, kk in rows])
        beta[j] = ridge(Xf, np.log(np.clip(t, 0.02, 0.98) / (1 - np.clip(t, 0.02, 0.98))), a.ridge)
    Epred = np.stack([TM.prior_predict(beta, feats[b]) for b in range(B)])    # (B,K,J)
    te = np.flatnonzero(z["split"] == 1)
    r_prior = {}
    for j in learn:
        s = avail[te] & (Mk[te, None, j] > 0)
        if s.sum() > 3:
            r_prior[TM.ITEMS[j]] = float(np.corrcoef(Epred[te][..., j][s], E_views[te][..., j][s])[0, 1])
    F = (z["fp_part"] > 0.02)[..., None] * np.ones((1, 1, J))
    out = {k: z[k] for k in z.files}
    out.update(Q=E_views * avail[..., None], qs=E_survey, w=np.broadcast_to(Mk, (B, J)).astype(float),
               qpred=Epred, F=F.astype(np.float32), avail=avail, Ptask=P, Ytask=Y, Mtask=Mk,
               task_items=np.array(TM.ITEMS))
    np.savez(os.path.join("data", "view_library_task.npz"), **out)
    np.savez(os.path.join("deploy", "task_prior.npz"), beta=beta, items=np.array(TM.ITEMS),
             features=np.array(TM.PRIOR_FEATURES), learn=np.array(learn))
    os.makedirs(OUT, exist_ok=True)
    S = dict(placements=B, views=K, items=[TM.ITEMS[j] for j in learn],
             mean_evidence_survey=float(np.mean(E_survey[:, learn])),
             mean_best_evidence_all_views=float(np.mean(np.maximum(E_survey, (E_views * avail[..., None]).max(1))[:, learn])),
             prior_vs_measured_evidence_r_test=r_prior)
    for j in learn:
        S["evidence_" + TM.ITEMS[j]] = dict(survey=float(E_survey[:, j][Mk[:, j] > 0].mean()) if (Mk[:, j] > 0).any() else None,
                                            straight=float(E_views[:, ~vs.rot[vs.level], j][avail[:, ~vs.rot[vs.level]]].mean()),
                                            tilted=float(E_views[:, vs.rot[vs.level], j][avail[:, vs.rot[vs.level]]].mean())
                                            if vs.rot.any() and avail[:, vs.rot[vs.level]].any() else None)
    json.dump(S, open(os.path.join(OUT, "task_library.json"), "w"), indent=2)
    print(json.dumps(S, indent=2))
    print("saved data/view_library_task.npz (RL environment) and deploy/task_prior.npz (Jetson)")


# ------------------------------------------------------------------ evaluation: part verdicts per planner
def verdict_of(Pb, seq, K, learn):
    probs = [Pb[K]] + [Pb[k] for k in seq if np.isfinite(Pb[k, 0])]
    probs = [p for p in probs if np.all(np.isfinite(p[learn]))]
    if not probs:
        return np.full(len(learn), 0.5)
    pf, _ = TM.fuse(np.array(probs)[:, learn])
    return pf


def time_to_decision(Pb, seq, T, K, learn, conf):
    """Robot time until every item is decided with calibrated confidence >= conf (inf if never)."""
    cur, t, probs = K, 0.0, [Pb[K]]
    if np.isfinite(Pb[K][learn]).all() and TM.confidence(Pb[K][learn]).min() >= conf:
        return 0.0
    for k in seq:
        t += T[cur, k]; cur = k
        if np.isfinite(Pb[k, 0]):
            probs.append(Pb[k])
        pf, _ = TM.fuse(np.array([p for p in probs if np.all(np.isfinite(p[learn]))])[:, learn])
        if TM.confidence(pf).min() >= conf:
            return t
    return np.inf


def cmd_evaluate(a):
    """Every planner chooses views on the held-out placements; the verdict uses only the images it took.
    Task-aware planners (policy_task*, greedy_evidence, oracle) act in the task library; task-agnostic ones
    (policy_quality, greedy_quality, setcover, nn_time, raster_*, random) in the original quality library, so each
    behaves exactly as it would on the robot. Budgets are fractions of the same full-inspection time."""
    from views import rollout
    from train_views import load_library, take, full_time
    from geo_attn import load_np
    vs, Dtk, split, names, meta = load_library(os.path.join("data", "view_library_task.npz"))
    _, Dq, _, names_q, _ = load_library(a.library)
    if [str(n) for n in names_q] != [str(n) for n in names]:
        raise SystemExit("view_library_task.npz does not match %s: run  task_colab.py library  again" % a.library)
    zt = np.load(os.path.join("data", "view_library_task.npz"))
    P, Y, Mk = zt["Ptask"], zt["Ytask"], zt["Mtask"]
    learn = [j for j in range(len(TM.ITEMS)) if Mk[:, j].any()]
    te = np.flatnonzero(split == 1)
    if not len(te):
        te = np.arange(len(split)); print("WARNING: no held-out placements; evaluating on all")
    tfull = full_time(Dq, vs)[te]
    envs = {"task": take(Dtk, te), "quality": take(Dq, te)}
    table = []
    for spec in a.policies:
        nm, p = spec.split("=") if "=" in spec else (os.path.basename(spec).replace(".npz", ""), spec)
        if os.path.exists(p):
            table.append((nm, "quality" if "quality" in nm else "task", "policy", load_np(p)))
        else:
            print("  (policy %s not found, skipped)" % p)
    table += [("greedy_evidence", "task", "greedy_pred", None), ("greedy_quality", "quality", "greedy_pred", None),
              ("setcover", "quality", "setcover", None), ("nn_time", "quality", "nn_time", None),
              ("raster_high", "quality", "raster_high", None), ("raster_low", "quality", "raster_low", None),
              ("random", "quality", "random", None), ("oracle", "task", "oracle", None)]
    res = {}
    for frac in FRACS + [1.0]:
        for nm, env, mm, W in table:
            Dt = envs[env]
            _, tu, seqs = rollout(Dt, vs, frac * tfull, mm, np.random.default_rng(0), W)
            ok, rec, fa, ttd = [], [], [], []
            for b, i in enumerate(te):
                pf = verdict_of(P[i], seqs[b], vs.K, learn)
                dec = (pf > 0.5).astype(int); truth = Y[i][learn].astype(int); mk = Mk[i][learn] > 0
                ok.append(bool(np.all((dec == truth) | ~mk)))
                for jj in range(len(learn)):
                    if mk[jj] and truth[jj] != TM.GOOD[learn[jj]]:
                        rec.append(bool(dec[jj] == truth[jj]))
                if not np.any((truth != TM.GOOD[learn]) & mk):
                    fa.append(bool(np.any((dec != TM.GOOD[learn]) & mk)))
                if frac == 1.0:
                    ttd.append(time_to_decision(P[i], seqs[b], Dt["T"][b], vs.K, learn, a.conf))
            r = dict(part_accuracy=float(np.mean(ok)), defect_recall=float(np.mean(rec)) if rec else None,
                     false_reject_rate=float(np.mean(fa)) if fa else None, mean_time_s=float(np.mean(tu)),
                     mean_views=float(np.mean([len(q) for q in seqs])), n_parts=len(ok), n_defects=len(rec),
                     n_good_parts=len(fa))
            if frac == 1.0:
                tt = np.array(ttd)
                r.update(decided_pct=float(100 * np.isfinite(tt).mean()),
                         median_time_to_confident_verdict_s=float(np.median(tt[np.isfinite(tt)])) if np.isfinite(tt).any() else None)
            res.setdefault(nm, {})[str(frac)] = r
    survey_only = []
    for i in te:
        pf = verdict_of(P[i], [], vs.K, learn); dec = (pf > 0.5).astype(int)
        survey_only.append(bool(np.all((dec == Y[i][learn]) | (Mk[i][learn] == 0))))
    S = dict(test_placements=[str(names[i]) for i in te], conf=a.conf, items=[TM.ITEMS[j] for j in learn],
             survey_only_part_accuracy=float(np.mean(survey_only)), results=res)
    os.makedirs(OUT, exist_ok=True)
    json.dump(S, open(os.path.join(OUT, "task_eval.json"), "w"), indent=2)
    print("held-out parts: %d; survey image only (no robot motion): part accuracy %.2f"
          % (len(te), S["survey_only_part_accuracy"]))
    print("%-18s" % "method" + "".join("  acc@%-4s rec@%-4s" % (f, f) for f in FRACS) + "  t_conf(s) decided")
    for nm, _, _, _ in table:
        rr = res[nm]; line = "%-18s" % nm
        for f in FRACS:
            x = rr[str(f)]
            line += "  %8.2f %8.2f" % (x["part_accuracy"], x["defect_recall"] if x["defect_recall"] is not None else float("nan"))
        x = rr["1.0"]
        line += "  %8s %6.0f%%" % ("%.1f" % x["median_time_to_confident_verdict_s"]
                                    if x["median_time_to_confident_verdict_s"] is not None else "-", x["decided_pct"])
        print(line)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for nm, _, _, W in table:
            xs = [res[nm][str(f)]["mean_time_s"] for f in FRACS]
            kw = dict(lw=2.2 if nm.startswith("policy_task") else 1.0, ls="--" if nm == "oracle" else "-")
            ax[0].plot(xs, [res[nm][str(f)]["part_accuracy"] for f in FRACS], "o", label=nm, **kw)
            ax[1].plot(xs, [res[nm][str(f)]["defect_recall"] if res[nm][str(f)]["defect_recall"] is not None else np.nan
                            for f in FRACS], "o", **kw)
        ax[0].axhline(S["survey_only_part_accuracy"], color="0.5", lw=0.8, ls=":")
        ax[0].set(xlabel="robot time used (s)", ylabel="part verdict accuracy")
        ax[1].set(xlabel="robot time used (s)", ylabel="defect recall")
        ax[0].legend(fontsize=7, frameon=False); fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_task_accuracy_vs_time.png"), dpi=300)
        print("figure: %s/fig_task_accuracy_vs_time.png" % OUT)
    except Exception as e:
        print("(plot skipped: %s)" % e)
    print("saved %s/task_eval.json" % OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["classify", "library", "evaluate"])
    ap.add_argument("--crops", default="data/task_crops.npz")
    ap.add_argument("--library", default="data/view_library.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--backbone", choices=["vit", "cnn"], default="vit",
                    help="vit: DINOv2 ViT-S/14 transformer (default, deployed); cnn: small CNN baseline")
    ap.add_argument("--suffix", default="", help="output name suffix, e.g. _cnn for the baseline run (not deployed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=3.0)
    ap.add_argument("--conf", type=float, default=0.9, help="confidence for the time-to-verdict measure")
    ap.add_argument("--policies", nargs="*", default=["policy_task=deploy/view_policy_attn_s0_task.npz",
                                                      "policy_task_mlp=deploy/view_policy_mlp_s0_task.npz",
                                                      "policy_quality=deploy/view_policy_attn_s0.npz"])
    a = ap.parse_args()
    {"classify": cmd_classify, "library": cmd_library, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
