#!/usr/bin/env python3
"""EXPERIMENT E3 (Colab): RL training of the view-selection policy on the REAL view library.

The environment replays real recorded images: every reward is a quality measured on an image the robot
actually took (no simulator). Policy gradient with a shared multi-rollout baseline (as in POMO,
Kwon et al. 2020). Two architectures with identical inputs:
  attn : transformer encoder over all candidate views + pointer decoder   (ours)
  mlp  : every view scored independently, no attention                    (ablation)
Ablations by input: --no-pose (pose flag zeroed), --no-time (time tokens zeroed).

  python3 train_views.py --arch attn --seed 0 --iters 2000
  python3 train_views.py --arch mlp  --seed 0 --iters 2000
  python3 train_views.py --eval-only --policy deploy/view_policy_attn_s0.npz
  python3 train_views.py --data data/view_library_task.npz --arch attn --seed 0     # TASK reward -> *_task.npz
Outputs: deploy/view_policy_<arch>_s<seed>.npz (NumPy weights for the Jetson),
         results_views/train_<tag>/curve.csv, test_summary.json
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from views import ViewSet, TOK_DIM, CTX_DIM, BatchEnv, rollout
from geo_attn import forward_np, load_np

METHODS = ["greedy_pred", "setcover", "nn_time", "raster_high", "raster_low", "random", "oracle"]
FRACS = [0.2, 0.35, 0.5]


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


class Attn(nn.Module):
    arch = "attn"

    def __init__(self, d=48, layers=2, ff=96, clip=10.0):
        super().__init__()
        self.d, self.L, self.clip = d, layers, clip
        self.emb_t, self.emb_c = nn.Linear(TOK_DIM, d), nn.Linear(CTX_DIM, d)
        mk = lambda a, b: nn.ModuleList([nn.Linear(a, b) for _ in range(layers)])
        self.q, self.k, self.v, self.ff1, self.ff2 = mk(d, d), mk(d, d), mk(d, d), mk(d, ff), mk(ff, d)
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.dec_q, self.dec_k = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, tok, ctx, mask):
        x = torch.cat([self.emb_c(ctx)[:, None], self.emb_t(tok)], 1)
        for l in range(self.L):
            Q, K, V = self.q[l](x), self.k[l](x), self.v[l](x)
            a = torch.softmax(Q @ K.transpose(1, 2) / self.d ** 0.5, -1)
            x = self.ln1[l](x + a @ V)
            x = self.ln2[l](x + self.ff2[l](gelu(self.ff1[l](x))))
        lg = self.clip * torch.tanh((self.dec_k(x[:, 1:]) @ self.dec_q(x[:, 0])[:, :, None]).squeeze(-1) / self.d ** 0.5)
        return lg.masked_fill(mask, float("-inf"))


class MLP(nn.Module):
    arch = "mlp"

    def __init__(self, d=96, clip=10.0):
        super().__init__()
        self.d, self.L, self.clip = d, 0, clip
        self.m1, self.m2, self.m3 = nn.Linear(TOK_DIM + CTX_DIM, d), nn.Linear(d, d), nn.Linear(d, 1)

    def forward(self, tok, ctx, mask):
        x = torch.cat([tok, ctx[:, None].expand(-1, tok.shape[1], -1)], -1)
        lg = self.clip * torch.tanh(self.m3(gelu(self.m2(gelu(self.m1(x))))).squeeze(-1) / self.d ** 0.5)
        return lg.masked_fill(mask, float("-inf"))


def save_np(model, path, meta):
    W = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    W.update(meta_arch=np.array(model.arch), meta_d=np.array(model.d), meta_L=np.array(model.L),
             meta_clip=np.array(model.clip), meta_tok_dim=np.array(TOK_DIM), meta_ctx_dim=np.array(CTX_DIM))
    for k, v in meta.items():
        W["info_" + k] = np.array(v)
    np.savez(path, **W)


# ------------------------------------------------------------------ data
def load_library(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    vs = ViewSet.from_meta(meta)
    keys = ["Q", "F", "qpred", "qs", "w", "T", "avail", "fp_part", "fp_edge", "fp_glare", "pose"]
    D = {k: z[k].astype(bool if k == "avail" else np.float64) for k in keys}
    return vs, D, z["split"], z["names"], meta


def take(D, idx):
    return {k: v[idx] for k, v in D.items()}


def full_time(D, vs):
    _, t, _ = rollout(D, vs, np.inf, "greedy_pred")
    return t


class Mods:                                   # input ablations applied to observations
    def __init__(self, no_pose=False, no_time=False):
        self.no_pose, self.no_time = no_pose, no_time

    def __call__(self, tok, ctx):
        if self.no_pose:
            ctx[:, 3] = 0.0
        if self.no_time:
            tok[:, :, 9] = 0.0
        return tok, ctx


def sample_batch(D, idx_pool, tfull, n_inst, R, rng):
    idx = rng.choice(idx_pool, n_inst)
    B = take(D, np.repeat(idx, R))
    B["T"] = B["T"] * rng.uniform(0.85, 1.15, (len(B["T"]), 1, 1)) * rng.normal(1, 0.05, B["T"].shape).clip(0.8, 1.2)
    B["avail"] = B["avail"] & (rng.random(B["avail"].shape) > 0.05)
    budget = np.repeat(rng.uniform(0.15, 0.6, n_inst) * tfull[idx], R)
    return B, budget


def policy_rollout(model, D, vs, budget, mods, sample, rng=None):
    env = BatchEnv(D, vs, budget)
    logp = torch.zeros(env.B); ent = torch.zeros(env.B)
    while not env.done().all():
        tok, ctx, m = env.obs(); tok, ctx = mods(tok, ctx)
        done = m.all(1); m[done, 0] = False
        lg = model(torch.tensor(tok), torch.tensor(ctx), torch.tensor(m))
        if sample:
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.sample()
            live = torch.tensor(~done, dtype=torch.float32)
            logp = logp + dist.log_prob(a) * live
            ent = ent + dist.entropy() * live
        else:
            a = lg.argmax(1)
        env.step(a.numpy())
    return env.improvement(), logp, ent


def np_policy_rollout(W, D, vs, budget, mods):
    env = BatchEnv(D, vs, budget)
    while not env.done().all():
        tok, ctx, m = env.obs(); tok, ctx = mods(tok, ctx)
        done = m.all(1)
        lg = forward_np(W, tok, ctx, m)
        env.step(np.where(done, 0, np.argmax(lg, 1)))
    return env.improvement()


def evaluate(D, vs, tfull, policy_fn, fracs=FRACS, rng_seed=0):
    """Improvement per (placement, budget) for the policy and every baseline."""
    rows = []
    for f in fracs:
        budget = f * tfull
        res = {"policy": policy_fn(D, budget)}
        for m in METHODS:
            if m == "random":
                res[m] = np.mean([rollout(D, vs, budget, m, np.random.default_rng(rng_seed + s))[0] for s in range(10)], 0)
            else:
                res[m] = rollout(D, vs, budget, m)[0]
        for i in range(len(tfull)):
            rows.append(dict(placement=i, frac=f, **{k: float(v[i]) for k, v in res.items()}))
    return rows


def summarise(rows):
    out = {}
    pol = np.array([r["policy"] for r in rows]); ora = np.array([r["oracle"] for r in rows])
    out["policy_pct_of_oracle"] = float(100 * pol.sum() / max(ora.sum(), 1e-9))
    comps = []
    for m in METHODS:
        x = np.array([r[m] for r in rows]); d = pol - x
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) and len(d) > 5 else float("nan")
        comps.append(dict(method=m, mean=float(x.mean()), pct_of_oracle=float(100 * x.sum() / max(ora.sum(), 1e-9)),
                          policy_minus_method=float(d.mean()), policy_win_pct=float(100 * np.mean(d > 1e-9)),
                          wilcoxon_p=p))
    out["policy_mean"] = float(pol.mean()); out["comparisons"] = comps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/view_library.npz")
    ap.add_argument("--arch", choices=["attn", "mlp"], default="attn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--n-inst", type=int, default=16)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-tilt", action="store_true",
                    help="ablation: remove the tilted candidate views (straight views only, same placements)")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    vs, D, split, names, meta = load_library(a.data)
    if vs.rot.any():
        print("tilted candidate levels: %s" % [tuple(vs.rpy[l]) for l in range(vs.L) if vs.rot[l]])
    if a.no_tilt:
        if not vs.rot.any():
            raise SystemExit("--no-tilt: this library has no tilted views")
        D["avail"][:, np.repeat(vs.rot, vs.P)] = False      # policy AND baselines see straight views only
    mods = Mods(a.no_pose, a.no_time)
    tag = "%s_s%d%s%s%s" % (a.arch, a.seed, "_nopose" if a.no_pose else "", "_notime" if a.no_time else "",
                            "_notilt" if a.no_tilt else "")
    if "task" in os.path.basename(a.data):              # task library (ring / O-ring / rust evidence): own names
        tag += "_task"
    out = os.path.join("results_views", "train_" + tag); os.makedirs(out, exist_ok=True)
    tr = np.flatnonzero(split == 0); te = np.flatnonzero(split == 1)
    if len(te) == 0:
        print("WARNING: no held-out placements (collect more); testing on training placements")
        te = tr
    poses = D["pose"]
    val = np.array([i for pz in (0.0, 1.0) for i in tr[poses[tr] == pz][:1]]) if len(tr) >= 8 else tr
    trn = np.array([i for i in tr if i not in set(val)]) if len(tr) >= 8 else tr
    tfull = full_time(D, vs)
    print("library: %d placements (train %d, val %d, test %d); %d candidate views; full inspection %.1f s mean"
          % (len(split), len(trn), len(val), len(te), vs.K, tfull.mean()))
    path = a.policy or os.path.join("deploy", "view_policy_%s.npz" % tag)
    os.makedirs("deploy", exist_ok=True)

    if not a.eval_only:
        model = Attn() if a.arch == "attn" else MLP()
        print("parameters:", sum(p.numel() for p in model.parameters()))
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        Dv = take(D, val); tv = tfull[val]
        oracle_v = sum(rollout(Dv, vs, f * tv, "oracle")[0].sum() for f in FRACS)
        best, curve, t_start = -1e9, [], time.time()
        for it in range(a.iters + 1):
            if it % a.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    v = sum(policy_rollout(model, Dv, vs, f * tv, mods, False)[0].sum() for f in FRACS)
                pct = 100 * v / max(oracle_v, 1e-9)
                curve.append(dict(iter=it, val_pct_of_oracle=pct, minutes=(time.time() - t_start) / 60))
                if pct > best:
                    best = pct
                    save_np(model, path, dict(arch=a.arch, seed=a.seed, iter=it, val_pct=pct, data=a.data,
                                              no_pose=a.no_pose, no_time=a.no_time, no_tilt=a.no_tilt))
                print("iter %5d  validation %.1f%% of oracle  (best %.1f%%)" % (it, pct, best))
                model.train()
                if it == a.iters:
                    break
            Bt, budget = sample_batch(D, trn, tfull, a.n_inst, a.rollouts, rng)
            r, logp, ent = policy_rollout(model, Bt, vs, budget, mods, True)
            r = torch.tensor(r, dtype=torch.float32).view(a.n_inst, a.rollouts)
            adv = (r - r.mean(1, keepdim=True)) / (r.std() + 1e-6)
            loss = -(adv.view(-1) * logp).mean() - a.ent * ent.mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        import csv
        with open(os.path.join(out, "curve.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curve[0])); w.writeheader(); w.writerows(curve)
        # parity: the NumPy deployment path must reproduce the PyTorch logits
        W = load_np(path); mdl = Attn() if a.arch == "attn" else MLP()
        mdl.load_state_dict({k: torch.tensor(W[k]) for k in mdl.state_dict()}); mdl.eval()
        tok, ctx, m = BatchEnv(take(D, te[:4]), vs, 1e9).obs(); m[m.all(1), 0] = False
        with torch.no_grad():
            lt = mdl(torch.tensor(tok), torch.tensor(ctx), torch.tensor(m)).numpy()
        ln = forward_np(W, tok, ctx, m); fin = np.isfinite(lt)
        parity = float(np.abs(lt[fin] - ln[fin]).max())
        print("PyTorch vs NumPy max logit difference: %.2e (must be < 1e-4)" % parity)
    else:
        parity = None

    W = load_np(path)
    Dt = take(D, te)
    rows = evaluate(Dt, vs, tfull[te], lambda Dx, b: np_policy_rollout(W, Dx, vs, b, mods))
    S = summarise(rows)
    S.update(data=a.data, arch=a.arch, seed=a.seed, no_pose=a.no_pose, no_time=a.no_time, no_tilt=a.no_tilt, policy=path, parity=parity,
             test_placements=[str(n) for n in names[te]], n_test_cases=len(rows),
             test_pose_counts={"horizontal": int((Dt["pose"] == 0).sum()), "vertical": int((Dt["pose"] == 1).sum())})
    for pz, nm in ((0.0, "horizontal"), (1.0, "vertical")):
        rr = [r for r in rows if Dt["pose"][r["placement"]] == pz]
        if rr:
            S["by_pose_" + nm] = summarise(rr)
    json.dump(S, open(os.path.join(out, "test_summary.json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(out, "test_rows.json"), "w"))
    print("\nHELD-OUT placements: policy reaches %.1f%% of the oracle (improvement %.4f)"
          % (S["policy_pct_of_oracle"], S["policy_mean"]))
    for c in S["comparisons"]:
        print("  %-12s %.1f%% of oracle | policy - method %+.4f, policy better in %.0f%%, Wilcoxon p=%.3g"
              % (c["method"], c["pct_of_oracle"], c["policy_minus_method"], c["policy_win_pct"], c["wilcoxon_p"]))
    print("saved", out)


if __name__ == "__main__":
    main()
