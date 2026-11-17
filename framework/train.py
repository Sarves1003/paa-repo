#!/usr/bin/env python3
"""Train and evaluate the geometry-attention policy on the hardware-grounded twin.

Training: REINFORCE with a shared per-instance baseline. Each instance is rolled out
K times by sampling, and the mean return of those K rollouts is the baseline
(as in POMO-style training). Validation uses greedy decoding against the classical
planners on a fixed set. The final test runs the NumPy deployment path.

  python3 train.py --task coverage --ik hold --iters 3000 --seed 0
  python3 train.py --task budget   --ik hold --iters 3000 --seed 0
  python3 train.py --eval-only --task coverage --ik hold --policy deploy/geo_attn_policy.npz

Run 3-5 seeds and report mean +/- SD across seeds (do not report the best seed only).
Run this on a PC/Colab with PyTorch and copy the .npz to the Jetson. Training directly on the
Jetson Nano 4 GB also works (NVIDIA's PyTorch 1.10 wheel for JetPack 4.6, CPU is enough;
memory use is a few hundred MB) but is slower: use --batch 16 --k 8 --threads 4 there.
"""
import argparse, csv, json, os, time
import numpy as np
import config as C
import cost_model as CM
import baselines as BL
from twin_env import InstanceSampler, BatchEnv, N
from geo_attn import forward_np, load_np


def load_surveys(path):
    if path and os.path.exists(path):
        z = np.load(path)
        return list(z["static"]), True
    return None, False


def run_policy_np(W, batch, greedy=True, rng=None):
    env = BatchEnv(batch); orders = [[] for _ in range(len(batch["value"]))]
    for _ in range(N):
        tok, ctx, mask = env.obs()
        if env.done.all():
            break
        lg = forward_np(W, tok, ctx, mask)
        a = lg.argmax(1)
        live = env.step(a)
        for b in np.flatnonzero(live):
            orders[b].append(int(a[b]))
    return orders


def eval_methods(sampler, batch, methods, rng):
    insts = sampler.instances(batch)
    out = {}
    for m in methods:
        res = [BL.evaluate(BL.plan(m, inst, rng), inst) for inst in insts]
        out[m] = np.array([BL.objective(r, inst) for r, inst in zip(res, insts)])
    return out, insts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["coverage", "budget"], default="coverage")
    ap.add_argument("--ik", choices=["hold", "position"], default="hold")
    ap.add_argument("--surveys", default=os.path.join("data", "surveys.npz"))
    ap.add_argument("--test-surveys", default=os.path.join("data", "surveys_test.npz"),
                    help="held-out survey images for the final test (never used in training)")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--policy", default=C.POLICY_PATH)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build cost model from a placeholder pose")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (Jetson Nano: 4 cores)")
    ap.add_argument("--arch", choices=["attn", "mlp"], default="attn",
                    help="attn = transformer policy (ours); mlp = ablation without attention")
    a = ap.parse_args()

    # grids at a 3x3 lattice of centres inside the fence: the robot centres the grid on the part
    from fence import load_fence
    fence = load_fence()
    lo = np.minimum(fence.poly.min(0) + fence.margin + C.GRID_HALF_MM, 0)
    hi = np.maximum(fence.poly.max(0) - fence.margin - C.GRID_HALF_MM, 0)
    centers = [(x, y) for x in np.linspace(lo[0], hi[0], 3) for y in np.linspace(lo[1], hi[1], 3)]
    print("building cost models for %d grid centres (part tracking) ..." % len(centers))
    M = CM.build_many(a.ik, centers, dry=a.dry_run)
    surveys, real = load_surveys(a.surveys)
    sampler = InstanceSampler(M, surveys, task=a.task)
    ref = "twoopt" if a.task == "coverage" else "budget_ls"
    methods = (["boustro", "nn_cart", "nn_time", "twoopt", "random"] if a.task == "coverage"
               else ["boustro", "nn_time", "greedy", "budget_ls", "random"])
    out_dir = a.out or os.path.join(C.LOGS_DIR, "training", f"{a.task}_{a.ik}_{a.arch}_seed{a.seed}")
    os.makedirs(out_dir, exist_ok=True)

    if not a.eval_only:
        import torch
        from geo_attn import make_model, save_np, n_params
        torch.set_num_threads(a.threads)
        torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
        model = make_model(a.arch); opt = torch.optim.Adam(model.parameters(), a.lr)
        print(f"policy ({a.arch}) parameters: {n_params(model)}; task={a.task}; ik={a.ik}; "
              f"surveys={'real (%d)' % len(surveys) if real else 'SYNTHETIC'}")
        vrng = np.random.default_rng(10_000 + a.seed)
        val = sampler.sample(a.n_val, vrng)
        val_base, _ = eval_methods(sampler, val, methods, np.random.default_rng(1))
        ref_mean = val_base[ref].mean()
        print("validation baselines: " + ", ".join(f"{m} {v.mean():.3f}" for m, v in val_base.items()))
        log = open(os.path.join(out_dir, "curve.csv"), "w", newline="")
        lw = csv.writer(log); lw.writerow(["iter", "train_return", "val_return", "val_gap_pct_to_" + ref, "elapsed_s"])
        best, t0 = -np.inf, time.time()
        for it in range(1, a.iters + 1):
            batch = sampler.sample(a.batch, rng)
            rep = {k: (np.repeat(v, a.k, 0) if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
            env = BatchEnv(rep); logp = torch.zeros(len(rep["value"])); ent = torch.zeros(())
            for _ in range(N):
                tok, ctx, mask = env.obs()
                if env.done.all():
                    break
                lg = model(torch.tensor(tok), torch.tensor(ctx), torch.tensor(mask))
                dist = torch.distributions.Categorical(logits=lg)
                act = dist.sample()
                live = torch.tensor(~env.done, dtype=torch.float32)
                logp = logp + dist.log_prob(act) * live
                ent = ent + (dist.entropy() * live).mean()
                env.step(act.numpy())
            R = env.reward().reshape(a.batch, a.k)
            adv = (R - R.mean(1, keepdims=True)).reshape(-1)
            loss = -(torch.tensor(adv, dtype=torch.float32) * logp).mean() - a.entropy * ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if it % a.eval_every == 0 or it == a.iters:
                tmp = os.path.join(out_dir, "tmp.npz"); save_np(model, tmp)
                W = load_np(tmp)
                orders = run_policy_np(W, val)
                insts = sampler.instances(val)
                pv = np.array([BL.objective(BL.evaluate(o, i), i) for o, i in zip(orders, insts)])
                gap = 100 * (ref_mean - pv.mean()) / abs(ref_mean)
                lw.writerow([it, float(R.mean()), float(pv.mean()), float(gap), round(time.time() - t0, 1)]); log.flush()
                print(f"it {it:5d}  train {R.mean():8.3f}  val {pv.mean():8.3f}  gap to {ref} {gap:+6.2f}%  "
                      f"({time.time() - t0:.0f}s)")
                if pv.mean() > best:
                    best = pv.mean()
                    save_np(model, a.policy, meta=dict(task=a.task, ik=a.ik, seed=a.seed, iters=it, arch=a.arch,
                                                       surveys="real" if real else "synthetic"))
        log.close()
        print(f"best validation return {best:.3f}; saved {a.policy}")

    # ---------------- test with the NumPy deployment path (held-out surveys if available)
    W = load_np(a.policy)
    test_surveys, test_real = load_surveys(a.test_surveys)
    tsampler = InstanceSampler(M, test_surveys, task=a.task) if test_real else sampler
    print("test set: " + ("HELD-OUT real surveys (%d)" % len(test_surveys) if test_real else
                          "same distribution as training (no held-out surveys found)"))
    sampler = tsampler
    trng = np.random.default_rng(99_999)
    test = sampler.sample(a.n_test, trng)
    base, insts = eval_methods(sampler, test, methods, np.random.default_rng(2))
    t_inf = []
    orders = []
    for b in range(len(insts)):
        one = {k: v[b:b + 1] for k, v in test.items()}
        env = BatchEnv(one); od = []
        for _ in range(N):
            tok, ctx, mask = env.obs()
            if env.done.all():
                break
            t = time.perf_counter(); lg = forward_np(W, tok, ctx, mask); t_inf.append((time.perf_counter() - t) * 1e3)
            act = lg.argmax(1); env.step(act); od.append(int(act[0]))
        orders.append(od)
    pol = np.array([BL.objective(BL.evaluate(o, i), i) for o, i in zip(orders, insts)])
    from scipy.stats import wilcoxon
    rows = []
    print(f"\nTEST ({a.n_test} instances, task={a.task}, ik={a.ik}); higher is better")
    print(f"  policy      {pol.mean():9.3f} ± {pol.std():.3f}")
    for m, v in base.items():
        d = pol - v
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        win = float((d > 1e-9).mean() * 100); tie = float((np.abs(d) <= 1e-9).mean() * 100)
        rows.append(dict(method=m, mean=float(v.mean()), sd=float(v.std()), policy_minus_method=float(d.mean()),
                         policy_win_pct=win, tie_pct=tie, wilcoxon_p=float(p)))
        print(f"  {m:11s} {v.mean():9.3f} ± {v.std():.3f}   policy-{m}: {d.mean():+.3f}  "
              f"win {win:4.1f}% tie {tie:4.1f}%  Wilcoxon p={p:.2g}")
    lat = np.array(t_inf)
    summary = dict(task=a.task, ik=a.ik, seed=a.seed, arch=str(W.get("meta_arch", "attn")),
                   policy_mean=float(pol.mean()), policy_sd=float(pol.std()),
                   comparisons=rows, numpy_latency_ms_mean=float(lat.mean()), numpy_latency_ms_p95=float(np.percentile(lat, 95)),
                   surveys="real" if real else "synthetic", test_set="held-out" if test_real else "train-distribution")
    json.dump(summary, open(os.path.join(out_dir, "test_summary.json"), "w"), indent=2)
    print(f"  NumPy inference per decision: {lat.mean():.2f} ms (p95 {np.percentile(lat, 95):.2f}) on this machine")
    print(f"saved {out_dir}/test_summary.json")


if __name__ == "__main__":
    main()
