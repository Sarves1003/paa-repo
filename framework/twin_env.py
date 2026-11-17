"""Batched inspection-ordering environment ("twin") and the observation builder
shared by training and the robot runner.

Instances come from
  * real grounded survey tokens (collect_surveys.py -> data/surveys.npz) with
    augmentation (flips, density noise, random cell drop-out), and
  * the hardware cost model (cost_model.py) with randomised timing scale and noise,
    so the policy is trained on the robot's own kinematics, not on Cartesian distance.
If no surveys exist yet, structured synthetic groove maps are used and the run
is labelled synthetic in its metadata.
"""
import numpy as np
from geo_attn import T_SCALE, TOK_DIM, CTX_DIM

GRID_N = 5
N = GRID_N * GRID_N


# ------------------------------------------------------------------ observations
def build_obs(static, value, T, avail, visited, cur, t_used, v_sum, budget, task_budget, step):
    """Batched: static (B,N,6) value (B,N) T (B,N,N) avail/visited (B,N) bool,
    cur/t_used/v_sum/budget/step (B,), task_budget (B,) bool.
    Returns tok (B,N,8), ctx (B,8), mask (B,N)."""
    B = static.shape[0]
    idx = np.arange(B)
    t_next = T[idx, cur]                                      # (B,N)
    tok = np.concatenate([static[:, :, :5], avail[:, :, None].astype(np.float32),
                          visited[:, :, None].astype(np.float32),
                          (t_next / T_SCALE)[:, :, None]], -1).astype(np.float32)
    n_av = np.maximum(avail.sum(1), 1)
    left = np.where(task_budget, (budget - t_used) / np.maximum(budget, 1e-6), 1.0)
    vtot = np.maximum((value * avail).sum(1), 1e-6)
    ctx = np.stack([(visited & avail).sum(1) / n_av, step / N, left,
                    static[idx, cur, 0], static[idx, cur, 1], v_sum / vtot,
                    task_budget.astype(np.float32), t_used / (N * T_SCALE)], 1).astype(np.float32)
    mask = visited | ~avail
    over = task_budget[:, None] & (t_used[:, None] + t_next > budget[:, None] + 1e-9)
    return tok, ctx, mask | over


# ------------------------------------------------------------------ instances
def synthetic_static(rng):
    """Groove-like density map on a 5x5 grid (used only when no real surveys exist)."""
    dens = rng.uniform(0.03, 0.12, (GRID_N, GRID_N))
    for _ in range(rng.integers(2, 5)):
        if rng.random() < 0.7:
            j = rng.integers(0, GRID_N); dens[:, j] += rng.uniform(0.1, 0.35)
        else:
            i = rng.integers(0, GRID_N); dens[i, :] += rng.uniform(0.1, 0.35)
    dens += rng.normal(0, 0.02, dens.shape)
    return _static_from_maps(np.clip(dens, 0, 0.6), rng.normal(0, 0.2, (GRID_N, GRID_N)),
                             rng.normal(0, 0.2, (GRID_N, GRID_N)), np.ones((GRID_N, GRID_N), bool))


def _static_from_maps(dens, gx, gy, obs):
    S = np.zeros((N, 6), np.float32)
    for k in range(N):
        i, j = divmod(k, GRID_N)
        S[k] = [(j / (GRID_N - 1) - 0.5) * 2, (i / (GRID_N - 1) - 0.5) * 2,
                dens[i, j] if obs[i, j] else 0.0, gx[i, j], gy[i, j], float(obs[i, j])]
    return S


def augment_static(S, rng):
    dens = S[:, 2].reshape(GRID_N, GRID_N).copy(); gx = S[:, 3].reshape(GRID_N, GRID_N).copy()
    gy = S[:, 4].reshape(GRID_N, GRID_N).copy(); obs = S[:, 5].reshape(GRID_N, GRID_N) > 0.5
    if rng.random() < 0.5:
        dens, gx, gy, obs = dens[:, ::-1], -gx[:, ::-1], gy[:, ::-1], obs[:, ::-1]
    if rng.random() < 0.5:
        dens, gx, gy, obs = dens[::-1], gx[::-1], -gy[::-1], obs[::-1]
    dens = np.clip(dens * rng.uniform(0.8, 1.2, dens.shape), 0, 1)
    return _static_from_maps(dens, gx, gy, obs)


class InstanceSampler:
    """cost_model: one cost-model dict or a LIST of them (grids at different centres, as produced by
    part tracking on the robot); each instance draws one, so the policy learns to follow the part."""
    def __init__(self, cost_model, surveys=None, task="coverage", budget_frac=(0.3, 0.7),
                 time_scale=(0.8, 1.25), time_noise=0.05, drop_p=0.08):
        self.Ms = cost_model if isinstance(cost_model, (list, tuple)) else [cost_model]
        self.surveys, self.task = surveys, task
        self.bf, self.ts, self.tn, self.dp = budget_frac, time_scale, time_noise, drop_p

    def sample(self, B, rng):
        from baselines import nn_time, evaluate
        static = np.zeros((B, N, 6), np.float32); T = np.zeros((B, N, N)); avail = np.zeros((B, N), bool)
        value = np.zeros((B, N)); budget = np.full(B, np.inf); start = np.zeros(B, int)
        D = np.zeros((B, N, N))
        for b in range(B):
            M = self.Ms[rng.integers(len(self.Ms))]
            S = (augment_static(self.surveys[rng.integers(len(self.surveys))], rng)
                 if self.surveys is not None and len(self.surveys) else synthetic_static(rng))
            static[b] = S
            noise = 1 + rng.normal(0, self.tn, (N, N)); noise = (noise + noise.T) / 2
            T[b] = M["T"] * rng.uniform(*self.ts) * noise
            D[b] = M["D_cart_mm"]; start[b] = int(M.get("start", 12))
            av = np.asarray(M["reach"], bool) & (rng.random(N) > self.dp)
            if self.task == "budget":
                av &= S[:, 5] > 0.5                       # value must have been observed
            avail[b] = av
            v = S[:, 2] * av
            value[b] = v / max(v.max(), 1e-6)
            if self.task == "budget":
                inst = dict(T=T[b], D_cart=D[b], reach=avail[b], value=value[b], start=start[b],
                            task="coverage", budget=np.inf)
                full = evaluate(nn_time(inst), inst)["time"]
                budget[b] = rng.uniform(*self.bf) * full
        return dict(static=static, value=value, T=T, avail=avail, budget=budget, D_cart=D,
                    task_budget=np.full(B, self.task == "budget"), start=start)

    def instances(self, batch):
        return [dict(T=batch["T"][b], D_cart=batch["D_cart"][b], reach=batch["avail"][b], value=batch["value"][b],
                     start=int(batch["start"][b]), task=self.task, budget=batch["budget"][b])
                for b in range(len(batch["value"]))]


# ------------------------------------------------------------------ batched rollout state
class BatchEnv:
    def __init__(self, batch):
        self.b = batch
        B = batch["value"].shape[0]
        self.cur = batch["start"].copy()
        self.visited = np.zeros((B, N), bool)
        self.t_used = np.zeros(B); self.v_sum = np.zeros(B); self.step_i = np.zeros(B)
        self.done = np.zeros(B, bool)

    def obs(self):
        b = self.b
        tok, ctx, mask = build_obs(b["static"], b["value"], b["T"], b["avail"], self.visited, self.cur,
                                   self.t_used, self.v_sum, b["budget"], b["task_budget"], self.step_i)
        self.done |= mask.all(1)
        mask[self.done, 0] = False            # dummy legal action for finished rows
        return tok, ctx, mask

    def step(self, a):
        live = ~self.done
        idx = np.flatnonzero(live)
        self.t_used[idx] += self.b["T"][idx, self.cur[idx], a[idx]]
        self.v_sum[idx] += self.b["value"][idx, a[idx]]
        self.visited[idx, a[idx]] = True
        self.cur[idx] = a[idx]; self.step_i[idx] += 1
        return live

    def reward(self):
        """Higher is better: coverage -> -time (s); budget -> collected value (normalised)."""
        vtot = np.maximum((self.b["value"] * self.b["avail"]).sum(1), 1e-6)
        return np.where(self.b["task_budget"], self.v_sum / vtot, -self.t_used)
