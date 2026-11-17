"""Classical planners and a shared evaluator, all operating on the same instance.

An instance is a dict:
  T       (n,n) move time between cells [s] (hardware-identified, cost_model.py)
  D_cart  (n,n) Cartesian distance [mm]
  reach   (n,)  bool, cell reachable with the chosen IK mode
  value   (n,)  inspection value (vision-derived, >= 0)
  start   int   cell under the camera at the start (locked pose = centre)
  task    "coverage" (visit all reachable cells, minimise time) or
          "budget"   (maximise collected value within `budget` seconds)
  budget  float seconds (budget task only)

Planners return a visiting order (list of cells). evaluate() executes an order
under the task rules so every method is scored identically.
"""
import numpy as np

GRID_N = 5


def evaluate(order, inst):
    T, reach, val = inst["T"], inst["reach"], inst["value"]
    budget = inst.get("budget", np.inf) if inst["task"] == "budget" else np.inf
    cur, t, v, seen, path = inst["start"], 0.0, 0.0, set(), []
    for c in order:
        if c in seen or not reach[c]:
            continue
        if t + T[cur, c] > budget + 1e-9:
            continue                     # skip cells that no longer fit; keep scanning
        t += T[cur, c]; v += val[c]; seen.add(c); path.append(c); cur = c
    n_reach = int(reach.sum())
    return dict(time=t, value=v, n=len(path), coverage=len(path) / max(n_reach, 1), path=path,
                cart_mm=float(sum(inst["D_cart"][a, b] for a, b in zip([inst["start"]] + path[:-1], path))))


def objective(res, inst):
    """Higher is better for every task (coverage: negative time; budget: value)."""
    return -res["time"] if inst["task"] == "coverage" else res["value"]


# ------------------------------------------------------------------ fixed orders
def boustro(inst):
    out = []
    for i in range(GRID_N):
        js = range(GRID_N) if i % 2 == 0 else reversed(range(GRID_N))
        out += [i * GRID_N + j for j in js]
    return [c for c in out if inst["reach"][c]]


def random_order(inst, rng):
    c = np.flatnonzero(inst["reach"]); rng.shuffle(c); return list(c)


def _nn(inst, M):
    rem = set(np.flatnonzero(inst["reach"])); cur = inst["start"]; out = []
    if cur in rem:
        out.append(cur); rem.discard(cur)
    while rem:
        nxt = min(rem, key=lambda c: (M[cur, c], c)); out.append(nxt); rem.discard(nxt); cur = nxt
    return out


def nn_cart(inst):
    return _nn(inst, inst["D_cart"])


def nn_time(inst):
    return _nn(inst, inst["T"])


def _seq_time(seq, T):
    return sum(T[a, b] for a, b in zip(seq[:-1], seq[1:]))


def two_opt(order, inst, max_rounds=50):
    """2-opt on joint-time for an open path with a fixed start."""
    T = inst["T"]; seq = [inst["start"]] + list(order); n = len(seq)
    best = _seq_time(seq, T)
    for _ in range(max_rounds):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = seq[i - 1], seq[i]
                c, d = seq[j], seq[j + 1] if j + 1 < n else None
                delta = T[a, c] - T[a, b] + ((T[b, d] - T[c, d]) if d is not None else 0.0)
                if delta < -1e-9:
                    seq[i:j + 1] = seq[i:j + 1][::-1]; best += delta; improved = True
        if not improved:
            break
    return seq[1:]


def two_opt_best(inst):
    cands = [two_opt(o, inst) for o in (nn_time(inst), nn_cart(inst), boustro(inst))]
    return min(cands, key=lambda o: _seq_time([inst["start"]] + o, inst["T"]))


# ------------------------------------------------------------------ budget task
def greedy_ratio(inst):
    T, val, reach, B = inst["T"], inst["value"], inst["reach"], inst["budget"]
    cur, t, out, rem = inst["start"], 0.0, [], set(np.flatnonzero(reach))
    while True:
        feas = [c for c in rem if t + T[cur, c] <= B]
        if not feas:
            return out
        c = max(feas, key=lambda c: (val[c] / T[cur, c], -c))
        out.append(c); rem.discard(c); t += T[cur, c]; cur = c


def budget_local_search(inst, rounds=20):
    """Greedy ratio, then alternate 2-opt (free time) and best feasible insertion."""
    order = greedy_ratio(inst)
    for _ in range(rounds):
        order = two_opt(order, inst)
        seq = [inst["start"]] + order
        t = _seq_time(seq, inst["T"])
        best = None
        for c in np.flatnonzero(inst["reach"]):
            if c in order:
                continue
            for p in range(1, len(seq) + 1):
                a = seq[p - 1]; b = seq[p] if p < len(seq) else None
                add = inst["T"][a, c] + ((inst["T"][c, b] - inst["T"][a, b]) if b is not None else 0.0)
                if t + add <= inst["budget"]:
                    score = inst["value"][c] / max(add, 1e-6)
                    if best is None or score > best[0]:
                        best = (score, c, p)
        if best is None:
            break
        _, c, p = best
        order = order[:p - 1] + [c] + order[p - 1:]
    return order


METHODS = {
    "boustro": lambda inst, rng: boustro(inst),
    "nn_cart": lambda inst, rng: nn_cart(inst),
    "nn_time": lambda inst, rng: nn_time(inst),
    "twoopt": lambda inst, rng: two_opt_best(inst),
    "greedy": lambda inst, rng: greedy_ratio(inst),
    "budget_ls": lambda inst, rng: budget_local_search(inst),
    "random": lambda inst, rng: random_order(inst, rng),
}


def plan(method, inst, rng=None):
    return METHODS[method](inst, rng or np.random.default_rng(0))
