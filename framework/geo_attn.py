"""Geometry-attention policy for inspection ordering.

Inputs per decision
  tok  (B, n, 8): per cell [x, y, edge_density, grad_x, grad_y, available,
                            visited, move_time_from_current / T_SCALE]
  ctx  (B, 8)   : [coverage, step/n, budget_left/budget, cur_x, cur_y,
                   value_collected/value_total, task_flag, time_used/(n*T_SCALE)]
  mask (B, n)   : True = cell not allowed now
Architecture
  token + context embeddings -> L x (single-head self-attention + FF, post-LN)
  -> pointer decoder: logit_k = C * tanh(<W_q h_ctx, W_k h_k> / sqrt(d))   (Kool et al., 2019)
The current position enters through `move_time_from_current`, so the policy can
learn travel-aware choices; the original policy had no such input.

GeoAttn (PyTorch) is used for training; forward_np (NumPy) is the deployment path
on the Jetson Nano and is tested for numerical parity with parity_test().
"""
import numpy as np

TOK_DIM, CTX_DIM = 8, 8
T_SCALE = 3.0           # seconds; normalises move times in the tokens

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _gelu_t(x):   # tanh-approximate GELU; identical to the NumPy path; works on torch 1.x
        return 0.5 * x * (1 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))

    class GeoAttn(nn.Module):
        def __init__(self, d=32, layers=2, ff=64, clip=10.0):
            super().__init__()
            self.d, self.L, self.clip = d, layers, clip
            self.emb_t = nn.Linear(TOK_DIM, d)
            self.emb_c = nn.Linear(CTX_DIM, d)
            self.q = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.k = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.v = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
            self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.ff1 = nn.ModuleList([nn.Linear(d, ff) for _ in range(layers)])
            self.ff2 = nn.ModuleList([nn.Linear(ff, d) for _ in range(layers)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.dec_q = nn.Linear(d, d)
            self.dec_k = nn.Linear(d, d)

        def forward(self, tok, ctx, mask=None):
            x = torch.cat([self.emb_c(ctx)[:, None], self.emb_t(tok)], 1)
            for l in range(self.L):
                Q, K, V = self.q[l](x), self.k[l](x), self.v[l](x)
                a = torch.softmax(Q @ K.transpose(1, 2) / self.d ** 0.5, -1)
                x = self.ln1[l](x + a @ V)
                x = self.ln2[l](x + self.ff2[l](_gelu_t(self.ff1[l](x))))
            qd = self.dec_q(x[:, 0])
            kd = self.dec_k(x[:, 1:])
            logits = self.clip * torch.tanh((kd @ qd[:, :, None]).squeeze(-1) / self.d ** 0.5)
            if mask is not None:
                logits = logits.masked_fill(mask, float("-inf"))
            return logits

    class CellMLP(nn.Module):
        """ABLATION: same inputs and decoder style, but NO attention: every cell is scored
        independently from [its token, the context]. If the transformer beats this, attention
        (relations between cells) is what helps."""
        arch = "mlp"

        def __init__(self, d=64, clip=10.0):
            super().__init__()
            self.d, self.L, self.clip = d, 0, clip
            self.m1 = nn.Linear(TOK_DIM + CTX_DIM, d)
            self.m2 = nn.Linear(d, d)
            self.m3 = nn.Linear(d, 1)

        def forward(self, tok, ctx, mask=None):
            x = torch.cat([tok, ctx[:, None].expand(-1, tok.shape[1], -1)], -1)
            h = _gelu_t(self.m2(_gelu_t(self.m1(x))))
            logits = self.clip * torch.tanh(self.m3(h).squeeze(-1) / self.d ** 0.5)
            if mask is not None:
                logits = logits.masked_fill(mask, float("-inf"))
            return logits

    def make_model(arch="attn"):
        return CellMLP() if arch == "mlp" else GeoAttn()

    def save_np(model, path, meta=None):
        W = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        W["meta_arch"] = np.array(getattr(model, "arch", "attn"))
        W["meta_d"], W["meta_L"], W["meta_clip"] = np.array(model.d), np.array(model.L), np.array(model.clip)
        W["meta_tok_dim"], W["meta_ctx_dim"] = np.array(TOK_DIM), np.array(CTX_DIM)
        if meta:
            for k, v in meta.items():
                W[f"info_{k}"] = np.array(v)
        np.savez(path, **W)

    def n_params(model):
        return sum(p.numel() for p in model.parameters())

except ImportError:          # Jetson deployment without torch: NumPy path only
    torch = None


# ------------------------------------------------------------------ NumPy deployment path
def load_np(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _lin(W, name, x):
    return x @ W[name + ".weight"].T + W[name + ".bias"]


def _ln(W, name, x, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * W[name + ".weight"] + W[name + ".bias"]


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def forward_np(W, tok, ctx, mask=None):
    tok = np.asarray(tok, np.float32); ctx = np.asarray(ctx, np.float32)
    single = tok.ndim == 2
    if single:
        tok, ctx = tok[None], ctx[None]
        mask = None if mask is None else np.asarray(mask)[None]
    d, L, clip = int(W["meta_d"]), int(W["meta_L"]), float(W["meta_clip"])
    if str(W.get("meta_arch", "attn")) == "mlp":                # ablation: no attention
        x = np.concatenate([tok, np.repeat(ctx[:, None], tok.shape[1], 1)], -1)
        h = _gelu(_lin(W, "m2", _gelu(_lin(W, "m1", x))))
        logits = clip * np.tanh(_lin(W, "m3", h)[..., 0] / np.sqrt(d))
        if mask is not None:
            logits = np.where(mask, -np.inf, logits)
        return logits[0] if single else logits
    x = np.concatenate([_lin(W, "emb_c", ctx)[:, None], _lin(W, "emb_t", tok)], 1)
    for l in range(L):
        Q, K, V = _lin(W, f"q.{l}", x), _lin(W, f"k.{l}", x), _lin(W, f"v.{l}", x)
        s = Q @ K.transpose(0, 2, 1) / np.sqrt(d)
        s = np.exp(s - s.max(-1, keepdims=True)); a = s / s.sum(-1, keepdims=True)
        x = _ln(W, f"ln1.{l}", x + a @ V)
        x = _ln(W, f"ln2.{l}", x + _lin(W, f"ff2.{l}", _gelu(_lin(W, f"ff1.{l}", x))))
    qd = _lin(W, "dec_q", x[:, 0]); kd = _lin(W, "dec_k", x[:, 1:])
    logits = clip * np.tanh(np.einsum("bnd,bd->bn", kd, qd) / np.sqrt(d))
    if mask is not None:
        logits = np.where(mask, -np.inf, logits)
    return logits[0] if single else logits


def parity_test(path=None, n=25, trials=20, seed=0):
    """Max |torch - numpy| logit difference on random inputs (should be < 1e-4)."""
    assert torch is not None, "parity test needs torch"
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    import tempfile, os
    worst = 0.0
    for model in (GeoAttn().eval(), CellMLP().eval()):
        worst = max(worst, _parity_one(model, path, n, trials, rng))
    return worst


def _parity_one(model, path, n, trials, rng):
    import tempfile, os
    p = path or os.path.join(tempfile.mkdtemp(), "p.npz")
    save_np(model, p); W = load_np(p)
    worst = 0.0
    for _ in range(trials):
        tok = rng.normal(size=(4, n, TOK_DIM)).astype(np.float32)
        ctx = rng.normal(size=(4, CTX_DIM)).astype(np.float32)
        with torch.no_grad():
            a = model(torch.tensor(tok), torch.tensor(ctx)).numpy()
        b = forward_np(W, tok, ctx)
        worst = max(worst, float(np.abs(a - b).max()))
    return worst
