"""End-to-end parametric portfolio policy for a G10 FX basket.

Implements hypothesis
``end-to-end-parametric-portfolio-policies-for-cross-asset-futures``:
a small causal self-attention ("transformer") policy network maps the
recent cross-sectional history of daily FX returns directly to
next-bar portfolio weights for nine G10 pairs.  The network is
trained end-to-end inside each walk-forward in-sample window by
maximising the annualised Sharpe ratio of the net-of-cost portfolio
return series (an L1-turnover penalty at ``COST_RATE`` proxies
transaction costs inside the loss, mirroring the paper's
cost-adjusted differentiable Sharpe objective).

Optimisation uses SPSA (simultaneous perturbation stochastic
approximation): a forward-pass-only stochastic optimiser of the same
end-to-end objective.  SPSA is used because the strategy sandbox
permits numpy/pandas/scipy/statsmodels/sklearn only -- no autodiff
framework (torch/jax) is available, so analytic backprop through the
attention block is replaced by a two-sided simultaneous-perturbation
gradient estimate.  The objective being optimised is identical.

Architecture (single head, single block, ~1.1k weights):
    return z-scores -> linear embed + tanh -> causal banded
    self-attention over the last ``lookback`` bars with a learned
    per-lag additive bias (relative-position encoding) -> residual
    add -> linear head + tanh -> per-bar raw asset scores ->
    gross-exposure normalisation (sum |w| <= 1).

Fixed (non-tunable) constants:
    COST_RATE   = 2e-4  -- 2 bp per unit of L1 turnover in the loss
    SPSA_C      = 0.02  -- SPSA perturbation half-width
    SEED        = 7     -- RNG seed; fits are fully reproducible
    MIN_HISTORY = 10    -- bars of history required before emitting
                           non-zero weights
    Z_CLIP      = 5.0   -- feature z-score clip
    RET_CLIP    = 0.1   -- daily-return outlier clip (bad prints)
    EVAL_EVERY  = 25    -- iterations between best-theta checkpoints
    ANNUALISER  = 252.0 -- trading days per year for Sharpe
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

COST_RATE = 2e-4
SPSA_C = 0.02
SEED = 7
MIN_HISTORY = 10
Z_CLIP = 5.0
RET_CLIP = 0.1
EVAL_EVERY = 25
ANNUALISER = 252.0


@dataclass
class E2EPortfolioPolicyConfig:
    """Configuration for the end-to-end parametric portfolio policy.

    Free hyperparameters (5 total):
        lookback:        attention context length in trading days.
                         Acceptable range: [21, 126].  Default 63
                         (one quarter), matching the paper's
                         quarterly walk-forward step.
        d_model:         width of the embedding / attention space.
                         Acceptable range: [8, 32].  Kept tiny to
                         bound the weight count (~1.1k) relative to
                         the number of daily training observations.
        train_iters:     SPSA iterations per walk-forward fit.
                         Acceptable range: [100, 1000].
        learning_rate:   SPSA base step size ``a0``.
                         Acceptable range: [0.01, 0.25].
        rebalance_band:  L1 no-trade band on the weight vector; a new
                         target is adopted only when
                         sum_i |target_i - current_i| exceeds this.
                         Controls turnover (the paper's transformer
                         "trades far less").  Acceptable range:
                         [0.0, 0.5].
    """

    lookback: int = 63
    d_model: int = 16
    train_iters: int = 400
    learning_rate: float = 0.08
    rebalance_band: float = 0.15


class Strategy:
    """Transformer-style end-to-end portfolio policy over G10 FX."""

    id = "end-to-end-parametric-portfolio-policies-for-cross-asset-futures"

    # Joint multi-asset strategy (CL-40n2 v2): generate_signals returns
    # a DataFrame of per-bar weights, one column per pair below.
    symbols = [
        "EURUSD",
        "USDJPY",
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDCAD",
        "USDCHF",
        "USDNOK",
        "USDSEK",
    ]
    execution_symbol = "EURUSD"  # alias anchor; P&L is portfolio-level

    def __init__(self, config=None):
        self.config = config if config is not None else E2EPortfolioPolicyConfig()
        self._n_assets = len(self.symbols)
        self._shapes = self._param_shapes()
        self._n_params = sum(int(np.prod(s)) for _, s in self._shapes)
        self._theta = None
        self._scale = None

    # ------------------------------------------------------------------
    # parameter bookkeeping
    # ------------------------------------------------------------------
    def _param_shapes(self):
        n, d, L = self._n_assets, self.config.d_model, self.config.lookback
        return [
            ("w_in", (n, d)),
            ("b_in", (d,)),
            ("w_q", (d, d)),
            ("w_k", (d, d)),
            ("w_v", (d, d)),
            ("pos", (L,)),
            ("w_out", (d, n)),
            ("b_out", (n,)),
        ]

    def _unpack(self, theta):
        params, i = {}, 0
        for name, shape in self._shapes:
            size = int(np.prod(shape))
            params[name] = theta[i : i + size].reshape(shape)
            i += size
        return params

    # ------------------------------------------------------------------
    # model forward pass (vectorised causal banded attention)
    # ------------------------------------------------------------------
    def _forward(self, theta, Z):
        """Map a (T, n_assets) z-scored return matrix to (T, n_assets)
        portfolio weights.  Row t depends only on rows <= t (causal)."""
        p = self._unpack(theta)
        T = Z.shape[0]
        L = self.config.lookback
        d = self.config.d_model
        if T == 0:
            return np.zeros((0, self._n_assets))

        E = np.tanh(Z @ p["w_in"] + p["b_in"])  # (T, d)
        Q = E @ p["w_q"]  # (T, d)
        K = E @ p["w_k"]  # (T, d)
        V = E @ p["w_v"]  # (T, d)

        pad = np.zeros((L - 1, d))
        # window t covers original rows [t-L+1, t]; window index j
        # corresponds to original row t - L + 1 + j (j = L-1 is "now").
        Kw = sliding_window_view(np.vstack([pad, K]), L, axis=0)  # (T, d, L)
        Vw = sliding_window_view(np.vstack([pad, V]), L, axis=0)  # (T, d, L)

        scores = np.einsum("td,tdl->tl", Q, Kw) / np.sqrt(d)
        scores = scores + p["pos"][None, :]  # learned lag bias
        jj = np.arange(L)[None, :]
        tt = np.arange(T)[:, None]
        valid = jj >= (L - 1 - tt)  # mask pre-history pad
        scores = np.where(valid, scores, -np.inf)
        scores = scores - scores.max(axis=1, keepdims=True)
        attn = np.exp(scores)
        attn = attn / attn.sum(axis=1, keepdims=True)

        H = E + np.einsum("tl,tdl->td", attn, Vw)  # residual add
        raw = np.tanh(H @ p["w_out"] + p["b_out"])  # (T, n)
        gross = np.abs(raw).sum(axis=1, keepdims=True)
        return raw / np.maximum(gross, 1.0)  # sum |w| <= 1

    # ------------------------------------------------------------------
    # end-to-end objective: negative net annualised Sharpe
    # ------------------------------------------------------------------
    def _neg_sharpe(self, theta, Z, R):
        W = self._forward(theta, Z)
        if W.shape[0] < 3:
            return 5.0
        gross_pnl = (W[:-1] * R[1:]).sum(axis=1)  # w_t earns r_{t+1}
        turnover = np.abs(np.diff(W, axis=0)).sum(axis=1)
        net = gross_pnl - COST_RATE * turnover
        burn = min(self.config.lookback, len(net) // 3)
        net = net[burn:]
        if len(net) < 3:
            return 5.0
        sd = net.std()
        if sd < 1e-8:
            return 5.0  # degenerate/constant
        return -(net.mean() / sd) * np.sqrt(ANNUALISER)

    # ------------------------------------------------------------------
    # SPSA training loop
    # ------------------------------------------------------------------
    def _train(self, Z, R):
        cfg = self.config
        rng = np.random.default_rng(SEED)
        theta = 0.1 * rng.standard_normal(self._n_params)
        best_theta = theta.copy()
        best_loss = self._neg_sharpe(theta, Z, R)
        A = 0.1 * cfg.train_iters
        for k in range(cfg.train_iters):
            ak = cfg.learning_rate / (k + 1 + A) ** 0.602
            ck = SPSA_C / (k + 1) ** 0.101
            delta = rng.choice(np.array([-1.0, 1.0]), size=self._n_params)
            loss_p = self._neg_sharpe(theta + ck * delta, Z, R)
            loss_m = self._neg_sharpe(theta - ck * delta, Z, R)
            ghat = (loss_p - loss_m) / (2.0 * ck) * delta
            theta = theta - ak * np.clip(ghat, -10.0, 10.0)
            if (k + 1) % EVAL_EVERY == 0:
                cur = self._neg_sharpe(theta, Z, R)
                if cur < best_loss:
                    best_loss, best_theta = cur, theta.copy()
        cur = self._neg_sharpe(theta, Z, R)
        if cur < best_loss:
            best_theta = theta.copy()
        return best_theta

    # ------------------------------------------------------------------
    # data plumbing
    # ------------------------------------------------------------------
    def _returns(self, data):
        prices = data.reindex(columns=self.symbols).astype(float).ffill()
        rets = prices.pct_change(fill_method=None).fillna(0.0).to_numpy()
        return np.clip(rets, -RET_CLIP, RET_CLIP)

    # ------------------------------------------------------------------
    # protocol surface
    # ------------------------------------------------------------------
    def fit(self, train_data):
        """Train the policy end-to-end on the in-sample window only."""
        R = self._returns(train_data)
        scale = R.std(axis=0)
        self._scale = np.where(scale < 1e-8, 1.0, scale)
        Z = np.clip(R / self._scale, -Z_CLIP, Z_CLIP)
        if R.shape[0] < 30:
            # Degenerate window: keep the (reproducible) seeded init
            # rather than fit noise.
            rng = np.random.default_rng(SEED)
            self._theta = 0.1 * rng.standard_normal(self._n_params)
            return
        self._theta = self._train(Z, R)

    def generate_signals(self, data):
        """Return per-bar portfolio weights, one column per pair.

        Row t is computable from data with ts <= t only (causal banded
        attention over past returns; feature scale frozen at fit time).
        """
        if self._theta is None:
            raise RuntimeError("fit() must be called before generate_signals()")
        cols = list(self.symbols)
        if len(data.index) == 0:
            return pd.DataFrame(np.zeros((0, len(cols))), index=data.index, columns=cols)
        R = self._returns(data)
        Z = np.clip(R / self._scale, -Z_CLIP, Z_CLIP)
        targets = self._forward(self._theta, Z)

        # L1 no-trade band: hold current weights until the target
        # drifts far enough; keeps turnover near the paper's
        # low-turnover transformer profile.
        band = self.config.rebalance_band
        W = np.zeros_like(targets)
        current = np.zeros(targets.shape[1])
        for t in range(targets.shape[0]):
            if t < MIN_HISTORY:
                continue
            if np.abs(targets[t] - current).sum() > band or not np.any(current):
                current = targets[t]
            W[t] = current
        return pd.DataFrame(W, index=data.index, columns=cols)
