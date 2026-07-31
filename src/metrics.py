"""Performance and benchmark-relative statistics.

Conventions, stated once so the tables are unambiguous:

  Annualised return    geometric, (1+cum)^(12/n)-1 on monthly returns.
  Excess return        geometric difference of annualised returns, fund minus
                       benchmark. This is what "beat the benchmark by x" means.
  Mean excess          arithmetic mean of monthly return differences, annualised
                       by x12. Used for tracking error, IR and t-stats because
                       those are defined on the arithmetic series.
  Tracking error       annualised standard deviation of the monthly difference.
  Information ratio    mean excess / tracking error.
  t-statistic          IR * sqrt(years). With four years of data you need an IR
                       near 1.0 before the excess is distinguishable from noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


# ------------------------------------------------------------------ basic returns


def cumulative(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def annualised(returns: pd.Series, periods: int = MONTHS) -> float:
    n = returns.notna().sum()
    if n == 0:
        return np.nan
    total = (1 + returns.dropna()).prod()
    if total <= 0:
        return np.nan
    return float(total ** (periods / n) - 1)


def ann_vol(returns: pd.Series, periods: int = MONTHS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe(returns: pd.Series, rf: pd.Series | None = None, periods: int = MONTHS) -> float:
    if rf is None or rf.empty:
        excess = returns
    else:
        excess = (returns - rf.reindex(returns.index)).dropna()
    if excess.std(ddof=1) == 0 or excess.empty:
        return np.nan
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods))


def growth_of(returns: pd.Series, start_value: float = 10_000.0) -> pd.Series:
    return start_value * (1 + returns.fillna(0)).cumprod()


# --------------------------------------------------------------------- drawdowns


def drawdown_series(prices: pd.Series) -> pd.Series:
    peak = prices.cummax()
    return prices / peak - 1.0


def max_drawdown(prices: pd.Series) -> dict:
    dd = drawdown_series(prices)
    if dd.empty or dd.isna().all():
        return {"max_dd": np.nan, "trough": None, "peak": None, "recovered": None,
                "months_to_trough": np.nan, "months_to_recover": np.nan}
    trough = dd.idxmin()
    peak = prices.loc[:trough].idxmax()
    after = prices.loc[trough:]
    peak_level = prices.loc[peak]
    recovered = after[after >= peak_level]
    rec_date = recovered.index[0] if len(recovered) else None

    def months_between(a, b):
        if a is None or b is None:
            return np.nan
        delta = b - a
        return delta.days / 30.44 if hasattr(delta, "days") else float(delta)

    return {
        "max_dd": float(dd.min()),
        "peak": peak,
        "trough": trough,
        "recovered": rec_date,
        "months_to_trough": months_between(peak, trough),
        "months_to_recover": months_between(trough, rec_date),
    }


# ----------------------------------------------------------- regression, hand rolled


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (beta, residuals, XtX_inv). X must already include a constant."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    return beta, resid, XtX_inv


def newey_west_se(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray, lags: int) -> np.ndarray:
    """HAC standard errors, Bartlett kernel. Needed whenever the dependent series
    overlaps, which it does for any rolling 12 month statistic."""
    n, k = X.shape
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv * (n / max(n - k, 1))
    return np.sqrt(np.diag(cov))


def capm(fund: pd.Series, bench: pd.Series, rf: pd.Series | None = None,
         periods: int = MONTHS) -> dict:
    """Single index regression of fund on benchmark, both over the risk free rate."""
    df = pd.concat({"f": fund, "b": bench}, axis=1).dropna()
    if rf is not None and not rf.empty:
        r = rf.reindex(df.index).fillna(0)
        df["f"] -= r
        df["b"] -= r
    if len(df) < 6:
        return {"alpha_ann": np.nan, "beta": np.nan, "alpha_t": np.nan, "r2": np.nan}

    y = df["f"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df["b"].to_numpy()])
    beta, resid, XtX_inv = ols(y, X)
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(XtX_inv * s2))
    tss = ((y - y.mean()) ** 2).sum()
    return {
        "alpha_ann": float((1 + beta[0]) ** periods - 1),
        "alpha_monthly": float(beta[0]),
        "beta": float(beta[1]),
        "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
        "r2": float(1 - (resid @ resid) / tss) if tss > 0 else np.nan,
    }


# -------------------------------------------------------------- relative statistics


def capture(fund: pd.Series, bench: pd.Series) -> dict:
    df = pd.concat({"f": fund, "b": bench}, axis=1).dropna()
    up, down = df[df["b"] > 0], df[df["b"] < 0]
    res = {}
    for label, sub in (("up", up), ("down", down)):
        if len(sub) < 3 or sub["b"].mean() == 0:
            res[f"{label}_capture"] = np.nan
        else:
            res[f"{label}_capture"] = float(sub["f"].mean() / sub["b"].mean())
        res[f"n_{label}"] = len(sub)
    return res


def rolling_excess(fund: pd.Series, bench: pd.Series, window: int = 12) -> pd.Series:
    """Compounded window return of the fund minus the benchmark, geometric."""
    df = pd.concat({"f": fund, "b": bench}, axis=1).dropna()
    if len(df) < window:
        return pd.Series(dtype=float)
    f = (1 + df["f"]).rolling(window).apply(np.prod, raw=True) - 1
    b = (1 + df["b"]).rolling(window).apply(np.prod, raw=True) - 1
    return (f - b).dropna()


def rolling_significance(roll: pd.Series, lags: int = 11) -> dict:
    """Mean of an overlapping rolling excess series with a HAC t-stat.

    The naive t-stat on 12 month overlapping windows is inflated by roughly
    sqrt(12). This is the corrected version.
    """
    s = roll.dropna()
    if len(s) < 12:
        return {"roll_mean": np.nan, "roll_t_nw": np.nan, "roll_t_naive": np.nan}
    y = s.to_numpy()
    X = np.ones((len(y), 1))
    beta, resid, XtX_inv = ols(y, X)
    se_nw = newey_west_se(X, resid, XtX_inv, lags)[0]
    se_naive = s.std(ddof=1) / np.sqrt(len(s))
    return {
        "roll_mean": float(beta[0]),
        "roll_t_nw": float(beta[0] / se_nw) if se_nw > 0 else np.nan,
        "roll_t_naive": float(beta[0] / se_naive) if se_naive > 0 else np.nan,
        "roll_pct_positive": float((s > 0).mean()),
        "roll_best": float(s.max()),
        "roll_worst": float(s.min()),
        "n_windows": int(len(s)),
    }


def pair_stats(fund_m: pd.Series, bench_m: pd.Series, fund_d: pd.Series,
               bench_d: pd.Series, rf_m: pd.Series | None = None,
               window: int = 12, nw_lags: int = 11,
               expense_ratio: float | None = None) -> dict:
    """Every statistic for one fund and one benchmark, on the common window."""
    m = pd.concat({"f": fund_m, "b": bench_m}, axis=1).dropna()
    if m.empty:
        return {}
    f, b = m["f"], m["b"]
    diff = f - b

    n_months = len(m)
    years = n_months / MONTHS
    fund_ann, bench_ann = annualised(f), annualised(b)
    te = ann_vol(diff)
    mean_excess_ann = float(diff.mean() * MONTHS)
    ir = mean_excess_ann / te if te and not np.isnan(te) and te > 0 else np.nan

    d = pd.concat({"f": fund_d, "b": bench_d}, axis=1).dropna()
    f_px = (1 + d["f"]).cumprod()
    b_px = (1 + d["b"]).cumprod()
    dd_f, dd_b = max_drawdown(f_px), max_drawdown(b_px)

    reg = capm(f, b, rf_m)
    cap = capture(f, b)
    roll = rolling_excess(f, b, window)
    roll_stats = rolling_significance(roll, nw_lags)

    out = {
        "start": m.index.min(),
        "end": m.index.max(),
        "n_months": n_months,
        "years": years,
        "fund_cum": cumulative(f),
        "bench_cum": cumulative(b),
        "fund_ann": fund_ann,
        "bench_ann": bench_ann,
        "excess_ann_geom": fund_ann - bench_ann,
        "excess_ann_mean": mean_excess_ann,
        "fund_vol": ann_vol(f),
        "bench_vol": ann_vol(b),
        "fund_sharpe": sharpe(f, rf_m),
        "bench_sharpe": sharpe(b, rf_m),
        "tracking_error": te,
        "info_ratio": ir,
        "t_stat": ir * np.sqrt(years) if ir == ir else np.nan,
        "hit_rate": float((diff > 0).mean()),
        "fund_max_dd": dd_f["max_dd"],
        "bench_max_dd": dd_b["max_dd"],
        "dd_diff": (dd_f["max_dd"] - dd_b["max_dd"]) if dd_f["max_dd"] == dd_f["max_dd"] else np.nan,
        "fund_dd_trough": dd_f["trough"],
        "bench_dd_trough": dd_b["trough"],
        "months_to_recover": dd_f["months_to_recover"],
        "worst_month_excess": float(diff.min()),
        "best_month_excess": float(diff.max()),
    }
    out.update({f"capm_{k}": v for k, v in reg.items()})
    out.update(cap)
    out.update(roll_stats)
    if expense_ratio is not None and out["excess_ann_geom"] == out["excess_ann_geom"]:
        out["expense_ratio"] = expense_ratio
        out["excess_before_fees"] = out["excess_ann_geom"] + expense_ratio
    out["_rolling"] = roll
    return out
