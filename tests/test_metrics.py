"""Sanity checks on the statistics. Run with: python -m tests.test_metrics"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import metrics as m


def approx(a, b, tol=1e-8):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_annualised():
    # 12 months of exactly 1% compounds to 1.01^12 - 1, annualised is the same
    r = pd.Series([0.01] * 12, index=pd.date_range("2021-01-31", periods=12, freq="ME"))
    approx(m.cumulative(r), 1.01 ** 12 - 1)
    approx(m.annualised(r), 1.01 ** 12 - 1)
    # 24 months annualises back to the same figure
    r24 = pd.Series([0.01] * 24, index=pd.date_range("2021-01-31", periods=24, freq="ME"))
    approx(m.annualised(r24), 1.01 ** 12 - 1, 1e-10)


def test_vol_and_sharpe():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.01, 0.04, 600),
                  index=pd.date_range("1990-01-31", periods=600, freq="ME"))
    approx(m.ann_vol(r), r.std(ddof=1) * np.sqrt(12), 1e-12)
    # with zero risk free the Sharpe is mean/sd scaled
    approx(m.sharpe(r), r.mean() / r.std(ddof=1) * np.sqrt(12), 1e-12)


def test_drawdown():
    px = pd.Series([100, 120, 60, 90, 130],
                   index=pd.date_range("2020-01-31", periods=5, freq="ME"))
    dd = m.max_drawdown(px)
    approx(dd["max_dd"], 60 / 120 - 1)
    assert dd["peak"] == px.index[1]
    assert dd["trough"] == px.index[2]
    assert dd["recovered"] == px.index[4]
    # a monotonic series has no drawdown
    approx(m.max_drawdown(pd.Series([1, 2, 3.0]))["max_dd"], 0.0)


def test_identical_series_is_a_zero():
    idx = pd.date_range("2021-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.008, 0.04, 48), index=idx)
    st = m.pair_stats(r, r, r, r)
    approx(st["excess_ann_geom"], 0.0, 1e-12)
    approx(st["tracking_error"], 0.0, 1e-12)
    approx(st["capm_beta"], 1.0, 1e-8)
    approx(st["capm_alpha_monthly"], 0.0, 1e-10)
    approx(st["dd_diff"], 0.0, 1e-12)
    assert st["_rolling"].abs().max() < 1e-12


def test_known_beta_and_alpha():
    idx = pd.date_range("2015-01-31", periods=240, freq="ME")
    rng = np.random.default_rng(2)
    b = pd.Series(rng.normal(0.007, 0.04, 240), index=idx)
    true_beta, true_alpha = 1.30, 0.0020
    f = true_alpha + true_beta * b + pd.Series(rng.normal(0, 0.005, 240), index=idx)
    reg = m.capm(f, b)
    assert abs(reg["beta"] - true_beta) < 0.02, reg["beta"]
    assert abs(reg["alpha_monthly"] - true_alpha) < 0.001, reg["alpha_monthly"]
    assert reg["r2"] > 0.95


def test_ir_and_tstat_are_consistent():
    idx = pd.date_range("2021-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(3)
    b = pd.Series(rng.normal(0.007, 0.04, 60), index=idx)
    f = b + pd.Series(rng.normal(0.0015, 0.006, 60), index=idx)
    st = m.pair_stats(f, b, f, b)
    diff = f - b
    approx(st["tracking_error"], diff.std(ddof=1) * np.sqrt(12), 1e-12)
    approx(st["excess_ann_mean"], diff.mean() * 12, 1e-12)
    approx(st["info_ratio"], (diff.mean() * 12) / (diff.std(ddof=1) * np.sqrt(12)), 1e-10)
    approx(st["t_stat"], st["info_ratio"] * np.sqrt(60 / 12), 1e-10)
    # t-stat from the IR should match a plain t-test on the monthly differences
    naive_t = diff.mean() / (diff.std(ddof=1) / np.sqrt(60))
    approx(st["t_stat"], naive_t, 1e-8)


def test_capture_ratios():
    idx = pd.date_range("2021-01-31", periods=6, freq="ME")
    b = pd.Series([0.02, 0.04, -0.02, -0.04, 0.06, -0.06], index=idx)
    f = b * 1.5
    cap = m.capture(f, b)
    approx(cap["up_capture"], 1.5, 1e-10)
    approx(cap["down_capture"], 1.5, 1e-10)
    assert cap["n_up"] == 3 and cap["n_down"] == 3


def test_newey_west_widens_se_under_autocorrelation():
    rng = np.random.default_rng(4)
    n = 400
    e = rng.normal(0, 1, n)
    # strongly positively autocorrelated series
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.8 * y[i - 1] + e[i]
    X = np.ones((n, 1))
    beta, resid, XtX_inv = m.ols(y, X)
    se_ols = np.sqrt(np.diag(XtX_inv * (resid @ resid / (n - 1))))[0]
    se_nw = m.newey_west_se(X, resid, XtX_inv, lags=11)[0]
    assert se_nw > se_ols * 1.5, (se_nw, se_ols)


def test_rolling_excess_and_overlap_correction():
    idx = pd.date_range("2018-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(5)
    b = pd.Series(rng.normal(0.007, 0.04, 120), index=idx)
    f = b + pd.Series(rng.normal(0.001, 0.005, 120), index=idx)
    roll = m.rolling_excess(f, b, 12)
    assert len(roll) == 120 - 12 + 1
    # the 12 month window is a compounded difference, so check one by hand
    seg_f = (1 + f.iloc[:12]).prod() - 1
    seg_b = (1 + b.iloc[:12]).prod() - 1
    approx(roll.iloc[0], seg_f - seg_b, 1e-12)
    sig = m.rolling_significance(roll, lags=11)
    approx(sig["roll_mean"], roll.mean(), 1e-10)
    # overlapping windows inflate the naive t-stat; the corrected one must be smaller
    assert abs(sig["roll_t_nw"]) < abs(sig["roll_t_naive"])


def test_short_history_is_handled():
    idx = pd.date_range("2025-01-31", periods=4, freq="ME")
    r = pd.Series([0.01, -0.02, 0.03, 0.00], index=idx)
    st = m.pair_stats(r, r * 0.5, r, r * 0.5)
    assert st["n_months"] == 4
    assert np.isnan(st["capm_beta"])          # too few points to regress
    assert st["_rolling"].empty               # not enough for a 12 month window


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  pass  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
