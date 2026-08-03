"""Independent verification of the built deliverables.

Recomputes the headline statistics from the monthly return panel in the
workbook and asserts the reported values match. The point is that this does
not import src.metrics, so a bug in the metrics module cannot make its own
output look correct.

    python tools/verify.py
    python tools/verify.py --xlsx output/dfa_benchmark_dataset.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 5e-5  # 0.5bp, comfortably inside display rounding
checks: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    checks.append((bool(ok), name, detail))


def geom(f: float, b: float) -> float:
    return (1.0 + f) / (1.0 + b) - 1.0


def nw_t(x: np.ndarray, lags: int = 11) -> float:
    n = len(x)
    m = x.mean()
    d = x - m
    var = (d @ d) / n
    for L in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - L / (lags + 1)) * ((d[L:] @ d[:-L]) / n)
    return m / np.sqrt(var / n) if var > 0 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="output/dfa_benchmark_dataset.xlsx")
    ap.add_argument("--pdf", default="output/dfa_chartbook.pdf")
    ap.add_argument("--md", default="output/summary_observations.md")
    args = ap.parse_args()

    xlsx = Path(args.xlsx)
    if not xlsx.exists():
        print(f"ERROR: {xlsx} not found. Run python -m src.run --source yahoo first.")
        return 2

    xl = pd.ExcelFile(xlsx)
    s = xl.parse("stats_all")
    monthly = xl.parse("returns_monthly", index_col=0)
    notes = xl.parse("notes")

    # ---------------------------------------------------------- structure
    check("decomposition" in xl.sheet_names, "decomposition sheet present")
    check("rolling_excess_12m" in xl.sheet_names, "rolling_excess_12m sheet present")
    n_funds = s.ticker.nunique()
    kinds = sorted(s.bm_kind.unique())
    check(kinds == ["broad", "peer", "style"], "three benchmark kinds", str(kinds))
    check(len(s) == n_funds * 3, f"{n_funds} funds x 3 comparisons = {len(s)} rows")
    check(bool((abs(s.years - s.n_months / 12) < 1e-9).all()), "years == n_months/12")

    # --------------------------------------------- returns reproduce from panel
    bad_f, bad_b, bad_e, bad_pre = [], [], [], []
    for _, r in s.iterrows():
        cols = [r.ticker, r.bm_code]
        if not set(cols) <= set(monthly.columns):
            continue
        a = monthly[cols].dropna()
        if len(a) != r.n_months:
            continue
        yrs = len(a) / 12
        fa = (1 + a[r.ticker]).prod() ** (1 / yrs) - 1
        ba = (1 + a[r.bm_code]).prod() ** (1 / yrs) - 1
        tag = f"{r.ticker}/{r.bm_kind}"
        if abs(fa - r.fund_ann) > TOL:
            bad_f.append(tag)
        if abs(ba - r.bench_ann) > TOL:
            bad_b.append(tag)
        if abs(geom(fa, ba) - r.excess_ann_geom) > TOL:
            bad_e.append(f"{tag} reported {r.excess_ann_geom*100:+.2f}% "
                         f"expected {geom(fa, ba)*100:+.2f}%")
        if r.excess_before_fees == r.excess_before_fees:
            want = geom(fa + r.expense_ratio, ba)
            if abs(want - r.excess_before_fees) > TOL:
                bad_pre.append(tag)

    check(not bad_f, "fund_ann reproduces from returns_monthly", ", ".join(bad_f[:4]))
    check(not bad_b, "bench_ann reproduces from returns_monthly", ", ".join(bad_b[:4]))
    check(not bad_e, "excess_ann_geom is GEOMETRIC, (1+rf)/(1+rb)-1",
          "; ".join(bad_e[:4]) + (f" ... {len(bad_e)} rows" if len(bad_e) > 4 else ""))
    check(not bad_pre, "excess_before_fees consistent with geometric convention",
          ", ".join(bad_pre[:4]))

    # the arithmetic version, explicitly ruled out
    arith = s.fund_ann - s.bench_ann
    n_arith = int((abs(arith - s.excess_ann_geom) < 1e-9).sum())
    check(n_arith <= 2, "excess is not arithmetic subtraction",
          f"{n_arith} of {len(s)} rows equal fund_ann - bench_ann")

    # ---------------------------------------------------------- statistics
    ir = s.excess_ann_mean / s.tracking_error
    check(bool((abs(ir - s.info_ratio) < 1e-6).all()),
          "info_ratio == excess_ann_mean / tracking_error")
    check(bool((abs(s.info_ratio * np.sqrt(s.years) - s.t_stat) < 1e-6).all()),
          "t_stat == info_ratio * sqrt(years)")
    check(bool((s.tracking_error >= 0).all()), "tracking error non-negative")
    check(bool((s.hit_rate.between(0, 1)).all()), "hit rate in [0,1]")

    # ------------------------------------------- Newey-West actually corrects
    roll = xl.parse("rolling_excess_12m", index_col=0)
    labelled = all(" vs " in str(c) for c in roll.columns)
    check(labelled, "rolling_excess_12m columns name their benchmark",
          f"got {list(roll.columns)[:3]}")
    mism, shrink_ok = [], True
    for c in roll.columns:
        tkr = str(c).split(" vs ")[0].strip()
        row = s[(s.ticker == tkr) & (s.bm_kind == "style")]
        if row.empty:
            continue
        row = row.iloc[0]
        mine = nw_t(np.asarray(roll[c].dropna(), float))
        if abs(mine - row.roll_t_nw) > 0.1:
            mism.append(f"{tkr} {mine:.2f} vs {row.roll_t_nw:.2f}")
        if abs(row.roll_t_nw) >= abs(row.roll_t_naive):
            shrink_ok = False
    check(not mism, "roll_t_nw reproduces independently", "; ".join(mism[:4]))
    check(shrink_ok, "Newey-West t is smaller than naive t on every row")

    # -------------------------------------------------- decomposition identity
    dec = xl.parse("decomposition")
    p = lambda v: np.nan if str(v).strip() in {"n/a", "nan", ""} else float(
        str(v).replace("%", "").replace("+", "")) / 100
    bad_dec = []
    for _, r in dec.iterrows():
        b, t_, i = p(r["vs broad"]), p(r["Style tilt effect"]), p(r["Implementation vs style"])
        if any(v != v for v in (b, t_, i)):
            continue
        if abs((1 + t_) * (1 + i) - (1 + b)) > 2e-4:
            bad_dec.append(f"{r.Ticker} {b*100:+.2f} vs tilt {t_*100:+.2f} x impl {i*100:+.2f}")
    check(not bad_dec, "decomposition composes: (1+broad)=(1+tilt)(1+impl)",
          "; ".join(bad_dec[:4]))

    # ---------------------------------------------------- documentation matches
    txt = " ".join(notes.astype(str).values.ravel()).lower()
    check("geometric" in txt, "notes sheet documents the geometric definition")
    check("(1 + style tilt)" in txt or "style tilt" in txt,
          "notes sheet documents the decomposition identity")

    pdf, md = Path(args.pdf), Path(args.md)
    check(pdf.exists() and pdf.stat().st_size > 100_000, "chartbook PDF written",
          f"{pdf.stat().st_size//1024}KB" if pdf.exists() else "missing")
    if md.exists():
        body = md.read_text()
        check("three benchmarks" in body.lower() or "passive peer" in body.lower(),
              "memo mentions the peer comparison")
        # A stripped em dash leaves a double space inside a table cell, which
        # silently merges two columns. Catch the symptom, not the character.
        mangled = [ln.strip()[:60] for ln in body.splitlines()
                   if ln.startswith("|") and "  " in ln.replace(" |", "").replace("| ", "")]
        check(not mangled, "no merged table cells from stripped separators",
              "; ".join(mangled[:2]))

    # ------------------------------------------------------------------ report
    width = max(len(n) for _, n, _ in checks) + 2
    n_fail = 0
    print(f"\nverifying {xlsx}\n")
    for ok, name, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}"
              + (f"  {detail}" if detail and not ok else ""))
        n_fail += not ok
    print(f"\n{len(checks) - n_fail} passed, {n_fail} failed")
    if not n_fail:
        print("Deliverables reconcile to the raw return panel.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
