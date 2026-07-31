"""Writes the observations memo, with the numbers filled in from the run.

The prose is deliberately hedged where the statistics are weak. If you loosen the
language, loosen it deliberately.
"""

from __future__ import annotations

import pandas as pd

from .config import Settings
from .tables import decomposition_table


def _pct(v, d=2, sign=True):
    if v is None or v != v:
        return "n/a"
    return f"{v * 100:{'+' if sign else ''}.{d}f}%"


def _verdict(row) -> str:
    t = row["t_stat"]
    e = row["excess_ann_geom"]
    if e != e or t != t:
        return "no read"
    if abs(t) >= 2:
        return "ahead, significant" if e > 0 else "behind, significant"
    if abs(t) >= 1:
        return "ahead, weak" if e > 0 else "behind, weak"
    return "indistinguishable"


def write(path, df: pd.DataFrame, settings: Settings, skipped: list[str]) -> None:
    demo = settings.source == "demo"
    lines: list[str] = []
    w = lines.append

    w("# DFA ETFs versus stated benchmarks: key observations")
    w("")
    if demo:
        w("> **This file was generated from simulated data.** Every figure below is a plumbing")
        w("> test, not a finding. Re-run with `data.source: yahoo` or `csv`.")
        w("")
    w(f"Generated {pd.Timestamp.today():%Y-%m-%d}. Source: `{settings.source}`. "
      f"Window convention: `{settings.a('window')}`.")
    w("")

    if df.empty:
        w("No results. Check the exceptions sheet in the workbook.")
        path.write_text("\n".join(lines))
        return

    style = df[df["bm_kind"] == "style"].copy()
    broad = df[df["bm_kind"] == "broad"].copy()
    style["verdict"] = style.apply(_verdict, axis=1)

    n = len(style)
    n_ahead = int((style["excess_ann_geom"] > 0).sum())
    n_sig = int((style["t_stat"].abs() >= 2).sum())
    n_sig_pos = int(((style["t_stat"] >= 2) & (style["excess_ann_geom"] > 0)).sum())
    med = style["excess_ann_geom"].median()
    med_te = style["tracking_error"].median()
    med_yrs = style["years"].median()

    w("## The short answer")
    w("")
    w(f"Across {n} funds with a median of {med_yrs:.1f} years of history, {n_ahead} are ahead of "
      f"their style benchmark and {n - n_ahead} are behind. The median annualised excess return is "
      f"{_pct(med)}. Only {n_sig} of {n} clear an absolute t-statistic of 2, and "
      f"{n_sig_pos} of those are positive.")
    w("")
    w(f"Median tracking error against the style benchmark is {_pct(med_te, 2, sign=False)}. At that "
      f"level of tracking error and this length of history, an excess return needs to be roughly "
      f"{_pct(2 * med_te / (med_yrs ** 0.5), 2, sign=False)} a year before it separates from noise. "
      f"That threshold, not the point estimates, is the main constraint on what can be claimed here.")
    w("")

    w("## Ranked by excess return versus the style benchmark")
    w("")
    w("| Fund | Sleeve | Style benchmark | Yrs | Excess ann | Pre-fee | TE | IR | t-stat | Read |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in style.sort_values("excess_ann_geom", ascending=False).itertuples():
        w(f"| {r.ticker} | {r.sleeve} | {r.bm_code} | {r.years:.1f} | "
          f"{_pct(r.excess_ann_geom)} | {_pct(getattr(r, 'excess_before_fees', float('nan')))} | "
          f"{_pct(r.tracking_error, 2, sign=False)} | {r.info_ratio:.2f} | {r.t_stat:.2f} | "
          f"{r.verdict} |")
    w("")

    w("## Where the value added sits")
    w("")
    dec = decomposition_table(df)
    if not dec.empty:
        w("Excess return against the broad regulatory benchmark splits into two pieces: the part "
          "delivered by the style tilt itself, and the part left over once the style index is the "
          "yardstick. The second piece is the closer read on implementation.")
        w("")
        w("| Fund | Sleeve | vs broad | Style tilt | Implementation | t-stat |")
        w("|---|---|---|---|---|---|")
        for r in dec.itertuples():
            w(f"| {r.Ticker} | {r.Sleeve} | {getattr(r, '_5')} | {getattr(r, '_6')} | "
              f"{getattr(r, '_7')} | {getattr(r, '_8')} |")
        w("")

    best = style.loc[style["excess_ann_geom"].idxmax()]
    worst = style.loc[style["excess_ann_geom"].idxmin()]
    w(f"The widest positive gap is {best['ticker']} at {_pct(best['excess_ann_geom'])} a year "
      f"(t = {best['t_stat']:.2f}), the widest negative gap is {worst['ticker']} at "
      f"{_pct(worst['excess_ann_geom'])} (t = {worst['t_stat']:.2f}).")
    w("")

    by_sleeve = style.groupby("sleeve")["excess_ann_geom"].agg(["median", "count"])
    w("Median excess return by sleeve:")
    w("")
    w("| Sleeve | Funds | Median excess ann |")
    w("|---|---|---|")
    for sleeve, row in by_sleeve.sort_values("median", ascending=False).iterrows():
        w(f"| {sleeve} | {int(row['count'])} | {_pct(row['median'])} |")
    w("")

    w("## Fees")
    w("")
    ahead_prefee = int((style["excess_before_fees"] > 0).sum()) if "excess_before_fees" in style else 0
    w(f"Reported returns are already net of fees. Adding the current expense ratio back, "
      f"{ahead_prefee} of {n} funds show a positive gross excess return, against {n_ahead} net. "
      f"The difference between those two counts is the set of funds where gross implementation is "
      f"positive but the fee consumes it. Note that Dimensional has cut expense ratios repeatedly "
      f"since these funds listed, so the current ratio understates the fee actually charged over "
      f"most of the window.")
    w("")

    w("## Caveats that matter for how this gets used")
    w("")
    caveats = [
        "Sample length. Three to five years is not enough to rank managers on realised excess "
        "return. The ranking above is mostly a ranking of which factor tilts paid off in this "
        "particular window.",
        "Benchmark choice does most of the work. Against the broad regulatory benchmark these funds "
        "look very different from how they look against a style index. Neither framing is wrong, "
        "but they answer different questions and should not be mixed in the same sentence.",
    ]
    if settings.source == "yahoo":
        caveats.append(
            "Benchmarks here are proxy ETFs, not index total return series. Each proxy carries its "
            "own fee and tracking error, which flatters the Dimensional fund by roughly the proxy's "
            "expense ratio. Anything client facing should be rebuilt on Bloomberg index series."
        )
        caveats.append(
            "Returns are market-price based rather than NAV based, so premium and discount noise is "
            "inside the excess return."
        )
    caveats.append(
        "Converted funds. Dimensional's published since-inception numbers splice predecessor mutual "
        "fund NAV history. This book starts at ETF listing unless that history is supplied, so it "
        "will not tie to the fact sheets."
    )
    caveats.append(
        "Survivorship and selection. The universe is the current lineup. No adjustment is made for "
        "funds or share classes that no longer exist."
    )
    for c in caveats:
        w(f"- {c}")
    w("")

    if skipped:
        w("## Skipped pairs")
        w("")
        for s in skipped:
            w(f"- {s}")
        w("")

    path.write_text("\n".join(lines))
