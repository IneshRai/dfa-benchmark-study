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


def _robustness_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Style result against peer result, per fund, with a verdict."""
    st = df[df["bm_kind"] == "style"].set_index("ticker")
    pe = df[df["bm_kind"] == "peer"]
    if pe.empty:
        return pd.DataFrame()
    pe = pe.set_index("ticker")
    rows = []
    for t in st.index:
        if t not in pe.index:
            continue
        a, b = st.loc[t], pe.loc[t]
        e_s, e_p = a["excess_ann_geom"], b["excess_ann_geom"]
        if e_s != e_s or e_p != e_p:
            continue
        if e_s * e_p < 0:
            verdict = "sign flips"
        elif e_s > 0:
            verdict = "both positive"
        else:
            verdict = "both negative"
        if verdict != "sign flips" and min(abs(a["t_stat"]), abs(b["t_stat"])) >= 2:
            verdict += ", significant"
        rows.append({
            "ticker": t, "sleeve": a["sleeve"],
            "style_bm": a["bm_code"], "e_style": e_s, "t_style": a["t_stat"],
            "peer_bm": b["bm_code"], "e_peer": e_p, "t_peer": b["t_stat"],
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


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
    w(f"Source: `{settings.source}`. Window convention: `{settings.a('window')}`.")
    w("")

    if df.empty:
        w("No results. Check the exceptions sheet in the workbook.")
        path.write_text("\n".join(lines))
        return

    style = df[df["bm_kind"] == "style"].copy()
    style["verdict"] = style.apply(_verdict, axis=1)
    rob = _robustness_frame(df)
    has_peer = not rob.empty

    n = len(style)
    n_ahead = int((style["excess_ann_geom"] > 0).sum())
    n_sig_pos = int(((style["t_stat"] >= 2) & (style["excess_ann_geom"] > 0)).sum())
    med_te = style["tracking_error"].median()
    med_yrs = style["years"].median()

    # ------------------------------------------------------------------ headline
    w("## The short answer")
    w("")
    if has_peer:
        flips = rob[rob["verdict"] == "sign flips"]
        robust = rob[(rob["verdict"].str.startswith("both positive"))
                     & (rob["t_style"].abs() >= 2) & (rob["t_peer"].abs() >= 2)]
        behind = rob[(rob["t_peer"] <= -2)]
        w(f"Of {len(rob)} funds tested against both a style index and a passive peer fund, "
          f"**{len(flips)} change sign depending on which comparator is used**. That is the "
          f"first thing to know about this data set. An apparent excess return that reverses "
          f"when the benchmark changes is a statement about the benchmark, not the fund.")
        w("")
        if len(robust):
            names = ", ".join(f"**{r.ticker}** ({_pct(r.e_style)} vs {r.style_bm}, "
                              f"{_pct(r.e_peer)} vs {r.peer_bm})" for r in robust.itertuples())
            w(f"Funds with a positive result that survives both comparisons at an absolute "
              f"t-statistic of 2 or better: {names}.")
        else:
            w("No fund shows a positive excess return that clears an absolute t-statistic of 2 "
              "against both comparators.")
        w("")
        if len(behind):
            names = ", ".join(f"**{r.ticker}** ({_pct(r.e_peer)} vs {r.peer_bm}, "
                              f"t = {r.t_peer:.2f})" for r in behind.itertuples())
            w(f"Funds significantly behind a passive peer: {names}.")
            w("")
    w(f"Across {n} funds with a median of {med_yrs:.1f} years of history, {n_ahead} are ahead of "
      f"their style benchmark and {n - n_ahead} are behind; {n_sig_pos} are ahead with an "
      f"absolute t-statistic of 2 or better. Median tracking error against the style benchmark "
      f"is {_pct(med_te, 2, sign=False)}, so an excess return needs to be roughly "
      f"{_pct(2 * med_te / (med_yrs ** 0.5), 2, sign=False)} a year before it separates from "
      f"noise. That threshold, not the point estimates, is the binding constraint.")
    w("")

    # -------------------------------------------------------------- robustness
    if has_peer:
        w("## Does the result survive a change of benchmark?")
        w("")
        w("| Fund | Sleeve | Style bm | Excess | t | Peer | Excess | t | Verdict |")
        w("|---|---|---|---|---|---|---|---|---|")
        order = {"sign flips": 0}
        for r in rob.sort_values("verdict", key=lambda c: c.map(lambda v: order.get(v, 1))) \
                    .itertuples():
            w(f"| {r.ticker} | {r.sleeve} | {r.style_bm} | {_pct(r.e_style)} | "
              f"{r.t_style:.2f} | {r.peer_bm} | {_pct(r.e_peer)} | {r.t_peer:.2f} | "
              f"{r.verdict} |")
        w("")
        w("The peer comparison is a different index family at a comparable fee. Where a peer is "
          "an actively managed systematic fund rather than an index tracker, the comparison mixes "
          "tilt design with implementation and should be read as \"would we rather own the other "
          "one\" rather than as a measure of execution.")
        w("")

    # ------------------------------------------------------- style-relative table
    w("## Ranked by excess return versus the style benchmark")
    w("")
    hdr = "| Fund | Sleeve | Style benchmark | Yrs | Excess ann | Pre-fee | TE | IR | t-stat | Read |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    if has_peer:
        hdr = hdr[:-1] + " Peer check |"
        sep = sep[:-1] + "---|"
    w(hdr); w(sep)
    rob_i = rob.set_index("ticker") if has_peer else None
    for r in style.sort_values("excess_ann_geom", ascending=False).itertuples():
        row = (f"| {r.ticker} | {r.sleeve} | {r.bm_code} | {r.years:.1f} | "
               f"{_pct(r.excess_ann_geom)} | {_pct(getattr(r, 'excess_before_fees', float('nan')))} | "
               f"{_pct(r.tracking_error, 2, sign=False)} | {r.info_ratio:.2f} | {r.t_stat:.2f} | "
               f"{r.verdict} |")
        if has_peer:
            v = rob_i["verdict"].get(r.ticker, "not tested")
            row = row[:-1] + f" {v} |"
        w(row)
    w("")

    # ----------------------------------------------------------- decomposition
    w("## Where the value added sits")
    w("")
    dec = decomposition_table(df)
    if not dec.empty:
        w("Excess return against the broad regulatory benchmark splits into the part delivered by "
          "the style tilt and the part left over once the style index is the yardstick. The second "
          "piece is the closer read on implementation.")
        w("")
        w("| Fund | Sleeve | vs broad | Style tilt | Implementation | t-stat |")
        w("|---|---|---|---|---|---|")
        for r in dec.itertuples():
            w(f"| {r.Ticker} | {r.Sleeve} | {getattr(r, '_5')} | {getattr(r, '_6')} | "
              f"{getattr(r, '_7')} | {getattr(r, '_8')} |")
        w("")

    # only name a "best" result if it actually holds up
    pool = style
    if has_peer:
        keep = rob[rob["verdict"].str.startswith("both")]["ticker"]
        pool = style[style["ticker"].isin(keep)]
    if len(pool):
        best = pool.loc[pool["excess_ann_geom"].idxmax()]
        worst = pool.loc[pool["excess_ann_geom"].idxmin()]
        qual = " among funds whose sign is stable across comparators" if has_peer else ""
        w(f"The widest positive gap{qual} is {best['ticker']} at "
          f"{_pct(best['excess_ann_geom'])} a year (t = {best['t_stat']:.2f}); the widest negative "
          f"is {worst['ticker']} at {_pct(worst['excess_ann_geom'])} (t = {worst['t_stat']:.2f}).")
        w("")

    by_sleeve = style.groupby("sleeve")["excess_ann_geom"].agg(["median", "count"])
    w("Median excess return by sleeve, versus style benchmark:")
    w("")
    w("| Sleeve | Funds | Median excess ann |")
    w("|---|---|---|")
    for sleeve, row in by_sleeve.sort_values("median", ascending=False).iterrows():
        w(f"| {sleeve} | {int(row['count'])} | {_pct(row['median'])} |")
    w("")

    # ------------------------------------------------------------------- fees
    w("## Fees")
    w("")
    ahead_prefee = int((style["excess_before_fees"] > 0).sum()) \
        if "excess_before_fees" in style else 0
    w(f"Reported returns are already net of fees. Adding the current expense ratio back, "
      f"{ahead_prefee} of {n} funds show a positive gross excess return against {n_ahead} net. "
      f"Dimensional has cut expense ratios repeatedly since these funds listed, so the current "
      f"ratio understates the fee actually charged over most of the window.")
    w("")

    # --------------------------------------------------------------- caveats
    w("## Caveats that matter for how this gets used")
    w("")
    caveats = [
        "Sample length. Three to five years is not enough to rank managers on realised excess "
        "return. Much of the spread above is a ranking of which factor tilts paid off in this "
        "particular window.",
        "Benchmark choice does most of the work. Against the broad regulatory benchmark these "
        "funds look very different from how they look against a style index. Neither framing is "
        "wrong, but they answer different questions and should not be mixed in one sentence.",
    ]
    if has_peer:
        caveats.append(
            "Comparator sensitivity is measured, not assumed. The robustness table above shows "
            "how many results reverse under a reasonable alternative benchmark. Treat any fund "
            "marked as flipping as having no established result."
        )
    if settings.source == "yahoo":
        caveats.append(
            "Benchmarks are proxy ETFs, not index total return series. Each proxy carries its own "
            "fee and tracking error, which flatters the Dimensional fund by roughly the proxy's "
            "expense ratio. Anything client facing should be rebuilt on Bloomberg index series."
        )
        caveats.append(
            "Returns are market-price based rather than NAV based, so premium and discount noise "
            "sits inside the excess return."
        )
    caveats.append(
        "Converted funds. Dimensional's published since-inception figures splice predecessor "
        "mutual fund NAV history. This book starts at ETF listing unless that history is "
        "supplied, so it will not tie to the fact sheets."
    )
    caveats.append(
        "Windows are not common. Each fund is measured from its own listing date, so the funds "
        "in the tables above are not all measured over the same period."
    )
    for c in caveats:
        w(f"- {c}")
    w("")

    if skipped:
        w("## Skipped pairs")
        w("")
        for sk in skipped:
            w(f"- {sk}")
        w("")

    path.write_text("\n".join(lines))
