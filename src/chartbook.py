"""Assembles the PDF chartbook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from . import charts
from .config import Settings
from .tables import decomposition_table, excess_table, performance_table, raw_frame

PAGE = (11.0, 8.5)   # landscape US letter
ROWS_PER_PAGE = 16


def _new_page(demo: bool, footer_left: str, footer_right: str = ""):
    fig = plt.figure(figsize=PAGE)
    if demo:
        charts.watermark(fig)
    charts.footer(fig, footer_left, footer_right)
    return fig


def _cover(pdf, settings: Settings, df: pd.DataFrame, demo: bool):
    fig = _new_page(demo, "")
    fig.text(0.06, 0.80, "Dimensional ETFs versus stated benchmarks", fontsize=24)
    fig.text(0.06, 0.735, "Since-inception implementation review", fontsize=14, color="0.35")

    n_funds = df["ticker"].nunique() if not df.empty else 0
    span = ""
    if not df.empty:
        span = f"{pd.to_datetime(df['start']).min():%b %Y} to {pd.to_datetime(df['end']).max():%b %Y}"
    lines = [
        f"Funds covered: {n_funds}",
        f"Comparison window: {span}",
        f"Data source: {settings.source}",
        f"Return basis: monthly total return, dividends reinvested",
        f"Window convention: {settings.a('window').replace('_', ' ')}",
    ]
    for i, line in enumerate(lines):
        fig.text(0.06, 0.55 - i * 0.045, line, fontsize=11)

    if demo:
        fig.text(0.06, 0.22,
                 "THIS COPY WAS BUILT ON SIMULATED DATA.\n"
                 "Every number is a plumbing test. Re-run with data.source: yahoo or csv.",
                 fontsize=12, color="0.25")
    pdf.savefig(fig)
    charts.close(fig)


def _methodology(pdf, settings: Settings, demo: bool, footer_left: str):
    fig = _new_page(demo, footer_left, "Methodology")
    fig.text(0.06, 0.92, "How to read this book", fontsize=16)

    body = [
        ("Three benchmarks per fund", [
            "Dimensional funds report against a broad regulatory benchmark (Russell 3000 for US funds,",
            "an MSCI aggregate for the rest) and, separately, against a style index that matches the",
            "fund's intended tilt. Both are shown because they answer different questions.",
            "",
            "Excess return versus the broad benchmark is almost entirely a factor tilt result. A small",
            "value fund beating the Russell 3000 tells you value worked, not that Dimensional executed",
            "well. Excess return versus the style index is the closer read on implementation: security",
            "selection inside the style, trading, securities lending, and fees.",
            "",
            "A third comparison is a passive or systematic competitor from a different index family",
            "at a comparable fee. It is a robustness check, not a measure of execution. Where the",
            "peer is itself a systematic fund rather than an index tracker, the comparison mixes",
            "tilt design with implementation and should be read as which you would rather own.",
        ]),
        ("What counts as evidence", [
            "Most of these funds have three to five years of history. Over four years, an excess return",
            "is only distinguishable from noise if the information ratio is near 1.0. The t-statistic",
            "column is IR times the square root of years. Bars on the summary pages are hatched where",
            "the t-statistic is below 2 in absolute value, which is most of them. Rolling 12-month",
            "windows overlap, so their t-statistics are Newey-West corrected with 11 lags.",
        ]),
        ("Known limitations", [
            "Index total return series are not free. Where the source is Yahoo, each benchmark is a",
            "proxy ETF, so the comparison carries the proxy's own fee and tracking error. The proxy",
            "quality column flags where the proxy is only approximate. Run the book from Bloomberg",
            "index series before anything leaves the desk.",
            "",
            "Seven of these funds are converted mutual funds. Dimensional's published since-inception",
            "figures splice the predecessor mutual fund NAV history, which starts in the late 1990s or",
            "2000s. Ticker history from Yahoo starts at ETF listing. This book uses the listing date",
            "unless the predecessor series is supplied, so it is not comparable to the fact sheet.",
            "",
            "Returns from Yahoo are market-price based, not NAV based. Premium and discount noise is",
            "small over multi-year windows but it is not zero.",
        ]),
    ]

    y = 0.855
    for heading, paras in body:
        fig.text(0.06, y, heading, fontsize=11, weight="bold")
        y -= 0.032
        for line in paras:
            fig.text(0.06, y, line, fontsize=9, color="0.15")
            y -= 0.0235
        y -= 0.018
    pdf.savefig(fig)
    charts.close(fig)


def _summary_bars(pdf, df: pd.DataFrame, demo: bool, footer_left: str):
    fig = _new_page(demo, footer_left, "Summary")
    kinds = [k for k in ("broad", "style", "peer") if (df["bm_kind"] == k).any()]
    gs = GridSpec(1, len(kinds), figure=fig, left=0.09, right=0.97, top=0.86,
                  bottom=0.12, wspace=0.55)
    fig.text(0.06, 0.93, "Annualised excess return since inception", fontsize=16)
    fig.text(0.06, 0.895, "Hatched bars are not statistically distinguishable from zero "
                          "(absolute t-statistic below 2)", fontsize=8, color="0.4")

    for i, kind in enumerate(kinds):
        sub = df[df["bm_kind"] == kind].sort_values("excess_ann_geom")
        ax = fig.add_subplot(gs[0, i])
        if sub.empty:
            ax.axis("off")
            continue
        labels = [f"{r.ticker}  vs {r.bm_code}" for r in sub.itertuples()]
        charts.excess_bar_chart(
            ax, labels, sub["excess_ann_geom"].tolist(), sub["t_stat"].tolist(),
            title={"broad": "vs broad regulatory benchmark",
                   "style": "vs style benchmark",
                   "peer": "vs passive peer fund"}[kind],
        )
    pdf.savefig(fig)
    charts.close(fig)


def _table_pages(pdf, table: pd.DataFrame, title: str, subtitle: str, demo: bool,
                 footer_left: str, tag: str, col_width=None):
    if table.empty:
        return
    chunks = [table.iloc[i:i + ROWS_PER_PAGE] for i in range(0, len(table), ROWS_PER_PAGE)]
    for j, chunk in enumerate(chunks):
        fig = _new_page(demo, footer_left, tag)
        fig.text(0.04, 0.93, title + (f"  ({j + 1} of {len(chunks)})" if len(chunks) > 1 else ""),
                 fontsize=16)
        if subtitle:
            fig.text(0.04, 0.895, subtitle, fontsize=8, color="0.4")
        ax = fig.add_axes([0.03, 0.10, 0.94, 0.76])
        charts.table_axis(ax, chunk.reset_index(drop=True), col_width=col_width, fontsize=6.8)
        pdf.savefig(fig)
        charts.close(fig)


def _fund_page(pdf, tkr: str, res: dict, monthly: pd.DataFrame, daily: pd.DataFrame,
               settings: Settings, demo: bool, footer_left: str):
    style = res.get((tkr, "style"))
    broad = res.get((tkr, "broad"))
    anchor = style or broad
    if anchor is None:
        return

    fig = _new_page(demo, footer_left, tkr)
    fig.text(0.04, 0.955, f"{tkr}  {anchor['fund_name']}", fontsize=15)
    sub = (f"{anchor['sleeve']}   |   {anchor['start']:%b %Y} to {anchor['end']:%b %Y}   |   "
           f"{anchor['n_months']} months   |   fee {anchor.get('expense_ratio', float('nan')) * 100:.2f}%")
    if anchor.get("short_history"):
        sub += "   |   SHORT HISTORY"
    fig.text(0.04, 0.925, sub, fontsize=8.5, color="0.4")

    gs = GridSpec(2, 2, figure=fig, left=0.07, right=0.96, top=0.885, bottom=0.34,
                  hspace=0.40, wspace=0.22)

    start = anchor["start"]
    codes = {}
    if broad is not None:
        codes["broad: " + broad["bm_code"]] = broad["bm_code"]
    if style is not None and (broad is None or style["bm_code"] != broad["bm_code"]):
        codes["style: " + style["bm_code"]] = style["bm_code"]

    m = monthly.loc[monthly.index >= start]
    series = {tkr: m[tkr].dropna()}
    for label, code in codes.items():
        if code in m.columns:
            series[label] = m[code].dropna()

    ax1 = fig.add_subplot(gs[0, 0])
    charts.growth_chart(ax1, series, log=True)

    ax2 = fig.add_subplot(gs[0, 1])
    rolls, seen = {}, set()
    for kind, label in (("broad", "vs broad"), ("style", "vs style"), ("peer", "vs peer")):
        st = res.get((tkr, kind))
        if st is None or st.get("_rolling") is None or st["bm_code"] in seen:
            continue
        seen.add(st["bm_code"])
        rolls[f"{label} ({st['bm_code']})"] = st["_rolling"]
    charts.rolling_excess_chart(ax2, rolls, int(settings.a("rolling_window_months")))

    ax3 = fig.add_subplot(gs[1, 0])
    d = daily.loc[daily.index >= start]
    px = {}
    for label, col in [(tkr, tkr)] + [(l, c) for l, c in codes.items()]:
        if col in d.columns:
            px[label] = (1 + d[col].dropna()).cumprod()
    charts.drawdown_chart(ax3, px)

    ax4 = fig.add_subplot(gs[1, 1])
    rel, seen_rel = {}, set()
    for kind, label in (("broad", "vs broad"), ("style", "vs style"), ("peer", "vs peer")):
        st = res.get((tkr, kind))
        if st is None or st["bm_code"] in seen_rel:
            continue
        seen_rel.add(st["bm_code"])
        code = st["bm_code"]
        if code in m.columns:
            pair = pd.concat({"f": m[tkr], "b": m[code]}, axis=1).dropna()
            rel[label] = (1 + pair["f"]) / (1 + pair["b"]) - 1
    charts.relative_chart(ax4, rel, title="Fund relative to benchmark, rebased to 100")

    stat_tbl = _fund_stat_table(tkr, res)
    n_rows = max(len(stat_tbl), 1)
    height = min(0.055 + 0.055 * n_rows, 0.22)
    ax5 = fig.add_axes([0.035, 0.245 - height, 0.93, height])
    charts.table_axis(ax5, stat_tbl, fontsize=6.0)
    fig.text(0.035, 0.055,
             "Excess and alpha are annualised. t-stat is the information ratio times the square "
             "root of years elapsed; treat anything under 2 as indistinguishable from zero. "
             "Roll t (NW) is Newey-West corrected for the overlap in 12-month windows.",
             fontsize=6.5, color="0.4")

    pdf.savefig(fig)
    charts.close(fig)


def _fund_stat_table(tkr: str, res: dict) -> pd.DataFrame:
    def f_pct(v, d=2, sign=False):
        if v is None or v != v:
            return "n/a"
        return f"{v * 100:{'+' if sign else ''}.{d}f}%"

    def f_num(v, d=2):
        return "n/a" if v is None or v != v else f"{v:.{d}f}"

    broad, style = res.get((tkr, "broad")), res.get((tkr, "style"))
    same = (broad is not None and style is not None
            and broad["bm_code"] == style["bm_code"])
    plan = ([("broad and style", broad)] if same
            else [(k, st) for k, st in (("broad", broad), ("style", style))
                  if st is not None])
    peer = res.get((tkr, "peer"))
    if peer is not None:
        plan = plan + [("peer", peer)]

    rows = []
    for kind, st in plan:
        rows.append({
            "Comparison": f"{kind} ({st['bm_code']})",
            "Proxy": f"{st.get('proxy_ticker', '')} [{st.get('proxy_quality', '')}]",
            "Fund": f_pct(st["fund_ann"], 1),
            "Bm": f_pct(st["bench_ann"], 1),
            "Excess": f_pct(st["excess_ann_geom"], 2, sign=True),
            "Pre-fee": f_pct(st.get("excess_before_fees"), 2, sign=True),
            "TE": f_pct(st["tracking_error"], 2),
            "IR": f_num(st["info_ratio"]),
            "t": f_num(st["t_stat"]),
            "Hit": f_pct(st["hit_rate"], 0),
            "Beta": f_num(st["capm_beta"]),
            "Alpha": f_pct(st["capm_alpha_ann"], 2, sign=True),
            "Alpha t": f_num(st["capm_alpha_t"]),
            "Up": f_num(st.get("up_capture")),
            "Down": f_num(st.get("down_capture")),
            "DD f": f_pct(st["fund_max_dd"], 1),
            "DD bm": f_pct(st["bench_max_dd"], 1),
            "Roll12m": f_pct(st.get("roll_mean"), 2, sign=True),
            "Roll t": f_num(st.get("roll_t_nw")),
            "Roll+": f_pct(st.get("roll_pct_positive"), 0),
        })
    return pd.DataFrame(rows)


def _robustness(pdf, df: pd.DataFrame, demo: bool, footer_left: str):
    """Does the sign of the excess return survive a change of benchmark?"""
    piv = df.pivot_table(index="ticker", columns="bm_kind",
                         values=["excess_ann_geom", "t_stat"])
    rows = []
    for tkr in piv.index:
        try:
            s, p = piv[("excess_ann_geom", "style")][tkr], piv[("excess_ann_geom", "peer")][tkr]
            ts, tp = piv[("t_stat", "style")][tkr], piv[("t_stat", "peer")][tkr]
        except KeyError:
            continue
        if s != s or p != p:
            continue
        sub = df[(df.ticker == tkr) & (df.bm_kind == "style")].iloc[0]
        pr = df[(df.ticker == tkr) & (df.bm_kind == "peer")].iloc[0]
        if s > 0 and p > 0:
            verdict = "holds, both positive"
        elif s < 0 and p < 0:
            verdict = "holds, both negative"
        else:
            verdict = "SIGN FLIPS"
        if verdict.startswith("holds") and min(abs(ts), abs(tp)) >= 2:
            verdict += ", significant"
        rows.append({
            "Ticker": tkr, "Sleeve": sub["sleeve"],
            "Style bm": sub["bm_code"], "Excess": f"{s * 100:+.2f}%", "t": f"{ts:.2f}",
            "Peer": pr["bm_code"], "Excess ": f"{p * 100:+.2f}%", "t ": f"{tp:.2f}",
            "Verdict": verdict,
        })
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return
    order = {"SIGN FLIPS": 0}
    tbl = tbl.sort_values("Verdict", key=lambda c: c.map(lambda v: order.get(v, 1)))

    fig = _new_page(demo, footer_left, "Robustness")
    fig.text(0.04, 0.93, "Does the result survive a change of benchmark?", fontsize=16)
    fig.text(0.04, 0.895,
             "Style benchmark against passive peer fund. A sign flip means the apparent result "
             "depended on the choice of comparator, not on the fund", fontsize=8, color="0.4")
    n_flip = int((tbl["Verdict"] == "SIGN FLIPS").sum())
    fig.text(0.04, 0.865, f"{n_flip} of {len(tbl)} funds flip sign.", fontsize=9)
    ax = fig.add_axes([0.03, 0.10, 0.94, 0.73])
    charts.table_axis(ax, tbl.reset_index(drop=True), fontsize=6.8,
                     col_width=[0.07, 0.15, 0.10, 0.09, 0.06, 0.09, 0.09, 0.06, 0.19])
    pdf.savefig(fig)
    charts.close(fig)


def _exceptions(pdf, skipped: list[str], benchmarks: pd.DataFrame, demo: bool, footer_left: str):
    fig = _new_page(demo, footer_left, "Data quality")
    fig.text(0.04, 0.93, "Data quality and exceptions", fontsize=16)

    import textwrap

    fig.text(0.04, 0.875, "Benchmark proxy quality", fontsize=11, weight="bold")
    bm = benchmarks.reset_index()
    tbl = bm[["bm_code", "bm_name", "proxy_ticker", "proxy_quality"]].rename(
        columns={"bm_code": "Code", "bm_name": "Benchmark", "proxy_ticker": "Proxy",
                 "proxy_quality": "Quality"})
    ax = fig.add_axes([0.035, 0.20, 0.45, 0.655])
    charts.table_axis(ax, tbl, fontsize=6.4, col_width=[0.20, 0.52, 0.13, 0.15])

    fig.text(0.53, 0.875, "Where the proxy is not the stated index", fontsize=11, weight="bold")
    y = 0.845
    for _, row in bm.iterrows():
        note = str(row["notes"]).strip()
        if not note or note.lower() == "nan" or row["proxy_quality"] == "exact":
            continue
        wrapped = textwrap.wrap(f"{row['bm_code']}: {note}", width=78)
        for k, line in enumerate(wrapped):
            fig.text(0.53 if k == 0 else 0.545, y, line, fontsize=7.2, color="0.15")
            y -= 0.021
        y -= 0.008
        if y < 0.24:
            break

    fig.text(0.04, 0.155, "Pairs skipped", fontsize=11, weight="bold")
    if skipped:
        for i, line in enumerate(skipped[:5]):
            fig.text(0.04, 0.125 - i * 0.021, "- " + line, fontsize=7.5, color="0.2")
        if len(skipped) > 5:
            fig.text(0.04, 0.125 - 5 * 0.021,
                     f"... and {len(skipped) - 5} more, see the exceptions sheet in the workbook",
                     fontsize=7.5, color="0.4")
    else:
        fig.text(0.04, 0.125, "None. Every configured pair produced an overlapping window.",
                 fontsize=7.5, color="0.2")
    pdf.savefig(fig)
    charts.close(fig)


def build(path, bundle: dict, settings: Settings) -> pd.DataFrame:
    results = bundle["results"]
    monthly, daily = bundle["monthly"], bundle["daily"]
    demo = settings.source == "demo"
    df = raw_frame(results)
    footer_left = (f"Source: {settings.source}. "
                   f"{'SIMULATED DATA' if demo else 'Total return, dividends reinvested'}.")

    with PdfPages(path) as pdf:
        _cover(pdf, settings, df, demo)
        _methodology(pdf, settings, demo, footer_left)
        if not df.empty:
            _summary_bars(pdf, df, demo, footer_left)
            _table_pages(pdf, performance_table(df, "style"),
                         "Performance summary, fund versus style benchmark",
                         "Cumulative and annualised total return, volatility, Sharpe ratio and "
                         "maximum drawdown over the common window",
                         demo, footer_left, "Performance")
            _table_pages(pdf, excess_table(df, "style"),
                         "Excess return summary, fund versus style benchmark",
                         "Excess is the geometric difference of annualised returns. "
                         "Pre-fee adds the expense ratio back. t-stat is IR times the square root of years",
                         demo, footer_left, "Excess")
            _table_pages(pdf, excess_table(df, "broad"),
                         "Excess return summary, fund versus broad regulatory benchmark",
                         "Mostly a factor tilt result rather than an implementation result",
                         demo, footer_left, "Excess")
            if (df["bm_kind"] == "peer").any():
                _table_pages(pdf, excess_table(df, "peer"),
                             "Excess return summary, fund versus passive peer fund",
                             "A different index family at a comparable fee. Where the sign here "
                             "disagrees with the style comparison, the style result was a proxy artifact",
                             demo, footer_left, "Peer")
                _robustness(pdf, df, demo, footer_left)
            dec = decomposition_table(df)
            _table_pages(pdf, dec,
                         "Where the excess return comes from",
                         "Total excess versus the broad benchmark, split into the part attributable "
                         "to the style tilt and the part remaining versus the style index",
                         demo, footer_left, "Decomposition",
                         col_width=[0.07, 0.15, 0.09, 0.09, 0.10, 0.13, 0.15, 0.09, 0.06])
            for tkr in df["ticker"].drop_duplicates():
                _fund_page(pdf, tkr, results, monthly, daily, settings, demo, footer_left)
        _exceptions(pdf, bundle["skipped"], bundle["benchmarks"], demo, footer_left)

        meta = pdf.infodict()
        meta["Title"] = "Dimensional ETFs versus stated benchmarks"
        meta["Subject"] = f"Since-inception benchmark relative review, source {settings.source}"
        meta["Keywords"] = "DFA, Dimensional, ETF, benchmark, excess return"
    return df
