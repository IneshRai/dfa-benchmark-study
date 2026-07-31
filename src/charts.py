"""Chart primitives. Black and white, no fills, no colour coding."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .config import SERIES_STYLE
from .metrics import drawdown_series, growth_of


def _pct(ax, axis="y", decimals=0):
    fmt = mticker.PercentFormatter(xmax=1, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def growth_chart(ax, series: dict[str, pd.Series], log: bool = True,
                 start_value: float = 10_000.0, title: str = "Growth of $10,000"):
    for key, rets in series.items():
        if rets is None or rets.empty:
            continue
        g = growth_of(rets, start_value)
        g = pd.concat([pd.Series([start_value], index=[g.index[0] - pd.offsets.MonthEnd(1)]), g])
        ax.plot(g.index, g.values, label=key, **SERIES_STYLE.get(_kind(key), {}))
    if log:
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ticks = np.geomspace(lo, hi, 6)
        mag = 10 ** np.floor(np.log10(ticks))
        ticks = np.unique(np.round(ticks / (mag / 2)) * (mag / 2))
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.legend(loc="upper left")
    return ax


def rolling_excess_chart(ax, rolls: dict[str, pd.Series], window: int = 12,
                         title: str | None = None):
    plotted = False
    for key, roll in rolls.items():
        if roll is None or roll.empty:
            continue
        ax.plot(roll.index, roll.values, label=key, **SERIES_STYLE.get(_kind(key), {}))
        plotted = True
    ax.axhline(0, color="0.2", linewidth=0.8)
    _pct(ax, decimals=0)
    ax.set_title(title or f"Rolling {window}-month excess return")
    if plotted:
        ax.legend(loc="lower left")
    else:
        ax.text(0.5, 0.5, "insufficient history", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="0.4")
    return ax


def drawdown_chart(ax, prices: dict[str, pd.Series], title: str = "Drawdown from prior peak"):
    for key, px in prices.items():
        if px is None or px.empty:
            continue
        dd = drawdown_series(px)
        ax.plot(dd.index, dd.values, label=key, **SERIES_STYLE.get(_kind(key), {}))
    ax.axhline(0, color="0.2", linewidth=0.8)
    _pct(ax, decimals=0)
    ax.set_title(title)
    ax.legend(loc="lower left")
    return ax


def relative_chart(ax, series: dict[str, pd.Series], title: str = "Cumulative relative performance"):
    """Fund divided by benchmark, rebased to 100. Rising means the fund is ahead."""
    for key, rets in series.items():
        if rets is None or rets.empty:
            continue
        rel = 100 * (1 + rets.fillna(0)).cumprod()
        rel = pd.concat([pd.Series([100.0], index=[rel.index[0] - pd.offsets.MonthEnd(1)]), rel])
        ax.plot(rel.index, rel.values, label=key, **SERIES_STYLE.get(_kind(key), {}))
    ax.axhline(100, color="0.2", linewidth=0.8)
    ax.set_title(title)
    ax.legend(loc="upper left")
    return ax


def excess_bar_chart(ax, labels: list[str], values: list[float], t_stats: list[float] | None = None,
                     title: str = "Annualised excess return vs benchmark",
                     xlabel: str = "Annualised excess return"):
    """Horizontal bars, hatched where the t-stat is below 2 in absolute value."""
    y = np.arange(len(labels))
    for i, v in enumerate(values):
        weak = t_stats is not None and (t_stats[i] != t_stats[i] or abs(t_stats[i]) < 2)
        ax.barh(y[i], v, height=0.65, color="0.35" if not weak else "white",
                edgecolor="0.15", linewidth=0.7,
                hatch="////" if weak else None)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="0.1", linewidth=0.8)
    _pct(ax, axis="x", decimals=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return ax


def table_axis(ax, df: pd.DataFrame, title: str | None = None, col_width=None,
               fontsize: float = 6.5, align_left_first: bool = True):
    """Render a DataFrame as a clean table on its own axis."""
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", pad=6)
    tbl = ax.table(
        cellText=df.astype(str).values,
        colLabels=df.columns,
        rowLabels=None,
        cellLoc="center",
        loc="upper center",
        colWidths=col_width,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.35)
    ncols = len(df.columns)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("0.75")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.92")
        elif row % 2 == 0:
            cell.set_facecolor("0.975")
        if align_left_first and col == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.03
    return ax


def _kind(label: str) -> str:
    low = label.lower()
    if "broad" in low:
        return "broad"
    if "style" in low:
        return "style"
    if "peer" in low:
        return "peer"
    return "fund"


def watermark(fig, text: str = "DEMO DATA - NOT FOR USE"):
    fig.text(0.5, 0.5, text, fontsize=34, color="0.85", ha="center", va="center",
             rotation=30, zorder=0, alpha=0.6)


def footer(fig, left: str, right: str = ""):
    fig.text(0.04, 0.015, left, fontsize=6, color="0.4", ha="left")
    if right:
        fig.text(0.96, 0.015, right, fontsize=6, color="0.4", ha="right")


def close(fig):
    plt.close(fig)
