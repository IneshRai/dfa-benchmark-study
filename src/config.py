"""Configuration loading and house plot style."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


@dataclass
class Settings:
    raw: dict

    @property
    def source(self) -> str:
        return self.raw["data"]["source"]

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.raw["data"]["start"])

    @property
    def end(self) -> pd.Timestamp:
        end = self.raw["data"]["end"]
        return pd.Timestamp(end) if end else pd.Timestamp(dt.date.today())

    @property
    def cache_dir(self) -> Path:
        return ROOT / self.raw["data"]["cache_dir"]

    @property
    def external_dir(self) -> Path:
        return ROOT / self.raw["data"]["external_dir"]

    @property
    def out_dir(self) -> Path:
        return ROOT / self.raw["output"]["dir"]

    def a(self, key):
        return self.raw["analysis"][key]

    def o(self, key):
        return self.raw["output"][key]

    def d(self, key):
        return self.raw["data"][key]


def load_settings(path: Path | None = None) -> Settings:
    path = path or CONFIG_DIR / "settings.yml"
    with open(path) as fh:
        return Settings(yaml.safe_load(fh))


def load_universe(include_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(CONFIG_DIR / "universe.csv")
    df["include"] = df["include"].astype(str).str.upper().eq("TRUE")
    if include_only:
        df = df[df["include"]].copy()
    return df.reset_index(drop=True)


def load_benchmarks() -> pd.DataFrame:
    df = pd.read_csv(CONFIG_DIR / "benchmarks.csv")
    return df.set_index("bm_code")


def set_style(font: str = "Arial", greyscale: bool = True) -> None:
    """House style: black and white, Arial, no chart junk."""
    import matplotlib as mpl
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in (font, "Liberation Sans", "DejaVu Sans"):
        if candidate in installed:
            chosen = candidate
            break
    else:
        chosen = "sans-serif"

    mpl.rcParams.update(
        {
            "font.family": chosen,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.grid": True,
            "grid.color": "0.85",
            "grid.linewidth": 0.5,
            "axes.edgecolor": "0.2",
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "lines.linewidth": 1.1,
        }
    )
    if greyscale:
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=["0.0", "0.45", "0.7"])


# Line styles used consistently across every chart in the book.
SERIES_STYLE = {
    "fund": {"color": "0.0", "linestyle": "-", "linewidth": 1.4},
    "broad": {"color": "0.45", "linestyle": "--", "linewidth": 1.0},
    "style": {"color": "0.0", "linestyle": ":", "linewidth": 1.0},
    "peer": {"color": "0.7", "linestyle": "-.", "linewidth": 1.0},
}
