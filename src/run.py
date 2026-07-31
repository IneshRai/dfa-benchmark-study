"""Entry point.

    python -m src.run                          # uses config/settings.yml as is
    python -m src.run --source demo            # simulated data, plumbing test
    python -m src.run --source csv             # Bloomberg exports in data/external
    python -m src.run --tickers DFAT,DFSV      # subset
    python -m src.run --no-cache --end 2026-06-30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import chartbook, summary
from .config import ROOT, load_settings, set_style
from .data import load_prices
from .tables import build_results, write_workbook


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DFA ETF versus benchmark chartbook")
    p.add_argument("--source", choices=["yahoo", "csv", "demo"])
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--tickers", help="comma separated subset of fund tickers")
    p.add_argument("--window", choices=["since_listing", "since_inception"])
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out", help="output directory override")
    p.add_argument("--settings", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = load_settings(Path(args.settings) if args.settings else None)

    if args.source:
        settings.raw["data"]["source"] = args.source
    if args.start:
        settings.raw["data"]["start"] = args.start
    if args.end:
        settings.raw["data"]["end"] = args.end
    if args.no_cache:
        settings.raw["data"]["cache"] = False
    if args.window:
        settings.raw["analysis"]["window"] = args.window
    if args.out:
        settings.raw["output"]["dir"] = args.out

    set_style(settings.o("font"), settings.o("greyscale"))
    out_dir = settings.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"source={settings.source} window={settings.a('window')} "
          f"start={settings.start.date()} end={settings.end.date()}")

    prices = load_prices(settings)
    print(f"loaded {prices.shape[1]} series, {prices.shape[0]} rows, "
          f"{prices.index.min().date()} to {prices.index.max().date()}")

    if args.tickers:
        keep = {t.strip().upper() for t in args.tickers.split(",")}
        from .config import CONFIG_DIR
        uni = pd.read_csv(CONFIG_DIR / "universe.csv")
        uni["include"] = uni["ticker"].isin(keep).map({True: "TRUE", False: "FALSE"})
        tmp = ROOT / "config" / "_universe_subset.csv"
        uni.to_csv(tmp, index=False)
        print(f"subset to {sorted(keep)}")
        import src.config as cfg
        _orig = cfg.load_universe

        def _patched(include_only: bool = True):
            df = pd.read_csv(tmp)
            df["include"] = df["include"].astype(str).str.upper().eq("TRUE")
            return (df[df["include"]] if include_only else df).reset_index(drop=True)

        cfg.load_universe = _patched
        import src.tables as tbl
        tbl.load_universe = _patched

    bundle = build_results(prices, settings)
    if not bundle["results"]:
        print("no results produced. Check config and the exceptions list:")
        for s in bundle["skipped"]:
            print("  -", s)
        return 1

    pdf_path = out_dir / settings.o("chartbook_name")
    df = chartbook.build(pdf_path, bundle, settings)
    print(f"wrote {pdf_path}")

    xlsx_path = out_dir / settings.o("dataset_name")
    write_workbook(xlsx_path, df, bundle["monthly"], bundle["results"], settings, bundle["skipped"])
    print(f"wrote {xlsx_path}")

    md_path = out_dir / settings.o("summary_name")
    summary.write(md_path, df, settings, bundle["skipped"])
    print(f"wrote {md_path}")

    if bundle["skipped"]:
        print(f"{len(bundle['skipped'])} pairs skipped, see the data quality page")
    return 0


if __name__ == "__main__":
    sys.exit(main())
