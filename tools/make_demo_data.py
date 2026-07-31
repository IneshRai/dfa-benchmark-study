"""Write simulated series to data/external in the shape a Bloomberg export takes.

Use this to test the csv path before you have real exports, and to see exactly what
column layout the loader expects.

    python tools/make_demo_data.py
    python -m src.run --source csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_benchmarks, load_settings, load_universe  # noqa: E402
from src.data import load_demo, required_tickers  # noqa: E402


def main() -> None:
    settings = load_settings()
    tickers, bm_map = required_tickers(settings)
    px = load_demo(settings, tickers)

    uni = load_universe()
    bms = load_benchmarks()

    fund_cols = [t for t in uni["ticker"] if t in px.columns]
    bm_cols = [c for c in bms.index if c in px.columns]

    out_dir = settings.external_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    funds = px[fund_cols].copy()
    funds.index.name = "date"
    funds.round(6).to_csv(out_dir / "demo_funds_nav.csv")

    # Benchmark file uses the Bloomberg ticker as the header so the loader's ticker
    # to bm_code mapping gets exercised too.
    rename = {c: bms.loc[c, "bbg_ticker"] for c in bm_cols}
    bench = px[bm_cols].rename(columns=rename).copy()
    bench.index.name = "date"
    bench.round(6).to_csv(out_dir / "demo_benchmarks_index.csv")

    print(f"wrote {out_dir / 'demo_funds_nav.csv'} ({len(fund_cols)} series)")
    print(f"wrote {out_dir / 'demo_benchmarks_index.csv'} ({len(bm_cols)} series)")
    print("these files are simulated and are gitignored")


if __name__ == "__main__":
    main()
