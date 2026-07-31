"""Price data loading.

Three interchangeable sources, all returning the same object: a wide DataFrame of
daily total-return index levels indexed by date, one column per ticker or bm_code.

  yahoo  yfinance adjusted closes. Free, keyless, but market-price based and it
         only covers ETFs, so index benchmarks come in as proxy ETFs.
  csv    wide CSVs dropped in data/external. This is the Bloomberg path: NAV based
         fund returns and real index total return series. Preferred for anything
         that leaves the desk.
  demo   simulated. Plumbing test only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings, load_benchmarks, load_universe


def _cache_path(settings: Settings, name: str) -> Path:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings.cache_dir / name


def required_tickers(settings: Settings) -> tuple[list[str], dict[str, str]]:
    """Fund tickers plus the proxy ticker needed for each benchmark code in use."""
    uni = load_universe()
    bms = load_benchmarks()

    codes: set[str] = set()
    for col in ("broad_bm", "style_bm"):
        codes.update(uni[col].dropna().unique())
    if settings.a("include_peer"):
        peers = uni["peer_bm"].dropna().unique().tolist()
    else:
        peers = []

    bm_map: dict[str, str] = {}
    for code in sorted(codes):
        if code not in bms.index:
            raise KeyError(f"benchmark code {code} in universe.csv is not in benchmarks.csv")
        proxy = bms.loc[code, "proxy_ticker"]
        if isinstance(proxy, str) and proxy.strip():
            bm_map[code] = proxy.strip()

    tickers = sorted(set(uni["ticker"]) | set(bm_map.values()) | set(peers))
    return tickers, bm_map


# --------------------------------------------------------------------------- yahoo


def load_yahoo(settings: Settings, tickers: list[str]) -> pd.DataFrame:
    cache = _cache_path(settings, "prices_yahoo.csv")
    if settings.d("cache") and cache.exists():
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in cached.columns]
        fresh_enough = cached.index.max() >= settings.end - pd.Timedelta(days=5)
        if not missing and fresh_enough:
            print(f"using cached prices through {cached.index.max().date()}")
            return cached

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install yfinance, or switch data.source to csv") from exc

    to_pull = sorted(set(tickers) | {settings.d("rf_ticker")})
    print(f"downloading {len(to_pull)} tickers from Yahoo")
    raw = yf.download(
        to_pull,
        start=settings.start,
        end=settings.end,
        auto_adjust=True,   # adjusted close = total return, dividends reinvested
        progress=False,
        actions=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw[["Close"]].rename(columns={"Close": to_pull[0]})

    px.index = pd.to_datetime(px.index).tz_localize(None)
    px = px.sort_index()
    px = px.dropna(how="all")

    empty = [c for c in px.columns if px[c].notna().sum() == 0]
    if empty:
        print(f"WARNING no data returned for: {', '.join(empty)}")
        px = px.drop(columns=empty)

    if settings.d("cache"):
        px.to_csv(cache)
    return px


# ----------------------------------------------------------------------------- csv


def load_csv(settings: Settings) -> pd.DataFrame:
    """Read every CSV in data/external and merge on date.

    Expected shape, which is what Bloomberg gives you out of a BDH pull once you
    strip the header rows:

        date,DFAC,DFAT,RU30INTR,RU20VATR
        2021-06-14,100.00,100.00,2431.11,9871.42

    Column names are matched against, in order: fund ticker, bm_code, bbg_ticker.
    Levels can be NAVs, index levels, or growth-of-1 series. Only returns matter.
    """
    # Files starting with an underscore are templates and examples, not data.
    files = [p for p in sorted(settings.external_dir.glob("*.csv"))
             if not p.name.startswith("_")]
    if not files:
        raise FileNotFoundError(
            f"no CSVs found in {settings.external_dir}. Export the series from Bloomberg "
            "or switch data.source to yahoo."
        )

    frames = []
    for path in files:
        df = pd.read_csv(path)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.loc[~df.index.duplicated(keep="last")]
        frames.append(df)
        print(f"read {path.name}: {df.shape[1]} series, {df.shape[0]} rows")

    px = pd.concat(frames, axis=1)
    px.index.name = "date"

    # A series supplied in more than one file: keep whichever version has the most
    # observations rather than whichever file happened to sort first.
    dupes = px.columns[px.columns.duplicated()].unique()
    if len(dupes):
        print(f"WARNING duplicate series across files, keeping the fullest history: "
              f"{', '.join(map(str, dupes))}")
        keep = {}
        for col in px.columns:
            block = px.loc[:, [col]] if (px.columns == col).sum() == 1 else px.loc[:, col]
            if isinstance(block, pd.DataFrame) and block.shape[1] > 1:
                best = block.notna().sum().idxmax()
                keep[col] = block.iloc[:, list(block.columns).index(best)] \
                    if isinstance(best, str) else block.iloc[:, int(block.notna().sum().argmax())]
            else:
                keep[col] = block.iloc[:, 0] if isinstance(block, pd.DataFrame) else block
        px = pd.DataFrame(keep)

    # Map Bloomberg tickers back onto bm_codes so the rest of the pipeline is agnostic.
    bms = load_benchmarks()
    rename = {}
    for code, row in bms.iterrows():
        bbg = str(row["bbg_ticker"])
        for candidate in (bbg, bbg.replace(" Index", ""), bbg.split()[0]):
            if candidate in px.columns and code not in px.columns:
                rename[candidate] = code
                break
    if rename:
        px = px.rename(columns=rename)
        print(f"mapped Bloomberg tickers to benchmark codes: {rename}")
    return px


# ---------------------------------------------------------------------------- demo


def load_demo(settings: Settings, tickers: list[str], seed: int = 11) -> pd.DataFrame:
    """Simulated series so the pipeline can be exercised without market data.

    Calibrated to plausible vols and correlations, with a deliberately small and
    mostly negative implementation drag on the funds. The numbers mean nothing.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(settings.start, settings.end)
    n = len(dates)

    # one common market factor plus a small-cap and a value factor
    mkt = rng.normal(0.00035, 0.0105, n)
    smb = rng.normal(0.00002, 0.0060, n)
    hml = rng.normal(0.00004, 0.0065, n)
    fx = rng.normal(0.0, 0.0045, n)

    loadings = {
        "IWV": (1.00, 0.05, 0.00, 0.0), "IWB": (0.99, -0.02, 0.00, 0.0),
        "IWD": (0.92, 0.02, 0.55, 0.0), "IWM": (1.02, 0.85, 0.10, 0.0),
        "IWN": (0.98, 0.85, 0.60, 0.0), "IJR": (1.00, 0.82, 0.15, 0.0),
        "ITOT": (1.00, 0.04, 0.00, 0.0), "VTV": (0.90, 0.00, 0.52, 0.0),
        "QUAL": (0.97, -0.10, -0.10, 0.0), "AVUV": (1.00, 0.90, 0.70, 0.0),
        "IDEV": (0.85, 0.05, 0.15, 1.0), "EFV": (0.83, 0.05, 0.60, 1.0),
        "SCZ": (0.88, 0.70, 0.25, 1.0), "DLS": (0.86, 0.72, 0.45, 1.0),
        "VXUS": (0.86, 0.10, 0.18, 1.0), "ACWI": (0.95, 0.03, 0.05, 0.4),
        "IEMG": (0.90, 0.20, 0.25, 1.2), "FNDE": (0.88, 0.15, 0.55, 1.2),
    }
    uni = load_universe()
    bm_codes = load_benchmarks()
    _, bm_map = required_tickers(settings)

    out = {}
    for tkr, (bm, s, h, f) in loadings.items():
        r = bm * mkt + s * smb + h * hml + f * fx + rng.normal(0, 0.0025, n)
        out[tkr] = r

    for _, row in uni.iterrows():
        proxy = bm_map.get(row["style_bm"]) or bm_map.get(row["broad_bm"]) or "IWV"
        base = out.get(proxy, out["IWV"])
        # small idiosyncratic implementation wedge, drawn once per fund
        wedge = rng.normal(0.0, 0.00008)
        r = base + wedge + rng.normal(0, 0.0020, n) - float(row["expense_ratio"]) / 252
        start = rng.integers(0, max(1, int(n * 0.35)))
        r = r.copy()
        r[:start] = np.nan
        out[row["ticker"]] = r

    for code in bm_codes.index:
        proxy = bm_map.get(code)
        if proxy and proxy in out and code not in out:
            out[code] = out[proxy]

    rets = pd.DataFrame(out, index=dates)
    px = (1 + rets.fillna(0)).cumprod() * 100
    px = px.where(rets.notna())
    px[settings.d("rf_ticker")] = 4.5
    return px


# --------------------------------------------------------------------------- entry


def load_prices(settings: Settings) -> pd.DataFrame:
    tickers, bm_map = required_tickers(settings)
    src = settings.source
    if src == "yahoo":
        px = load_yahoo(settings, tickers)
    elif src == "csv":
        px = load_csv(settings)
    elif src == "demo":
        px = load_demo(settings, tickers)
    else:
        raise ValueError(f"unknown data.source {src}")

    # Alias proxy ETF columns onto benchmark codes when the source did not supply
    # a real index series. Keeps downstream code source agnostic.
    for code, proxy in bm_map.items():
        if code not in px.columns and proxy in px.columns:
            px[code] = px[proxy]

    px = px.loc[(px.index >= settings.start) & (px.index <= settings.end)]
    return px.sort_index()


def daily_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return prices.pct_change()


def monthly_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Month-end compounded returns. Last partial month is dropped."""
    me = prices.resample("ME").last()
    rets = me.pct_change()
    if len(rets) and prices.index.max() < prices.index.max().to_period("M").end_time.normalize():
        rets = rets.iloc[:-1]
    return rets.dropna(how="all")


def risk_free_monthly(prices: pd.DataFrame, rf_ticker: str) -> pd.Series:
    """^IRX is an annualised discount rate in percent. Convert to a monthly rate."""
    if rf_ticker not in prices.columns:
        return pd.Series(dtype=float)
    ann = prices[rf_ticker].resample("ME").last() / 100.0
    return (1 + ann.clip(lower=0)) ** (1 / 12) - 1
