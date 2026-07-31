# dfa-benchmark-study

Chartbook and supporting dataset comparing Dimensional (DFA) ETFs to their stated
benchmarks since inception. Built to answer one question: have these funds delivered
performance meaningfully different from their benchmarks, and if so, where.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.run --source demo     # simulated data, proves the plumbing, watermarked
python -m src.run --source yahoo    # real data, proxy ETF benchmarks
python -m src.run --source csv      # real data, Bloomberg index series (preferred)

python -m tests.test_metrics        # 10 checks on the statistics
```

Outputs land in `output/`:

| File | What it is |
|---|---|
| `dfa_chartbook.pdf` | The chartbook. Cover, methodology, summary bars, three summary tables, a decomposition table, one page per fund, data quality page. |
| `dfa_benchmark_dataset.xlsx` | Supporting dataset: benchmark mapping, full stat block, monthly returns, rolling excess panel, exceptions. |
| `summary_observations.md` | Written observations with the numbers filled in from the run. |

## The one methodological decision that matters

Each fund is compared against **two** benchmarks, because they answer different questions.

**Broad regulatory benchmark.** Since the SEC's tailored shareholder report rules,
Dimensional reports US equity funds against the Russell 3000 and non-US funds against
an MSCI aggregate, regardless of the fund's actual tilt. DFAT, a small-cap value fund,
carries the Russell 3000 as its primary benchmark and the Russell 2000 Value as a
secondary. Excess return against the broad index is therefore mostly a **factor tilt**
result: it tells you whether small value paid off in the window, not whether Dimensional
executed well.

**Style benchmark.** The index that matches the fund's intended exposure. Excess return
here is the closer read on **implementation**: security selection inside the style,
patient trading, securities lending revenue, and fees.

The decomposition table splits total excess against the broad benchmark into these two
pieces. If you only report one number, report the style-relative one, and say which
benchmark it is against.

## Statistical conventions

| Quantity | Definition |
|---|---|
| Annualised return | Geometric, from monthly returns |
| Excess return | Geometric difference of annualised returns, fund minus benchmark |
| Tracking error | Annualised standard deviation of the monthly return difference |
| Information ratio | Annualised mean monthly excess divided by tracking error |
| t-statistic | IR times the square root of years. Equivalent to a t-test on the monthly differences |
| Rolling window t-stat | Newey-West with 11 lags, because 12-month windows overlap |
| Alpha and beta | Single-index regression of fund on benchmark, both over the 13-week T-bill |

**Read the t-stats before the point estimates.** Most of these funds have three to five
years of history. With a 3% tracking error and four years of data, an excess return has
to be around 3% a year before it separates from noise. Bars on the summary pages are
hatched where the absolute t-statistic is below 2, which is most of them. That is the
honest answer to the brief, not a defect in the analysis.

## Data sources

**`--source yahoo`** — keyless, no account, but two compromises. Index total return
series are not free, so each benchmark is represented by a proxy ETF, which carries its
own expense ratio and tracking error and flatters the Dimensional fund by roughly the
proxy's fee. And Yahoo's adjusted closes are market-price based, not NAV based. Fine for
iteration, not for anything client facing. The `proxy_quality` column in
`config/benchmarks.csv` flags where the proxy is only approximate; the data quality page
in the chartbook reproduces it.

**`--source csv`** — the Bloomberg path. Drop wide CSVs into `data/external/`; see the
README in that directory for the layout and the `=BDH` formulas. Use
`TOT_RETURN_INDEX_NET_DVDS` for the funds so you get NAV-based total return, and real
index total return series for the benchmarks. **The Bloomberg tickers in
`config/benchmarks.csv` are unverified best guesses** — `bbg_verified` is FALSE on every
row. Confirm each one on the terminal and flip the flag as you go.

`data/external/` is gitignored. Bloomberg data is licensed and must not be committed.

Run `python tools/make_demo_data.py` to see the expected CSV layout with dummy numbers.

## Since inception means two different things here

Seven of these funds are converted mutual funds:

| ETF | Listed | Predecessor mutual fund |
|---|---|---|
| DFUS, DFAC, DFAS, DFAT | 14 Jun 2021 | Tax-managed US portfolios |
| DFIV, DFAX | 13 Sep 2021 | Tax-managed international portfolios |
| DFUV | 9 May 2022 | Tax-managed US marketwide value |

Dimensional's published since-inception figures splice the predecessor mutual fund NAV
history, which for several funds starts in the late 1990s. Yahoo's ticker history starts
at ETF listing. This book uses **listing date** by default (`window: since_listing`), so
it will not tie to the fact sheets. To run the longer history, pull the predecessor NAV
series from Bloomberg using the `predecessor_ticker` column in `config/universe.csv`,
splice it into the fund column of your CSV, and set `window: since_inception`.

Predecessor tickers marked in the config are best guesses for DFAC and DFUV. Verify.

## Configuration

Everything is driven by three files, no code changes needed for routine updates.

- **`config/universe.csv`** — one row per fund. `broad_bm`, `style_bm` and `peer_bm` are
  benchmark codes. Set `include` to FALSE to drop a fund. `list_date` of `auto` means the
  window is derived from the first date with data.
- **`config/benchmarks.csv`** — one row per benchmark code, with its Yahoo proxy, proxy
  quality rating, and Bloomberg ticker.
- **`config/settings.yml`** — source, dates, rolling window, Newey-West lags, output names.

Expense ratios in `universe.csv` are current as of July 2026. Dimensional has cut them
repeatedly since these funds listed, so the current figure understates the fee actually
charged over most of the window. The `excess_before_fees` column adds the current ratio
back and is therefore a lower bound on gross implementation.

## Layout

```
config/          universe, benchmark mapping, run settings
src/config.py    settings loading, house plot style
src/data.py      three interchangeable data sources
src/metrics.py   all statistics, including hand-rolled OLS and Newey-West
src/charts.py    chart primitives, black and white
src/tables.py    analysis engine, table shaping, workbook export
src/chartbook.py PDF assembly
src/summary.py   written observations, numbers filled in from the run
src/run.py       CLI
tests/           10 checks on the statistics
tools/           demo data generator
```

No API keys anywhere. Nothing in this repo needs a credential.

## Extending it

- Add a fund: one row in `universe.csv`.
- Add a benchmark: one row in `benchmarks.csv`, then reference the code.
- Third comparison against a passive peer ETF: set `include_peer: true`. The `peer_bm`
  column is already populated.
- Fixed income funds: excluded on purpose. Duration and credit make the excess return
  decomposition mean something different, and the drawdown comparison is not informative
  the same way. Add them as a separate book rather than more rows here.
