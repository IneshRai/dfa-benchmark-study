# External data drop

Put wide CSVs here and run with `--source csv`. Nothing in this directory is
committed: Bloomberg and other vendor series are licensed and must not go into git.

Shape, one date column first then one column per series:

    date,DFAC,DFAT,RU30INTR Index,RU20VATR Index
    2021-06-14,100.00,100.00,2431.11,9871.42

Column names are matched against fund tickers, then `bm_code` values, then
`bbg_ticker` values from `config/benchmarks.csv`. Levels can be NAVs, index
levels or growth-of-one series; only the returns are used.

Bloomberg pull, one row per series, then paste as values and delete the header rows:

    =BDH("DFAC US Equity","TOT_RETURN_INDEX_NET_DVDS","6/14/2021","","Dir=V","Sort=A")
    =BDH("RU30INTR Index","PX_LAST","6/14/2021","","Dir=V","Sort=A")

Use `TOT_RETURN_INDEX_NET_DVDS` for funds so you get NAV based total return
rather than market price. Index tickers in `benchmarks.csv` are unverified best
guesses; confirm each one on the terminal before you trust the output.
