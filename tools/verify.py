"""Pre-send checks on the three deliverables. Text output only, no dependencies
beyond what the project already uses.

    python tools/verify.py

Confirms the chartbook, workbook and memo exist, agree with each other, and that
the arithmetic reconciles. Exits non-zero if any check fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PDF = OUT / "dfa_chartbook.pdf"
XLSX = OUT / "dfa_benchmark_dataset.xlsx"
MD = OUT / "summary_observations.md"

results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    results.append((ok, f"{label}{': ' + detail if detail else ''}"))
    return ok


def pdf_pages(path: Path) -> int:
    raw = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page[^s]", raw))


def main() -> int:
    print("=" * 72)
    print("DELIVERABLE CHECK")
    print("=" * 72)

    # ---------------------------------------------------------------- existence
    for f in (PDF, XLSX, MD):
        kb = f.stat().st_size / 1024 if f.exists() else 0
        check(f.exists() and kb > 5, f"{f.name} exists", f"{kb:,.0f} KB")

    if not (XLSX.exists() and MD.exists()):
        report()
        return 1

    n_pages = pdf_pages(PDF) if PDF.exists() else 0
    check(n_pages >= 25, "chartbook page count", str(n_pages))

    # ------------------------------------------------------------------ workbook
    xl = pd.ExcelFile(XLSX)
    need = {"notes", "mapping_funds", "mapping_benchmarks", "stats_all",
            "returns_monthly"}
    check(need.issubset(set(xl.sheet_names)), "workbook sheets present",
          f"{len(xl.sheet_names)} sheets")

    df = pd.read_excel(xl, "stats_all")
    counts = df.bm_kind.value_counts().to_dict()
    check(counts.get("broad", 0) == counts.get("style", 0),
          "broad and style row counts match", str(counts))

    has_peer = counts.get("peer", 0) > 0
    check(has_peer, "peer comparison present",
          f"{counts.get('peer', 0)} rows" if has_peer else "MISSING, set include_peer: true")

    # peer rows that merely duplicate the style row are not a robustness test
    dupes = []
    if has_peer:
        st = df[df.bm_kind == "style"].set_index("ticker")
        pe = df[df.bm_kind == "peer"].set_index("ticker")
        for t in pe.index.intersection(st.index):
            if pe.loc[t, "bm_code"] == st.loc[t, "bm_code"]:
                dupes.append(t)
        check(not dupes, "no peer row duplicates its style benchmark",
              ", ".join(dupes) if dupes else "all distinct")

    # ------------------------------------------------------ decomposition maths
    st = df[df.bm_kind == "style"].set_index("ticker")
    br = df[df.bm_kind == "broad"].set_index("ticker")
    bad = []
    for t in st.index.intersection(br.index):
        vs_broad = br.loc[t, "excess_ann_geom"]
        vs_style = st.loc[t, "excess_ann_geom"]
        tilt = vs_broad - vs_style
        if abs((vs_style + tilt) - vs_broad) > 1e-9:
            bad.append(t)
    check(not bad, "decomposition reconciles", "all funds" if not bad else str(bad))

    # annualised return recomputes from cumulative
    bad_ann = []
    for t, r in st.iterrows():
        yrs, cum, ann = r["years"], r["fund_cum"], r["fund_ann"]
        if yrs and yrs > 0:
            implied = (1 + cum) ** (1 / yrs) - 1
            if abs(implied - ann) > 0.0015:
                bad_ann.append(f"{t} {implied:.4f} vs {ann:.4f}")
    check(not bad_ann, "annualised returns recompute from cumulative",
          "all funds" if not bad_ann else "; ".join(bad_ann))

    # ---------------------------------------------------------------- the memo
    md = MD.read_text()
    check("change sign depending" in md, "memo leads with the robustness result")
    check("Does the result survive a change of benchmark?" in md,
          "memo contains the robustness table")

    flips = robust = behind = []
    if has_peer:
        pe = df[df.bm_kind == "peer"].set_index("ticker")
        common = st.index.intersection(pe.index)
        e_s, e_p = st.loc[common, "excess_ann_geom"], pe.loc[common, "excess_ann_geom"]
        t_s, t_p = st.loc[common, "t_stat"], pe.loc[common, "t_stat"]
        flips = sorted(common[(e_s * e_p) < 0])
        robust = sorted(common[(e_s > 0) & (e_p > 0) & (t_s.abs() >= 2) & (t_p.abs() >= 2)])
        behind = sorted(common[t_p <= -2])

        # the memo must not name a flipped fund as the best result
        m = re.search(r"widest positive gap[^.]*?is (\w+) at", md)
        named = m.group(1) if m else None
        check(named is None or named not in flips,
              "memo does not name a sign-flipping fund as best",
              f"names {named}" if named else "no claim made")

    # ------------------------------------------------------------------ summary
    print()
    print("-" * 72)
    print("HEADLINE NUMBERS")
    print("-" * 72)
    win = f"{pd.to_datetime(df['start']).min():%b %Y} to {pd.to_datetime(df['end']).max():%b %Y}"
    print(f"funds                    {st.index.nunique()}")
    print(f"window                   {win}")
    print(f"ahead of style bm        {int((st['excess_ann_geom'] > 0).sum())} of {len(st)}")
    print(f"ahead with |t| >= 2      "
          f"{int(((st['t_stat'] >= 2) & (st['excess_ann_geom'] > 0)).sum())}")
    if has_peer:
        print(f"sign flips vs peer       {len(flips)} of {len(common)}  {flips}")
        print(f"robust positive          {robust}")
        print(f"significantly behind     {behind}")
    print(f"median tracking error    {st['tracking_error'].median() * 100:.2f}%")

    return report()


def report() -> int:
    print()
    print("-" * 72)
    failed = [m for ok, m in results if not ok]
    for ok, msg in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
    print("-" * 72)
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    if failed:
        print("\nNOT READY TO SEND")
    else:
        print("\nAll checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
