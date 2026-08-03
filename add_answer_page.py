#!/usr/bin/env python3
"""Add a computed 'The answer' page as page 2 of the chartbook.

Run from the repo root (the folder containing src/ and config/):

    python add_answer_page.py

Every figure on the page is derived from the results frame, so the page cannot
drift out of line with the tables behind it when the data updates. Matches on
exact file content, writes nothing if an anchor is missing, and is safe to run
twice.
"""

import pathlib
import sys

ANSWER_FN = '''
def _answer(pdf, df: pd.DataFrame, demo: bool, footer_left: str):
    """One page stating the conclusion, computed from df so it cannot go stale."""
    fig = _new_page(demo, footer_left, "Answer")
    fig.text(0.06, 0.92, "The answer", fontsize=16)

    style = df[df["bm_kind"] == "style"].set_index("ticker")
    peer = df[df["bm_kind"] == "peer"].set_index("ticker")
    n = len(style)
    med_te = style["tracking_error"].median()
    med_yrs = style["years"].median()
    threshold = med_te * 2.0 / np.sqrt(med_yrs) if med_yrs > 0 else float("nan")

    both = style.index.intersection(peer.index)
    flips = [t for t in both
             if np.sign(style.loc[t, "excess_ann_geom"]) != np.sign(peer.loc[t, "excess_ann_geom"])]
    within = int((style["excess_ann_geom"].abs() < 0.01).sum())
    verb = "reverses" if len(flips) == 1 else "reverse"

    survivors = [t for t in both
                 if style.loc[t, "excess_ann_geom"] > 0 and style.loc[t, "t_stat"] >= 2
                 and peer.loc[t, "excess_ann_geom"] > 0 and peer.loc[t, "t_stat"] >= 2]
    if survivors:
        surv = "; ".join(
            f"{t} at {style.loc[t, 'excess_ann_geom']*100:+.2f}% versus "
            f"{style.loc[t, 'bm_code']} and {peer.loc[t, 'excess_ann_geom']*100:+.2f}% "
            f"versus {peer.loc[t, 'bm_code']}" for t in survivors)
        verdict = f"Positives surviving both comparators at an absolute t of 2 or better: {surv}."
    else:
        verdict = ("No fund shows a positive excess return that survives both comparators "
                   "at an absolute t of 2 or better.")

    top = style["excess_ann_geom"].idxmax()
    top_line = (f"Largest apparent value-add is {top} in {style.loc[top, 'sleeve']}, "
                f"{style.loc[top, 'excess_ann_geom']*100:+.2f}% a year against "
                f"{style.loc[top, 'bm_code']} (t = {style.loc[top, 't_stat']:.2f}).")
    if top in peer.index:
        pe = peer.loc[top, "excess_ann_geom"]
        if np.sign(pe) != np.sign(style.loc[top, "excess_ann_geom"]):
            top_line += (f" That reverses to {pe*100:+.2f}% against {peer.loc[top, 'bm_code']}, "
                         "so treat it as unmeasured rather than as a result.")
    if str(style.loc[top, "proxy_quality"]).lower() == "poor":
        top_line += (f" The {style.loc[top, 'bm_code']} proxy is flagged poor, "
                     "which is the likely explanation.")

    body = [
        ("Have these funds beaten their benchmarks?", [
            "Mostly no, and the question is harder to answer than it looks.",
            "",
            f"Across {n} funds, median tracking error against the style index is {med_te*100:.2f}% "
            f"a year. At a median {med_yrs:.1f} years of history an excess return has to clear "
            f"roughly {threshold*100:.2f}% a year before it separates from noise. {within} of {n} "
            "funds sit within 1% of their style index, which is itself the finding: these are "
            "index-like products and they track like them.",
            "",
            f"{len(flips)} of {n} results {verb} sign depending on whether the comparator is the fund's "
            "style index or a passive competitor from a different index family. A result that flips "
            "when the yardstick changes is a statement about the benchmark, not the fund, so those "
            "should be read as unestablished rather than as small positives.",
            "",
            verdict,
        ]),
        ("Where has the value-add occurred?", [
            top_line,
            "",
            "Read the style-relative column for implementation and the broad-benchmark column for "
            "factor tilt. A small value fund beating the Russell 3000 tells you value worked in "
            "this window, not that Dimensional executed well.",
        ]),
        ("What would change this answer", [
            "Benchmarks here are proxy ETFs rather than index total return series, so every "
            "comparison carries the proxy's own fee and tracking error. Rebuild on Bloomberg index "
            "series before treating any single number as final.",
        ]),
    ]

    y = 0.855
    for heading, paras in body:
        fig.text(0.06, y, heading, fontsize=11, weight="bold")
        y -= 0.032
        for para in paras:
            lines = textwrap.wrap(para, width=112) if para else [""]
            for line in lines:
                fig.text(0.06, y, line, fontsize=9, color="0.15")
                y -= 0.0235
        y -= 0.018
    pdf.savefig(fig)
    charts.close(fig)

'''

HEADER = ('"""Assembles the PDF chartbook."""\n\nfrom __future__ import annotations\n'
          '\nimport textwrap')


def main() -> int:
    path = pathlib.Path("src/chartbook.py")
    if not path.exists():
        print("ERROR: no src/chartbook.py here. cd to the repo root first.")
        return 1

    text = path.read_text()
    done = []

    # 1. module-level textwrap import. A function-local one already exists in
    #    _exceptions, so check the header rather than searching the whole file.
    if not text.startswith(HEADER):
        anchor = "import matplotlib.pyplot as plt"
        if text.count(anchor) != 1:
            print(f"ERROR: import anchor appears {text.count(anchor)} times")
            return 1
        text = text.replace(anchor, "import textwrap\n\n" + anchor, 1)
        done.append("module-level import textwrap")

    # 2. the page builder itself
    if "def _answer(pdf" not in text:
        anchor = "\ndef _methodology(pdf, settings: Settings"
        if text.count(anchor) != 1:
            print(f"ERROR: _methodology anchor appears {text.count(anchor)} times")
            return 1
        text = text.replace(anchor, ANSWER_FN + anchor, 1)
        done.append("_answer page builder")

    # 3. wire it into build() between the cover and the methodology page
    old = ("        _cover(pdf, settings, df, demo)\n"
           "        _methodology(pdf, settings, demo, footer_left)")
    new = ("        _cover(pdf, settings, df, demo)\n"
           "        if not df.empty:\n"
           "            _answer(pdf, df, demo, footer_left)\n"
           "        _methodology(pdf, settings, demo, footer_left)")
    if new not in text:
        if text.count(old) != 1:
            print(f"ERROR: build() anchor appears {text.count(old)} times")
            return 1
        text = text.replace(old, new, 1)
        done.append("call _answer as page 2")

    if not done:
        print("Already applied, nothing to do.")
        return 0

    path.write_text(text)
    print("wrote src/chartbook.py")
    for d in done:
        print("  applied:", d)
    print("\nNext: python -m src.run --source yahoo --out /tmp/dfa_out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
