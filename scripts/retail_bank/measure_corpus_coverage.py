#!/usr/bin/env python
"""Report a corpus's use-case coverage against the declared matrix, and gate on it.

usage:
  measure_corpus_coverage.py --corpus router      [--split train] [--gate]
  measure_corpus_coverage.py --corpus alignment   [--split train] [--gate]

Without ``--gate`` this prints the category totals, the phrasing-form
distribution, the full intent × form × turn matrix, and every declared cell
below its target ranked worst first -- that ranking is the authoring order.
With ``--gate`` it also exits non-zero if any declared cell or category is
below its ``minimum``, which is what `make verify` runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hello_slm.banking_corpus_coverage import (  # noqa: E402
    CATEGORIES,
    FORMS,
    evaluate,
    load_spec,
    measure,
)

CORPORA = {
    "router": REPO_ROOT / "data/banking-conversation-router-v8-first-turn-mutation",
    "alignment": REPO_ROOT / "data/banking-servicing-alignment-v5",
}
SPEC = REPO_ROOT / "configs/corpus-coverage.toml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=tuple(CORPORA), required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="also write the matrix here")
    args = parser.parse_args()

    path = CORPORA[args.corpus] / f"{args.split}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    report = measure(rows)
    spec = load_spec(SPEC, args.corpus)
    shortfalls = evaluate(report, spec)

    print(f"--- {args.corpus} / {args.split}: {report.rows} rows\n")
    print("categories")
    for name in CATEGORIES:
        have = report.category_counts[name]
        share = f"{have / report.rows:6.1%}" if report.rows else "   n/a"
        print(f"  {name:22s} {have:6d} {share}")

    print("\nphrasing form")
    for form in FORMS:
        have = report.form_counts[form]
        print(f"  {form:14s} {have:6d} {have / report.rows:6.1%}")

    print("\nintent × form  (first_turn / multi_turn)")
    header = "  " + f"{'intent':28s}" + "".join(f"{form:>22s}" for form in FORMS)
    print(header)
    for intent in sorted(report.intents, key=lambda i: (i == "unlabelled", i)):
        cells = "".join(
            f"{report.cell(intent, form, 'first_turn'):>10d} "
            f"/{report.cell(intent, form, 'multi_turn'):<9d}"
            for form in FORMS
        )
        print(f"  {intent:28s}{cells}")

    print("\ndeclared cells below target  (blocking first; this is the authoring order)")
    if not shortfalls:
        print("  none")
    for item in shortfalls:
        flag = "BLOCK" if item.blocks else "     "
        print(
            f"  {flag} {item.what:52s} have {item.have:5d}"
            f"   min {item.minimum:4d}   target {item.target:4d}"
        )

    if args.json is not None:
        payload = {
            "corpus": args.corpus,
            "split": args.split,
            "rows": report.rows,
            "categories": dict(report.category_counts),
            "forms": dict(report.form_counts),
            "cells": {" × ".join(k): v for k, v in sorted(report.cells.items())},
            "shortfalls": [item.__dict__ for item in shortfalls],
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    blocking = [item for item in shortfalls if item.blocks]
    if args.gate and blocking:
        print(f"\ncoverage gate: {len(blocking)} declared minimum(s) not met")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
