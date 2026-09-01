#!/usr/bin/env python
"""Report how much of an eval split is a paraphrase of its training split.

Two metrics, because the obvious one is misleading on its own.

*4-gram overlap* is what caught the alignment corpus, where evaluation rows
were the training questions with a different trailing phrase. But raw 4-gram
overlap over-reports badly: every split shares the realizer's style directives
("please keep the answer concise"), and a banking corpus shares its own
vocabulary ("my card ending in", "freeze the active card"). Counting those as
contamination would argue for training a model to avoid ordinary English and
its own domain's nouns. So grams are filtered to the *identifying* ones -- a
gram that occurs in a single eval scenario family -- and known instruction
boilerplate is discounted.

*Nearest-neighbour similarity* is the metric that actually distinguishes the
two situations. A test question that differs from a training question only in
a trailing phrase scores above 0.95; one that shares a domain vocabulary but
asks something else scores around 0.75. Judge a corpus on the shape of this
distribution, not on the gram count.

usage:
  measure_split_contamination.py [--corpus alignment|base] [--split test|validation]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hello_slm.banking_tool_sft_data import (  # noqa: E402
    current_user_text,
    generate_records,
    normalized_user_text,
)

#: Style directives the realizer appends to every split. Shared phrasing here
#: carries no task content and a model gains nothing from having seen it.
BOILERPLATE = (
    "please keep the answer concise",
    "use the information from this conversation",
    "check the signed in profile rather than asking for a private id",
    "i am referring to the result from the previous turn",
    "i am going through my accounts",
    "i am continuing from the details above",
    "i am confirming the details before continuing",
    "tell me what you find and what happens next",
)


def core(text: str) -> str:
    normalized = normalized_user_text(text)
    for phrase in BOILERPLATE:
        normalized = normalized.replace(phrase, " ")
    return " ".join(normalized.split())


def grams(text: str, size: int = 4) -> set[tuple[str, ...]]:
    words = text.split()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def family(record: Any) -> str:
    return str(record.get("metadata", {}).get("scenario_family") or record.get("record_id"))


def report(train: list[Any], evaluation: list[Any], *, label: str) -> None:
    owners: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for record in evaluation:
        for gram in grams(core(current_user_text(record))):
            owners[gram].add(family(record))
    identifying = {gram for gram, families in owners.items() if len(families) == 1}

    echoed = sum(
        1 for record in train if grams(core(current_user_text(record))) & identifying
    )
    train_cores = [core(current_user_text(record)) for record in train]
    similarities = []
    for record in evaluation:
        question = core(current_user_text(record))
        best = 0.0
        for candidate in train_cores:
            matcher = SequenceMatcher(a=question, b=candidate, autojunk=False)
            if matcher.real_quick_ratio() <= best or matcher.quick_ratio() <= best:
                continue
            best = max(best, matcher.ratio())
        similarities.append(best)

    print(f"--- {label}: {len(evaluation)} eval rows against {len(train)} train rows")
    median = statistics.median(similarities)
    print(f"    train rows echoing an identifying eval 4-gram : {echoed}")
    print(f"    nearest-train similarity, median              : {median:.3f}")
    for threshold in (0.95, 0.90, 0.80):
        near = sum(value >= threshold for value in similarities)
        print(f"    eval rows with a >= {threshold:.2f} near-duplicate       : {near}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("base", "alignment"), default="base")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    args = parser.parse_args()

    if args.corpus == "base":
        records = generate_records(pilot_count=1200)
        train = [r for r in records if r["metadata"]["split"] == "train"]
        evaluation = [r for r in records if r["metadata"]["split"] == args.split]
    else:
        from hello_slm.banking_servicing_alignment_data import (
            build_servicing_alignment_splits,
        )

        splits, _ = build_servicing_alignment_splits()
        train, evaluation = splits["train"], splits[args.split]

    report(train, evaluation, label=f"{args.corpus} / {args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
