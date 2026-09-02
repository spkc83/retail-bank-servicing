#!/usr/bin/env python
"""Score the learned router and the SLM as classifiers on the same held-out turns.

The question is *who should decide the turn*, so only the classifier varies:
both arms are scored against the same gold labels from the router's own test
split, and in serving both feed the identical harness (see
``poc/.../model_router.py``). Nothing here measures the harness.

The gold labels come from
``data/banking-conversation-router-v8-first-turn-mutation/test.jsonl``, the
split the shipped router was evaluated on -- 4,921 rows carrying
``action_name``, ``intent``, ``entity_resolution_name`` and ``domain_name``
alongside the turn text and its history.

Reported per arm:

* per-head accuracy, and **exact-tuple** accuracy, which is what the harness
  actually consumes -- three right heads and one wrong still routes wrongly;
* seconds per turn, because a DistilBERT pass and a 9B pass are not the same
  cost and a demo that ignores that is not an honest comparison;
* for the model arm, how often it proposed something the joint decoder would
  never emit (``illegal_tuple_rate``) and how often it returned nothing usable
  (``unparsable_rate``). Those are the model's failures, not the harness's, and
  they stay visible rather than being repaired into the score.

usage:
  compare_routing_classifiers.py --arm router [--limit N]
  compare_routing_classifiers.py --arm model  [--limit N]   # needs the 9B
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
POC = REPO_ROOT / "poc/retail-bank-customer-service-poc"
TEST_SPLIT = REPO_ROOT / "data/banking-conversation-router-v8-first-turn-mutation/test.jsonl"
ROUTER_ARTIFACT = REPO_ROOT / "artifacts/banking-conversation-router-v8-first-turn-mutation"

sys.path.insert(0, str(POC))

#: gold field -> how the harness reads that head off a decision.
#: Intent is resolved exactly as ``_generation_plan`` resolves it, because what
#: matters is the value the harness acts on. Reading only "fine_intent" scored
#: the learned router at 0.0 on 347 labelled rows: it reports the intent under
#: "intent" and leaves "fine_intent" unset unless the V6 refinement fires.
HEADS = {
    "domain_name": lambda d: d.get("domain"),
    "intent": lambda d: d.get("fine_intent") or d.get("intent"),
    "action_name": lambda d: d.get("action"),
    "entity_resolution_name": lambda d: d.get("entity_resolution"),
}


def load_rows(limit: int | None, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in TEST_SPLIT.open(encoding="utf-8")]
    if limit is not None and limit < len(rows):
        # Sampled, not truncated: the split is grouped by trajectory, so the
        # first N rows are not a cross-section of it.
        rows = random.Random(seed).sample(rows, limit)
    return rows


def history_of(row: dict[str, Any]) -> list[dict[str, str]]:
    history = row.get("history") or []
    return [
        {"role": str(turn.get("role", "")), "content": str(turn.get("content", ""))}
        for turn in history
        if isinstance(turn, dict) and turn.get("role") in {"user", "assistant"}
    ]


def build_router():
    from router import LearnedBankingRouter  # type: ignore[import-not-found]

    return LearnedBankingRouter.from_artifact_dir(ROUTER_ARTIFACT)


def build_model_router():
    from local_gpu_runtime import LocalGraniteRuntime  # type: ignore[import-not-found]
    from model_router import ModelConversationRouter  # type: ignore[import-not-found]

    return ModelConversationRouter(LocalGraniteRuntime.load(), revision="local")


def score(arm: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    classifier = build_router() if arm == "router" else build_model_router()
    correct: Counter[str] = Counter()
    labelled: Counter[str] = Counter()
    exact = 0
    exact_graded = 0
    illegal = 0
    unparsable = 0
    elapsed = 0.0

    for row in rows:
        text = str(row.get("current_text") or row.get("text") or "")
        if not text.strip():
            continue
        started = time.perf_counter()
        try:
            decision = classifier.classify(text, history_of(row))
        except Exception:  # noqa: BLE001 - a crash is a wrong answer, not a stop
            decision = {}
        elapsed += time.perf_counter() - started

        if "action" not in decision:
            unparsable += 1
        if decision.get("constraint_diagnostics"):
            illegal += 1

        # Only 2,136 of the 4,921 rows carry a gold intent; the rest are masked
        # with -100 / None because no intent applies. Scoring those as misses
        # measured the label distribution, not the classifier -- it put the
        # router's intent accuracy at 0.558 when the labelled subset is what
        # the head was ever trained or gated on. Each head is scored over the
        # rows that actually carry it.
        hits = 0
        graded = 0
        for gold_field, read in HEADS.items():
            gold = row.get(gold_field)
            if gold is None:
                continue
            graded += 1
            labelled[gold_field] += 1
            if read(decision) == gold:
                correct[gold_field] += 1
                hits += 1
        if graded:
            exact += hits == graded
            exact_graded += 1

    total = len(rows)
    result = {
        "arm": arm,
        "rows": total,
        "exact_tuple_accuracy": round(exact / exact_graded, 4) if exact_graded else 0.0,
        "exact_tuple_rows": exact_graded,
        "seconds_per_turn": round(elapsed / total, 4) if total else 0.0,
        "unparsable_rate": round(unparsable / total, 4) if total else 0.0,
    }
    # The router cannot emit an illegal tuple -- its joint decoder enumerates the
    # legal ones, and its constraint notes are ordinary operation rather than a
    # rejected proposal. Reporting the rate for both arms compared two different
    # things under one name.
    if arm == "model":
        result["illegal_tuple_rate"] = round(illegal / total, 4) if total else 0.0
    for gold_field in HEADS:
        seen = labelled[gold_field]
        result[f"{gold_field}_accuracy"] = round(correct[gold_field] / seen, 4) if seen else None
        result[f"{gold_field}_rows"] = seen
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("router", "model"), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=711)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load_rows(args.limit, args.seed)
    print(f"scoring {args.arm} on {len(rows)} held-out turns", flush=True)
    result = score(args.arm, rows)
    result["seed"] = args.seed
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
