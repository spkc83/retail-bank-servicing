from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

ROUTER_SPLITS = ("train", "validation", "test")

# CLINC intents that overlap the retail-banking capabilities represented in
# Banking77. They strengthen the binary domain head but do not supervise the
# 77-way intent head because their label ontology is different.
CLINC_SUPPORTED_BANKING_LABELS = frozenset(
    {
        "account_blocked",
        "apr",
        "balance",
        "card_declined",
        "credit_limit",
        "credit_limit_change",
        "damaged_card",
        "direct_deposit",
        "exchange_rate",
        "expiration_date",
        "freeze_account",
        "international_fees",
        "interest_rate",
        "min_payment",
        "new_card",
        "order_checks",
        "pay_bill",
        "pin_change",
        "replacement_card_duration",
        "report_fraud",
        "report_lost_card",
        "rewards_balance",
        "routing",
        "spending_history",
        "transactions",
        "transfer",
    }
)

# Customer-service conversation is part of the supported application domain
# even when a turn does not express one of the 77 Banking77 intents. These rows
# supervise only the binary domain head.
CLINC_CONVERSATIONAL_IN_DOMAIN_LABELS = frozenset(
    {
        "are_you_a_bot",
        "goodbye",
        "greeting",
        "thank_you",
    }
)

PII_PATTERNS = (
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]?){12,}\b"),
    re.compile(r"\b(?:\+?1[ -.]?)?(?:\(?\d{3}\)?[ -.]?)\d{3}[ -.]\d{4}\b"),
)


def render_router_input(current: str, *, previous_user: str | None = None) -> str:
    rendered = f"[CURRENT]\n{current.strip()}"
    if previous_user and previous_user.strip():
        rendered += f"\n[PREVIOUS_USER]\n{previous_user.strip()}"
    return rendered


def normalize_router_text(text: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in text).split()
    )


def build_router_splits(
    banking_rows: Sequence[dict[str, Any]],
    clinc_payload: dict[str, list[list[str]]],
    *,
    validation_fraction: float = 0.15,
    seed: int = 7101,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")

    intents = sorted({str(row["label"]) for row in banking_rows})
    if not intents:
        raise ValueError("banking_rows must contain intent labels")
    intent_to_index = {intent: index for index, intent in enumerate(intents)}
    split_banking = _split_banking77(
        banking_rows,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    splits: dict[str, list[dict[str, Any]]] = {split: [] for split in ROUTER_SPLITS}

    for split, rows in split_banking.items():
        for row in rows:
            intent = str(row["label"])
            current = str(row["text"]).strip()
            splits[split].append(
                _example(
                    current=current,
                    previous_user=None,
                    domain_label=1,
                    intent_label=intent_to_index[intent],
                    intent=intent,
                    example_kind="banking77_single",
                    source="PolyAI/banking77",
                    source_split=str(row["split"]),
                    source_label=intent,
                    license_name="CC-BY-4.0",
                )
            )

    clinc_split_keys = {
        "train": ("train", "oos_train"),
        "validation": ("val", "oos_val"),
        "test": ("test", "oos_test"),
    }
    for split, source_keys in clinc_split_keys.items():
        for source_key in source_keys:
            for pair in clinc_payload.get(source_key, []):
                if len(pair) != 2:
                    raise ValueError(f"invalid CLINC row in {source_key}: {pair!r}")
                current, source_label = (str(pair[0]).strip(), str(pair[1]).strip())
                supported_banking = source_label in CLINC_SUPPORTED_BANKING_LABELS
                supported_conversation = (
                    source_label in CLINC_CONVERSATIONAL_IN_DOMAIN_LABELS
                )
                supported = supported_banking or supported_conversation
                if supported_banking:
                    example_kind = "clinc_supported_banking"
                elif supported_conversation:
                    example_kind = "clinc_conversational_in_domain"
                elif source_label == "oos":
                    example_kind = "clinc_oos"
                else:
                    example_kind = "clinc_nonbanking"
                splits[split].append(
                    _example(
                        current=current,
                        previous_user=None,
                        domain_label=1 if supported else 0,
                        intent_label=-100,
                        intent=None,
                        example_kind=example_kind,
                        source="UCI/clinc150",
                        source_split=source_key,
                        source_label=source_label,
                        license_name="CC-BY-4.0",
                    )
                )

    for split in ROUTER_SPLITS:
        splits[split].extend(_same_intent_transitions(splits[split]))
        splits[split].extend(_banking_to_ood_transitions(splits[split], seed=seed))

    deduplicated, duplicates_removed = _deduplicate_across_splits(splits)
    report = {
        "intent_count": len(intents),
        "intent_labels": intents,
        "banking77_test_rows": len(split_banking["test"]),
        "cross_split_duplicates_removed": duplicates_removed,
        "pii_matches": _count_pii_matches(
            str(row["text"]) for rows in deduplicated.values() for row in rows
        ),
        "split_counts": {split: len(rows) for split, rows in deduplicated.items()},
        "kind_counts": {
            split: _counts_by(rows, "example_kind") for split, rows in deduplicated.items()
        },
        "domain_counts": {
            split: _counts_by(rows, "domain_label") for split, rows in deduplicated.items()
        },
    }
    return deduplicated, report


def _split_banking77(
    rows: Sequence[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in ROUTER_SPLITS}
    train_by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_split = str(row.get("split", "")).lower()
        if source_split == "test":
            split_rows["test"].append(dict(row))
        elif source_split == "train":
            train_by_intent[str(row["label"])].append(dict(row))
        else:
            raise ValueError(f"unsupported Banking77 source split: {source_split!r}")

    for intent, intent_rows in sorted(train_by_intent.items()):
        ranked = sorted(
            intent_rows,
            key=lambda row: _stable_rank(
                seed,
                intent,
                str(row.get("source_row_id", "")),
                str(row["text"]),
            ),
        )
        validation_count = max(1, round(len(ranked) * validation_fraction))
        split_rows["validation"].extend(ranked[:validation_count])
        split_rows["train"].extend(ranked[validation_count:])

    for split in ROUTER_SPLITS:
        split_rows[split].sort(
            key=lambda row: (
                str(row["label"]),
                int(row.get("source_row_id", 0)),
                str(row["text"]),
            )
        )
    return split_rows


def _same_intent_transitions(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["example_kind"] == "banking77_single":
            by_intent[str(row["intent"])].append(row)

    transitions: list[dict[str, Any]] = []
    for intent, intent_rows in sorted(by_intent.items()):
        if len(intent_rows) < 2:
            continue
        for index, current in enumerate(intent_rows):
            previous = intent_rows[index - 1]
            transitions.append(
                _example(
                    current=str(current["current_text"]),
                    previous_user=str(previous["current_text"]),
                    domain_label=1,
                    intent_label=int(current["intent_label"]),
                    intent=intent,
                    example_kind="same_intent_followup",
                    source="PolyAI/banking77",
                    source_split=str(current["source_split"]),
                    source_label=intent,
                    license_name="CC-BY-4.0",
                )
            )
    return transitions


def _banking_to_ood_transitions(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    banking = [row for row in rows if row["example_kind"] == "banking77_single"]
    ood = [
        row
        for row in rows
        if row["domain_label"] == 0
        and row["example_kind"] in {"clinc_nonbanking", "clinc_oos"}
    ]
    if not banking:
        return []

    ordered_banking = sorted(
        banking,
        key=lambda row: _stable_rank(seed, str(row["text"])),
    )
    transitions = []
    for index, current in enumerate(ood):
        previous = ordered_banking[index % len(ordered_banking)]
        transitions.append(
            _example(
                current=str(current["current_text"]),
                previous_user=str(previous["current_text"]),
                domain_label=0,
                intent_label=-100,
                intent=None,
                example_kind="banking_to_ood_transition",
                source=str(current["source"]),
                source_split=str(current["source_split"]),
                source_label=str(current["source_label"]),
                license_name=str(current["license"]),
            )
        )
    return transitions


def _example(
    *,
    current: str,
    previous_user: str | None,
    domain_label: int,
    intent_label: int,
    intent: str | None,
    example_kind: str,
    source: str,
    source_split: str,
    source_label: str,
    license_name: str,
) -> dict[str, Any]:
    return {
        "text": render_router_input(current, previous_user=previous_user),
        "current_text": current,
        "previous_user": previous_user,
        "domain_label": domain_label,
        "intent_label": intent_label,
        "intent": intent,
        "example_kind": example_kind,
        "source": source,
        "source_split": source_split,
        "source_label": source_label,
        "license": license_name,
    }


def _deduplicate_across_splits(
    splits: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    seen_in_higher_priority_split: set[str] = set()
    duplicates_removed = 0
    kept: dict[str, list[dict[str, Any]]] = {split: [] for split in ROUTER_SPLITS}
    for split in ("test", "validation", "train"):
        current_split_values: set[str] = set()
        for row in splits[split]:
            normalized = normalize_router_text(str(row["text"]))
            if normalized in seen_in_higher_priority_split:
                duplicates_removed += 1
                continue
            kept[split].append(row)
            current_split_values.add(normalized)
        seen_in_higher_priority_split.update(current_split_values)
    return kept, duplicates_removed


def _count_pii_matches(texts: Iterable[str]) -> int:
    return sum(bool(pattern.search(text)) for text in texts for pattern in PII_PATTERNS)


def _counts_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def _stable_rank(seed: int, *parts: str) -> str:
    value = "\0".join((str(seed), *parts)).encode()
    return hashlib.sha256(value).hexdigest()
