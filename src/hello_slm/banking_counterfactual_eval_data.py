from __future__ import annotations

import ast
import hashlib
import heapq
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from hello_slm.banking_tool_sft_data import (
    BANKING_TOOL_SFT_CONTRACT,
    SYSTEM_PROMPT,
    normalized_user_text,
    public_tool_manifest,
    validate_records,
)
from hello_slm.config import canonical_json_bytes, file_sha256

COUNTERFACTUAL_MANIFEST_CONTRACT = "banking-counterfactual-eval-manifest/v1"
COUNTERFACTUAL_GENERATOR_VERSION = "banking-counterfactual-eval/v1"
COUNTERFACTUAL_ALLOWED_USE = "counterfactual-evaluation"
DEFAULT_OUTPUT_DIR = Path("data/banking-counterfactual-eval-v1")
DEFAULT_STAGE1_TRAIN = Path("data/banking-v3-tool-sft/train.jsonl")
DEFAULT_STAGE2_TRAIN = Path("data/banking-servicing-alignment-v4/train.jsonl")
DEFAULT_POC_APP = Path("poc/retail-bank-customer-service-poc/app.py")
DEFAULT_POC_BANK = Path("poc/retail-bank-customer-service-poc/synthetic_bank.json")
MAX_TRAINING_PROMPT_SIMILARITY = 0.90
COUNTERFACTUAL_GATE_CONTRACT = "banking-counterfactual-eval-gate/v1"

_PERFECT_COUNTERFACTUAL_METRICS = (
    "tool_name_accuracy",
    "tool_argument_accuracy",
    "multi_tool_exact_sequence",
    "clarification_appropriateness",
    "grounded_final_factuality",
    "no_tool_faq_quality",
    "ood_small_talk_response_path",
)
_ZERO_COUNTERFACTUAL_METRICS = (
    "malformed_tool_call_rate",
    "unsupported_private_arguments",
    "credential_request_rate",
    "in_domain_false_refusal",
    "ood_false_accept",
)


class CounterfactualEvalDataError(ValueError):
    """Raised when the clean benchmark contract or contamination gate fails."""


@dataclass(frozen=True)
class PairVariant:
    suffix: str
    result: dict[str, Any]
    final: str
    grounding_facts: tuple[str, ...]
    varied_facts: tuple[str, ...]


@dataclass(frozen=True)
class ReadPair:
    pair_id: str
    scenario_family: str
    prompt: str
    tool_name: str
    arguments: dict[str, Any]
    variants: tuple[PairVariant, PairVariant]


def build_counterfactual_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair in _read_pairs():
        records.extend(_records_for_read_pair(pair))
    records.extend(_action_records())
    records.extend(_coverage_records())
    validate_counterfactual_records(records)
    return records


def write_counterfactual_benchmark(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    stage1_train: Path = DEFAULT_STAGE1_TRAIN,
    stage2_train: Path = DEFAULT_STAGE2_TRAIN,
    poc_app: Path = DEFAULT_POC_APP,
    poc_bank: Path = DEFAULT_POC_BANK,
) -> dict[str, Any]:
    records = build_counterfactual_records()
    audit = audit_counterfactual_records(
        records,
        stage1_train=stage1_train,
        stage2_train=stage2_train,
        poc_app=poc_app,
        poc_bank=poc_bank,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = output_dir / "test.jsonl"
    test_path.write_bytes(_jsonl_bytes(records))
    report = {
        "contract": "banking-counterfactual-eval-preparation-report/v1",
        "generator_version": COUNTERFACTUAL_GENERATOR_VERSION,
        "record_count": len(records),
        "counterfactual_pair_count": len(
            {
                str(record["metadata"]["counterfactual_pair_id"])
                for record in records
                if record["metadata"].get("counterfactual_pair_id")
            }
        ),
        "scenario_family_counts": _counts(
            str(record["metadata"]["scenario_family"]) for record in records
        ),
        "tool_counts": _counts(
            str(call["name"]) for record in records for call in record["expected"]["tool_calls"]
        ),
        "audit": audit,
    }
    manifest = {
        "format_version": 1,
        "name": "retail-bank-counterfactual-eval-v1",
        "contract": COUNTERFACTUAL_MANIFEST_CONTRACT,
        "schema_version": BANKING_TOOL_SFT_CONTRACT,
        "generator_version": COUNTERFACTUAL_GENERATOR_VERSION,
        "training_allowed": False,
        "allowed_use": [COUNTERFACTUAL_ALLOWED_USE],
        "splits": {
            "test": {
                "path": "test.jsonl",
                "record_count": len(records),
                "bytes": test_path.stat().st_size,
                "sha256": file_sha256(test_path),
                "included": True,
                "allowed_use": [COUNTERFACTUAL_ALLOWED_USE],
            }
        },
        "source_inputs": audit["source_inputs"],
        "report": report,
    }
    _write_json(output_dir / "preparation-report.json", report)
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(_render_readme(manifest), encoding="utf-8")
    validate_counterfactual_manifest(output_dir / "manifest.json")
    return manifest


def validate_counterfactual_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("contract") != COUNTERFACTUAL_MANIFEST_CONTRACT:
        raise CounterfactualEvalDataError("not a counterfactual evaluation manifest")
    if manifest.get("schema_version") != BANKING_TOOL_SFT_CONTRACT:
        raise CounterfactualEvalDataError("counterfactual schema version mismatch")
    if manifest.get("training_allowed") is not False:
        raise CounterfactualEvalDataError("counterfactual manifest must forbid training")
    if manifest.get("allowed_use") != [COUNTERFACTUAL_ALLOWED_USE]:
        raise CounterfactualEvalDataError("counterfactual manifest has invalid allowed_use")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"test"}:
        raise CounterfactualEvalDataError("counterfactual manifest may declare only the test split")
    entry = splits["test"]
    if not isinstance(entry, Mapping):
        raise CounterfactualEvalDataError("counterfactual test split must be an object")
    if entry.get("allowed_use") != [COUNTERFACTUAL_ALLOWED_USE]:
        raise CounterfactualEvalDataError("counterfactual test split has invalid allowed_use")
    declared = Path(str(entry.get("path", "")))
    if not declared.name or declared.is_absolute() or ".." in declared.parts:
        raise CounterfactualEvalDataError("counterfactual test path must be manifest-relative")
    test_path = path.parent / declared
    if not test_path.is_file():
        raise CounterfactualEvalDataError("counterfactual test split is unavailable")
    if file_sha256(test_path) != str(entry.get("sha256")):
        raise CounterfactualEvalDataError("counterfactual test split digest mismatch")
    records = _read_jsonl(test_path)
    if len(records) != int(entry.get("record_count", -1)):
        raise CounterfactualEvalDataError("counterfactual test record_count mismatch")
    if test_path.stat().st_size != int(entry.get("bytes", -1)):
        raise CounterfactualEvalDataError("counterfactual test byte count mismatch")
    validate_counterfactual_records(records)
    return manifest


def validate_counterfactual_records(
    records: Sequence[dict[str, Any]],
    *,
    stage1_train: Path = DEFAULT_STAGE1_TRAIN,
    stage2_train: Path = DEFAULT_STAGE2_TRAIN,
    poc_app: Path = DEFAULT_POC_APP,
    poc_bank: Path = DEFAULT_POC_BANK,
) -> None:
    if not records:
        raise CounterfactualEvalDataError("counterfactual benchmark is empty")
    seen_ids: set[str] = set()
    for record in records:
        record_id = str(record.get("record_id", ""))
        if not record_id or record_id in seen_ids:
            raise CounterfactualEvalDataError(f"duplicate counterfactual record: {record_id}")
        seen_ids.add(record_id)
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("trainable") is not False:
            raise CounterfactualEvalDataError(f"{record_id} must be non-trainable")
        if metadata.get("split") != "test":
            raise CounterfactualEvalDataError(f"{record_id} must belong to test")
        if record.get("provenance", {}).get("generator_version") != (
            COUNTERFACTUAL_GENERATOR_VERSION
        ):
            raise CounterfactualEvalDataError(f"{record_id} generator version mismatch")
        # Reuse the production record validator one row at a time. Deliberate
        # counterfactual pairs may share a current prompt, which training data may not.
        validate_records([record])
    audit_counterfactual_records(
        records,
        stage1_train=stage1_train,
        stage2_train=stage2_train,
        poc_app=poc_app,
        poc_bank=poc_bank,
    )
    _validate_pair_invariants(records)


def counterfactual_gate_failures(
    report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return failures for the small, evaluation-only counterfactual suite."""
    metrics = report.get("metrics")
    record_reports = report.get("records")
    if not isinstance(metrics, Mapping):
        return ["counterfactual report is missing metrics"]
    if not isinstance(record_reports, Mapping):
        return ["counterfactual report is missing record results"]
    failures: list[str] = []
    if report.get("record_count") != len(records):
        failures.append(f"record_count={report.get('record_count')!r} must equal {len(records)}")
    for name in _PERFECT_COUNTERFACTUAL_METRICS:
        failures.extend(_counterfactual_metric_failure(metrics, name, expected=1.0))
    for name in _ZERO_COUNTERFACTUAL_METRICS:
        failures.extend(_counterfactual_metric_failure(metrics, name, expected=0.0))
    for record in records:
        pair_id = record.get("metadata", {}).get("counterfactual_pair_id")
        if not pair_id:
            continue
        record_id = str(record["record_id"])
        result = record_reports.get(record_id)
        if not isinstance(result, Mapping):
            failures.append(f"counterfactual pair record is missing: {record_id}")
            continue
        if result.get("tool_argument_accuracy") is not True:
            failures.append(f"counterfactual tool selection failed: {record_id}")
        if result.get("grounded_final_factuality") is not True:
            failures.append(f"counterfactual grounding failed: {record_id}")
    return failures


def _counterfactual_metric_failure(
    metrics: Mapping[str, Any],
    name: str,
    *,
    expected: float,
) -> list[str]:
    metric = metrics.get(name)
    if not isinstance(metric, Mapping):
        return [f"missing counterfactual metric: {name}"]
    denominator = metric.get("denominator")
    score = metric.get("score")
    if not isinstance(denominator, int) or denominator < 1:
        return [f"counterfactual metric has no evaluated rows: {name}"]
    if not isinstance(score, int | float) or float(score) != expected:
        return [f"{name}={score!r} must equal {expected:.1f}"]
    return []


def audit_counterfactual_records(
    records: Sequence[dict[str, Any]],
    *,
    stage1_train: Path = DEFAULT_STAGE1_TRAIN,
    stage2_train: Path = DEFAULT_STAGE2_TRAIN,
    poc_app: Path = DEFAULT_POC_APP,
    poc_bank: Path = DEFAULT_POC_BANK,
) -> dict[str, Any]:
    input_paths = (stage1_train, stage2_train, poc_app, poc_bank)
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise CounterfactualEvalDataError(f"contamination inputs are unavailable: {missing}")
    training_rows = [*_read_jsonl(stage1_train), *_read_jsonl(stage2_train)]
    training_user_texts = {
        normalized_user_text(text)
        for record in training_rows
        for text in _role_texts(record, "user")
    }
    training_current_texts = sorted(
        {normalized_user_text(_last_user_text(record)) for record in training_rows}
    )
    training_final_texts = {
        normalized_user_text(_final_assistant_text(record)) for record in training_rows
    }
    training_templates = {
        str(record.get("split_keys", {}).get("template_id", "")) for record in training_rows
    }
    benchmark_users = {
        normalized_user_text(text) for record in records for text in _role_texts(record, "user")
    }
    benchmark_finals = {normalized_user_text(_final_assistant_text(record)) for record in records}
    benchmark_templates = {str(record["split_keys"]["template_id"]) for record in records}
    user_overlap = sorted(benchmark_users & training_user_texts)
    if user_overlap:
        raise CounterfactualEvalDataError(f"training user-text overlap: {user_overlap[:3]}")
    final_overlap = sorted(benchmark_finals & training_final_texts)
    if final_overlap:
        raise CounterfactualEvalDataError(f"training final-target overlap: {final_overlap[:3]}")
    template_overlap = sorted(benchmark_templates & training_templates)
    if template_overlap:
        raise CounterfactualEvalDataError(f"training template overlap: {template_overlap[:3]}")
    poc_texts = _python_literal_texts(poc_app)
    poc_user_overlap = sorted(benchmark_users & poc_texts)
    if poc_user_overlap:
        raise CounterfactualEvalDataError(f"POC prompt overlap: {poc_user_overlap[:3]}")
    poc_facts = _poc_fact_values(json.loads(poc_bank.read_text(encoding="utf-8")))
    varied_facts = {
        str(fact) for record in records for fact in record["metadata"].get("varied_facts", [])
    }
    poc_fact_overlap = sorted(
        fact for fact in varied_facts if any(form in poc_facts for form in _fact_search_forms(fact))
    )
    if poc_fact_overlap:
        raise CounterfactualEvalDataError(f"POC fact overlap: {poc_fact_overlap[:3]}")
    training_blob = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) for record in training_rows
    ).casefold()
    training_fact_overlap = sorted(
        fact
        for fact in varied_facts
        if any(form in training_blob for form in _fact_search_forms(fact))
    )
    if training_fact_overlap:
        raise CounterfactualEvalDataError(f"training fact overlap: {training_fact_overlap[:3]}")
    similarity_rows: list[dict[str, str | float]] = []
    indexed_prompts = [
        (candidate, _character_ngrams(candidate)) for candidate in training_current_texts
    ]
    for current in sorted({_last_user_text(record) for record in records}):
        normalized = normalized_user_text(current)
        best = _nearest_prompt(normalized, indexed_prompts)
        score = SequenceMatcher(None, normalized, best, autojunk=False).ratio()
        similarity_rows.append(
            {"benchmark": current, "nearest_training": best, "similarity": round(score, 6)}
        )
    maximum_similarity = max(float(row["similarity"]) for row in similarity_rows)
    if maximum_similarity >= MAX_TRAINING_PROMPT_SIMILARITY:
        offender = max(similarity_rows, key=lambda row: float(row["similarity"]))
        raise CounterfactualEvalDataError(
            "training prompt similarity exceeds threshold: "
            f"{offender['similarity']} for {offender['benchmark']!r}"
        )
    return {
        "status": "pass",
        "source_inputs": {
            str(path): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in input_paths
        },
        "exact_training_user_overlaps": 0,
        "exact_training_final_overlaps": 0,
        "training_template_overlaps": 0,
        "poc_prompt_overlaps": 0,
        "poc_fact_overlaps": 0,
        "training_fact_overlaps": 0,
        "maximum_training_prompt_similarity": maximum_similarity,
        "maximum_allowed_training_prompt_similarity": MAX_TRAINING_PROMPT_SIMILARITY,
        "nearest_training_prompts": similarity_rows,
    }


def _read_pairs() -> tuple[ReadPair, ...]:
    return (
        ReadPair(
            pair_id="accounts-returned-facts",
            scenario_family="counterfactual_accounts",
            prompt=(
                "Inventory the deposit accounts returned for this signed-in profile, "
                "including each display name, ending digits, and available balance."
            ),
            tool_name="list_accounts",
            arguments={},
            variants=(
                PairVariant(
                    suffix="a",
                    result={
                        "accounts": [
                            {
                                "name": "Quill Observatory Checking",
                                "last4": "1014",
                                "type": "checking",
                                "currency": "USD",
                                "available_balance_cents": 843127,
                                "current_balance_cents": 845500,
                                "status": "active",
                            }
                        ]
                    },
                    final=(
                        "Quill Observatory Checking ending in 1014 has USD 8,431.27 "
                        "available and USD 8,455.00 current."
                    ),
                    grounding_facts=(
                        "account.last4=1014",
                        "account.name=Quill Observatory Checking",
                        "account.balance=8,431.27",
                    ),
                    varied_facts=(
                        "1014",
                        "Quill Observatory Checking",
                        "8,431.27",
                        "8,455.00",
                    ),
                ),
                PairVariant(
                    suffix="b",
                    result={
                        "accounts": [
                            {
                                "name": "Mosaic Trail Checking",
                                "last4": "1119",
                                "type": "checking",
                                "currency": "USD",
                                "available_balance_cents": 294618,
                                "current_balance_cents": 299118,
                                "status": "active",
                            }
                        ]
                    },
                    final=(
                        "Mosaic Trail Checking ending in 1119 has USD 2,946.18 "
                        "available and USD 2,991.18 current."
                    ),
                    grounding_facts=(
                        "account.last4=1119",
                        "account.name=Mosaic Trail Checking",
                        "account.balance=2,946.18",
                    ),
                    varied_facts=(
                        "1119",
                        "Mosaic Trail Checking",
                        "2,946.18",
                        "2,991.18",
                    ),
                ),
            ),
        ),
        ReadPair(
            pair_id="cards-returned-facts",
            scenario_family="counterfactual_cards",
            prompt=(
                "Describe every payment card in the returned profile and identify its "
                "current status and ending digits."
            ),
            tool_name="list_cards",
            arguments={},
            variants=(
                PairVariant(
                    suffix="a",
                    result={
                        "cards": [
                            {
                                "name": "Orchid Transit Debit",
                                "last4": "1126",
                                "status": "active",
                                "network": "Visa",
                            }
                        ]
                    },
                    final="Orchid Transit Debit ending in 1126 is active on the Visa network.",
                    grounding_facts=(
                        "card.last4=1126",
                        "card.name=Orchid Transit Debit",
                        "card.status=active",
                    ),
                    varied_facts=("1126", "Orchid Transit Debit"),
                ),
                PairVariant(
                    suffix="b",
                    result={
                        "cards": [
                            {
                                "name": "Copper Meridian Debit",
                                "last4": "1156",
                                "status": "frozen",
                                "network": "Mastercard",
                            }
                        ]
                    },
                    final=(
                        "Copper Meridian Debit ending in 1156 is frozen on the Mastercard network."
                    ),
                    grounding_facts=(
                        "card.last4=1156",
                        "card.name=Copper Meridian Debit",
                        "card.status=frozen",
                    ),
                    varied_facts=("1156", "Copper Meridian Debit"),
                ),
            ),
        ),
        ReadPair(
            pair_id="transactions-returned-facts",
            scenario_family="counterfactual_transactions",
            prompt=(
                "Read back the two newest ledger entries returned by the bank, including "
                "the merchant labels and amounts."
            ),
            tool_name="list_transactions",
            arguments={"limit": 2},
            variants=(
                PairVariant(
                    suffix="a",
                    result={
                        "transactions": [
                            {
                                "description": "Juniper Observatory Shop",
                                "amount_cents": -7122,
                                "currency": "USD",
                                "status": "posted",
                            },
                            {
                                "description": "Polar Archive Membership",
                                "amount_cents": -7771,
                                "currency": "USD",
                                "status": "posted",
                            },
                        ]
                    },
                    final=(
                        "The newest entries are Juniper Observatory Shop for USD 71.22 "
                        "and Polar Archive Membership for USD 77.71; both are posted."
                    ),
                    grounding_facts=(
                        "transaction.description=Juniper Observatory Shop",
                        "transaction.description=Polar Archive Membership",
                        "transaction.amount=71.22",
                        "transaction.amount=77.71",
                    ),
                    varied_facts=(
                        "Juniper Observatory Shop",
                        "Polar Archive Membership",
                        "71.22",
                        "77.71",
                    ),
                ),
                PairVariant(
                    suffix="b",
                    result={
                        "transactions": [
                            {
                                "description": "Cobalt Lantern Books",
                                "amount_cents": -2682,
                                "currency": "USD",
                                "status": "posted",
                            },
                            {
                                "description": "Glass Prairie Cinema",
                                "amount_cents": -7206,
                                "currency": "USD",
                                "status": "posted",
                            },
                        ]
                    },
                    final=(
                        "The newest entries are Cobalt Lantern Books for USD 26.82 and "
                        "Glass Prairie Cinema for USD 72.06; both are posted."
                    ),
                    grounding_facts=(
                        "transaction.description=Cobalt Lantern Books",
                        "transaction.description=Glass Prairie Cinema",
                        "transaction.amount=26.82",
                        "transaction.amount=72.06",
                    ),
                    varied_facts=(
                        "Cobalt Lantern Books",
                        "Glass Prairie Cinema",
                        "26.82",
                        "72.06",
                    ),
                ),
            ),
        ),
        ReadPair(
            pair_id="transfers-returned-facts",
            scenario_family="counterfactual_transfers",
            prompt=(
                "Summarize the transfer queue returned for this profile, naming each "
                "recipient, amount, and completion state."
            ),
            tool_name="list_transfers",
            arguments={},
            variants=(
                PairVariant(
                    suffix="a",
                    result={
                        "transfers": [
                            {
                                "recipient": "Aster Field Studio",
                                "amount_cents": 61247,
                                "currency": "USD",
                                "status": "pending",
                            }
                        ]
                    },
                    final=("Aster Field Studio has one pending transfer for USD 612.47."),
                    grounding_facts=(
                        "transfer.recipient=Aster Field Studio",
                        "transfer.amount=612.47",
                        "transfer.status=pending",
                    ),
                    varied_facts=("Aster Field Studio", "612.47"),
                ),
                PairVariant(
                    suffix="b",
                    result={
                        "transfers": [
                            {
                                "recipient": "Signal Harbor Repairs",
                                "amount_cents": 38000,
                                "currency": "USD",
                                "status": "completed",
                            }
                        ]
                    },
                    final=("Signal Harbor Repairs has one completed transfer for USD 380.00."),
                    grounding_facts=(
                        "transfer.recipient=Signal Harbor Repairs",
                        "transfer.amount=380.00",
                        "transfer.status=completed",
                    ),
                    varied_facts=("Signal Harbor Repairs", "380.00"),
                ),
            ),
        ),
        ReadPair(
            pair_id="cases-returned-facts",
            scenario_family="counterfactual_service_cases",
            prompt=(
                "Summarize the returned service-case list and state precisely when each case began."
            ),
            tool_name="list_service_cases",
            arguments={},
            variants=(
                PairVariant(
                    suffix="a",
                    result={
                        "service_cases": [
                            {
                                "subject": "Replace damaged travel card",
                                "case_type": "card_damage",
                                "status": "open",
                                "created_at": "2026-08-11T09:17:00Z",
                            }
                        ]
                    },
                    final=(
                        "Replace damaged travel card is an open card damage case created "
                        "on August 11, 2026 at 09:17 UTC."
                    ),
                    grounding_facts=(
                        "case.subject=Replace damaged travel card",
                        "case.case_type=card_damage",
                        "case.status=open",
                        "case.created_at=2026-08-11T09:17:00Z",
                    ),
                    varied_facts=(
                        "Replace damaged travel card",
                        "card_damage",
                        "2026-08-11T09:17:00Z",
                    ),
                ),
                PairVariant(
                    suffix="b",
                    result={
                        "service_cases": [
                            {
                                "subject": "Trace delayed cash deposit",
                                "case_type": "deposit_trace",
                                "status": "closed",
                                "created_at": "2026-09-03T16:42:00Z",
                            }
                        ]
                    },
                    final=(
                        "Trace delayed cash deposit is a closed deposit trace case created "
                        "on September 3, 2026 at 16:42 UTC."
                    ),
                    grounding_facts=(
                        "case.subject=Trace delayed cash deposit",
                        "case.case_type=deposit_trace",
                        "case.status=closed",
                        "case.created_at=2026-09-03T16:42:00Z",
                    ),
                    varied_facts=(
                        "Trace delayed cash deposit",
                        "deposit_trace",
                        "2026-09-03T16:42:00Z",
                    ),
                ),
            ),
        ),
    )


def _records_for_read_pair(pair: ReadPair) -> list[dict[str, Any]]:
    records = []
    for variant, counterpart in (
        (pair.variants[0], pair.variants[1]),
        (pair.variants[1], pair.variants[0]),
    ):
        record_id = f"cf_{pair.pair_id}_{variant.suffix}"
        records.append(
            _tool_record(
                record_id=record_id,
                scenario_family=pair.scenario_family,
                user=pair.prompt,
                tool_plan=((pair.tool_name, pair.arguments, variant.result),),
                final=variant.final,
                grounding_facts=variant.grounding_facts,
                forbidden_facts=counterpart.varied_facts,
                varied_facts=variant.varied_facts,
                pair_id=pair.pair_id,
                variant=variant.suffix,
            )
        )
    return records


def _action_records() -> list[dict[str, Any]]:
    return [
        _tool_record(
            record_id="cf_freeze_indigo_card",
            scenario_family="counterfactual_card_action",
            user=("Freeze the Indigo Harbor Debit card whose displayed ending digits are 1445."),
            tool_plan=(
                (
                    "freeze_card",
                    {"last4": "1445"},
                    {
                        "card": {
                            "name": "Indigo Harbor Debit",
                            "last4": "1445",
                            "status": "frozen",
                        },
                        "simulated": True,
                    },
                ),
            ),
            final="Indigo Harbor Debit ending in 1445 is now frozen.",
            grounding_facts=("card.last4=1445", "card.status=frozen"),
            forbidden_facts=("4821", "Everyday Visa Debit"),
            varied_facts=("1445", "Indigo Harbor Debit"),
        ),
        _tool_record(
            record_id="cf_replace_context_orchid",
            scenario_family="counterfactual_card_anaphora",
            user="Replace the usable card identified in the prior returned card list.",
            tool_plan=(
                (
                    "replace_card",
                    {"last4": "1400"},
                    {
                        "card": {
                            "name": "Violet Compass Debit",
                            "last4": "1400",
                            "status": "replacement_pending",
                        },
                        "simulated": True,
                    },
                ),
            ),
            final=("Replacement is pending for Violet Compass Debit ending in 1400."),
            grounding_facts=("card.last4=1400", "card.status=replacement_pending"),
            forbidden_facts=("4821", "Everyday Visa Debit"),
            varied_facts=("1400", "Violet Compass Debit"),
            pre_messages=_card_context("Violet Compass Debit", "1400"),
        ),
        _tool_record(
            record_id="cf_cancel_aster_transfer",
            scenario_family="counterfactual_transfer_action",
            user=(
                "The pending payment addressed to Ember Cartography Lab should be cancelled now."
            ),
            tool_plan=(
                (
                    "cancel_transfer",
                    {"recipient": "Ember Cartography Lab"},
                    {
                        "transfer": {
                            "recipient": "Ember Cartography Lab",
                            "amount_cents": 44731,
                            "currency": "USD",
                            "status": "cancelled",
                        },
                        "simulated": True,
                    },
                ),
            ),
            final=("The USD 447.31 transfer to Ember Cartography Lab is now cancelled."),
            grounding_facts=(
                "transfer.recipient=Ember Cartography Lab",
                "transfer.status=cancelled",
            ),
            forbidden_facts=("River Consulting",),
            varied_facts=("Ember Cartography Lab", "447.31"),
        ),
        _tool_record(
            record_id="cf_dispute_glacier_purchase",
            scenario_family="counterfactual_transaction_action",
            user=(
                "I reject the posted charge labeled Glacier Loom Workshop; open a "
                "dispute for that merchant."
            ),
            tool_plan=(
                (
                    "dispute_transaction",
                    {"description": "Glacier Loom Workshop"},
                    {
                        "transaction": {
                            "description": "Glacier Loom Workshop",
                            "amount_cents": -10234,
                            "currency": "USD",
                            "status": "posted",
                            "disputed": True,
                        },
                        "simulated": True,
                    },
                ),
            ),
            final="A dispute is open for the Glacier Loom Workshop charge of USD 102.34.",
            grounding_facts=(
                "transaction.description=Glacier Loom Workshop",
                "transaction.disputed=true",
            ),
            forbidden_facts=("North Harbor Market",),
            varied_facts=("Glacier Loom Workshop", "102.34"),
        ),
    ]


def _coverage_records() -> list[dict[str, Any]]:
    return [
        _tool_record(
            record_id="cf_multi_accounts_then_transfers",
            scenario_family="counterfactual_multi_tool",
            user=(
                "First inventory the returned deposit accounts; afterward summarize the "
                "returned transfer queue."
            ),
            tool_plan=(
                (
                    "list_accounts",
                    {},
                    {
                        "accounts": [
                            {
                                "name": "Silver Atlas Checking",
                                "last4": "1417",
                                "available_balance_cents": 671255,
                                "current_balance_cents": 671255,
                                "currency": "USD",
                                "status": "active",
                            }
                        ]
                    },
                ),
                (
                    "list_transfers",
                    {},
                    {
                        "transfers": [
                            {
                                "recipient": "Lumen Prairie Design",
                                "amount_cents": 23318,
                                "currency": "USD",
                                "status": "pending",
                            }
                        ]
                    },
                ),
            ),
            final=(
                "Silver Atlas Checking ending in 1417 has USD 6,712.55 available. "
                "Lumen Prairie Design has a pending transfer for USD 233.18."
            ),
            grounding_facts=(
                "account.last4=1417",
                "transfer.recipient=Lumen Prairie Design",
                "transfer.status=pending",
            ),
            forbidden_facts=("1792", "River Consulting"),
            varied_facts=("1417", "Silver Atlas Checking", "Lumen Prairie Design", "233.18"),
            multi_tool=True,
        ),
        _no_tool_record(
            record_id="cf_clarify_unspecified_card",
            scenario_family="counterfactual_clarification",
            user=(
                "A payment card needs attention, but I have not identified which card "
                "the request concerns."
            ),
            final=(
                "Which payment card should I use? Share only the last four digits shown "
                "in the application."
            ),
            path="clarification",
            grounding_facts=("missing_field=last4",),
        ),
        _no_tool_record(
            record_id="cf_faq_variable_savings_rate",
            scenario_family="counterfactual_banking_faq",
            user=(
                "In plain language, what does a variable savings yield imply about "
                "future interest earnings?"
            ),
            final=(
                "A variable savings interest rate may rise or fall, so future earnings "
                "can change even when the deposited balance stays constant."
            ),
            path="no_tool_banking_faq",
            grounding_facts=("faq=variable savings interest rate",),
        ),
        _no_tool_record(
            record_id="cf_ood_telescope_optics",
            scenario_family="counterfactual_ood",
            user=("Compare wide-field telescope eyepieces for observing a dim nebula."),
            final=(
                "I cannot compare astronomy equipment; this assistant is limited to "
                "retail banking service requests."
            ),
            path="ood",
            grounding_facts=("domain=out_of_domain",),
            path_markers=("retail banking",),
        ),
    ]


def _tool_record(
    *,
    record_id: str,
    scenario_family: str,
    user: str,
    tool_plan: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    final: str,
    grounding_facts: Sequence[str],
    forbidden_facts: Sequence[str],
    varied_facts: Sequence[str],
    pair_id: str | None = None,
    variant: str | None = None,
    pre_messages: Sequence[Mapping[str, Any]] = (),
    multi_tool: bool = False,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [_message("system", SYSTEM_PROMPT, loss=False)]
    messages.extend(json.loads(json.dumps(list(pre_messages))))
    messages.append(_message("user", user, loss=False))
    ordered_calls = []
    expected_calls = []
    for index, (tool_name, arguments, result) in enumerate(tool_plan):
        call_id = f"call_{record_id}_{index}"
        ordered_calls.append(call_id)
        expected_calls.append({"name": tool_name, "arguments": dict(arguments)})
        messages.append(_tool_call(call_id, tool_name, arguments, loss=True))
        messages.append(_tool_result(call_id, tool_name, result))
    messages.append(_message("assistant", final, loss=True))
    expected = {
        "requires_tool": True,
        "ordered_calls": ordered_calls,
        "tool_calls": expected_calls,
        "final_state_hash": f"sha256:{_sha256_text(record_id + '|counterfactual')}",
        "grounding_facts": list(grounding_facts),
        "forbidden_facts": list(forbidden_facts),
        "path": "tool_success",
    }
    if multi_tool:
        expected["multi_tool"] = True
    return _record_envelope(
        record_id=record_id,
        scenario_family=scenario_family,
        messages=messages,
        expected=expected,
        varied_facts=varied_facts,
        pair_id=pair_id,
        variant=variant,
    )


def _no_tool_record(
    *,
    record_id: str,
    scenario_family: str,
    user: str,
    final: str,
    path: str,
    grounding_facts: Sequence[str],
    path_markers: Sequence[str] = (),
) -> dict[str, Any]:
    expected = {
        "requires_tool": False,
        "ordered_calls": [],
        "tool_calls": [],
        "final_state_hash": None,
        "grounding_facts": list(grounding_facts),
        "forbidden_facts": [],
        "path": path,
    }
    if path_markers:
        expected["path_markers"] = list(path_markers)
    return _record_envelope(
        record_id=record_id,
        scenario_family=scenario_family,
        messages=[
            _message("system", SYSTEM_PROMPT, loss=False),
            _message("user", user, loss=False),
            _message("assistant", final, loss=True),
        ],
        expected=expected,
        varied_facts=(),
    )


def _record_envelope(
    *,
    record_id: str,
    scenario_family: str,
    messages: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Any],
    varied_facts: Sequence[str],
    pair_id: str | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "record_type": "counterfactual_tool_evaluation",
        "trainable": False,
        "scenario_family": scenario_family,
        "path": expected["path"],
        "split": "test",
        "split_group": f"counterfactual|{record_id}",
        "varied_facts": list(varied_facts),
    }
    if pair_id is not None:
        metadata["counterfactual_pair_id"] = pair_id
        metadata["counterfactual_variant"] = variant
    return {
        "schema_version": BANKING_TOOL_SFT_CONTRACT,
        "record_id": record_id,
        "messages": [dict(message) for message in messages],
        "expected": dict(expected),
        "split_keys": {
            "scenario_family": scenario_family,
            "state_seed": f"counterfactual-state-{record_id}",
            "customer_id": f"counterfactual-customer-{record_id}",
            "template_id": f"counterfactual-v1-{scenario_family}",
            "realization_seed": f"counterfactual-realization-{record_id}",
        },
        "provenance": {
            "source": "self-authored-synthetic",
            "license": "MIT",
            "generator_version": COUNTERFACTUAL_GENERATOR_VERSION,
            "teacher_model": None,
            "teacher_prompt_hash": None,
        },
        "validation": {
            "tool_manifest_hash": _tool_manifest_hash(),
            "replay_hash": None,
            "replay_verified": False,
            "final_state_verified": False,
            "schema_accepted": True,
            "accepted": False,
            "training_allowed": False,
        },
        "metadata": metadata,
    }


def _card_context(name: str, last4: str) -> list[dict[str, Any]]:
    call_id = "context_cf_replace_context_orchid_0"
    return [
        _message(
            "user",
            "Before taking an action, inspect the card list returned for this profile.",
            loss=False,
        ),
        _tool_call(call_id, "list_cards", {}, loss=False),
        _tool_result(
            call_id,
            "list_cards",
            {"cards": [{"name": name, "last4": last4, "status": "active"}]},
        ),
        _message(
            "assistant",
            f"The returned list contains one active card: {name} ending in {last4}.",
            loss=False,
        ),
    ]


def _message(role: str, content: str | None, *, loss: bool) -> dict[str, Any]:
    return {"role": role, "content": content, "loss": loss}


def _tool_call(
    call_id: str,
    name: str,
    arguments: Mapping[str, Any],
    *,
    loss: bool,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "loss": loss,
        "tool_calls": [
            {
                "id": call_id,
                "index": 0,
                "type": "function",
                "function": {"name": name, "arguments": dict(arguments)},
            }
        ],
    }


def _tool_result(
    call_id: str,
    name: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": {"ok": True, "result": dict(result)},
        "loss": False,
    }


def _validate_pair_invariants(records: Sequence[Mapping[str, Any]]) -> None:
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        pair_id = record.get("metadata", {}).get("counterfactual_pair_id")
        if pair_id:
            pairs[str(pair_id)].append(record)
    for pair_id, variants in pairs.items():
        if len(variants) != 2:
            raise CounterfactualEvalDataError(
                f"counterfactual pair {pair_id} must contain exactly two variants"
            )
        if {str(record["metadata"].get("counterfactual_variant")) for record in variants} != {
            "a",
            "b",
        }:
            raise CounterfactualEvalDataError(f"counterfactual pair {pair_id} needs a/b variants")
        prompts = [_messages_before_first_target(record) for record in variants]
        for record, prompt in zip(variants, prompts, strict=True):
            prompt_text = json.dumps(prompt, ensure_ascii=False, sort_keys=True).casefold()
            own_facts = tuple(str(fact) for fact in record["metadata"].get("varied_facts", []))
            if not own_facts:
                raise CounterfactualEvalDataError(
                    f"counterfactual pair {pair_id} has no varied facts"
                )
            for fact in own_facts:
                if fact.casefold() in prompt_text:
                    raise CounterfactualEvalDataError(
                        f"counterfactual fact {fact!r} is visible before its tool result"
                    )
        if prompts[0] != prompts[1]:
            raise CounterfactualEvalDataError(
                f"counterfactual pair {pair_id} must have identical first-phase prompts"
            )
        results = [_target_tool_results(record) for record in variants]
        if results[0] == results[1]:
            raise CounterfactualEvalDataError(
                f"counterfactual pair {pair_id} must vary canonical tool results"
            )
        for record in variants:
            counterpart = variants[1] if record is variants[0] else variants[0]
            counterpart_facts = {
                str(fact) for fact in counterpart["metadata"].get("varied_facts", [])
            }
            if not counterpart_facts <= set(record["expected"].get("forbidden_facts", [])):
                raise CounterfactualEvalDataError(
                    f"counterfactual pair {pair_id} must forbid counterpart facts"
                )


def _messages_before_first_target(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    for index, message in enumerate(record["messages"]):
        if (
            message.get("role") == "assistant"
            and message.get("tool_calls")
            and message.get("loss", True) is not False
        ):
            return [dict(item) for item in record["messages"][:index]]
    raise CounterfactualEvalDataError(f"{record.get('record_id')} has no target tool call")


def _target_tool_results(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    target_ids = {
        str(call["id"])
        for message in record["messages"]
        if message.get("role") == "assistant"
        and message.get("tool_calls")
        and message.get("loss", True) is not False
        for call in message.get("tool_calls", [])
    }
    return [
        dict(message)
        for message in record["messages"]
        if message.get("role") == "tool" and str(message.get("tool_call_id", "")) in target_ids
    ]


def _role_texts(record: Mapping[str, Any], role: str) -> list[str]:
    return [
        str(message["content"])
        for message in record.get("messages", [])
        if message.get("role") == role and isinstance(message.get("content"), str)
    ]


def _last_user_text(record: Mapping[str, Any]) -> str:
    for message in reversed(record.get("messages", [])):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    raise CounterfactualEvalDataError(f"{record.get('record_id')} has no user text")


def _final_assistant_text(record: Mapping[str, Any]) -> str:
    for message in reversed(record.get("messages", [])):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return str(message["content"])
    raise CounterfactualEvalDataError(f"{record.get('record_id')} has no final target")


def _python_literal_texts(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            normalized = normalized_user_text(node.value)
            if len(normalized.split()) >= 3:
                values.add(normalized)
    return values


def _poc_fact_values(payload: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    _collect_scalar_values(payload, values)
    return values


def _collect_scalar_values(
    item: Any,
    values: set[str],
    *,
    field: str | None = None,
) -> None:
    if isinstance(item, Mapping):
        for key, value in item.items():
            _collect_scalar_values(value, values, field=str(key))
        return
    if isinstance(item, list):
        for value in item:
            _collect_scalar_values(value, values, field=field)
        return
    if item is None:
        return
    values.add(normalized_user_text(str(item)))
    if field and field.endswith("_cents") and isinstance(item, int):
        absolute = abs(item)
        values.add(f"{absolute // 100}.{absolute % 100:02d}")
        values.add(f"{absolute // 100:,}.{absolute % 100:02d}")


def _fact_search_forms(fact: str) -> frozenset[str]:
    raw = fact.strip().casefold()
    normalized = normalized_user_text(raw)
    forms = {raw, normalized}
    if re.fullmatch(r"[0-9][0-9,]*\.[0-9]{2}", raw):
        decimal = raw.replace(",", "")
        forms.add(decimal)
        forms.add(decimal.replace(".", ""))
    return frozenset(forms)


def _character_ngrams(value: str, size: int = 3) -> frozenset[str]:
    padded = f"  {value}  "
    return frozenset(
        padded[index : index + size] for index in range(max(1, len(padded) - size + 1))
    )


def _nearest_prompt(
    query: str,
    indexed_prompts: Sequence[tuple[str, frozenset[str]]],
) -> str:
    query_ngrams = _character_ngrams(query)
    # Trigram Dice similarity is a cheap, conservative shortlist for near-copy
    # prompts. The expensive edit ratio is evaluated only for the 32 closest
    # lexical candidates instead of every training row.
    shortlist = heapq.nlargest(
        32,
        indexed_prompts,
        key=lambda item: (
            2 * len(query_ngrams & item[1]) / max(1, len(query_ngrams) + len(item[1]))
        ),
    )
    if not shortlist:
        raise CounterfactualEvalDataError("training prompt corpus is empty")
    return max(
        (candidate for candidate, _ in shortlist),
        key=lambda candidate: SequenceMatcher(None, query, candidate, autojunk=False).ratio(),
    )


def _tool_manifest_hash() -> str:
    digest = hashlib.sha256(canonical_json_bytes(public_tool_manifest())).hexdigest()
    return f"sha256:{digest}"


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(sorted(counts.items()))


def _render_readme(manifest: Mapping[str, Any]) -> str:
    report = manifest["report"]
    return f"""# Retail Bank Counterfactual Evaluation v1

This dataset is an evaluation-only counterfactual benchmark for the pinned
Granite servicing model. It contains no train or validation split and must not
be used for SFT, remediation, prompt development, or model selection.

- Records: `{report["record_count"]}`
- Counterfactual pairs: `{report["counterfactual_pair_count"]}`
- Training allowed: `false`
- Manifest contract: `{COUNTERFACTUAL_MANIFEST_CONTRACT}`
- Contamination audit: `{report["audit"]["status"]}`

Each pair presents an identical first-phase prompt but changes the canonical
tool result. The changed values are absent before the result and the other
variant's values are forbidden in the final response.

See `docs/13-counterfactual-evaluation.md` for preparation, scoring, and
interpretation instructions.
"""
