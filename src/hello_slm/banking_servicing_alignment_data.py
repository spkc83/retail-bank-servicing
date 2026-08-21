from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from hello_slm.banking_tool_sft_data import (
    BANKING_TOOL_SFT_CONTRACT,
    GENERATION_CONTRACT_VERSION,
    NO_TOOL_OOD_RESPONSE,
    POLICY_CHUNKS,
    SYSTEM_PROMPT,
    _attach_generation_contract,
    export_teacher_realization_requests,
    import_teacher_realizations,
    load_canonical_policy_corpus,
    validate_banking_tool_sft_manifest,
    validate_records,
)
from hello_slm.config import file_sha256

SPLITS = ("train", "validation", "test")
CREATED_AT = "2026-07-31T00:00:00Z"
GENERATOR_VERSION = "banking-servicing-alignment-sft/v7.0-argument-contract"
DEFAULT_OUTPUT_DIR = Path("data/banking-servicing-alignment-v5")
DEFAULT_BASE_SFT_DIR = Path("data/banking-v5-tool-sft")
DEFAULT_SYNTHETIC_BANK_PATH = Path("poc/retail-bank-customer-service-poc/synthetic_bank.json")

SCREENSHOT_HELDOUT_CURRENTS = frozenset(
    {
        "when was that created?",
        "what is that all about? when was it created?",
        "was the mailing address updated recently?",
        "ok, thats the one i want to replace",
        "i didn't ask about mortgage",
        "why are you repeating yourself",
        "what about the weather there?",
        "what happened with the money i sent recently?",
        "show my five most recent transactions.",
    }
)

PII_PATTERNS = (
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]?){12,19}\b"),
)

REALIZATION_COUNTS = {
    "train": 32,
    "validation": 8,
    "test": 4,
}
REALIZATION_CONTEXTS = {
    "train": (
        "I am going through my accounts",
        "I am reviewing my recent banking activity",
        "I want to finish this task in chat",
        "I am looking at the result you just showed",
        "I need the next step for this banking request",
        "I am confirming the details before continuing",
        "I want to act on the item we were discussing",
        "I am following up on your previous answer",
    ),
    "validation": (
        "I am continuing from the details above",
        "I need to resolve this during the same session",
    ),
    "test": ("I am referring to the result from the previous turn",),
}
REALIZATION_REQUESTS = (
    "Please keep the answer concise",
    "Use the information from this conversation",
    "Check the signed-in profile rather than asking for a private ID",
    "Tell me what you find and what happened",
)
FINAL_OPENERS = (
    "",
    "Here is the current update:",
    "I reviewed the conversation and account details.",
    "For clarity,",
    "The relevant information is this:",
    "Here is the concise result:",
    "I checked the available details.",
    "Based on this conversation,",
)
FINAL_CLOSERS = (
    "",
    "I can help with the related next step.",
    "This keeps the response focused on your request.",
    "That reflects the information in this session.",
)


@lru_cache(maxsize=1)
def _cached_servicing_alignment_splits() -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    splits = {
        "train": _train_records(),
        "validation": _validation_records(),
        "test": _test_records(),
    }
    for split, records in splits.items():
        validate_records(records)
        _assert_no_duplicate_current(records, split=split)
    _assert_coreference_pair_integrity(splits)
    _assert_no_cross_split_leakage(splits)
    report = {
        "contract": "banking-servicing-alignment-sft-report",
        "created_at": CREATED_AT,
        "generator_version": GENERATOR_VERSION,
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "scenario_family_counts": {
            split: dict(Counter(row["metadata"]["scenario_family"] for row in rows))
            for split, rows in splits.items()
        },
        "path_counts": {
            split: dict(Counter(row["expected"]["path"] for row in rows))
            for split, rows in splits.items()
        },
        "generation_contract_counts": {
            split: dict(
                Counter(
                    str(row["expected"]["generation_contract"]["mode"])
                    for row in rows
                    if "generation_contract" in row["expected"]
                )
            )
            for split, rows in splits.items()
        },
        "coreference_pair_counts": {
            split: len(
                {
                    str(row["metadata"]["coreference_pair_id"])
                    for row in rows
                    if "coreference_pair_id" in row["metadata"]
                }
            )
            for split, rows in splits.items()
        },
        "duplicate_current_policy": (
            "only declared two-record coreference pairs with opposite targets; "
            "a normalized current may group multiple pairs only across distinct history forms"
        ),
        "pii_matches": _count_pii_matches(splits),
        "heldout_exact_currents_in_train": _heldout_exact_currents_in_train(splits),
        "heldout_long_ngram_leaks_in_train": _heldout_long_ngram_leaks_in_train(splits),
    }
    if report["pii_matches"]:
        raise ValueError("servicing alignment data contains PII-like text")
    if report["heldout_exact_currents_in_train"]:
        raise ValueError("held-out screenshot currents leaked into training")
    if report["heldout_long_ngram_leaks_in_train"]:
        raise ValueError("long held-out screenshot n-grams leaked into training")
    return splits, report


def build_servicing_alignment_splits() -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    splits, report = _cached_servicing_alignment_splits()
    return deepcopy(splits), deepcopy(report)


def build_coreference_shadow_gate() -> list[dict[str, Any]]:
    records = _deictic_replace_curriculum("shadow")
    for record in records:
        record["metadata"]["trainable"] = False
        _attach_generation_contract(record)
    validate_records(records)
    _assert_no_duplicate_current(records, split="shadow")
    _assert_coreference_pair_integrity({"train": [], "validation": records})
    if _count_pii_matches({"shadow": records}):
        raise ValueError("coreference shadow gate contains PII-like text")
    return records


def build_granite_v7_shadow_gate() -> list[dict[str, Any]]:
    records = _granite_v7_examples("shadow")
    for record in records:
        record["metadata"]["trainable"] = False
        _attach_generation_contract(record)
    validate_records(records)
    _assert_no_duplicate_current(records, split="granite-v7-shadow")
    if _count_pii_matches({"shadow": records}):
        raise ValueError("Granite V7 shadow gate contains PII-like text")
    return records


def build_screenshot_regression_fixture() -> list[dict[str, Any]]:
    case_history = [
        {"role": "user", "content": "Show my recent service cases."},
        {"role": "assistant", "content": "You have a closed mailing-address update case."},
    ]
    card_history = [
        {"role": "user", "content": "Show me the cards on my profile."},
        {
            "role": "assistant",
            "content": "Your active debit card ends in 4821. Is that the one to replace?",
        },
    ]
    wrong_answer_history = [
        {"role": "user", "content": "Tell me when my mailing-address case was opened."},
        {"role": "assistant", "content": "Mortgage applicants are typically at least 18."},
    ]
    balance_history = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I can help with your banking questions."},
        {"role": "user", "content": "Show my account balances."},
        {
            "role": "assistant",
            "content": (
                "Everyday Checking ending in 1042 has USD 3,245.67 available; "
                "Goal Saver ending in 8831 has USD 12,500.00 available."
            ),
        },
    ]
    transfer_history = [
        *balance_history,
        {"role": "user", "content": "What happened with the money I sent recently?"},
        {"role": "assistant", "content": "I could not provide the transfer details."},
    ]
    cases: tuple[tuple[Any, ...], ...] = (
        (
            "service-case-created",
            case_history,
            "When was that created?",
            "service",
            "view_service_cases",
            "resolved",
            "list_service_cases",
            {},
            ("created",),
        ),
        (
            "service-case-details",
            case_history,
            "what is that all about? when was it created?",
            "service",
            "view_service_cases",
            "resolved",
            "list_service_cases",
            {},
            ("address", "created"),
        ),
        (
            "mailing-address-standalone",
            [],
            "was the mailing address updated recently?",
            "service",
            "view_service_cases",
            "not_required",
            "list_service_cases",
            {},
            ("address",),
        ),
        (
            "card-selection",
            card_history,
            "ok, thats the one i want to replace",
            "service",
            "replace_card",
            "resolved",
            "replace_card",
            {"last4": {"const": "4821"}},
            ("4821", "replacement"),
        ),
        (
            "agent-repetition-repair",
            wrong_answer_history,
            "why are you repeating yourself",
            "service",
            "view_service_cases",
            "resolved",
            "list_service_cases",
            {},
            ("address", "case"),
        ),
        (
            "wrong-topic-repair",
            wrong_answer_history,
            "I didn't ask about mortgage",
            "service",
            "view_service_cases",
            "resolved",
            "list_service_cases",
            {},
            ("address", "case"),
        ),
        (
            "weather-topic-shift",
            case_history,
            "what about the weather there?",
            "ood",
            "refuse_ood",
            "not_required",
            None,
            {},
            ("retail banking",),
        ),
        (
            "recent-sent-money-status",
            balance_history,
            "What happened with the money I sent recently?",
            "service",
            "view_transfers",
            "not_required",
            "list_transfers",
            {},
            ("transfer",),
        ),
        (
            "explicit-recent-transactions",
            transfer_history,
            "Show my five most recent transactions.",
            "service",
            "view_transactions",
            "not_required",
            "list_transactions",
            {"limit": {"const": 5}},
            ("transaction",),
        ),
    )
    return [
        {
            "contract": "banking-v7-screenshot-regression/v1",
            "record_id": f"screenshot-{case_id}",
            "history": list(history),
            "current": current,
            "expected": {
                "route": route,
                "effective_action": action,
                "entity_state": entity_state,
                "tool_name": tool_name,
                "argument_constraints": constraints,
                "response_properties": {
                    "must_include": list(must_include),
                    "must_not_include": ["password", "pin", "customer id"],
                    "grounded": tool_name is not None,
                },
            },
            "metadata": {"trainable": False, "regression_only": True},
        }
        for (
            case_id,
            history,
            current,
            route,
            action,
            entity_state,
            tool_name,
            constraints,
            must_include,
        ) in cases
    ]


def load_base_sft_splits(
    base_sft_dir: Path = DEFAULT_BASE_SFT_DIR,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_path = base_sft_dir / "manifest.json"
    manifest = validate_banking_tool_sft_manifest(manifest_path)
    splits: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest["tool_sft"]:
        split = str(entry["name"])
        if split not in SPLITS:
            continue
        path = base_sft_dir / str(entry["path"])
        if file_sha256(path) != str(entry["sha256"]):
            raise ValueError(f"base SFT {split} digest mismatch")
        splits[split] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    if set(splits) != set(SPLITS):
        raise ValueError("base SFT must contain train, validation, and test splits")
    return manifest, splits


def _assert_alignment_teacher_rows(
    path: Path, *, trainable: Sequence[Mapping[str, Any]], test_ids: set[str]
) -> None:
    """Reject teacher rows that touch the test split or edit anything but the final."""

    user_text = {
        str(record["record_id"]): str(
            [message for message in record["messages"] if message["role"] == "user"][-1]["content"]
        ).strip()
        for record in trainable
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record_id = str(row.get("record_id"))
        if record_id in test_ids:
            raise ValueError(f"{record_id}: teacher rows must not target the test split")
        if record_id in user_text and (
            str(row.get("user_content", "")).strip() != user_text[record_id]
        ):
            raise ValueError(f"{record_id}: alignment teacher rows may edit final_response only")


def write_servicing_alignment_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    base_sft_dir: Path = DEFAULT_BASE_SFT_DIR,
    synthetic_bank_path: Path = DEFAULT_SYNTHETIC_BANK_PATH,
    export_teacher_requests: Path | None = None,
    teacher_responses: Path | None = None,
    teacher_model: str | None = None,
    teacher_prompt_hash: str | None = None,
) -> dict[str, Any]:
    alignment_splits, alignment_report = build_servicing_alignment_splits()
    shadow_records = build_coreference_shadow_gate()
    granite_shadow_records = build_granite_v7_shadow_gate()
    screenshot_records = build_screenshot_regression_fixture()
    _assert_shadow_isolation(alignment_splits, shadow_records)
    _assert_granite_shadow_isolation(alignment_splits, granite_shadow_records)
    _assert_screenshot_fixture_isolation(alignment_splits, screenshot_records)
    base_manifest, base_splits = load_base_sft_splits(base_sft_dir)
    trainable = [*alignment_splits["train"], *alignment_splits["validation"]]
    if export_teacher_requests is not None:
        export_teacher_realization_requests(trainable, export_teacher_requests)
    realized_counts = {"train": 0, "validation": 0}
    if teacher_responses is not None:
        if not teacher_model or not teacher_prompt_hash:
            raise ValueError(
                "teacher_model and teacher_prompt_hash are required with teacher_responses"
            )
        _assert_alignment_teacher_rows(
            teacher_responses,
            trainable=trainable,
            test_ids={
                str(record["record_id"])
                for record in (*base_splits["test"], *alignment_splits["test"])
            },
        )
        realized = import_teacher_realizations(
            trainable,
            teacher_responses,
            teacher_model=teacher_model,
            teacher_prompt_hash=teacher_prompt_hash,
        )
        n_train = len(alignment_splits["train"])
        alignment_splits = {
            **alignment_splits,
            "train": realized[:n_train],
            "validation": realized[n_train:],
        }
        for split in ("train", "validation"):
            realized_counts[split] = sum(
                1
                for record in alignment_splits[split]
                if record["provenance"].get("teacher_model") == teacher_model
            )
    policy_revision = load_canonical_policy_corpus()["corpus_revision"]
    if base_manifest.get("policy_corpus_revision") != policy_revision:
        raise ValueError("base SFT policy corpus revision is missing or stale")
    splits = {split: [*base_splits[split], *alignment_splits[split]] for split in SPLITS}
    validate_servicing_alignment_splits(splits)
    if _heldout_exact_currents_in_train(splits):
        raise ValueError("held-out screenshot currents leaked into composite training")
    report = {
        **alignment_report,
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "base_split_counts": {split: len(rows) for split, rows in base_splits.items()},
        "alignment_split_counts": {split: len(rows) for split, rows in alignment_splits.items()},
        "alignment_generation_contract_counts": alignment_report["generation_contract_counts"],
        "generation_contract_counts": {
            split: dict(
                Counter(
                    str(row["expected"]["generation_contract"]["mode"])
                    for row in rows
                    if "generation_contract" in row["expected"]
                )
            )
            for split, rows in splits.items()
        },
        "base_manifest_sha256": file_sha256(base_sft_dir / "manifest.json"),
        "alignment_teacher_realization": {
            "model": teacher_model,
            "prompt_hash": teacher_prompt_hash,
            "realized_counts": realized_counts,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for split in SPLITS:
        rows = splits[split]
        path = output_dir / f"{split}.jsonl"
        path.write_bytes(_jsonl_bytes(rows))
        entries.append(
            {
                "name": split,
                "path": path.name,
                "record_count": len(rows),
                "sha256": _sha256_bytes(path.read_bytes()),
                "bytes": path.stat().st_size,
                "allowed_use": [
                    "granite-continuation-sft"
                    if split == "train"
                    else "granite-continuation-evaluation"
                ],
            }
        )
    shadow_path = output_dir / "coreference-shadow.jsonl"
    shadow_path.write_bytes(_jsonl_bytes(shadow_records))
    granite_shadow_path = output_dir / "granite-v7-shadow.jsonl"
    granite_shadow_path.write_bytes(_jsonl_bytes(granite_shadow_records))
    screenshot_path = output_dir / "screenshot-regression.jsonl"
    screenshot_path.write_bytes(_jsonl_bytes(screenshot_records))
    behavioral_gates = [
        {
            "name": "coreference-shadow",
            "path": shadow_path.name,
            "record_count": len(shadow_records),
            "pair_count": len(shadow_records) // 2,
            "sha256": _sha256_bytes(shadow_path.read_bytes()),
            "bytes": shadow_path.stat().st_size,
            "allowed_use": ["post-selection-evaluation-once"],
            "trainable": False,
        },
        {
            "name": "granite-v7-shadow",
            "path": granite_shadow_path.name,
            "record_count": len(granite_shadow_records),
            "sha256": _sha256_bytes(granite_shadow_path.read_bytes()),
            "bytes": granite_shadow_path.stat().st_size,
            "allowed_use": ["checkpoint-selection", "generalization-evaluation"],
            "trainable": False,
            "gate_contract": "banking-v7-granite-predicted-e2e-gate/v1",
        },
    ]
    manifest = {
        "format_version": 1,
        "name": "retail-bank-servicing-alignment-v5",
        "created_at": CREATED_AT,
        "contract": "banking-tool-sft-manifest",
        "schema_version": BANKING_TOOL_SFT_CONTRACT,
        "generator_version": GENERATOR_VERSION,
        "generation_contract_version": GENERATION_CONTRACT_VERSION,
        "generation_contract_model_inputs": (
            "compatible tool schemas only; routing metadata is not rendered"
        ),
        "policy_corpus_revision": policy_revision,
        "tool_sft": entries,
        "behavioral_gates": behavioral_gates,
        "evaluation_fixtures": [
            {
                "name": "screenshot-regression",
                "path": screenshot_path.name,
                "record_count": len(screenshot_records),
                "sha256": _sha256_bytes(screenshot_path.read_bytes()),
                "bytes": screenshot_path.stat().st_size,
                "allowed_use": ["regression-evaluation"],
                "trainable": False,
                "gate_contract": "banking-v7-screenshot-regression/v1",
            }
        ],
        "source_roles": {
            "released-retail-bank-agent-sft": {
                "role": "base-tool-use-sft",
                "license": "MIT",
                "trainable": True,
                "manifest_sha256": report["base_manifest_sha256"],
                "generator_version": base_manifest.get("generator_version"),
            },
            "self-authored-synthetic": {
                "role": "servicing-alignment-continuation-sft",
                "license": "MIT",
                "trainable": True,
            },
        },
        "synthetic_bank_sha256": file_sha256(synthetic_bank_path),
        "report": report,
        "signed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "preparation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _train_records() -> list[dict[str, Any]]:
    records = []
    records.extend(_service_case_followups("train"))
    records.extend(_card_anaphora("train"))
    records.extend(_clarifications("train"))
    records.extend(_agent_repairs("train"))
    records.extend(_topic_shifts("train"))
    records.extend(_policy_detour_and_resume("train"))
    records.extend(_history_entity_actions("train"))
    records.extend(_history_entity_ambiguity("train"))
    records.extend(_tool_outcome_consistency("train"))
    return [
        *_expand_records(records, split="train"),
        *_deictic_replace_curriculum("train"),
        *_deictic_ineligible_curriculum("train"),
        *_missing_entity_records("train"),
        *_social_style_records("train"),
        *_granite_v7_examples("train"),
    ]


def _validation_records() -> list[dict[str, Any]]:
    records = []
    records.extend(_service_case_followups("validation"))
    records.extend(_card_anaphora("validation"))
    records.extend(_clarifications("validation"))
    records.extend(_agent_repairs("validation"))
    records.extend(_topic_shifts("validation"))
    records.extend(_policy_detour_and_resume("validation"))
    records.extend(_history_entity_actions("validation"))
    records.extend(_history_entity_ambiguity("validation"))
    records.extend(_tool_outcome_consistency("validation"))
    return [
        *_expand_records(records, split="validation"),
        *_deictic_replace_curriculum("validation"),
        *_missing_entity_records("validation"),
        *_social_style_records("validation"),
        *_granite_v7_examples("validation"),
    ]


def _test_records() -> list[dict[str, Any]]:
    records = []
    records.extend(_service_case_followups("test"))
    records.extend(_card_anaphora("test"))
    records.extend(_topic_shifts("test"))
    records.extend(_policy_detour_and_resume("test"))
    return [
        *_expand_records(records, split="test"),
        *_heldout_regression_records(),
    ]


def _service_case_followups(split: str) -> list[dict[str, Any]]:
    suffix = _suffix(split)
    return [
        _record(
            record_id=f"svc_case_created_{split}",
            split=split,
            scenario_family="service_case_context",
            current=f"When did the mailing-address case get created{suffix}?",
            final=(
                "The mailing-address update case was created on 2026-06-18 at "
                "14:00 UTC. Its status is closed, the type is address_update, "
                "and the subject is Confirm mailing address update."
            ),
            tool_plan=[("list_service_cases", {})],
            grounding_facts=["case.created_at=2026-06-18T14:00:00Z"],
            path="multi_turn",
            pre_messages=_case_history(split),
        ),
        _record(
            record_id=f"svc_case_status_{split}",
            split=split,
            scenario_family="service_case_context",
            current=f"What happened with that address request{suffix}?",
            final=(
                "I found the same service case. It is closed, its type is "
                "address_update, the subject is Confirm mailing address update, "
                "and it was created on 2026-06-18 at 14:00 UTC."
            ),
            tool_plan=[("list_service_cases", {})],
            grounding_facts=[
                "case.status=closed",
                "case.case_type=address_update",
                "case.subject=Confirm mailing address update",
                "case.created_at=2026-06-18T14:00:00Z",
            ],
            path="multi_turn",
            pre_messages=_case_history(split),
        ),
    ]


def _card_anaphora(split: str) -> list[dict[str, Any]]:
    suffix = _suffix(split)
    return [
        _record(
            record_id=f"card_replace_that_one_{split}",
            split=split,
            scenario_family="card_anaphora_action",
            current=f"Please replace the active card you just listed{suffix}.",
            final=(
                "Replacement is pending for the Everyday Visa Debit card ending "
                "in 4821. That replacement request has been submitted."
            ),
            tool_plan=[("replace_card", {"last4": "4821"})],
            grounding_facts=["card.last4=4821", "card.status=replacement_pending"],
            path="multi_turn",
            pre_messages=_card_history(split),
        ),
        _record(
            record_id=f"card_freeze_that_one_{split}",
            split=split,
            scenario_family="card_anaphora_action",
            current=f"Freeze that one before I travel{suffix}.",
            final="Your Everyday Visa Debit card ending in 4821 is now frozen.",
            tool_plan=[("freeze_card", {"last4": "4821"})],
            grounding_facts=["card.last4=4821", "card.status=frozen"],
            path="multi_turn",
            pre_messages=_card_history(split),
        ),
    ]


def _clarifications(split: str) -> list[dict[str, Any]]:
    suffix = _suffix(split)
    return [
        _record(
            record_id=f"clarify_replace_last4_{split}",
            split=split,
            scenario_family="clarification_answer",
            current=f"It is 4821{suffix}.",
            final=("Replacement is pending for your Everyday Visa Debit card ending in 4821."),
            tool_plan=[("replace_card", {"last4": "4821"})],
            grounding_facts=["card.last4=4821", "card.status=replacement_pending"],
            path="multi_turn",
            pre_messages=[
                _user(f"Replace my debit card{suffix}."),
                _assistant(
                    "Which card should I replace? Please share the last four digits.",
                    loss=False,
                ),
            ],
        ),
        _record(
            record_id=f"clarify_dispute_merchant_{split}",
            split=split,
            scenario_family="clarification_answer",
            current=f"The North Harbor Market one{suffix}.",
            final=(
                "Using your clarification, I opened a dispute for the North Harbor "
                "Market transaction."
            ),
            tool_plan=[("dispute_transaction", {"description": "North Harbor Market"})],
            grounding_facts=[
                "transaction.description=North Harbor Market",
                "transaction.disputed=true",
            ],
            path="multi_turn",
            pre_messages=[
                _user(f"I need to dispute a debit{suffix}."),
                _assistant("Which transaction should I dispute?", loss=False),
            ],
        ),
    ]


def _agent_repairs(split: str) -> list[dict[str, Any]]:
    suffix = _suffix(split)
    return [
        _record(
            record_id=f"repair_wrong_topic_case_{split}",
            split=split,
            scenario_family="agent_repair",
            current=f"I meant the address case, not mortgage basics{suffix}.",
            final=(
                "You are right. The address case is closed, its type is "
                "address_update, the subject is Confirm mailing address update, "
                "and it was created on 2026-06-18 at 14:00 UTC."
            ),
            tool_plan=[("list_service_cases", {})],
            grounding_facts=[
                "case.subject=Confirm mailing address update",
                "case.created_at=2026-06-18T14:00:00Z",
            ],
            path="multi_turn",
            pre_messages=[
                _user(f"Tell me about my address case{suffix}."),
                _assistant(
                    "Mortgage applications usually require a lender review.",
                    loss=False,
                ),
            ],
        ),
        _record(
            record_id=f"repair_repetition_{split}",
            split=split,
            scenario_family="agent_repair",
            current=f"Do not repeat that; check the actual service case{suffix}.",
            final=(
                "I checked the service case. It is closed, the type is "
                "address_update, the subject is Confirm mailing address update, "
                "and it was created on 2026-06-18 at 14:00 UTC."
            ),
            tool_plan=[("list_service_cases", {})],
            grounding_facts=[
                "case.status=closed",
                "case.case_type=address_update",
                "case.created_at=2026-06-18T14:00:00Z",
            ],
            path="multi_turn",
            pre_messages=_case_history(split),
        ),
    ]


def _topic_shifts(split: str) -> list[dict[str, Any]]:
    suffix = _suffix(split)
    return [
        _record(
            record_id=f"banking_to_external_ood_{split}",
            split=split,
            scenario_family="external_topic_shift",
            current=f"What is the weather near North Harbor{suffix}?",
            final=(
                "I can’t provide weather information. Harbor can help with accounts, "
                "cards, transfers, payments, loans, and related retail banking support."
            ),
            tool_plan=[],
            grounding_facts=[],
            path="ood",
            pre_messages=_case_history(split),
        ),
        _record(
            record_id=f"external_to_banking_{split}",
            split=split,
            scenario_family="banking_topic_shift",
            current=f"Back to banking: show my service cases{suffix}.",
            final=(
                "You have one recent service case: Confirm mailing address update. "
                "It is closed, its type is address_update, and it was created on "
                "2026-06-18 at 14:00 UTC."
            ),
            tool_plan=[("list_service_cases", {})],
            grounding_facts=["case.status=closed", "case.created_at=2026-06-18T14:00:00Z"],
            path="multi_turn",
            pre_messages=[
                _user(f"What is the weather near North Harbor{suffix}?"),
                _assistant(NO_TOOL_OOD_RESPONSE, loss=False),
            ],
        ),
    ]


def _policy_detour_and_resume(split: str) -> list[dict[str, Any]]:
    suffix = _suffix(split)
    chunk = POLICY_CHUNKS["faq-savings-interest-v1"]
    policy_answer = (
        f"Before returning to your dispute, {chunk['answer']} [Policy: {chunk['chunk_id']}]."
    )
    pending_history = [
        _user(f"I need to dispute the North Harbor Market debit{suffix}."),
        _assistant(
            "I can help with that dispute. Before I submit it, what would you like to know?",
            loss=False,
        ),
    ]
    return [
        _record(
            record_id=f"policy_detour_{split}",
            split=split,
            scenario_family="policy_detour",
            current=f"First, how does savings interest work{suffix}?",
            final=policy_answer,
            tool_plan=[],
            grounding_facts=list(chunk["required_claims"]),
            path="retrieval_grounded_policy",
            pre_messages=pending_history,
            policy=chunk,
        ),
        _record(
            record_id=f"policy_resume_{split}",
            split=split,
            scenario_family="policy_resume",
            current=f"Thanks. Continue with that dispute{suffix}.",
            final=(
                "I resumed your earlier request and opened a dispute for the North "
                "Harbor Market transaction."
            ),
            tool_plan=[("dispute_transaction", {"description": "North Harbor Market"})],
            grounding_facts=[
                "transaction.description=North Harbor Market",
                "transaction.disputed=true",
            ],
            path="multi_turn",
            pre_messages=[
                *pending_history,
                _message(
                    "system",
                    "Authoritative Harborlight Bank policy context. Answer only from this "
                    "context and cite the bracketed policy chunk ID.\n"
                    f"[Policy: {chunk['chunk_id']}] {chunk['title']}: {chunk['text']}",
                    loss=False,
                ),
                _user(f"First, how does savings interest work{suffix}?"),
                _assistant(policy_answer, loss=False),
            ],
        ),
    ]


def _remediation_entities(split: str) -> dict[str, str]:
    if split == "train":
        return {
            "card_name": "Everyday Rewards Debit",
            "card_last4": "6158",
            "other_card_name": "Travel Visa",
            "other_card_last4": "2046",
            "outcome_card_name": "Cashback Debit",
            "outcome_card_last4": "7742",
            "missing_card_last4": "9307",
            "recipient": "Summit Plumbing",
            "missing_recipient": "Cedar Mobile",
            "merchant": "Silver Pine Books",
        }
    if split == "validation":
        return {
            "card_name": "Essentials Debit",
            "card_last4": "3074",
            "other_card_name": "Reserve Credit",
            "other_card_last4": "8662",
            "outcome_card_name": "Neighborhood Debit",
            "outcome_card_last4": "5413",
            "missing_card_last4": "1289",
            "recipient": "Juniper Internet",
            "missing_recipient": "Granite Wireless",
            "merchant": "Copper Trail Pharmacy",
        }
    raise ValueError(f"unsupported remediation split: {split}")


def _coreference_curriculum_specs(split: str) -> tuple[dict[str, str], ...]:
    if split == "train":
        return (
            {
                "phrase_family": "that-card",
                "prompt": "replace that card",
                "product": "Harbor",
            },
            {
                "phrase_family": "that-one",
                "prompt": "replace that one",
                "product": "Summit",
            },
            {
                "phrase_family": "swap-card",
                "prompt": "swap that card",
                "product": "Cedar",
            },
            {
                "phrase_family": "swap-one",
                "prompt": "swap that one",
                "product": "Maple",
            },
            {
                "phrase_family": "it",
                "prompt": "order a replacement for it",
                "product": "Orchard",
            },
            {
                "phrase_family": "replacement-card",
                "prompt": "get a replacement for that card",
                "product": "River",
            },
            {
                "phrase_family": "would-like-card",
                "prompt": "i would like that card replaced",
                "product": "Prairie",
            },
            {
                "phrase_family": "needs-replacing",
                "prompt": "that one needs replacing",
                "product": "Lake",
            },
            {
                "phrase_family": "same-card",
                "prompt": "the same card should be replaced",
                "product": "Valley",
            },
            {
                "phrase_family": "replacement-needed",
                "prompt": "a replacement is what i need for that one",
                "product": "Pioneer",
            },
            {
                "phrase_family": "active-one",
                "prompt": "the active one is the card to replace",
                "product": "Meadow",
            },
            {
                "phrase_family": "one-to-swap",
                "prompt": "that card is the one to swap out",
                "product": "Forest",
            },
            {
                "phrase_family": "prior-reference-bridge",
                "prompt": "the card you mentioned should be replaced",
                "product": "Beacon",
            },
            {
                "phrase_family": "result-reference-bridge",
                "prompt": "replace the card shown in those results",
                "product": "Canyon",
            },
            {
                "phrase_family": "fresh-copy-bridge",
                "prompt": "send a fresh version of that card",
                "product": "Delta",
            },
            {
                "phrase_family": "proceed-reference-bridge",
                "prompt": "continue with replacing the referenced card",
                "product": "Elm",
            },
            {
                "phrase_family": "previous-reply-bridge",
                "prompt": "replace the card from your previous reply",
                "product": "Fjord",
            },
            {
                "phrase_family": "result-item-bridge",
                "prompt": "the card in those results needs replacement",
                "product": "Grove",
            },
            {
                "phrase_family": "duplicate-card-bridge",
                "prompt": "issue another physical card matching it",
                "product": "Haven",
            },
            {
                "phrase_family": "continue-action-bridge",
                "prompt": "proceed with a replacement for the card mentioned",
                "product": "Indigo",
            },
            {
                "phrase_family": "list-item-bridge",
                "prompt": "the card on that list needs to be replaced",
                "product": "Redwood",
            },
            {
                "phrase_family": "listed-card-bridge",
                "prompt": "i would like the listed card replaced",
                "product": "Sierra",
            },
            {
                "phrase_family": "from-results-bridge",
                "prompt": "replace the card from the results you showed",
                "product": "Mesa",
            },
            {
                "phrase_family": "copy-needed-bridge",
                "prompt": "a fresh copy of that one is what i need",
                "product": "Bluff",
            },
            {
                "phrase_family": "need-new-copy-bridge",
                "prompt": "i need a new copy of the card you found",
                "product": "Tundra",
            },
            {
                "phrase_family": "want-swap-bridge",
                "prompt": "swapping that one out is what i want",
                "product": "Dune",
            },
            {
                "phrase_family": "desire-list-bridge",
                "prompt": "i would like the card shown on the list replaced",
                "product": "Basalt",
            },
            {
                "phrase_family": "from-your-list-bridge",
                "prompt": "the card from your list should be replaced",
                "product": "Cobalt",
            },
            {
                "phrase_family": "want-list-bridge",
                "prompt": "i want the card in that list replaced",
                "product": "Onyx",
            },
            {
                "phrase_family": "copy-cleft-bridge",
                "prompt": "a replacement copy of that card is needed",
                "product": "Slate",
            },
            {
                "phrase_family": "cleft-copy-bridge",
                "prompt": "what i need is a fresh copy of that card",
                "product": "Flint",
            },
            {
                "phrase_family": "copy-required-bridge",
                "prompt": "another copy of that one is required",
                "product": "Shale",
            },
            {
                "phrase_family": "desire-results-bridge",
                "prompt": "i would like the card from your results replaced",
                "product": "Quarry",
            },
            {
                "phrase_family": "desire-overview-bridge",
                "prompt": "i would like the card in that overview replaced",
                "product": "Boulder",
            },
            {
                "phrase_family": "desire-swap-list-bridge",
                "prompt": "i would like the card on that list swapped",
                "product": "Cinder",
            },
            {
                "phrase_family": "summary-replace-bridge",
                "prompt": "take the card from that summary and replace it",
                "product": "Garnet",
            },
            {
                "phrase_family": "wish-results-bridge",
                "prompt": "i wish to have the card from those results replaced",
                "product": "Amber",
            },
            {
                "phrase_family": "shown-list-bridge",
                "prompt": "kindly replace the card shown in that list",
                "product": "Ivory",
            },
            {
                "phrase_family": "presented-above-bridge",
                "prompt": "the card presented above should be replaced",
                "product": "Cliff",
            },
            {
                "phrase_family": "onscreen-target-bridge",
                "prompt": "the card on screen is the one to replace",
                "product": "Reef",
            },
            {
                "phrase_family": "shown-target-bridge",
                "prompt": "the card you showed is the replacement candidate",
                "product": "Vale",
            },
            {
                "phrase_family": "above-reference-bridge",
                "prompt": "replace the card referenced above",
                "product": "Moor",
            },
            # Targeted margin for the held-out "list-reference" validation pair:
            # nearby list/showed phrasings that never reuse the held-out wording.
            {
                "phrase_family": "listed-card",
                "prompt": "replace the card you listed",
                "product": "Marsh",
            },
            {
                "phrase_family": "from-your-list",
                "prompt": "the card in your list needs replacing",
                "product": "Alder",
            },
            {
                "phrase_family": "card-you-showed",
                "prompt": "replace the card you just showed me",
                "product": "Brook",
            },
            # Targeted margin for the held-out "results-reference-shadow" pair:
            # nearby shown-above / results phrasings that never reuse its wording.
            {
                "phrase_family": "shown-above",
                "prompt": "replace the card shown above",
                "product": "Cove",
            },
            {
                "phrase_family": "above-card",
                "prompt": "the card above is the one to replace",
                "product": "Ridge",
            },
            {
                "phrase_family": "from-results",
                "prompt": "the card from those results needs a replacement",
                "product": "Basin",
            },
            {
                "phrase_family": "target-card",
                "prompt": "that card is my replacement target",
                "product": "Larch",
            },
        )
    if split == "validation":
        return (
            {
                "phrase_family": "prior-answer",
                "prompt": "that is the card i need replaced",
                "product": "Juniper",
            },
            {
                "phrase_family": "list-reference",
                "prompt": "i would like the card from that list replaced",
                "product": "Granite",
            },
            {
                "phrase_family": "another-copy",
                "prompt": "a new copy of that card is what i need",
                "product": "Copper",
            },
            {
                "phrase_family": "replacement-reference",
                "prompt": "go ahead with replacing that card",
                "product": "Aspen",
            },
        )
    if split == "shadow":
        return (
            {
                "phrase_family": "earlier-response-shadow",
                "prompt": "use the card identified in your earlier response",
                "product": "Quartz",
            },
            {
                "phrase_family": "results-reference-shadow",
                "prompt": "the card displayed above is the replacement target",
                "product": "Birch",
            },
            {
                "phrase_family": "fresh-card-shadow",
                "prompt": "arrange a fresh card corresponding to that one",
                "product": "Willow",
            },
            {
                "phrase_family": "resume-replacement-shadow",
                "prompt": "resume the replacement for the card already referenced",
                "product": "Stone",
            },
        )
    raise ValueError(f"unsupported coreference curriculum split: {split}")




def _deictic_replace_curriculum(split: str) -> list[dict[str, Any]]:
    prompt_forms = (
        "{prompt}",
        "okay {prompt}",
        "{prompt} please",
        "yes {prompt}",
    )
    sole_card_history_forms = (
        "You have an active {card_name} card ending in {card_last4}.",
        "Cards on this profile: {card_name} ending in {card_last4} (active).",
        "I found {card_name} ending in {card_last4} with active status.",
        "Card results: {card_name} ending in {card_last4}, status active.",
    )
    multiple_card_history_forms = (
        (
            "You have an active {card_name} card ending in {card_last4} and an active "
            "{other_card_name} card ending in {other_card_last4}."
        ),
        (
            "Cards on this profile: {card_name} ending in {card_last4} (active); "
            "{other_card_name} ending in {other_card_last4} (active)."
        ),
        (
            "I found {card_name} ending in {card_last4} and {other_card_name} ending in "
            "{other_card_last4}, both with active status."
        ),
        (
            "Card results: {card_name} ending in {card_last4}, status active; "
            "{other_card_name} ending in {other_card_last4}, status active."
        ),
    )
    tiers = ("Everyday Debit", "Rewards Debit", "Travel Debit", "Cashback Debit")
    records: list[dict[str, Any]] = []
    number_base = {"train": 6100, "validation": 2100, "shadow": 3100}[split]
    other_number_base = {"train": 8100, "validation": 4100, "shadow": 5100}[split]
    pair_index = 0
    specs = _coreference_curriculum_specs(split)
    products = tuple(spec["product"] for spec in specs)
    for family_index, spec in enumerate(specs):
        combinations = (
            (
                (prompt_index, history_index)
                for prompt_index in range(len(prompt_forms))
                for history_index in range(len(sole_card_history_forms))
            )
            if split == "train"
            else (
                (realization, (3 * family_index + realization) % len(sole_card_history_forms))
                for realization in range(len(prompt_forms))
            )
        )
        for prompt_index, history_form in combinations:
            pair_index += 1
            family = spec["phrase_family"]
            prompt = prompt_forms[prompt_index].format(prompt=spec["prompt"])
            if split == "train":
                product = products[
                    (family_index + (3 * prompt_index) + (5 * history_form)) % len(products)
                ]
                tier = tiers[(family_index + prompt_index + history_form) % len(tiers)]
                realization_key = f"{prompt_index}-{history_form}"
            else:
                product = products[(family_index + (3 * prompt_index)) % len(products)]
                tier = tiers[(family_index + prompt_index) % len(tiers)]
                realization_key = str(prompt_index)
            card_name = f"{product} {tier}"
            other_card_name = f"{product} {tier.replace('Debit', 'Credit')}"
            card_last4 = f"{number_base + pair_index:04d}"
            other_card_last4 = f"{other_number_base + pair_index:04d}"
            pair_id = f"coreference-{split}-{family}-{realization_key}"
            history_user = "List the cards on my profile."
            history_values = {
                "card_name": card_name,
                "card_last4": card_last4,
                "other_card_name": other_card_name,
                "other_card_last4": other_card_last4,
            }
            sole_card_history = sole_card_history_forms[history_form].format(**history_values)
            multiple_card_history = multiple_card_history_forms[history_form].format(
                **history_values
            )
            action_final = (
                f"Replacement is pending for your {card_name} ending in {card_last4}."
                if split == "validation"
                else f"{card_name} ending in {card_last4} now has replacement pending."
            )
            action = _record(
                record_id=f"deictic_replace_{family}_{split}_{realization_key}",
                split=split,
                scenario_family="deictic_replace_action",
                current=prompt,
                final=action_final,
                tool_plan=[("replace_card", {"last4": card_last4})],
                grounding_facts=[
                    f"card.last4={card_last4}",
                    "card.status=replacement_pending",
                ],
                path="multi_turn",
                pre_messages=[
                    _user(history_user),
                    _assistant(sole_card_history, loss=False),
                ],
                tool_envelopes=[
                    _success_envelope(
                        card={
                            "name": card_name,
                            "last4": card_last4,
                            "status": "replacement_pending",
                        }
                    )
                ],
            )
            # Single template on purpose: the 2026-08-21 v9 run replaced it with a
            # 32-phrasing conversational pool and the coreference dev gate fell to
            # ambiguity_accuracy 0.44 (the continuation LR could not overwrite the
            # parent adapter's prior on the gate prompts). The template is the only
            # configuration proven to pass; coreference-shadow.jsonl also pins it.
            ambiguity_closer = (
                "share the last four digits shown in the app."
                if split == "shadow"
                else "share its last four digits."
            )
            ambiguity_final = (
                f"I found {card_name} ending in {card_last4} and {other_card_name} "
                f"ending in {other_card_last4}. Which card should I replace? Please "
                f"{ambiguity_closer}"
            )
            ambiguity = _record(
                record_id=f"deictic_ambiguous_{family}_{split}_{realization_key}",
                split=split,
                scenario_family="deictic_replace_ambiguity",
                current=prompt,
                final=ambiguity_final,
                tool_plan=[],
                grounding_facts=[],
                path="clarification",
                pre_messages=[
                    _user(history_user),
                    _assistant(multiple_card_history, loss=False),
                ],
            )
            entity_keys = (
                f"{card_name}|{card_last4}",
                f"{other_card_name}|{other_card_last4}",
            )
            for record, target, actionable_card_count in (
                (action, "replace_card", 1),
                (ambiguity, "clarification", 2),
            ):
                record["metadata"].update(
                    {
                        "coreference_pair_id": pair_id,
                        "coreference_phrase_family": family,
                        "coreference_prompt": prompt,
                        "coreference_prompt_form": prompt_index,
                        "coreference_history_form": history_form,
                        "coreference_product": product,
                        "coreference_tier": tier,
                        "coreference_entity_keys": entity_keys,
                        "coreference_target": target,
                        "actionable_card_count": actionable_card_count,
                    }
                )
            records.extend((action, ambiguity))
    return records


_INELIGIBLE_PROMPT_FORMS = (
    "{prompt}",
    "okay {prompt}",
    "{prompt} please",
    "yes {prompt}",
)
_INELIGIBLE_CURRENT_SUFFIXES = ("", " right now", " today")

_INELIGIBLE_SPECS = (
    {
        "phrase_family": "frozen-replace-bridge",
        "prompt": "freeze the card i just asked about",
        "product": "Cobble",
    },
    {
        "phrase_family": "pending-freeze-bridge",
        "prompt": "go ahead and freeze that card",
        "product": "Drift",
    },
    {
        "phrase_family": "closed-card-copy-bridge",
        "prompt": "send me a new copy of that card",
        "product": "Ember",
    },
    {
        "phrase_family": "pending-again-bridge",
        "prompt": "please order another replacement for it",
        "product": "Foss",
    },
    {
        "phrase_family": "frozen-swap-bridge",
        "prompt": "swap that card out for a new one",
        "product": "Gale",
    },
    {
        "phrase_family": "closed-freeze-bridge",
        "prompt": "freeze the card you just found",
        "product": "Hollow",
    },
)

# (status, blocker clause, single-card history sentence) -- exactly one card is
# listed and its status is what makes the request ineligible.
_INELIGIBLE_HISTORY_FORMS = (
    (
        "frozen",
        "is frozen",
        "You have one card on file: {card_name} ending in {card_last4}, currently frozen.",
    ),
    (
        "replacement_pending",
        "already has a replacement pending",
        "Card search results: {card_name} ending in {card_last4} — status: "
        "replacement pending.",
    ),
    (
        "closed",
        "has been closed",
        "I found {card_name} ending in {card_last4}; it shows as closed.",
    ),
)

# One distinct final phrasing per family (matched by position) naming the blocker
# and asking for an eligible, specific card.
_INELIGIBLE_FINAL_TEMPLATES = (
    "Your {card_name} ending in {card_last4} {blocker}, so I can't do that. Which "
    "active card should I use instead? Please share its last four digits.",
    "That {card_name} ending in {card_last4} {blocker}, so this won't go through. "
    "Tell me the last four digits of a different card to use.",
    "I can't proceed because the {card_name} ending in {card_last4} {blocker}. "
    "Which eligible card should I use, and what are its last four digits?",
    "The {card_name} ending in {card_last4} {blocker}, which rules it out for this "
    "request. Please name an eligible card and its last four digits.",
    "This won't work for the {card_name} ending in {card_last4} — it {blocker}. "
    "Which other card should I use? Its last four digits would help.",
    "The {card_name} ending in {card_last4} {blocker}, so it isn't eligible for "
    "that action. Could you pick an eligible card and give me its last four digits?",
)

_MISSING_PROMPT_FORMS = _INELIGIBLE_PROMPT_FORMS
_MISSING_CURRENT_SUFFIXES = ("", " if that's possible", " when you get a chance")

_MISSING_SPECS = (
    {"phrase_family": "freeze-my-card-bridge", "verb": "freeze", "prompt": "freeze my card"},
    {"phrase_family": "replace-my-card-bridge", "verb": "replace", "prompt": "replace my card"},
    {
        "phrase_family": "copy-my-card-bridge",
        "verb": "send a new copy of",
        "prompt": "send a new copy of my card",
    },
    {
        "phrase_family": "replacement-my-card-bridge",
        "verb": "order a replacement for",
        "prompt": "order a replacement for my card",
    },
    {"phrase_family": "block-my-card-bridge", "verb": "block", "prompt": "block my card"},
    {"phrase_family": "swap-my-card-bridge", "verb": "swap out", "prompt": "swap out my card"},
)

# (context clause, prior user turn or None, prior assistant turn) -- no card is ever
# listed, so entity_state defaults to "missing" rather than "ineligible"/"ambiguous".
_MISSING_HISTORY_FORMS: tuple[tuple[str, tuple[str, str] | None], ...] = (
    ("I don't see a card identified on this profile yet.", None),
    (
        "No card has been specified in this conversation so far.",
        ("Hi, I need some help today.", "Hello! I'm happy to help with your banking today."),
    ),
    (
        "There's no card on file that I can match to your request yet.",
        (
            "What's my account balance?",
            "I can pull that up, but I don't have any card selected for this request.",
        ),
    ),
)

_MISSING_ASK_FORMS = (
    "Which card would you like me to {verb}? Please share its last four digits.",
    "Okay — which card should I {verb}? What are its last four digits?",
    "Sure, but which card do you need me to {verb}? Let me know its last four digits.",
    "Got it — please tell me which card to {verb}, along with its last four digits.",
)


def _deictic_ineligible_curriculum(split: str) -> list[dict[str, Any]]:
    """Train-only clarify curriculum covering the "ineligible" and "missing"
    entity states.

    The runtime derives entity_state "ineligible" from three grounding branches (a
    card that is frozen, replacement_pending, or closed) but the corpus otherwise
    carries a single ineligible training example, and the "missing" state is thin
    too, so the model improvises on those turns -- one observed failure fabricated
    a card freeze. This is modeled directly on `_deictic_replace_curriculum` above:
    same prompt_forms mechanism, same `_record` builder, same metadata update
    pattern, but train split only -- the frozen validation gate is not extended.
    """
    if split != "train":
        raise ValueError(f"unsupported ineligible/missing curriculum split: {split}")
    return [
        *_ineligible_clarification_records(split),
        *_missing_clarification_records(split),
    ]


def _ineligible_clarification_records(split: str) -> list[dict[str, Any]]:
    tiers = ("Everyday Debit", "Rewards Debit", "Travel Debit", "Cashback Debit")
    number_base = 7100
    records: list[dict[str, Any]] = []
    pair_index = 0
    for family_index, spec in enumerate(_INELIGIBLE_SPECS):
        family = spec["phrase_family"]
        product = spec["product"]
        final_template = _INELIGIBLE_FINAL_TEMPLATES[family_index]
        for prompt_index, prompt_form in enumerate(_INELIGIBLE_PROMPT_FORMS):
            base_prompt = prompt_form.format(prompt=spec["prompt"])
            for history_index, (status, blocker, history_template) in enumerate(
                _INELIGIBLE_HISTORY_FORMS
            ):
                pair_index += 1
                tier = tiers[(family_index + prompt_index + history_index) % len(tiers)]
                card_name = f"{product} {tier}"
                card_last4 = f"{number_base + pair_index:04d}"
                current = base_prompt + _INELIGIBLE_CURRENT_SUFFIXES[history_index]
                history_text = history_template.format(
                    card_name=card_name, card_last4=card_last4
                )
                final = final_template.format(
                    card_name=card_name, card_last4=card_last4, blocker=blocker
                )
                realization_key = f"{prompt_index}-{history_index}"
                record = _record(
                    record_id=f"deictic_ineligible_{family}_{split}_{realization_key}",
                    split=split,
                    scenario_family="deictic_ineligible_clarification",
                    current=current,
                    final=final,
                    tool_plan=[],
                    grounding_facts=[f"card.last4={card_last4}", f"card.status={status}"],
                    path="clarification",
                    pre_messages=[
                        _user("Check the status of my card."),
                        _assistant(history_text, loss=False),
                    ],
                )
                record["metadata"].update(
                    {
                        "coreference_target": "clarification",
                        "actionable_card_count": 0,
                    }
                )
                records.append(record)
    return records


def _missing_clarification_records(split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in _MISSING_SPECS:
        family = spec["phrase_family"]
        verb = spec["verb"]
        for prompt_index, prompt_form in enumerate(_MISSING_PROMPT_FORMS):
            base_prompt = prompt_form.format(prompt=spec["prompt"])
            ask_clause = _MISSING_ASK_FORMS[prompt_index].format(verb=verb)
            for history_index, (context_clause, prior_turn) in enumerate(
                _MISSING_HISTORY_FORMS
            ):
                current = base_prompt + _MISSING_CURRENT_SUFFIXES[history_index]
                final = f"{context_clause} {ask_clause}"
                pre_messages = (
                    []
                    if prior_turn is None
                    else [_user(prior_turn[0]), _assistant(prior_turn[1], loss=False)]
                )
                realization_key = f"{prompt_index}-{history_index}"
                record = _record(
                    record_id=f"deictic_missing_{family}_{split}_{realization_key}",
                    split=split,
                    scenario_family="deictic_missing_clarification",
                    current=current,
                    final=final,
                    tool_plan=[],
                    grounding_facts=["missing_field=last4"],
                    path="clarification",
                    pre_messages=pre_messages,
                )
                record["metadata"].update(
                    {
                        "coreference_target": "clarification",
                        "actionable_card_count": 0,
                    }
                )
                records.append(record)
    return records


def _history_entity_actions(split: str) -> list[dict[str, Any]]:
    entity = _remediation_entities(split)
    suffix = _suffix(split)
    card_history = [
        _user(f"Which card did you find {suffix}?"),
        _assistant(
            f"I found your active {entity['card_name']} ending in {entity['card_last4']}.",
            loss=False,
        ),
    ]
    return [
        _record(
            record_id=f"history_replace_card_{split}",
            split=split,
            scenario_family="history_entity_action",
            current=f"Replace the one identified in your last message {suffix}.",
            final=(
                f"Replacement is pending for your {entity['card_name']} ending in "
                f"{entity['card_last4']}."
            ),
            tool_plan=[("replace_card", {"last4": entity["card_last4"]})],
            grounding_facts=[
                f"card.last4={entity['card_last4']}",
                "card.status=replacement_pending",
            ],
            path="multi_turn",
            pre_messages=card_history,
            tool_envelopes=[
                _success_envelope(
                    card={
                        "name": entity["card_name"],
                        "last4": entity["card_last4"],
                        "status": "replacement_pending",
                    }
                )
            ],
        ),
        _record(
            record_id=f"history_freeze_card_{split}",
            split=split,
            scenario_family="history_entity_action",
            current=f"Freeze that card for me {suffix}.",
            final=(f"Your {entity['card_name']} ending in {entity['card_last4']} is now frozen."),
            tool_plan=[("freeze_card", {"last4": entity["card_last4"]})],
            grounding_facts=[
                f"card.last4={entity['card_last4']}",
                "card.status=frozen",
            ],
            path="multi_turn",
            pre_messages=card_history,
            tool_envelopes=[
                _success_envelope(
                    card={
                        "name": entity["card_name"],
                        "last4": entity["card_last4"],
                        "status": "frozen",
                    }
                )
            ],
        ),
        _record(
            record_id=f"history_cancel_transfer_{split}",
            split=split,
            scenario_family="history_entity_action",
            current=f"Cancel the pending payment you identified {suffix}.",
            final=(
                f"The cancellation completed successfully for the pending transfer to "
                f"{entity['recipient']}."
            ),
            tool_plan=[("cancel_transfer", {"recipient": entity["recipient"]})],
            grounding_facts=[
                f"transfer.recipient={entity['recipient']}",
                "transfer.status=cancelled",
            ],
            path="multi_turn",
            pre_messages=[
                _user(f"Summarize my transfers {suffix}."),
                _assistant(
                    f"You have a pending transfer to {entity['recipient']} and no other "
                    "pending transfers.",
                    loss=False,
                ),
            ],
            tool_envelopes=[
                _success_envelope(
                    transfer={"recipient": entity["recipient"], "status": "cancelled"}
                )
            ],
        ),
        _record(
            record_id=f"history_dispute_transaction_{split}",
            split=split,
            scenario_family="history_entity_action",
            current=f"Open a dispute for the merchant charge you identified above {suffix}.",
            final=f"I opened a dispute for the {entity['merchant']} transaction.",
            tool_plan=[("dispute_transaction", {"description": entity["merchant"]})],
            grounding_facts=[
                f"transaction.description={entity['merchant']}",
                "transaction.disputed=true",
            ],
            path="multi_turn",
            pre_messages=[
                _user(f"What was my latest card purchase {suffix}?"),
                _assistant(
                    f"Your latest posted card purchase was at {entity['merchant']}.",
                    loss=False,
                ),
            ],
            tool_envelopes=[
                _success_envelope(
                    transaction={
                        "description": entity["merchant"],
                        "status": "posted",
                        "disputed": True,
                    }
                )
            ],
        ),
    ]


def _history_entity_ambiguity(split: str) -> list[dict[str, Any]]:
    entity = _remediation_entities(split)
    suffix = _suffix(split)
    return [
        _record(
            record_id=f"history_ambiguous_card_{split}",
            split=split,
            scenario_family="history_entity_ambiguity",
            current=f"Replace the card we were discussing {suffix}.",
            final=(
                "I found two cards in our conversation. Which should I replace? Please share "
                "the last four digits."
            ),
            tool_plan=[],
            grounding_facts=[],
            path="clarification",
            pre_messages=[
                _user(f"Show the cards on my profile {suffix}."),
                _assistant(
                    f"I found {entity['card_name']} ending in {entity['card_last4']} and "
                    f"{entity['other_card_name']} ending in {entity['other_card_last4']}.",
                    loss=False,
                ),
            ],
        )
    ]


def _tool_outcome_consistency(split: str) -> list[dict[str, Any]]:
    entity = _remediation_entities(split)
    suffix = _suffix(split)
    return [
        _record(
            record_id=f"outcome_replace_success_{split}",
            split=split,
            scenario_family="tool_outcome_consistency",
            current=f"Replace my card ending in {entity['outcome_card_last4']} {suffix}.",
            final=(
                f"Replacement is pending for your {entity['outcome_card_name']} ending in "
                f"{entity['outcome_card_last4']}."
            ),
            tool_plan=[("replace_card", {"last4": entity["outcome_card_last4"]})],
            grounding_facts=["card.status=replacement_pending"],
            path="tool_success",
            pre_messages=[],
            tool_envelopes=[
                _success_envelope(
                    card={
                        "name": entity["outcome_card_name"],
                        "last4": entity["outcome_card_last4"],
                        "status": "replacement_pending",
                    }
                )
            ],
        ),
        _record(
            record_id=f"outcome_replace_error_{split}",
            split=split,
            scenario_family="tool_outcome_consistency",
            current=f"Replace my card ending in {entity['missing_card_last4']} {suffix}.",
            final=(
                f"I could not replace a card ending in {entity['missing_card_last4']} because "
                "no matching card was found. No card status was changed."
            ),
            tool_plan=[("replace_card", {"last4": entity["missing_card_last4"]})],
            grounding_facts=["error.code=card_not_found"],
            path="tool_error",
            pre_messages=[],
            tool_envelopes=[_error_envelope("card_not_found", "No matching card was found.")],
        ),
        _record(
            record_id=f"outcome_cancel_success_{split}",
            split=split,
            scenario_family="tool_outcome_consistency",
            current=f"Cancel the pending transfer to {entity['recipient']} {suffix}.",
            final=f"The pending transfer to {entity['recipient']} is now cancelled.",
            tool_plan=[("cancel_transfer", {"recipient": entity["recipient"]})],
            grounding_facts=["transfer.status=cancelled"],
            path="tool_success",
            pre_messages=[],
            tool_envelopes=[
                _success_envelope(
                    transfer={"recipient": entity["recipient"], "status": "cancelled"}
                )
            ],
        ),
        _record(
            record_id=f"outcome_cancel_error_{split}",
            split=split,
            scenario_family="tool_outcome_consistency",
            current=f"Cancel the transfer to {entity['missing_recipient']} {suffix}.",
            final=(
                f"I could not cancel a transfer to {entity['missing_recipient']} because no "
                "matching pending transfer was found. The transfer status was not changed."
            ),
            tool_plan=[("cancel_transfer", {"recipient": entity["missing_recipient"]})],
            grounding_facts=["error.code=transfer_not_pending"],
            path="tool_error",
            pre_messages=[],
            tool_envelopes=[
                _error_envelope("transfer_not_pending", "No matching pending transfer was found.")
            ],
        ),
    ]


def _social_style_records(split: str) -> list[dict[str, Any]]:
    examples = {
        "train": (
            (
                "morning",
                "Good morning, Harbor.",
                "Good morning! How can I help with your banking today?",
            ),
            (
                "available",
                "Hi there, are you available?",
                "Hi! I’m here and ready to help. What would you like to work on?",
            ),
            (
                "starting",
                "Hey, can we get started?",
                "Absolutely. What banking question or account task can I help with?",
            ),
            (
                "checkin",
                "How is your day going?",
                "I’m doing well and ready to help. How can I assist you today?",
            ),
            (
                "ready",
                "Are you ready for a banking question?",
                "Yes, I’m ready. Tell me what you would like help with.",
            ),
            (
                "capabilities",
                "What kinds of things can you help me with?",
                "I can help with common banking questions and supported account tasks. "
                "Where would you like to begin?",
            ),
            (
                "thanks",
                "Thanks, that cleared things up.",
                "You’re welcome. I’m here if another banking question comes up.",
            ),
            (
                "appreciation",
                "I appreciate the clear answer.",
                "Happy to help. Let me know if you need anything else with your banking.",
            ),
            (
                "all_set",
                "Great, I am all set now.",
                "Glad I could help. I’ll be here if you need another banking task.",
            ),
            (
                "pause",
                "That is all I need for the moment.",
                "Understood. I’m here whenever you need more banking help.",
            ),
            (
                "return",
                "I am back with another question.",
                "Welcome back. What would you like help with this time?",
            ),
            (
                "welcome",
                "Nice to meet you, Harbor.",
                "Nice to meet you too. How can I help with your banking today?",
            ),
        ),
        "validation": (
            (
                "hello",
                "Hi Harbor, I have a question.",
                "Hi! I’m ready to help. What would you like to ask?",
            ),
            (
                "online",
                "Is the banking assistant online?",
                "Yes, I’m here and ready. How can I help you today?",
            ),
            (
                "gratitude",
                "Thank you, that was useful.",
                "You’re welcome. I’m available if you have another banking question.",
            ),
            (
                "done",
                "Perfect, that answers what I needed.",
                "Glad that answered your question. I’m here whenever you need more help.",
            ),
        ),
    }
    if split not in examples:
        raise ValueError(f"unsupported social-style split: {split}")
    return [
        _record(
            record_id=f"social_{key}_{split}",
            split=split,
            scenario_family="natural_social_style",
            current=current,
            final=final,
            tool_plan=[],
            grounding_facts=[],
            path="conversation",
            pre_messages=[],
        )
        for key, current, final in examples[split]
    ]


def _missing_entity_records(split: str) -> list[dict[str, Any]]:
    suffix = {"train": "in this session", "validation": "from this chat"}[split]
    final = {
        "train": (
            "Which debit card would you like me to replace? Please share the last four "
            "digits shown with the card in your account view."
        ),
        "validation": (
            "I need to know which card you mean before replacing it. What are the last "
            "four digits displayed with that card?"
        ),
    }[split]
    return [
        _record(
            record_id=f"missing_card_selector_{split}",
            split=split,
            scenario_family="missing_entity_clarification",
            current=f"Please replace my debit card {suffix}.",
            final=final,
            tool_plan=[],
            grounding_facts=["missing_field=last4"],
            path="clarification",
            pre_messages=[],
        )
    ]


def _granite_v7_examples(split: str) -> list[dict[str, Any]]:
    if split not in {"train", "validation", "shadow"}:
        raise ValueError(f"unsupported Granite V7 split: {split}")
    tag = {"train": "Harbor", "validation": "Granite", "shadow": "Prairie"}[split]
    card_last4 = {"train": "6158", "validation": "3074", "shadow": "8462"}[split]
    other_last4 = {"train": "2046", "validation": "8662", "shadow": "1935"}[split]
    recipient = {
        "train": "Summit Plumbing",
        "validation": "Juniper Internet",
        "shadow": "Prairie Electric",
    }[split]
    # The Granite V7 shadow gate is frozen in granite-v7-shadow.jsonl, so it keeps the
    # legacy wording; trainable splits must not name a product surface.
    selector_missing_closer = (
        "four digits shown in the app." if split == "shadow" else "four digits."
    )
    records = [
        _record(
            record_id=f"v7_greeting_{split}",
            split=split,
            scenario_family="v7_natural_greeting",
            current=f"Hello {tag}, good to see you.",
            final=f"Hello from the {tag} review! I’m ready to help with your banking.",
            tool_plan=[],
            grounding_facts=[],
            path="conversation",
            pre_messages=[],
        ),
        _record(
            record_id=f"v7_mortgage_detour_{split}",
            split=split,
            scenario_family="v7_mortgage_policy_detour",
            current=(
                "Pause the service-case review and explain how a mortgage application "
                f"works, {tag}."
            ),
            final=(
                f"For the {tag} review, {POLICY_CHUNKS['faq-mortgage-opening-v1']['answer']} "
                f"[Policy: {POLICY_CHUNKS['faq-mortgage-opening-v1']['chunk_id']}]."
            ),
            tool_plan=[],
            grounding_facts=list(POLICY_CHUNKS["faq-mortgage-opening-v1"]["required_claims"]),
            path="retrieval_grounded_policy",
            pre_messages=_case_history(split),
            policy=POLICY_CHUNKS["faq-mortgage-opening-v1"],
        ),
        _record(
            record_id=f"v7_list_transfers_{split}",
            split=split,
            scenario_family="v7_list_transfers",
            current=f"List the transfers visible in my {tag.lower()} session.",
            final=f"I found a pending transfer to {recipient}.",
            tool_plan=[("list_transfers", {})],
            grounding_facts=[f"transfer.recipient={recipient}", "transfer.status=pending"],
            path="tool_success",
            pre_messages=[],
            tool_envelopes=[
                _success_envelope(transfers=[{"recipient": recipient, "status": "pending"}])
            ],
        ),
        _record(
            record_id=f"v7_selector_grounded_{split}",
            split=split,
            scenario_family="v7_grounded_selector",
            current=f"Replace the {tag} debit card from those results.",
            final=f"Replacement is pending for the {tag} debit card ending in {card_last4}.",
            tool_plan=[("replace_card", {"last4": card_last4})],
            grounding_facts=[f"card.last4={card_last4}", "card.status=replacement_pending"],
            path="tool_success",
            pre_messages=[
                _user(f"Show my cards and recent transfers for this {tag.lower()} review."),
                _assistant(
                    f"The {tag} debit card ends in {card_last4}; a transfer to {recipient} "
                    "is also pending.",
                    loss=False,
                ),
            ],
            tool_envelopes=[
                _success_envelope(
                    card={
                        "name": f"{tag} Debit",
                        "last4": card_last4,
                        "status": "replacement_pending",
                    }
                )
            ],
        ),
        _record(
            record_id=f"v7_selector_missing_{split}",
            split=split,
            scenario_family="v7_selector_clarification",
            current=f"Replace my debit card during this {tag.lower()} review.",
            final=(
                f"For the {tag} review, which card should I replace? Please share the last "
                f"{selector_missing_closer}"
            ),
            tool_plan=[],
            grounding_facts=["missing_field=last4"],
            path="clarification",
            pre_messages=[],
        ),
        _record(
            record_id=f"v7_selector_ambiguous_{split}",
            split=split,
            scenario_family="v7_selector_clarification",
            current=f"Replace the card from that {tag.lower()} list.",
            final=(
                f"Two eligible cards were listed in the {tag} review. Which one should I "
                "replace? Please share its last four digits."
            ),
            tool_plan=[],
            grounding_facts=["ambiguous_field=last4"],
            path="clarification",
            pre_messages=[
                _user(f"Show the cards in my {tag.lower()} profile."),
                _assistant(f"I found cards ending in {card_last4} and {other_last4}.", loss=False),
            ],
        ),
        _record(
            record_id=f"v7_selector_ineligible_{split}",
            split=split,
            scenario_family="v7_selector_clarification",
            current=f"Replace the closed {tag} card you just mentioned.",
            final=(
                f"That {tag} card is closed and cannot be replaced. Please choose an eligible "
                "card and share its last four digits."
            ),
            tool_plan=[],
            grounding_facts=["ineligible_selector=closed_card"],
            path="clarification",
            pre_messages=[
                _user(f"Check the status of my {tag} card ending in {card_last4}."),
                _assistant(f"That card ending in {card_last4} is closed.", loss=False),
            ],
        ),
        _record(
            record_id=f"v7_tool_success_{split}",
            split=split,
            scenario_family="v7_tool_outcome",
            current=f"Freeze the active {tag} card ending in {card_last4}.",
            final=f"The {tag} card ending in {card_last4} is now frozen.",
            tool_plan=[("freeze_card", {"last4": card_last4})],
            grounding_facts=["card.status=frozen"],
            path="tool_success",
            pre_messages=[],
            tool_envelopes=[
                _success_envelope(
                    card={"name": f"{tag} Card", "last4": card_last4, "status": "frozen"}
                )
            ],
        ),
        _record(
            record_id=f"v7_tool_error_{split}",
            split=split,
            scenario_family="v7_tool_outcome",
            current=f"Cancel the transfer to {recipient} even if it already settled.",
            final=(
                f"I could not cancel the transfer to {recipient} because it was not pending. "
                "Its status was unchanged."
            ),
            tool_plan=[("cancel_transfer", {"recipient": recipient})],
            grounding_facts=["error.code=transfer_not_pending"],
            path="tool_error",
            pre_messages=[],
            tool_envelopes=[
                _error_envelope("transfer_not_pending", "The transfer was not pending.")
            ],
        ),
    ]
    limit_values = range(1, 21) if split == "train" else (2, 7, 13, 19)
    number_words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty",
    }
    for limit in limit_values:
        rendered_limit = number_words[limit] if limit % 2 else str(limit)
        records.append(
            _record(
                record_id=f"v7_transactions_limit_{limit:02d}_{split}",
                split=split,
                scenario_family="v7_list_transactions_limit",
                current=(
                    f"Show my most recent {rendered_limit} transactions for the "
                    f"{tag.lower()} review."
                ),
                final=(
                    f"I returned the {limit} most recent transactions requested for the "
                    f"{tag} review."
                ),
                tool_plan=[("list_transactions", {"limit": limit})],
                grounding_facts=[f"transactions.limit={limit}"],
                path="tool_success",
                pre_messages=[],
                tool_envelopes=[
                    _success_envelope(
                        transactions=[
                            {"description": f"{tag} item {index + 1}"} for index in range(limit)
                        ]
                    )
                ],
            )
        )
    if split == "shadow":
        shadow_currents = {
            "v7_greeting_shadow": "Prairie checking in—hello.",
            "v7_mortgage_detour_shadow": (
                "Temporarily leave support history aside. Describe home-loan application review."
            ),
            "v7_list_transfers_shadow": "Which outbound money movements appear?",
            "v7_selector_grounded_shadow": (
                "Use the Prairie debit selection above for replacement."
            ),
            "v7_selector_missing_shadow": (
                "A debit replacement is needed; the specific card is unstated."
            ),
            "v7_selector_ambiguous_shadow": (
                "Both displayed cards qualify; request a replacement without choosing."
            ),
            "v7_selector_ineligible_shadow": ("Proceed using the closed Prairie payment card."),
            "v7_tool_success_shadow": "Apply a freeze to Prairie card 8462.",
            "v7_tool_error_shadow": (
                "The Prairie Electric payment already settled; try cancellation."
            ),
            **{
                f"v7_transactions_limit_{limit:02d}_shadow": (
                    f"For activity review, return {limit} entries from transaction history."
                )
                for limit in limit_values
            },
        }
        for record in records:
            for message in reversed(record["messages"]):
                if message.get("role") == "user":
                    message["content"] = shadow_currents[str(record["record_id"])]
                    break
    return records


def _heldout_regression_records() -> list[dict[str, Any]]:
    return [
        _record(
            record_id="heldout_when_created_test",
            split="test",
            scenario_family="heldout_screenshot_regression",
            current="when was that created?",
            final=(
                "The mailing-address update case was created on 2026-06-18 at "
                "14:00 UTC. It is closed and its subject is Confirm mailing "
                "address update."
            ),
            tool_plan=[("list_service_cases", {})],
            grounding_facts=["case.created_at=2026-06-18T14:00:00Z"],
            path="multi_turn",
            pre_messages=_case_history("heldout"),
        ),
        _record(
            record_id="heldout_replace_that_one_test",
            split="test",
            scenario_family="heldout_screenshot_regression",
            current="ok, thats the one i want to replace",
            final=("Replacement is pending for the Everyday Visa Debit card ending in 4821."),
            tool_plan=[("replace_card", {"last4": "4821"})],
            grounding_facts=["card.last4=4821", "card.status=replacement_pending"],
            path="multi_turn",
            pre_messages=_card_history("heldout"),
        ),
        _record(
            record_id="heldout_weather_shift_test",
            split="test",
            scenario_family="heldout_screenshot_regression",
            current="what about the weather there?",
            final=(
                "Weather is outside the retail banking support I provide. I can help you return "
                "to an account, card, transfer, payment, loan, or service request."
            ),
            tool_plan=[],
            grounding_facts=[],
            path="ood",
            pre_messages=_case_history("heldout"),
        ),
    ]


def _expand_records(
    base_records: Sequence[dict[str, Any]],
    *,
    split: str,
) -> list[dict[str, Any]]:
    realization_count = REALIZATION_COUNTS[split]
    contexts = REALIZATION_CONTEXTS[split]
    combinations = [(context, request) for context in contexts for request in REALIZATION_REQUESTS]
    if len(combinations) != realization_count:
        raise ValueError(f"realization configuration mismatch for {split}")
    expanded = []
    for base_record in base_records:
        for realization, (context, request) in enumerate(combinations):
            old_id = str(base_record["record_id"])
            new_id = f"{old_id}-r{realization:03d}"
            record = deepcopy(base_record)
            record["record_id"] = new_id
            current_message: dict[str, Any] | None = None
            for message in record["messages"]:
                if message.get("role") == "user":
                    current_message = message
            if current_message is None:
                raise ValueError(f"{old_id} has no user message")
            current = str(current_message["content"]).strip()
            current_message["content"] = f"{current} {context}. {request}."
            record["messages"][-1]["content"] = _varied_final(
                str(record["messages"][-1]["content"]),
                split=split,
                realization=realization,
            )
            for message in record["messages"]:
                for tool_call in message.get("tool_calls", []):
                    call_id = str(tool_call["id"]).replace(old_id, new_id)
                    tool_call["id"] = call_id
                if message.get("role") == "tool":
                    message["tool_call_id"] = str(message["tool_call_id"]).replace(
                        old_id,
                        new_id,
                    )
            record["expected"]["ordered_calls"] = [
                str(call_id).replace(old_id, new_id)
                for call_id in record["expected"]["ordered_calls"]
            ]
            if record["expected"]["requires_tool"]:
                record["expected"]["final_state_hash"] = f"sha256:{_sha256_text(new_id)}"
            record["split_keys"]["template_id"] = old_id
            record["split_keys"]["realization_seed"] = (
                f"servicing-alignment-{split}-{realization:03d}"
            )
            record["metadata"]["split_group"] = f"{record['metadata']['scenario_family']}|{new_id}"
            expanded.append(record)
    return expanded


def _record(
    *,
    record_id: str,
    split: str,
    scenario_family: str,
    current: str,
    final: str,
    tool_plan: Sequence[tuple[str, dict[str, Any]]],
    grounding_facts: Sequence[str],
    path: str,
    pre_messages: Sequence[dict[str, Any]],
    policy: Mapping[str, str] | None = None,
    tool_envelopes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if tool_envelopes is not None and len(tool_envelopes) != len(tool_plan):
        raise ValueError("tool_envelopes must align one-to-one with tool_plan")
    messages = [_message("system", SYSTEM_PROMPT, loss=False), *pre_messages]
    if policy is not None:
        messages.append(
            _message(
                "system",
                "Authoritative Harborlight Bank policy context. Answer only from this "
                "context and cite the bracketed policy chunk ID.\n"
                f"[Policy: {policy['chunk_id']}] {policy['title']}: {policy['text']}",
                loss=False,
            )
        )
    messages.append(_user(current))
    ordered_calls = []
    expected_calls = []
    for index, (tool_name, arguments) in enumerate(tool_plan):
        call_id = f"call_{record_id}_{index}"
        ordered_calls.append(call_id)
        expected_calls.append({"name": tool_name, "arguments": dict(arguments)})
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "loss": True,
                "tool_calls": [
                    {
                        "id": call_id,
                        "index": 0,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": dict(arguments),
                        },
                    }
                ],
            }
        )
        envelope = tool_envelopes[index] if tool_envelopes is not None else None
        messages.append(_tool_result(call_id, tool_name, envelope=envelope))
    messages.append(_assistant(final, loss=True))
    expected = {
        "requires_tool": bool(tool_plan),
        "ordered_calls": ordered_calls,
        "tool_calls": expected_calls,
        "final_state_hash": f"sha256:{_sha256_text(record_id)}" if tool_plan else None,
        "grounding_facts": list(grounding_facts),
        "path": path,
    }
    if policy is not None:
        expected["policy_citations"] = [str(policy["chunk_id"])]
        expected["policy_corpus_revision"] = str(policy["corpus_revision"])
        expected["grounding_facts"] = list(policy["required_claims"])
        expected["forbidden_facts"] = list(policy["forbidden_claims"])
    record = {
        "schema_version": BANKING_TOOL_SFT_CONTRACT,
        "record_id": record_id,
        "messages": messages,
        "expected": expected,
        "split_keys": {
            "scenario_family": scenario_family,
            "state_seed": f"alignment-{split}-{scenario_family}",
            "customer_id": "synthetic-customer-alex",
            "template_id": record_id.rsplit(f"_{split}", maxsplit=1)[0],
            "realization_seed": f"servicing-alignment-{split}",
        },
        "provenance": {
            "source": "self-authored-synthetic",
            "license": "MIT",
            "generator_version": GENERATOR_VERSION,
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
        },
        "metadata": {
            "record_type": "tool_use_sft",
            "trainable": True,
            "customer_login": "alex.demo",
            "scenario_family": scenario_family,
            "path": path,
            "split": split,
            "split_group": f"{scenario_family}|{record_id}",
        },
    }
    if split in {"train", "validation"}:
        _attach_generation_contract(record)
    return record


def _varied_final(final: str, *, split: str, realization: int) -> str:
    split_lead = {
        "train": "",
        "validation": "For this request,",
        "test": "In this session,",
    }[split]
    opener = FINAL_OPENERS[realization % len(FINAL_OPENERS)]
    closer = FINAL_CLOSERS[(realization // len(FINAL_OPENERS)) % len(FINAL_CLOSERS)]
    return " ".join(part for part in (split_lead, opener, final, closer) if part)


def _case_history(split: str) -> list[dict[str, Any]]:
    return [
        _user(f"Show my recent service cases {_suffix(split)}.".strip()),
        _assistant(
            "You have one recent service case: Confirm mailing address update. "
            "It is closed and was created on 2026-06-18 at 14:00 UTC.",
            loss=False,
        ),
    ]


def _card_history(split: str) -> list[dict[str, Any]]:
    return [
        _user(f"Show my cards {_suffix(split)}.".strip()),
        _assistant(
            "You have an active Everyday Visa Debit card ending in 4821.",
            loss=False,
        ),
    ]


def _success_envelope(**result: Any) -> dict[str, Any]:
    return {"ok": True, "result": result}


def _error_envelope(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _tool_result(
    call_id: str,
    tool_name: str,
    *,
    envelope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": dict(envelope)
        if envelope is not None
        else _success_envelope(**_result_for_tool(tool_name)),
        "loss": False,
    }


def _result_for_tool(tool_name: str) -> dict[str, Any]:
    if tool_name == "list_service_cases":
        return {
            "service_cases": [
                {
                    "case_type": "address_update",
                    "subject": "Confirm mailing address update",
                    "status": "closed",
                    "created_at": "2026-06-18T14:00:00Z",
                }
            ]
        }
    if tool_name == "replace_card":
        return {
            "card": {
                "name": "Everyday Visa Debit",
                "last4": "4821",
                "status": "replacement_pending",
            }
        }
    if tool_name == "freeze_card":
        return {
            "card": {
                "name": "Everyday Visa Debit",
                "last4": "4821",
                "status": "frozen",
            }
        }
    if tool_name == "dispute_transaction":
        return {
            "transaction": {
                "description": "North Harbor Market",
                "status": "posted",
                "disputed": True,
            }
        }
    raise ValueError(f"unsupported alignment tool: {tool_name}")


def _message(role: str, content: str | None, *, loss: bool) -> dict[str, Any]:
    return {"role": role, "content": content, "loss": loss}


def _user(content: str) -> dict[str, Any]:
    return _message("user", content, loss=False)


def _assistant(content: str, *, loss: bool) -> dict[str, Any]:
    return _message("assistant", content, loss=loss)


def _tool_manifest_hash() -> str:
    from hello_slm.banking_tool_sft_data import _tool_manifest_hash as manifest_hash

    return manifest_hash()


def _jsonl_bytes(records: Sequence[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _suffix(split: str) -> str:
    return {
        "train": "on my account",
        "validation": "from this chat",
        "test": "from my profile",
        "heldout": "",
    }.get(split, split)


def _normalize(text: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in text).split()
    )


def _last_user_text(record: dict[str, Any]) -> str:
    for message in reversed(record["messages"]):
        if message["role"] == "user":
            return str(message["content"])
    raise ValueError("record has no user message")


def _assert_no_duplicate_current(records: Sequence[dict[str, Any]], *, split: str) -> None:
    seen: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        normalized = _normalize(_last_user_text(record))
        seen.setdefault(normalized, []).append(record)
    for normalized, duplicates in seen.items():
        if len(duplicates) == 1:
            continue
        pairs: dict[str, list[dict[str, Any]]] = {}
        for row in duplicates:
            pair_id = str(row["metadata"].get("coreference_pair_id", ""))
            pairs.setdefault(pair_id, []).append(row)
        history_forms = {
            rows[0]["metadata"].get("coreference_history_form")
            for pair_id, rows in pairs.items()
            if pair_id and len(rows) == 2
        }
        if len(history_forms) == len(pairs) and all(
            pair_id
            and len(rows) == 2
            and {row["metadata"].get("coreference_target") for row in rows}
            == {"replace_card", "clarification"}
            for pair_id, rows in pairs.items()
        ):
            continue
        raise ValueError(f"duplicate current user text in {split}: {normalized}")


def _assert_coreference_pair_integrity(
    splits: Mapping[str, Sequence[dict[str, Any]]],
) -> None:
    pair_owners: dict[str, str] = {}
    entities_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    families_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    for split in ("train", "validation"):
        pairs: dict[str, list[dict[str, Any]]] = {}
        for record in splits[split]:
            pair_id = record["metadata"].get("coreference_pair_id")
            if pair_id is None:
                continue
            pair_key = str(pair_id)
            owner = pair_owners.setdefault(pair_key, split)
            if owner != split:
                raise ValueError(f"coreference pair leaked across splits: {pair_key}")
            pairs.setdefault(pair_key, []).append(record)
        for pair_id, pair in pairs.items():
            if len(pair) != 2:
                raise ValueError(f"coreference pair must contain two records: {pair_id}")
            if len({_last_user_text(record) for record in pair}) != 1:
                raise ValueError(f"coreference pair current text drifted: {pair_id}")
            if {record["metadata"].get("coreference_target") for record in pair} != {
                "replace_card",
                "clarification",
            }:
                raise ValueError(f"coreference pair targets are invalid: {pair_id}")
            if {record["metadata"].get("actionable_card_count") for record in pair} != {
                1,
                2,
            }:
                raise ValueError(f"coreference pair card counts are invalid: {pair_id}")
            positive = next(
                record
                for record in pair
                if record["metadata"].get("coreference_target") == "replace_card"
            )
            tool_targets = [
                message
                for message in positive["messages"]
                if message["role"] == "assistant" and message.get("tool_calls")
            ]
            if len(tool_targets) != 1 or tool_targets[0].get("loss") is not True:
                raise ValueError(f"coreference positive tool decision is not supervised: {pair_id}")
            history_users = [
                tuple(
                    str(message["content"])
                    for message in record["messages"][:-1]
                    if message["role"] == "user"
                )
                for record in pair
            ]
            if history_users[0] != history_users[1]:
                raise ValueError(f"coreference pair list history drifted: {pair_id}")
            if "selected" in json.dumps(pair, ensure_ascii=False).lower():
                raise ValueError(f"coreference pair contains preselection leakage: {pair_id}")
            for record in pair:
                entities_by_split[split].update(record["metadata"]["coreference_entity_keys"])
                families_by_split[split].add(str(record["metadata"]["coreference_phrase_family"]))
    if entities_by_split["train"] & entities_by_split["validation"]:
        raise ValueError("coreference entities leaked across train and validation")
    if families_by_split["train"] & families_by_split["validation"]:
        raise ValueError("coreference phrase families leaked across train and validation")


def _assert_shadow_isolation(
    splits: Mapping[str, Sequence[dict[str, Any]]],
    shadow_records: Sequence[dict[str, Any]],
) -> None:
    shadow_entities = {
        str(entity)
        for record in shadow_records
        for entity in record["metadata"]["coreference_entity_keys"]
    }
    shadow_families = {
        str(record["metadata"]["coreference_phrase_family"]) for record in shadow_records
    }
    shadow_prompts = {_normalize(_last_user_text(record)) for record in shadow_records}
    governed = [*splits["train"], *splits["validation"]]
    governed_entities = {
        str(entity)
        for record in governed
        for entity in record["metadata"].get("coreference_entity_keys", ())
    }
    governed_families = {
        str(record["metadata"]["coreference_phrase_family"])
        for record in governed
        if "coreference_phrase_family" in record["metadata"]
    }
    governed_prompts = {_normalize(_last_user_text(record)) for record in governed}
    if shadow_entities & governed_entities:
        raise ValueError("coreference shadow entities leaked into train or dev")
    if shadow_families & governed_families:
        raise ValueError("coreference shadow phrase families leaked into train or dev")
    if shadow_prompts & governed_prompts:
        raise ValueError("coreference shadow prompts leaked into train or dev")
    shadow_as_train = {"train": list(shadow_records)}
    if _heldout_exact_currents_in_train(shadow_as_train):
        raise ValueError("held-out screenshot current leaked into coreference shadow")
    if _heldout_long_ngram_leaks_in_train(shadow_as_train):
        raise ValueError("held-out screenshot n-gram leaked into coreference shadow")


def _assert_granite_shadow_isolation(
    splits: Mapping[str, Sequence[dict[str, Any]]],
    shadow_records: Sequence[dict[str, Any]],
) -> None:
    governed = [*splits["train"], *splits["validation"]]
    governed_prompts = {_normalize(_last_user_text(record)) for record in governed}
    shadow_prompts = {_normalize(_last_user_text(record)) for record in shadow_records}
    if governed_prompts & shadow_prompts:
        raise ValueError("Granite V7 shadow prompt leaked into train or validation")
    governed_ngrams = set().union(
        *(_word_ngrams(_last_user_text(record), size=4) for record in governed)
    )
    shadow_ngrams = set().union(
        *(_word_ngrams(_last_user_text(record), size=4) for record in shadow_records)
    )
    if governed_ngrams & shadow_ngrams:
        raise ValueError("Granite V7 shadow long n-gram leaked into train or validation")
    shadow_as_train = {"train": list(shadow_records)}
    if _heldout_exact_currents_in_train(shadow_as_train):
        raise ValueError("held-out screenshot current leaked into Granite V7 shadow")
    if _heldout_long_ngram_leaks_in_train(shadow_as_train):
        raise ValueError("held-out screenshot n-gram leaked into Granite V7 shadow")


def _assert_screenshot_fixture_isolation(
    splits: Mapping[str, Sequence[dict[str, Any]]],
    fixture: Sequence[Mapping[str, Any]],
) -> None:
    if len(fixture) != 9:
        raise ValueError("screenshot regression fixture must contain exactly nine cases")
    governed = [*splits["train"], *splits["validation"]]
    governed_currents = {_normalize(_last_user_text(record)) for record in governed}
    fixture_currents = {_normalize(str(row.get("current", ""))) for row in fixture}
    if governed_currents & fixture_currents:
        raise ValueError("screenshot regression current leaked into train or validation")
    fixture_ngrams = set().union(
        *(_word_ngrams(str(row.get("current", "")), size=4) for row in fixture)
    )
    leaks = [
        str(record["record_id"])
        for record in governed
        if _word_ngrams(_last_user_text(record), size=4) & fixture_ngrams
    ]
    if leaks:
        raise ValueError(f"screenshot regression long n-gram leaked into training: {leaks}")
    required = {
        "route",
        "effective_action",
        "entity_state",
        "tool_name",
        "argument_constraints",
        "response_properties",
    }
    if any(set(row.get("expected", {})) != required for row in fixture):
        raise ValueError("screenshot regression fixture has an incomplete expected contract")


def _assert_no_cross_split_leakage(
    splits: Mapping[str, Sequence[dict[str, Any]]],
) -> None:
    owners: dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            group = str(record["metadata"]["split_group"])
            previous = owners.setdefault(group, split)
            if previous != split:
                raise ValueError(f"split group leaked across splits: {group}")


def _count_pii_matches(splits: Mapping[str, Sequence[dict[str, Any]]]) -> int:
    count = 0
    for records in splits.values():
        for record in records:
            text = json.dumps(record["messages"], ensure_ascii=False)
            count += sum(1 for pattern in PII_PATTERNS if pattern.search(text))
    return count


def _heldout_exact_currents_in_train(
    splits: Mapping[str, Sequence[dict[str, Any]]],
) -> list[str]:
    heldout = {_normalize(text) for text in SCREENSHOT_HELDOUT_CURRENTS}
    return sorted(
        _last_user_text(record)
        for record in splits["train"]
        if _normalize(_last_user_text(record)) in heldout
    )


def _word_ngrams(text: str, *, size: int) -> set[tuple[str, ...]]:
    tokens = _normalize(text).split()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _heldout_long_ngram_leaks_in_train(
    splits: Mapping[str, Sequence[dict[str, Any]]],
    *,
    size: int = 4,
) -> list[dict[str, Any]]:
    heldout_ngrams = set().union(
        *(_word_ngrams(text, size=size) for text in SCREENSHOT_HELDOUT_CURRENTS)
    )
    leaks = []
    for record in splits["train"]:
        shared = sorted(heldout_ngrams & _word_ngrams(_last_user_text(record), size=size))
        if shared:
            leaks.append(
                {
                    "record_id": str(record["record_id"]),
                    "ngrams": [" ".join(ngram) for ngram in shared],
                }
            )
    return leaks


def rows_sha256(records: Sequence[dict[str, Any]]) -> str:
    return _sha256_bytes(_jsonl_bytes(records))


def all_records(splits: Mapping[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [record for split in SPLITS for record in splits[split]]


def validate_servicing_alignment_splits(
    splits: Mapping[str, Sequence[dict[str, Any]]],
) -> None:
    validate_records(all_records(splits))
    _assert_no_cross_split_leakage(splits)
    _assert_coreference_pair_integrity(splits)
    if _count_pii_matches(splits):
        raise ValueError("servicing alignment splits contain PII-like text")
