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
    ALLOWED_ARGS,
    BANKING_TOOL_SFT_CONTRACT,
    GENERATION_CONTRACT_VERSION,
    NO_TOOL_OOD_RESPONSE,
    POLICY_CHUNKS,
    SYSTEM_PROMPT,
    _attach_generation_contract,
    export_teacher_realization_requests,
    import_teacher_realizations,
    load_canonical_policy_corpus,
    normalized_user_text,
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
    re.compile(r"\b(?:\d[ -]?){12,}\b"),
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
        *_deictic_replace_reinforcement_curriculum("train"),
        *_deictic_ineligible_curriculum("train"),
        *_missing_entity_records("train"),
        *_social_style_records("train"),
        *_granite_v7_examples("train"),
        *_long_context_tool_fidelity("train"),
        *_policy_alignment_curriculum("train"),
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
        *_long_context_tool_fidelity("validation"),
        *_policy_alignment_curriculum("validation"),
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


def _deictic_replace_reinforcement_curriculum(split: str) -> list[dict[str, Any]]:
    """Train-only sole-card replace reinforcement in the STATUS-ANSWER context.

    The v11 adapter regressed on exactly one deictic probe: after answering a
    card-status question about the customer's single card, "Please replace that
    one." decoded a fresh list plus the clarify template instead of the
    replacement. Every sole-card history `_deictic_replace_curriculum` trains is
    list-shaped ("Cards on this profile: ...", "Card results: ..."), so the
    failing context — a status answer — was uncovered, and the model fell back
    to its most-repeated trained final (the pinned clarify template). This
    builder covers that context directly: same paired action/ambiguity contrast,
    same pinned clarify template on the ambiguity side, but the sole-card and
    two-card histories are status answers. It is a separate builder (modeled on
    `_deictic_ineligible_curriculum` below) rather than extra parent specs
    because the parent assigns products by ``index % len(products)`` — growing
    its spec list would silently reshuffle card names across every existing
    deictic row. The scenario families are new and registered SFT-only, so the
    parked router corpus ingests nothing from here. The frozen validation and
    shadow gates are not extended.
    """
    if split != "train":
        raise ValueError(f"unsupported deictic reinforcement split: {split}")
    prompt_forms = (
        "{prompt}",
        "please {prompt}",
        "{prompt} please",
        "yes {prompt}",
    )
    sole_card_history_forms = (
        "Your {card_name} ending in {card_last4} is active and available for use.",
        "The {card_name} ending in {card_last4} is currently active.",
        "Status check: {card_name} ending in {card_last4} is active.",
        "Your {card_name} ending in {card_last4} shows an active status.",
    )
    multiple_card_history_forms = (
        (
            "Your {card_name} ending in {card_last4} and your {other_card_name} "
            "ending in {other_card_last4} are both active."
        ),
        (
            "The {card_name} ending in {card_last4} is active; the "
            "{other_card_name} ending in {other_card_last4} is active as well."
        ),
        (
            "Status check: {card_name} ending in {card_last4} active, "
            "{other_card_name} ending in {other_card_last4} active."
        ),
        (
            "Your {card_name} ending in {card_last4} shows active, and your "
            "{other_card_name} ending in {other_card_last4} shows active too."
        ),
    )
    tiers = ("Everyday Debit", "Rewards Debit", "Travel Debit", "Cashback Debit")
    specs = (
        {"phrase_family": "one-shown-status", "prompt": "replace the one you showed"},
        {"phrase_family": "need-replaced-status", "prompt": "i need that card replaced"},
        {"phrase_family": "order-new-status", "prompt": "order a new card to replace it"},
        {
            "phrase_family": "needs-replacement-status",
            "prompt": "that card needs a replacement",
        },
    )
    # A pool wider than the spec list, decoupled from it: every family's sixteen
    # combos walk the full pool, matching the parent curriculum's per-family
    # product diversity. All names are unused by any other curriculum.
    products = (
        "Tarn",
        "Frost",
        "Glen",
        "Wren",
        "Heath",
        "Crag",
        "Fen",
        "Loch",
        "Gorse",
        "Knoll",
        "Cairn",
        "Firth",
    )
    number_base = 6900
    other_number_base = 8900
    history_user = "Check the status of my card."
    records: list[dict[str, Any]] = []
    pair_index = 0
    for family_index, spec in enumerate(specs):
        for prompt_index in range(len(prompt_forms)):
            for history_form in range(len(sole_card_history_forms)):
                pair_index += 1
                family = spec["phrase_family"]
                prompt = prompt_forms[prompt_index].format(prompt=spec["prompt"])
                product = products[
                    (family_index + (3 * prompt_index) + (5 * history_form))
                    % len(products)
                ]
                tier = tiers[(family_index + prompt_index + history_form) % len(tiers)]
                realization_key = f"{prompt_index}-{history_form}"
                card_name = f"{product} {tier}"
                other_card_name = f"{product} {tier.replace('Debit', 'Credit')}"
                card_last4 = f"{number_base + pair_index:04d}"
                other_card_last4 = f"{other_number_base + pair_index:04d}"
                pair_id = f"coreference-reinforce-{split}-{family}-{realization_key}"
                history_values = {
                    "card_name": card_name,
                    "card_last4": card_last4,
                    "other_card_name": other_card_name,
                    "other_card_last4": other_card_last4,
                }
                sole_card_history = sole_card_history_forms[history_form].format(
                    **history_values
                )
                multiple_card_history = multiple_card_history_forms[history_form].format(
                    **history_values
                )
                action_final = (
                    f"{card_name} ending in {card_last4} now has replacement pending."
                )
                action = _record(
                    record_id=f"deictic_replace_{family}_{split}_{realization_key}",
                    split=split,
                    scenario_family="deictic_replace_reinforcement_action",
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
                # The pinned clarify template, verbatim: the 2026-08-21 v9 run
                # proved any rephrasing of it collapses the ambiguity gate.
                ambiguity_final = (
                    f"I found {card_name} ending in {card_last4} and {other_card_name} "
                    f"ending in {other_card_last4}. Which card should I replace? Please "
                    "share its last four digits."
                )
                ambiguity = _record(
                    record_id=f"deictic_ambiguous_{family}_{split}_{realization_key}",
                    split=split,
                    scenario_family="deictic_replace_reinforcement_ambiguity",
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


# ---------------------------------------------------------------------------
# Long-context tool fidelity.
#
# Measured V9 defect: once a session renders to roughly 980 tokens the adapter
# answers a lexically misleading final user turn by inventing a tool that does not
# exist -- list_addresses, get_statement, list_pin_requests, list_disputes --
# instead of calling the one tool its contract exposes. Every row below puts a
# misleading current turn at the end of an ordinary, long servicing session and
# labels exactly one tool call, so generation_contract_for_record yields a
# single-tool contract and the trainer renders exactly that one schema.
# ---------------------------------------------------------------------------

LONG_CONTEXT_FAMILY = "long_context_tool_fidelity"
# Fresh, non-overlapping number bases so no synthetic entity collides with the
# coreference curriculum (train 6100/8100, validation 2100/4100, shadow 3100/5100).
LONG_CONTEXT_NUMBER_BASES = {"train": 7300, "validation": 9300}
LONG_CONTEXT_TRAIN_HISTORY_BUNDLES = 5
# Rendered-length guard. ToolWireAdapter._select_whole_chain_suffix silently drops
# the earliest user chains once a row exceeds max_seq_len (2048 in the continuation
# run). A row that lost its history would still train, on the wrong thing, so the
# ceiling is asserted at build time.
#
# The trainer's own render needs the published tokenizer and the granite chat
# template, which a pure data generator must not depend on, so the build asserts a
# calibrated char proxy instead. Measured over all 224 rows of this curriculum with
# the granite-4.1-8b tokenizer through ToolWireAdapter._render_messages (system
# prompt + turn guidance + the one exposed tool schema + the full message chain),
# the ratio of _long_context_prompt_chars to rendered tokens fell between 3.01 and
# 3.50. The two constants below sit just outside that window, so dividing the char
# count by them brackets the real render from both sides rather than estimating it.
# tests/test_banking_servicing_alignment_data.py re-measures the exact rendered
# lengths against the real tokenizer whenever it is available locally; the observed
# distribution at the time of writing was 818 to 1311 tokens.
LONG_CONTEXT_CHARS_PER_TOKEN_FLOOR = 2.95
LONG_CONTEXT_CHARS_PER_TOKEN_CEILING = 3.55
LONG_CONTEXT_MIN_RENDERED_TOKENS = 800
LONG_CONTEXT_MAX_RENDERED_TOKENS = 1700
# Deliberately steered away from SCREENSHOT_HELDOUT_CURRENTS and both shadow gates:
# the 4-gram assertions are strict and this curriculum's nouns (address, statement,
# transactions, money sent) sit right next to the held-out wordings.
LONG_CONTEXT_CURRENT_FRAMES = (
    "Before we wrap this up I still need you to {ask}",
    "One last item and then I am done, so please {ask}",
    "While we are both here I would like you to {ask}",
    "There is one loose end left over, so go ahead and {ask}",
)
_LONGCTX_PLACE_STEMS = (
    "Alder Row",
    "Bright Quay",
    "Cobble Lane",
    "Dunmore",
    "Elmfield",
    "Foxglove",
    "Garrick Wharf",
    "Havenside",
    "Ironbridge",
    "Juniper Walk",
)
_LONGCTX_TRADES = (
    "Grocers",
    "Optics",
    "Bakery",
    "Hardware",
    "Cycles",
    "Florist",
    "Bookbinders",
    "Creamery",
    "Outfitters",
    "Stationers",
    "Ceramics",
    "Fishmongers",
    "Ironworks",
    "Herbalists",
    "Tailors",
    "Cobblers",
    "Luthiers",
    "Glassworks",
    "Apothecary",
    "Provisions",
)
_LONGCTX_RECIPIENTS = (
    "Marchwood Utilities",
    "Ferngate Landscaping",
    "Oakhill Dental Care",
    "Pinewharf Storage",
    "Quarrybank Roofing",
    "Redmoor Insurance",
    "Saltmarsh Broadband",
    "Thornbury Childcare",
    "Uppermill Fitness",
    "Vinegate Cleaning",
    "Westhaven Removals",
    "Yarrowfield Heating",
)
_LONGCTX_SERVICES = (
    "Utilities",
    "Landscaping",
    "Dental Care",
    "Storage",
    "Roofing",
    "Insurance",
    "Broadband",
    "Childcare",
    "Fitness",
    "Cleaning",
    "Removals",
    "Heating",
    "Plumbing",
    "Glaziers",
    "Veterinary",
    "Opticians",
    "Tutoring",
    "Laundry",
    "Catering",
    "Security",
)
_LONGCTX_LOCALITIES = (
    "Ashfield",
    "Brackenmoor",
    "Colnebridge",
    "Dunhollow",
    "Edenvale",
    "Fernhurst",
    "Greystone",
    "Hartsmere",
    "Inglewood",
    "Kelsford",
    "Lowmarsh",
    "Marlbury",
    "Netherby",
    "Oakmere",
    "Pendlecombe",
    "Rushmere",
    "Stanwick",
    "Thorpeleigh",
    "Wexley",
    "Yarnton",
)
# Train and validation walk the name pools from different starting points. Without
# that, a train row and a validation row at the same index share every name and their
# finals differ only by the single digit of the number base, which
# _assert_no_fuzzy_final_duplicates rejects at 0.995 similarity.
_LONGCTX_POOL_OFFSETS = {"train": 0, "validation": 7}
_LONGCTX_CARD_PRODUCTS = (
    "Harborview",
    "Stonebridge",
    "Fairwater",
    "Lantern Bay",
    "Northgate",
)
_LONGCTX_CARD_TIERS = (
    "Everyday Debit",
    "Rewards Debit",
    "Travel Debit",
    "Cashback Debit",
)
# Ordinary servicing chatter. History turns are never loss-bearing and are not
# checked for uniqueness (only the last user turn is), so the same filler can back
# many rows; the per-row entities carry the variety instead. Keep every string clear
# of TRAINABLE_TEXT_BANNED_WORDS.
_LONGCTX_PLAIN_FILLERS = (
    (
        "Quick question before anything else. If a payment lands before my salary "
        "does, what happens to the balance in the meantime?",
        "If a payment posts before the credit arrives, the account moves into its "
        "arranged overdraft rather than bouncing the payment. Interest accrues daily "
        "on the overdrawn portion and is collected once a month, so a single day of "
        "being short costs very little. I can switch on an alert that reaches you the "
        "same day the balance goes below zero, which most people find is enough "
        "warning to move something across. If the shortfall is going to run for more "
        "than a few days, it is usually cheaper to move funds over from savings than "
        "to sit in the overdraft. Either way the charge only ever applies to the "
        "amount you are actually short by, never to the whole balance, and it stops "
        "the moment the account is back in credit.",
    ),
    (
        "And when does the salary credit normally reach me each month?",
        "Incoming salary credits are normally applied on the last working day of the "
        "month, as soon as the sending bank releases them. When that day falls on a "
        "weekend the credit arrives on the Friday before instead, so the money is "
        "already sitting there when the weekend payments run. Employers occasionally "
        "release early in December, and when they do the credit still lands on the day "
        "they send it rather than being held back until the usual date. Nothing on "
        "your side has to change for any of that to work. If a credit ever fails to "
        "arrive on the day you expected it, the sending employer is the right place "
        "to start, and I can tell you exactly what has and has not reached us.",
    ),
    (
        "I am heading abroad in a fortnight. Do I have to tell you before I travel?",
        "There is no need to register a trip any more. Spending abroad is judged "
        "against the ordinary pattern of the account rather than against a notice, so "
        "most people travel without anything being queried at all. The one piece of "
        "advice worth following is to carry a second means of payment, because if a "
        "particular transaction does get held you will want something else in your "
        "pocket while we sort it out. Cash withdrawals abroad carry a separate charge, "
        "which is set out in the fee summary for the account. Payments in another "
        "currency are converted at the rate applying on the day they settle rather "
        "than the day you spend, so the two figures can differ a little.",
    ),
    (
        "What is the ceiling on a contactless tap at the moment?",
        "A single contactless payment goes through up to one hundred, and there is a "
        "cumulative total across consecutive taps before the terminal asks for the PIN "
        "again. Entering the PIN once clears that running total immediately, so the "
        "next tap starts from zero. Some retailers set their own lower limit on the "
        "terminal, and when that happens the ceiling you run into is theirs rather "
        "than ours. Higher-value purchases always go through the chip, which is why a "
        "large payment asks for the PIN even on the first attempt. If a tap is "
        "refused for no obvious reason, inserting the card and entering the PIN once "
        "normally clears whatever caused it and lets you carry on.",
    ),
    (
        "If I wanted to add my partner to the account later on, is that a big job?",
        "Adding a second holder means you both complete identity verification, and "
        "after that each of you has your own card and your own sign-in. Existing "
        "scheduled payments carry across untouched, so nothing has to be arranged a "
        "second time. Both holders can see the full history from the day the account "
        "opened rather than only from the day the second person joined, which "
        "occasionally surprises people. Removing a holder later is a separate request "
        "and needs agreement from both of you. Until the second holder is added "
        "everything on the account stays under your name alone, so nothing you "
        "arrange between now and then has to be undone afterwards.",
    ),
    (
        "One thing I can never remember is when the savings interest actually posts.",
        "Savings interest is worked out daily and paid on the first working day of "
        "the following month. It arrives as a single credit line rather than a run of "
        "small ones, which makes it easy to pick out when you look back over the "
        "month. The rate applies to the whole balance rather than in bands, so you do "
        "not have to work out where a threshold falls. If you move money out partway "
        "through the month you still keep the interest earned on the days it was "
        "sitting there. Interest is reported back to you once a year in a single "
        "summary that covers every savings balance you hold with us rather than "
        "each one separately.",
    ),
    (
        "Is it still possible to book time with somebody in a branch?",
        "Branch bookings can usually be made for the same week and run to about half "
        "an hour. Bring one photographic identity document if the visit involves "
        "changing anything held on the profile, and otherwise there is nothing you "
        "need to prepare beforehand. Where the matter is something I can finish here, "
        "it is normally quicker to deal with it in this conversation than to travel "
        "in. If a written signature is genuinely required I will say so before you "
        "book the slot. Whatever the two of us settle here is written into the "
        "record straight away, so nobody in a branch would ask you to go over the "
        "same ground a second time.",
    ),
    (
        "Could I move my correspondence over to electronic delivery?",
        "Correspondence can be switched to electronic delivery whenever you like, and "
        "the change takes effect from the next cycle onward. Anything issued before "
        "the switch stays available to download for seven years, so nothing is lost in "
        "the move. Regulatory notices still arrive on paper where the rules require "
        "it, and those are the only items that keep coming through the door. You can "
        "switch back at any point and there is no charge either way. Delivery "
        "preferences sit on the profile rather than on an individual account, so a "
        "single change covers everything you hold with us today and anything you "
        "open later.",
    ),
)
# Bundle -> the plain fillers behind each row. Bundles 0-1 are the three-exchange
# tier, 2-3 the four-exchange tier, and bundle 4 the four-exchange tier whose middle
# two exchanges are tool-backed (see _long_context_history).
_LONGCTX_HISTORY_BUNDLES = (
    (0, 1),
    (2, 3),
    (4, 5, 6),
    (7, 0, 3),
    (5,),
)
_LONGCTX_TOOL_BACKED_BUNDLE = 4
# Tier C, the 20% tier. Read decoys spend their extra length on two context tool-call
# pairs plus one plain exchange; write decoys cannot repeat their call without
# performing the write twice, so they buy the same length with plain exchanges only.
_LONGCTX_TIER_C_READ_FILLERS = (5, 2)
_LONGCTX_TIER_C_WRITE_FILLERS = (5, 2, 6, 1)
# 24 validation rows as (decoy, phrasing, bundle): ten tier-A, ten tier-B and four
# tier-C rows covering all ten decoys, a strict subset of the train cross-product.
_LONGCTX_VALIDATION_PLAN = (
    (0, 0, 0),
    (1, 1, 1),
    (2, 2, 0),
    (3, 3, 1),
    (4, 0, 0),
    (5, 1, 1),
    (6, 2, 0),
    (7, 3, 1),
    (8, 0, 0),
    (9, 1, 1),
    (0, 2, 2),
    (1, 3, 3),
    (2, 0, 2),
    (3, 1, 3),
    (4, 2, 2),
    (5, 3, 3),
    (6, 0, 2),
    (7, 1, 3),
    (8, 2, 2),
    (9, 3, 3),
    (0, 1, 4),
    (2, 3, 4),
    (5, 0, 4),
    (7, 2, 4),
)


def _long_context_entities(split: str, index: int) -> dict[str, str]:
    reference = LONG_CONTEXT_NUMBER_BASES[split] + index
    position = index + _LONGCTX_POOL_OFFSETS[split]
    group = position // len(_LONGCTX_PLACE_STEMS)
    stem = _LONGCTX_PLACE_STEMS[position % len(_LONGCTX_PLACE_STEMS)]
    recipient_stem = _LONGCTX_PLACE_STEMS[(position + 3) % len(_LONGCTX_PLACE_STEMS)]
    product = _LONGCTX_CARD_PRODUCTS[position % len(_LONGCTX_CARD_PRODUCTS)]
    tier_index = (position // len(_LONGCTX_CARD_PRODUCTS)) % len(_LONGCTX_CARD_TIERS)
    return {
        # Every current turn and every final carries this per-row reference, which
        # makes both globally unique by construction without a phrasing pool. The
        # word-level names carry the rest of the variety, which is what keeps two
        # finals of the same decoy far enough apart for the fuzzy-duplicate check.
        "last4": f"{reference:04d}",
        "case_ref": f"HB-{reference}",
        "amount": f"{reference / 100:.2f}",
        "savings": f"{(reference + 33900) / 100:.2f}",
        "merchant": f"{stem} {_LONGCTX_TRADES[group % len(_LONGCTX_TRADES)]}",
        "recipient": f"{recipient_stem} {_LONGCTX_SERVICES[group % len(_LONGCTX_SERVICES)]}",
        "card_name": f"{product} {_LONGCTX_CARD_TIERS[tier_index]}",
        # Digit-free on purpose: the final quotes it, and PII_PATTERNS matches any
        # 12-19 digit run.
        "address": f"{stem} House, {_LONGCTX_LOCALITIES[group % len(_LONGCTX_LOCALITIES)]}",
    }


def _long_context_decoys(entity: dict[str, str], suffix: str) -> tuple[dict[str, Any], ...]:
    """Ten misleading-noun / correct-tool pairs covering all nine exposed tools.

    Each entry carries the one tool the row may ever name, the envelope that tool
    returns, and a final that states only what that envelope contains. Two rules are
    load-bearing and are asserted at build time:

    * `_assert_long_context_contract_tools` - no message anywhere in the row may name
      a tool other than `tool`, because `training_tools_for_record` renders exactly
      that one schema. A history call to anything else would demonstrate the very
      defect this curriculum corrects.
    * every numeric token and every spelled-out count in `final` has to be readable
      out of `envelope`; `read_history` decoys additionally reuse `envelope` verbatim
      for their context calls, so a row cannot contradict itself across turns.

    The four write tools are in ENTITY_REQUIRED_TOOLS, so their contract asserts
    entity_state="resolved". Each of those anchors therefore states the selector --
    the last four digits, the merchant, the recipient -- in an earlier assistant
    turn, so the row never teaches the model to guess one. Where the customer asks
    for an action the manifest cannot perform (a lost-card report, a stop payment, a
    chargeback, a PIN reissue), the final says so rather than silently substituting.
    """

    return (
        {
            "key": "address_case",
            "decoy_tool": "list_addresses",
            "tool": "list_service_cases",
            "arguments": {},
            # Repeating a read call later in a session is ordinary, so these decoys
            # carry the tier-C context pairs; the write decoys below cannot without
            # performing the write twice, and use a longer plain history instead.
            "read_history": True,
            "anchor": (
                f"Remind me which service requests are still sitting open {suffix}.",
                f"One request is open: {entity['case_ref']}, raised on 2026-07-02 to "
                f"confirm {entity['address']} as the mailing address. Nothing else is "
                "outstanding on the profile, and the two before it were closed off "
                "earlier in the spring without needing anything from you.",
            ),
            "ask": f"pin down the mailing address entry filed under {entity['case_ref']}",
            "final": (
                f"Case {entity['case_ref']} is the mailing address entry you asked "
                f"about. It is still open, it was raised on 2026-07-02, and the address "
                f"it is confirming is {entity['address']}. Nothing else is recorded "
                "against it."
            ),
            "history_summary": (
                f"The case list comes back with {entity['case_ref']}, an open mailing "
                f"address confirmation for {entity['address']} raised on 2026-07-02."
            ),
            "grounding": [
                f"service_case.reference={entity['case_ref']}",
                f"service_case.requested_address={entity['address']}",
                "service_case.status=open",
                "service_case.created_at=2026-07-02T09:30:00Z",
            ],
            "envelope": _success_envelope(
                service_cases=[
                    {
                        "case_type": "address_update",
                        "reference": entity["case_ref"],
                        "requested_address": entity["address"],
                        "subject": "Confirm change of mailing address",
                        "status": "open",
                        "created_at": "2026-07-02T09:30:00Z",
                    }
                ]
            ),
        },
        {
            "key": "statement_lines",
            "decoy_tool": "get_statement",
            "tool": "list_transactions",
            "arguments": {"limit": 6},
            "read_history": True,
            "anchor": (
                f"Tell me what has hit the account since the month began {suffix}.",
                f"Six entries have posted since 2026-07-01, one of them "
                f"{entity['amount']} at {entity['merchant']}. The others are everyday "
                "purchases spread across the first fortnight, and none of the six is "
                "still pending.",
            ),
            "ask": (
                f"read out the monthly statement lines around the {entity['amount']} "
                f"charge at {entity['merchant']}"
            ),
            "final": (
                f"Six entries came back on the account. The {entity['merchant']} charge "
                f"is {entity['amount']} and posted on 2026-07-14, and the rest are "
                "everyday purchases from earlier in the month."
            ),
            "context_arguments": {"limit": 2},
            "context_envelope": _success_envelope(
                transactions=[
                    {
                        "description": entity["merchant"],
                        "amount": entity["amount"],
                        "posted_at": "2026-07-14",
                        "status": "posted",
                    },
                    {
                        "description": "Kingsford Transit",
                        "amount": "18.40",
                        "posted_at": "2026-07-12",
                        "status": "posted",
                    },
                ]
            ),
            "history_summary": (
                f"The two most recent entries are {entity['amount']} at "
                f"{entity['merchant']} on 2026-07-14 and 18.40 at Kingsford Transit."
            ),
            "grounding": [
                f"transaction.description={entity['merchant']}",
                f"transaction.amount={entity['amount']}",
                "transaction.posted_at=2026-07-14",
                "transaction.status=posted",
            ],
            "envelope": _success_envelope(
                transactions=[
                    {
                        "description": entity["merchant"],
                        "amount": entity["amount"],
                        "posted_at": "2026-07-14",
                        "status": "posted",
                    },
                    {
                        "description": "Kingsford Transit",
                        "amount": "18.40",
                        "posted_at": "2026-07-12",
                        "status": "posted",
                    },
                    {
                        "description": "Ravensworth Energy",
                        "amount": "64.20",
                        "posted_at": "2026-07-08",
                        "status": "posted",
                    },
                    {
                        "description": "Halewood Newsagent",
                        "amount": "3.95",
                        "posted_at": "2026-07-05",
                        "status": "posted",
                    },
                    {
                        "description": "Bexley Lane Cafe",
                        "amount": "11.75",
                        "posted_at": "2026-07-03",
                        "status": "posted",
                    },
                    {
                        "description": "Cranmore Water",
                        "amount": "29.60",
                        "posted_at": "2026-07-02",
                        "status": "posted",
                    },
                ]
            ),
        },
        {
            "key": "pin_reissue",
            "decoy_tool": "list_pin_requests",
            "tool": "list_cards",
            "arguments": {},
            "read_history": True,
            "anchor": (
                f"Which card am I actually holding on this profile {suffix}?",
                f"You hold one active card, the {entity['card_name']} ending in "
                f"{entity['last4']}, and it has been active since 2026-03-11. There is "
                "no second card on the profile, so anything you ask me about a card "
                "refers to that one.",
            ),
            "ask": (
                f"check whether a PIN reissue is showing against the "
                f"{entity['card_name']} ending in {entity['last4']}"
            ),
            "final": (
                f"Your card list shows one card, the {entity['card_name']} ending in "
                f"{entity['last4']}, active since 2026-03-11. The card record does not "
                "carry PIN request history, so I cannot tell you from here whether a "
                "reissue is under way. Say the word and I will raise it with the card "
                "team."
            ),
            "history_summary": (
                f"The card list comes back with one card, the {entity['card_name']} "
                f"ending in {entity['last4']}, active since 2026-03-11."
            ),
            "grounding": [
                f"card.last4={entity['last4']}",
                "card.status=active",
                "card.opened_at=2026-03-11",
            ],
            "envelope": _success_envelope(
                cards=[
                    {
                        "name": entity["card_name"],
                        "last4": entity["last4"],
                        "status": "active",
                        "opened_at": "2026-03-11",
                    }
                ]
            ),
        },
        {
            "key": "dispute_status",
            "decoy_tool": "list_disputes",
            "tool": "list_transactions",
            "arguments": {"limit": 8},
            "read_history": True,
            "anchor": (
                f"Read me back the card purchase I asked about earlier {suffix}.",
                f"The purchase was {entity['amount']} at {entity['merchant']}, posted on "
                "2026-07-09. It has settled and no claim has been raised against it. "
                "The merchant took the money in one go rather than in parts, so there "
                "is only the single entry to look at.",
            ),
            "ask": (
                f"see where the dispute status sits for the {entity['amount']} charge at "
                f"{entity['merchant']}"
            ),
            "final": (
                f"The {entity['merchant']} charge is {entity['amount']} and posted on "
                "2026-07-09, with no claim recorded against it. Say the word and I will "
                "raise a dispute for you."
            ),
            "context_arguments": {"limit": 2},
            "context_envelope": _success_envelope(
                transactions=[
                    {
                        "description": entity["merchant"],
                        "amount": entity["amount"],
                        "posted_at": "2026-07-09",
                        "status": "posted",
                        "disputed": False,
                    },
                    {
                        "description": "Ravensworth Energy",
                        "amount": "64.20",
                        "posted_at": "2026-07-08",
                        "status": "posted",
                        "disputed": False,
                    },
                ]
            ),
            "history_summary": (
                f"The two most recent entries are {entity['amount']} at "
                f"{entity['merchant']} on 2026-07-09, with no claim on it, and 64.20 "
                "at Ravensworth Energy."
            ),
            "grounding": [
                f"transaction.description={entity['merchant']}",
                f"transaction.amount={entity['amount']}",
                "transaction.posted_at=2026-07-09",
                "transaction.disputed=false",
            ],
            "envelope": _success_envelope(
                transactions=[
                    {
                        "description": entity["merchant"],
                        "amount": entity["amount"],
                        "posted_at": "2026-07-09",
                        "status": "posted",
                        "disputed": False,
                    },
                    {
                        "description": "Ravensworth Energy",
                        "amount": "64.20",
                        "posted_at": "2026-07-08",
                        "status": "posted",
                        "disputed": False,
                    },
                    {
                        "description": "Bexley Lane Cafe",
                        "amount": "11.75",
                        "posted_at": "2026-07-03",
                        "status": "posted",
                        "disputed": False,
                    },
                ]
            ),
        },
        {
            "key": "standing_order",
            "decoy_tool": "list_standing_orders",
            "tool": "list_transfers",
            "arguments": {},
            "read_history": True,
            "anchor": (
                f"What money is queued to leave the account this month {suffix}?",
                f"One payment is queued: {entity['amount']} to {entity['recipient']}, "
                "due on 2026-07-28 and still pending. Everything else you set up this "
                "month has already gone through and settled at the receiving end.",
            ),
            "ask": (
                f"confirm the standing order sending {entity['amount']} to "
                f"{entity['recipient']} each month"
            ),
            "final": (
                f"Your transfer list shows one payment waiting, {entity['amount']} to "
                f"{entity['recipient']}, due on 2026-07-28 and still pending. The "
                "transfer record does not mark it as recurring, so I cannot confirm a "
                "monthly standing order from here."
            ),
            "history_summary": (
                f"The transfer list comes back with one pending payment, "
                f"{entity['amount']} to {entity['recipient']} due on 2026-07-28."
            ),
            "grounding": [
                f"transfer.recipient={entity['recipient']}",
                f"transfer.amount={entity['amount']}",
                "transfer.due_on=2026-07-28",
                "transfer.status=pending",
            ],
            "envelope": _success_envelope(
                transfers=[
                    {
                        "recipient": entity["recipient"],
                        "amount": entity["amount"],
                        "due_on": "2026-07-28",
                        "status": "pending",
                    }
                ]
            ),
        },
        {
            "key": "balance_position",
            "decoy_tool": "get_balance_sheet",
            "tool": "list_accounts",
            "arguments": {},
            "read_history": True,
            "anchor": (
                f"Give me a rough idea of where I stand overall {suffix}.",
                f"Your everyday account is holding {entity['amount']} available today "
                f"and the savings behind it holds {entity['savings']}. Both are open, and "
                "no charge is waiting to come off either balance in the next few days.",
            ),
            "ask": (
                f"walk me through the balance sheet position before I commit "
                f"{entity['amount']} anywhere"
            ),
            "final": (
                f"You hold two accounts. Everyday Checking has {entity['amount']} "
                f"available and Reserve Savings has {entity['savings']}. Both accounts are "
                "open and neither balance is negative, so the money is there if you want it."
            ),
            "history_summary": (
                f"The account list comes back with Everyday Checking at "
                f"{entity['amount']} and Reserve Savings at {entity['savings']}, both open."
            ),
            "grounding": [
                f"account.everyday_checking.available={entity['amount']}",
                f"account.reserve_savings.available={entity['savings']}",
                "account.status=open",
            ],
            "envelope": _success_envelope(
                accounts=[
                    {
                        "name": "Everyday Checking",
                        "available": entity["amount"],
                        "status": "open",
                    },
                    {
                        "name": "Reserve Savings",
                        "available": entity["savings"],
                        "status": "open",
                    },
                ]
            ),
        },
        {
            "key": "lost_card",
            "decoy_tool": "report_lost_card",
            "tool": "freeze_card",
            "arguments": {"last4": entity["last4"]},
            # A write decoy: repeating the call in the history would freeze the card
            # twice, so tier C reaches its length with more plain exchanges instead.
            "read_history": False,
            "anchor": (
                f"Which card is currently live on the profile {suffix}?",
                f"The live card is your {entity['card_name']} ending in "
                f"{entity['last4']}, active since 2026-03-11 with nothing held against "
                "it. It is the only card issued on the profile, and no earlier card is "
                "still open behind it.",
            ),
            "ask": (
                "file the lost-card report, because I cannot find the "
                f"{entity['card_name']} ending in {entity['last4']}"
            ),
            "final": (
                "I cannot file a lost-card report from here, so I have frozen the card "
                f"instead. Your {entity['card_name']} ending in {entity['last4']} now "
                "shows as frozen, and nothing further can be charged to it while you "
                "look for it."
            ),
            "history_summary": "",
            "grounding": [
                f"card.last4={entity['last4']}",
                "card.status=frozen",
            ],
            "envelope": _success_envelope(
                card={
                    "name": entity["card_name"],
                    "last4": entity["last4"],
                    "status": "frozen",
                }
            ),
        },
        {
            "key": "new_card_order",
            "decoy_tool": "order_card",
            "tool": "replace_card",
            "read_history": False,
            "arguments": {"last4": entity["last4"]},
            "anchor": (
                f"Confirm for me which card is worn out on the profile {suffix}.",
                f"That is your {entity['card_name']} ending in {entity['last4']}. It is "
                "still active, though the magnetic stripe has been failing at terminals. "
                "Nothing else on the profile is in that condition, so it is the only "
                "one worth doing anything about.",
            ),
            "ask": (
                f"put through the new card order for the {entity['card_name']} ending in "
                f"{entity['last4']}"
            ),
            "final": (
                "I have raised a replacement rather than a separate new-card order. "
                f"Your {entity['card_name']} ending in {entity['last4']} now shows "
                "replacement pending on the card record."
            ),
            "history_summary": "",
            "grounding": [
                f"card.last4={entity['last4']}",
                "card.status=replacement_pending",
            ],
            "envelope": _success_envelope(
                card={
                    "name": entity["card_name"],
                    "last4": entity["last4"],
                    "status": "replacement_pending",
                }
            ),
        },
        {
            "key": "chargeback",
            "decoy_tool": "open_chargeback",
            "tool": "dispute_transaction",
            "read_history": False,
            "arguments": {"description": entity["merchant"]},
            "anchor": (
                f"Which purchase was it that I flagged earlier {suffix}?",
                f"The one you flagged was {entity['amount']} at {entity['merchant']}, "
                "posted on 2026-07-09 and already settled. It is the only entry from "
                "that merchant in the period we looked at, so there is no risk of "
                "picking up the wrong one.",
            ),
            "ask": (
                f"start the chargeback on the {entity['amount']} charge at {entity['merchant']}"
            ),
            "final": (
                f"I have raised a dispute on the {entity['amount']} charge at "
                f"{entity['merchant']} rather than a separate chargeback. It still "
                "shows as posted on the account while the claim is looked at."
            ),
            "history_summary": "",
            "grounding": [
                f"transaction.description={entity['merchant']}",
                f"transaction.amount={entity['amount']}",
                "transaction.status=posted",
                "transaction.disputed=true",
            ],
            "envelope": _success_envelope(
                transaction={
                    "description": entity["merchant"],
                    "amount": entity["amount"],
                    "status": "posted",
                    "disputed": True,
                }
            ),
        },
        {
            "key": "stop_payment",
            "decoy_tool": "stop_payment",
            "tool": "cancel_transfer",
            "read_history": False,
            "arguments": {"recipient": entity["recipient"]},
            "anchor": (
                f"Remind me who the queued payment is going to {suffix}.",
                f"It is going to {entity['recipient']}, {entity['amount']} due on "
                "2026-07-28, and it has not left the account yet. That is the only "
                "queued payment on the account, so nothing else is waiting behind it "
                "to go out.",
            ),
            "ask": (
                f"place a stop payment on the {entity['amount']} heading out to "
                f"{entity['recipient']}"
            ),
            "final": (
                "A stop payment is not something I can place, so I have cancelled the "
                f"transfer instead. The {entity['amount']} to {entity['recipient']} "
                "that was due on 2026-07-28 is now cancelled and will not leave the "
                "account."
            ),
            "history_summary": "",
            "grounding": [
                f"transfer.recipient={entity['recipient']}",
                f"transfer.amount={entity['amount']}",
                "transfer.due_on=2026-07-28",
                "transfer.status=cancelled",
            ],
            "envelope": _success_envelope(
                transfer={
                    "recipient": entity["recipient"],
                    "amount": entity["amount"],
                    "due_on": "2026-07-28",
                    "status": "cancelled",
                }
            ),
        },
    )


def _long_context_context_pair(
    record_id: str,
    context_index: int,
    decoy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """One tool-backed prior exchange: user, context call, its result, a summary.

    The call names the row's own target tool with the row's own arguments and returns
    the row's own envelope, which is what keeps the whole chain inside the single
    schema `training_tools_for_record` renders and stops the history contradicting the
    loss-bearing turn. The id is context_{record_id}_{n} with loss=False, content=None
    and index 0, exactly as validate_records demands of a non-target call, and the
    tool result follows it immediately so the correlation check stays satisfied.
    """

    if context_index == 0:
        user_text = "While we are on it, pull that up for me before we go any further."
        summary = str(decoy["history_summary"])
    else:
        user_text = "Can you run it once more? I want to be sure nothing moved since."
        summary = (
            "That comes back exactly as it did a moment ago, so nothing has moved on "
            "the record while we have been talking."
        )
    call_id = f"context_{record_id}_{context_index}"
    tool_name = str(decoy["tool"])
    arguments = dict(decoy.get("context_arguments", decoy["arguments"]))
    envelope = decoy.get("context_envelope", decoy["envelope"])
    return [
        _user(user_text),
        {
            "role": "assistant",
            "content": None,
            "loss": False,
            "tool_calls": [
                {
                    "id": call_id,
                    "index": 0,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }
            ],
        },
        _tool_result(call_id, tool_name, envelope=envelope),
        _assistant(summary, loss=False),
    ]


def _long_context_history(
    record_id: str,
    decoy: Mapping[str, Any],
    bundle: int,
) -> list[dict[str, Any]]:
    anchor_user, anchor_assistant = decoy["anchor"]
    messages = [_user(anchor_user), _assistant(anchor_assistant, loss=False)]
    read_history = bool(decoy["read_history"])
    if bundle == _LONGCTX_TOOL_BACKED_BUNDLE and read_history:
        messages.extend(_long_context_context_pair(record_id, 0, decoy))
        messages.extend(_long_context_context_pair(record_id, 1, decoy))
        fillers = _LONGCTX_TIER_C_READ_FILLERS
    elif bundle == _LONGCTX_TOOL_BACKED_BUNDLE:
        fillers = _LONGCTX_TIER_C_WRITE_FILLERS
    else:
        fillers = _LONGCTX_HISTORY_BUNDLES[bundle]
    for filler_index in fillers:
        filler_user, filler_assistant = _LONGCTX_PLAIN_FILLERS[filler_index]
        messages.extend([_user(filler_user), _assistant(filler_assistant, loss=False)])
    return messages


def _long_context_prompt_chars(record: Mapping[str, Any]) -> int:
    """Characters the chat template will render, minus its own scaffolding."""

    total = 0
    for message in record["messages"]:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif content is not None:
            total += len(json.dumps(content, ensure_ascii=False, sort_keys=True))
        for call in message.get("tool_calls") or ():
            total += len(json.dumps(call["function"], ensure_ascii=False, sort_keys=True))
    return total


def _long_context_rendered_token_bounds(record: Mapping[str, Any]) -> tuple[int, int]:
    """Bracket the trainer's rendered token count without a tokenizer.

    Returns (floor, ceiling); the real granite render sits between them for every
    row of this curriculum. See the calibration note on the constants above.
    """

    chars = _long_context_prompt_chars(record)
    return (
        int(chars / LONG_CONTEXT_CHARS_PER_TOKEN_CEILING),
        int(chars / LONG_CONTEXT_CHARS_PER_TOKEN_FLOOR) + 1,
    )


def _long_context_tool_fidelity(split: str) -> list[dict[str, Any]]:
    """Long multi-turn sessions whose misleading final turn must still route right.

    Train takes the whole 10 x 4 x 5 cross-product (200 rows); validation takes a
    24-row subset of the same shape on a disjoint entity base. Never wired into
    _test_records: the frozen 215-row composite digest must not move.
    """

    suffix = _suffix(split)
    if split == "train":
        plan = tuple(
            (decoy_index, phrasing, bundle)
            for decoy_index in range(10)
            for phrasing in range(len(LONG_CONTEXT_CURRENT_FRAMES))
            for bundle in range(LONG_CONTEXT_TRAIN_HISTORY_BUNDLES)
        )
    elif split == "validation":
        plan = _LONGCTX_VALIDATION_PLAN
    else:
        raise ValueError(f"unsupported long-context split: {split}")
    records: list[dict[str, Any]] = []
    for index, (decoy_index, phrasing, bundle) in enumerate(plan):
        entity = _long_context_entities(split, index)
        decoy = _long_context_decoys(entity, suffix)[decoy_index]
        record_id = f"longctx_{decoy['key']}_{split}_p{phrasing}b{bundle}"
        current = LONG_CONTEXT_CURRENT_FRAMES[phrasing].format(ask=decoy["ask"]) + f" {suffix}."
        records.append(
            _record(
                record_id=record_id,
                split=split,
                scenario_family=LONG_CONTEXT_FAMILY,
                current=current,
                final=decoy["final"],
                tool_plan=[(decoy["tool"], dict(decoy["arguments"]))],
                grounding_facts=decoy["grounding"],
                path="multi_turn",
                pre_messages=_long_context_history(record_id, decoy, bundle),
                tool_envelopes=[decoy["envelope"]],
            )
        )
    _assert_long_context_contract_tools(records)
    _assert_long_context_render_budget(records)
    return records


def _assert_long_context_contract_tools(records: Sequence[dict[str, Any]]) -> None:
    """Every tool named anywhere in a row must be the one tool its contract exposes.

    training_tools_for_record (scripts/retail_bank/cloud_train_tool_sft.py) renders
    exactly the target tool's schema for a single-tool contract, so a history call to
    any other tool would show the model calling something the prompt does not expose
    -- which is the defect this curriculum exists to correct, demonstrated at
    loss=False immediately before the loss-bearing turn.
    """

    offenders = []
    for record in records:
        target = str(record["expected"]["tool_calls"][0]["name"])
        for message in record["messages"]:
            for call in message.get("tool_calls") or ():
                if str(call["function"]["name"]) != target:
                    offenders.append((str(record["record_id"]), call["function"]["name"]))
            if message.get("role") == "tool" and str(message["name"]) != target:
                offenders.append((str(record["record_id"]), message["name"]))
    if offenders:
        raise ValueError(f"long-context rows name a tool outside their contract: {offenders}")


def _assert_long_context_render_budget(records: Sequence[dict[str, Any]]) -> None:
    over = []
    short = []
    for record in records:
        floor, ceiling = _long_context_rendered_token_bounds(record)
        if ceiling > LONG_CONTEXT_MAX_RENDERED_TOKENS:
            over.append(str(record["record_id"]))
        # The proxy floor is deliberately pessimistic, so it is compared against a
        # relaxed target; the exact 800-token design floor is asserted in the tests
        # against the real render.
        if floor < LONG_CONTEXT_MIN_RENDERED_TOKENS - 120:
            short.append(str(record["record_id"]))
    if over:
        raise ValueError(f"long-context rows may exceed the render budget: {over}")
    if short:
        raise ValueError(f"long-context rows are too short to reproduce the defect: {short}")


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


# --- Policy-alignment curriculum (v11) ---------------------------------------
#
# Four zero-tool behaviour families targeting what the guidance-free bare probes
# showed the tuned model still does: write requested poems, solicit PIN digits and
# account last-fours, promise statement delivery, and invent request status. Every
# training row carries TURN GUIDANCE at render time, so the behaviour only survives
# with the guidance stripped if the mapping itself is repeated enough to reach the
# weights — the shipped weather refusal (one seed, 32 realizations) is the proof
# and the dosing model. Each seed here therefore repeats one mapping 28 times in
# train (7 user frames x 4 subjects) with the surface varied by entity, not by a
# wide phrasing pool (the 2026-08-21 pool regression), and 2 times in validation
# on a held-back subject. The families bypass _expand_records deliberately: the
# realizer's opener/closer scaffolding is forbidden in shipped finals, and only
# teacher realization may strip it — these finals are self-authored clean instead.
#
# Finals obey the runtime response-policy guard by construction: no digits, no
# question marks, no completed-retrieval or account-state claims, no promised
# unsupported actions, and the per-path markers validate_records requires ("retail
# banking" on ood; "account numbers"/"customer ids" on hard_negative). The exact
# bare-probe texts and their entities (an ocean poem, a January statement, the
# checking balance, a PIN change request) are deliberately absent so the probes
# stay held-out generalization checks, and every family is excluded from the
# router corpus via _SFT_ONLY_SCENARIO_FAMILIES.

POLICY_ALIGNMENT_FAMILIES = (
    "scope_refusal",
    "credential_hygiene",
    "capability_boundary",
    "no_evidence_honesty",
)
_POLICY_ALIGNMENT_TRAIN_ROWS_PER_SEED = 28
_POLICY_ALIGNMENT_VALIDATION_ROWS_PER_SEED = 2

_POLICY_ALIGNMENT_SEEDS: dict[str, tuple[dict[str, Any], ...]] = {
    "scope_refusal": (
        {
            "key": "creative_writing",
            "path": "ood",
            "subjects": {
                "train": (
                    "the autumn rain",
                    "the mountain sunrise",
                    "the city at night",
                    "a quiet forest",
                ),
                "validation": ("the winter sky",),
            },
            "frames": (
                "Could you write a short poem about {s} for me?",
                "Write a little poem about {s} while we chat.",
                "I'd love a quick poem about {s} — can you make one up?",
                "Please compose a few rhyming lines about {s}.",
                "Can you put together a verse about {s} real quick?",
                "Give me a short poem on {s} before we continue.",
                "Would you write me something poetic about {s}?",
            ),
            "finals": (
                "Writing a poem about {s} is outside what I can help with — my role"
                " covers retail banking only, so accounts, cards, and transfers are"
                " where I can be useful.",
                "I'll have to pass on a poem about {s}; I stick to retail banking"
                " topics like your accounts, cards, and transfers.",
                "A poem about {s} isn't something I can offer here. Retail banking"
                " questions — accounts, cards, transfers, payments — are what I can"
                " take on.",
                "I can't compose poetry about {s}. What I can do is retail banking"
                " support, from card questions to transfers and payments.",
                "Poetry about {s} falls outside my scope. I help with retail"
                " banking, so feel free to ask about your accounts, cards, or"
                " transfers.",
                "I'm not able to write about {s} — I only handle retail banking"
                " matters such as accounts, cards, and payments.",
                "That poem about {s} will have to come from elsewhere; here I can"
                " help with retail banking, like your accounts, cards, and"
                " transfers.",
            ),
        },
        {
            "key": "lifestyle",
            "path": "ood",
            "subjects": {
                "train": (
                    "mushroom risotto",
                    "lentil soup",
                    "roasted vegetables",
                    "banana bread",
                ),
                "validation": ("herb flatbread",),
            },
            "frames": (
                "Do you have a good recipe for {s}?",
                "How do I cook {s} at home?",
                "Share a simple way to make {s} tonight.",
                "What ingredients go into {s}?",
                "Walk me through preparing {s} for dinner.",
                "Any tips for making {s} taste better?",
                "What's the best way to prepare {s}?",
            ),
            "finals": (
                "A recipe for {s} isn't something I can help with — retail banking"
                " is my whole world, so accounts, cards, and transfers are where"
                " I'm useful.",
                "Cooking advice on {s} is outside my range; I handle retail banking"
                " questions like accounts, transfers, and cards.",
                "I'll leave {s} to the cooks — retail banking is what I cover, from"
                " card issues to transfers and payments.",
                "When it comes to {s}, I can't be your guide. Retail banking"
                " support — accounts, cards, payments — is what I offer.",
                "Kitchen questions about {s} are beyond me here. My focus stays on"
                " retail banking, so accounts and transfers are fair game.",
                "I can't advise on {s}, I'm afraid; retail banking matters like"
                " cards, payments, and transfers are my territory.",
                "Help with {s} isn't on my menu — retail banking is. Ask me about"
                " accounts, cards, or transfers anytime.",
            ),
        },
        {
            "key": "general_knowledge",
            "path": "ood",
            "subjects": {
                "train": (
                    "the Harbor City Rowers",
                    "the Northside Chess Club",
                    "the Bayview Volleyball Club",
                    "the Riverside Cycling Crew",
                ),
                "validation": ("the Lakeside Runners",),
            },
            "frames": (
                "Who won the {s} match last night?",
                "What was the score in the {s} game?",
                "Can you tell me how the {s} season is going?",
                "Give me an update on the {s} standings.",
                "Did the {s} squad win over the weekend?",
                "What's happening with {s} this season?",
                "How did {s} do in their last game?",
            ),
            "finals": (
                "Scores and standings for {s} aren't something I can look into —"
                " retail banking is my only field, so accounts, cards, and"
                " transfers are where I can help.",
                "I don't follow {s}, so I can't report on the games; retail banking"
                " questions about accounts, cards, or transfers are what I answer.",
                "News about {s} sits outside my coverage. Retail banking —"
                " accounts, transfers, payments — is what I'm here for.",
                "The season for {s} isn't in my wheelhouse; my help is limited to"
                " retail banking, from cards to transfers.",
                "I can't speak to results for {s}. Retail banking support like"
                " accounts, cards, and payments is my entire role.",
                "Sports updates on {s} aren't available from me — retail banking is"
                " where I work, so bring me account, card, or transfer questions.",
                "For {s} results you'll need another source; my lane is retail"
                " banking, covering accounts, cards, and transfers.",
            ),
        },
    ),
    "credential_hygiene": (
        {
            "key": "reveal_request",
            "path": "hard_negative",
            "subjects": {
                "train": (
                    "debit card",
                    "credit card",
                    "checking account",
                    "savings account",
                ),
                "validation": ("joint account",),
            },
            "frames": (
                "Can you read me the full number on my {s}?",
                "I need the complete number for my {s} — can you show it?",
                "Please display the whole number tied to my {s}.",
                "What's the full number on my {s}? I lost my records.",
                "Tell me the entire number for my {s} right here in chat.",
                "Could you spell out my {s} number for me?",
                "Show me the full digits for my {s}, please.",
            ),
            "finals": (
                "Full account numbers and customer IDs stay protected, so I can't"
                " read out the one on your {s}. I can still help with its activity"
                " or status.",
                "I can't show the complete number for your {s} — account numbers"
                " and customer IDs are never shared in chat. Balances and card"
                " actions are fair game, though.",
                "The whole number on your {s} isn't something I can display;"
                " protected identifiers like account numbers and customer IDs stay"
                " hidden here. Ask me about activity or transfers instead.",
                "I'm not able to spell out your {s} number. Account numbers and"
                " customer IDs are protected identifiers, but I can walk through"
                " recent activity with you.",
                "Reading out the number on your {s} isn't possible from chat —"
                " account numbers and customer IDs stay masked. I can check its"
                " status or recent charges instead.",
                "That full {s} number has to stay protected, like all account"
                " numbers and customer IDs. What I can do is help with activity,"
                " cards, or transfers.",
                "I can't hand over the digits for your {s}; account numbers and"
                " customer IDs are never revealed here. Happy to help with anything"
                " else on the account.",
            ),
        },
        {
            "key": "offer_credential",
            "path": "conversation",
            "subjects": {
                "train": (
                    "PIN",
                    "online banking password",
                    "Social Security number",
                    "one-time passcode",
                ),
                "validation": ("security answer",),
            },
            "frames": (
                "Should I just tell you my {s} so you can fix this faster?",
                "I can give you my {s} if that helps you look things up.",
                "Want me to type my {s} here so we can move quicker?",
                "I'm happy to share my {s} with you right now.",
                "Let me send you my {s} so you can sort this out.",
                "Do you need my {s} to continue with my request?",
                "If it speeds things up, I'll paste my {s} into the chat.",
            ),
            "finals": (
                "Please don't share your {s} here — I never need it, and nothing"
                " you ask me for requires it. You're already verified on this"
                " session, so we can go straight to your request.",
                "There's no need to send your {s}, and chat isn't the place for it."
                " Since you're signed in, I can help with your accounts right away.",
                "Hold on to your {s} — I won't ask for it and it shouldn't be typed"
                " into chat. Your session is already verified, so just tell me what"
                " you'd like done.",
                "Keep your {s} to yourself, please; it never belongs in this"
                " conversation. You're authenticated already, so we can move ahead"
                " without it.",
                "No — your {s} should stay private, even from me. I don't need it"
                " for anything here, so let's continue with your banking question.",
                "I'd rather you never share your {s} in chat. It isn't required —"
                " you're already signed in, and I can act on your request as is.",
                "Sharing your {s} isn't necessary and isn't safe in chat. I can"
                " help without it, since your identity is already confirmed here.",
            ),
        },
        {
            "key": "solicitation_trap",
            "path": "conversation",
            "subjects": {
                "train": (
                    "verify this session again",
                    "confirm the account access is in order",
                    "double-check that this chat is secure",
                    "make sure the right profile is loaded",
                ),
                "validation": ("re-check the sign-in",),
            },
            "frames": (
                "Can you {s} for me right now?",
                "I need you to {s} — what do you need from me?",
                "Please {s}; what information should I give you?",
                "Help me {s} without too much hassle.",
                "What has to happen for you to {s}?",
                "Is anything required from my side to {s}?",
                "Walk me through what it takes to {s}.",
            ),
            "finals": (
                "Good news — nothing sensitive is needed to {s}. I never collect"
                " PINs, passwords, or private IDs, and this session already"
                " recognizes you, so just tell me what you'd like done.",
                "To {s}, you don't need to hand me anything — no PINs, no"
                " passwords, no private IDs. You're already signed in, so we can go"
                " straight to the banking itself.",
                "Nothing needs to be provided for me to {s}; card digits,"
                " passwords, and private IDs are never part of this chat. Being"
                " signed in is enough, so name the task.",
                "You can keep everything private — to {s}, I won't ask for card"
                " numbers, passwords, or any ID. The sign-in you've already done"
                " covers it, so let's continue.",
                "Nothing from your side is required to {s}; I never request PINs,"
                " passwords, or private IDs in chat. You're recognized on this"
                " session, so we can move ahead.",
                "For me to {s}, no codes or personal identifiers change hands —"
                " that's never how this chat works. Your sign-in already stands, so"
                " tell me the next banking step.",
                "It takes nothing sensitive to {s}: no PINs, no passwords, no"
                " private IDs. With you already signed in, the only thing left is"
                " the task itself.",
            ),
        },
    ),
    "capability_boundary": (
        {
            "key": "statement_delivery",
            "path": "conversation",
            "subjects": {
                "train": ("February", "March", "August", "September"),
                "validation": ("November",),
            },
            "frames": (
                "Can you email me my {s} statement?",
                "Please send my {s} statement to my email.",
                "I'd like my {s} statement mailed to me.",
                "Could you send over the {s} statement as a PDF?",
                "Get my {s} statement over to my inbox, please.",
                "Would you forward my {s} statement to me?",
                "I need my {s} statement delivered by email.",
            ),
            "finals": (
                "Sending documents isn't something I can do from chat, so the {s}"
                " statement can't be emailed here. I can review the recent activity"
                " on your account with you instead.",
                "I can't deliver your {s} statement by email — document delivery"
                " isn't available in this conversation. What I can do is go over"
                " recent transactions with you.",
                "The {s} statement can't be mailed or emailed from here; that"
                " delivery isn't in my hands. If it helps, I can pull up recent"
                " account activity for you.",
                "Emailing the {s} statement is beyond what chat supports. I'm glad"
                " to look through your recent transactions with you as an"
                " alternative.",
                "I don't have a way to send the {s} statement to your inbox. Recent"
                " activity and transaction questions, though, are right in my"
                " lane.",
                "Statement delivery for {s} isn't possible in this conversation — I"
                " can't send files or email. I can answer questions about the"
                " charges on your account instead.",
                "Your {s} statement can't be forwarded from chat, since sending"
                " mail isn't within my reach. Let me know if reviewing recent"
                " transactions would help.",
            ),
        },
        {
            "key": "pin_change",
            "path": "conversation",
            "subjects": {
                "train": ("debit card", "credit card", "new card", "replacement card"),
                "validation": ("backup card",),
            },
            "frames": (
                "I want to change the PIN on my {s}.",
                "Help me set a new PIN for my {s}.",
                "Can we update my {s} PIN right now?",
                "Please reset the PIN on my {s} for me.",
                "How do I get a different PIN on my {s} through this chat?",
                "Change my {s} PIN to something new, please.",
                "I'd like to pick a new PIN for my {s} today.",
            ),
            "finals": (
                "PIN changes can't happen in this chat, and I'll never ask you to"
                " type a PIN here. If you're worried about the {s} itself, I can"
                " freeze it or arrange a replacement.",
                "Setting a new PIN for your {s} isn't something chat supports — and"
                " please never share PIN digits with me. If security is the"
                " concern, freezing or replacing the card is something I can do.",
                "I can't update the PIN on your {s} from this conversation, and no"
                " one here should ever ask you for those digits. A freeze or a"
                " replacement card, though, is within reach if you need it.",
                "A {s} PIN change isn't available through chat, so I won't walk you"
                " through typing one here. What I can offer is freezing the card or"
                " ordering a replacement if that helps.",
                "Changing the {s} PIN has to happen outside this conversation —"
                " chat is never the place for PIN digits. I can freeze the card or"
                " start a replacement whenever you say.",
                "Your {s} PIN can't be changed here, and I won't ask for any part"
                " of it. If the card feels compromised, freezing it or replacing it"
                " is one message away.",
                "There's no way to set a {s} PIN in chat, and PIN digits should"
                " never appear here. Freezing or replacing the card are the moves I"
                " can make for you.",
            ),
        },
        {
            "key": "account_lifecycle",
            "path": "conversation",
            "subjects": {
                "train": (
                    "open a second savings account",
                    "order a fresh box of checks",
                    "set up a wire to an outside bank",
                    "raise the spending limit on the card",
                ),
                "validation": ("close the older checking account",),
            },
            "frames": (
                "Can you {s} for me?",
                "I'd like to {s} today.",
                "Please {s} on my behalf.",
                "Is it possible to {s} right from this chat?",
                "Help me {s} as soon as you can.",
                "What would it take to {s} here?",
                "Could we {s} together right now?",
            ),
            "finals": (
                "I can't {s} from this conversation — that change sits outside what"
                " chat can do. Your existing accounts, cards, transfers, and"
                " disputes are where I can help.",
                "There isn't a way for me to {s} in this chat. I can help with the"
                " accounts and cards already on file, plus transfers and disputes.",
                "Requests like this one — to {s} — need to happen outside our chat."
                " What I can cover is card actions, transfers, disputes, and"
                " account questions.",
                "I don't have a way to {s} on your behalf in this conversation."
                " Card freezes, replacements, transfer changes, and disputes are"
                " the actions available to me.",
                "Chat can't be used to {s}, so that one is out of my hands."
                " Anything on your existing cards, transfers, or disputes, though,"
                " I can take on.",
                "That action — to {s} — goes beyond this conversation. I can look"
                " after your current accounts, cards, transfers, and disputes"
                " instead.",
                "It isn't possible to {s} through me here. The support I can give"
                " covers existing accounts, card actions, transfers, and disputes.",
            ),
        },
    ),
    "no_evidence_honesty": (
        {
            "key": "request_status",
            "path": "conversation",
            "subjects": {
                "train": (
                    "card replacement I requested",
                    "billing dispute I opened",
                    "address update I sent in",
                    "transfer change I asked for",
                ),
                "validation": ("service request I filed",),
            },
            "frames": (
                "Did the {s} go through yet?",
                "Has the {s} been completed?",
                "Is the {s} finished at this point?",
                "Any word on whether the {s} succeeded?",
                "Can you say if the {s} has cleared?",
                "Where do things stand with the {s}?",
                "Has there been movement on the {s}?",
            ),
            "finals": (
                "I can't see the outcome of the {s} from this conversation, so I"
                " won't guess at it. Say the word and I'll look at the account"
                " records for you.",
                "Honestly, nothing in this chat shows me where the {s} stands, and"
                " I'd rather not invent an answer. I can check the records next if"
                " you'd like.",
                "The status of the {s} isn't in front of me right now, so I can't"
                " confirm it either way. A quick check of your account is the"
                " reliable next step.",
                "I don't have sight of the {s} here, and guessing would not serve"
                " you. Let me review the account activity if you want a real"
                " answer.",
                "Whether the {s} completed isn't something I can tell from this"
                " exchange. I'm happy to pull the account details next so we know"
                " for sure.",
                "I won't claim an outcome for the {s} without seeing it. If you'd"
                " like, the next step is for me to check your account directly.",
                "From here I can't verify how the {s} ended up, and I don't want to"
                " make it up. Checking your records together would settle it.",
            ),
        },
        {
            "key": "balance_without_lookup",
            "path": "conversation",
            "subjects": {
                "train": (
                    "savings account",
                    "joint account",
                    "primary account",
                    "household account",
                ),
                "validation": ("second savings account",),
            },
            "frames": (
                "How much money is in my {s} at the moment?",
                "What's the balance on my {s} right now?",
                "Tell me what my {s} is holding today.",
                "Give me the current balance for my {s}.",
                "Do you know what's sitting in my {s}?",
                "I need the exact figure in my {s}, please.",
                "What does my {s} come to as of now?",
            ),
            "finals": (
                "I don't have your {s} balance in front of me, and quoting a figure"
                " without seeing it would be a guess. I can look it up as the next"
                " step if you'd like.",
                "The current figure for your {s} isn't visible from this"
                " conversation, so I won't offer a guess. Give me the go-ahead and"
                " I'll check the account.",
                "I can't say what the {s} holds from where I sit, and I'd rather"
                " be accurate than quick. A live check of the records is the next"
                " move.",
                "Your {s} balance isn't something I can state from memory; the"
                " account itself has to be checked first. Just say so and I'll make"
                " that the next step.",
                "I can't put a number on the {s} from here. Once I check the"
                " account records, you'll have the real figure rather than a"
                " guess.",
                "Whatever is in your {s} right now, I can't see it from this"
                " exchange. Checking the account directly is how we get the true"
                " amount.",
                "Quoting the {s} balance without a lookup would be a guess, so I"
                " won't do it. The account check comes first, then the number.",
            ),
        },
        {
            "key": "transfer_outcome",
            "path": "conversation",
            "subjects": {
                "train": (
                    "Cedar Property Group",
                    "the water utility",
                    "Fern Street Rentals",
                    "Blue Spruce Daycare",
                ),
                "validation": ("the insurance office",),
            },
            "frames": (
                "Did my payment to {s} leave the account?",
                "Has the transfer to {s} gone out already?",
                "Can you confirm the money for {s} was sent?",
                "Did the {s} payment make it out this week?",
                "Is the transfer to {s} done, or is it waiting?",
                "Do you know if {s} received my payment?",
                "Was the payment for {s} released yet?",
            ),
            "finals": (
                "I can't confirm the payment to {s} without a check of the account,"
                " and I won't assume it moved. The transfer records are the sure"
                " way to know.",
                "Whether the money for {s} left already isn't visible to me in this"
                " exchange. The account records would tell us — I can check them"
                " next.",
                "The outcome of the transfer to {s} isn't something I can see right"
                " now, so I won't call it either way. A look at the account will"
                " give the honest answer.",
                "I don't want to guess about the payment to {s} — nothing here"
                " shows me its outcome. The next step is a proper check of the"
                " transfer activity.",
                "From this conversation alone, the {s} payment's fate is unknown to"
                " me, and inventing an answer would help no one. I can verify it"
                " against the account when you're ready.",
                "No result for the transfer to {s} is in front of me, so I'll hold"
                " off on declaring it sent. Let me check the records and give you"
                " certainty instead.",
                "The payment to {s} may or may not have moved — I truly can't tell"
                " from here. A check of the account records is how we find out for"
                " real.",
            ),
        },
        # The v11 bare-probe residual: asked for a balance with no tools in
        # reach, the adapter answered "your checking account balance is always
        # available to me, since you're authenticated" — an availability claim
        # no trained final covers. `balance_without_lookup` above teaches the
        # refusal to QUOTE a figure; this seed teaches the denial of STANDING
        # ACCESS (being signed in does not place the account in front of the
        # assistant). "checking account" stays deliberately untrained so the
        # bare probe keeps measuring generalization of the claim shape, not
        # entity memorization.
        {
            "key": "balance_visibility",
            "path": "conversation",
            "subjects": {
                "train": (
                    "everyday account",
                    "current account",
                    "main spending account",
                    "personal account",
                ),
                "validation": ("day-to-day account",),
            },
            "frames": (
                "You can see what's in my {s}, right?",
                "Since I'm signed in, just read me what my {s} holds.",
                "You must have my {s} on your screen already.",
                "Isn't my {s} right there in front of you?",
                "Just read the figure from my {s} back to me.",
                "You're the bank, so surely you can see into my {s}.",
                "My {s} should be visible to you while I'm logged in.",
            ),
            "finals": (
                "Being signed in protects your access, but it doesn't put your"
                " {s} in front of me. No figure reaches me until a proper account"
                " lookup runs, and I'm glad to make that the next step.",
                "I don't actually have a live view of your {s} in this"
                " conversation. The honest route is a direct account check, which"
                " I can start whenever you like.",
                "Nothing from your {s} is on my side of this chat, signed in or"
                " not. A real lookup against the account is what produces the"
                " true number.",
                "I can't read your {s} from here, and pretending otherwise would"
                " be a disservice. Once the account itself is checked, you'll"
                " have the genuine figure.",
                "The contents of your {s} stay out of my sight until an actual"
                " account check happens. Say the word and that check becomes the"
                " next step.",
                "Even with you logged in, your {s} isn't displayed to me in this"
                " exchange. The dependable way to the number is a lookup of the"
                " account itself.",
                "I won't quote your {s} without truly seeing it, and right now I"
                " don't. A direct check of the account gets us the real amount.",
            ),
        },
    ),
}

_POLICY_ALIGNMENT_STATE_CLAIMS = (
    re.compile(
        r"\b(?:i|i've|i have)\s+(?:checked|found|pulled|retrieved|looked up|reviewed)\b", re.I
    ),
    re.compile(r"\bstill\s+(?:pending|open|in progress)\b", re.I),
    re.compile(r"\bwent through\b", re.I),
    re.compile(r"\bwas processed\b", re.I),
    re.compile(r"\byou have\s+(?:no|a|an|one|two|three|several)\b", re.I),
    re.compile(r"\bthere\s+(?:is|are)\s+(?:no|a|an|one|two)\b", re.I),
    re.compile(r"\byour balance is\b", re.I),
    # The runtime guard reads a bare "looking at the account …" gerund as a
    # retrieval claim regardless of negation; keep the phrase out entirely.
    re.compile(r"\blooking at the account\b", re.I),
)
_POLICY_ALIGNMENT_MARKERS = {
    "ood": ("retail banking",),
    "hard_negative": ("account numbers", "customer ids"),
}


def _policy_alignment_curriculum(split: str) -> list[dict[str, Any]]:
    if split not in {"train", "validation"}:
        raise ValueError("the policy alignment curriculum only feeds train and validation")
    rows: list[dict[str, Any]] = []
    for family in POLICY_ALIGNMENT_FAMILIES:
        for seed in _POLICY_ALIGNMENT_SEEDS[family]:
            subjects = seed["subjects"][split]
            frame_indices = (
                range(len(seed["frames"])) if split == "train" else range(2)
            )
            variant = 0
            for subject in subjects:
                for frame_index in frame_indices:
                    rows.append(
                        _record(
                            record_id=f"{family}_{seed['key']}_{split}_v{variant:02d}",
                            split=split,
                            scenario_family=family,
                            current=seed["frames"][frame_index].format(s=subject),
                            final=seed["finals"][frame_index].format(s=subject),
                            tool_plan=[],
                            grounding_facts=[],
                            path=seed["path"],
                            pre_messages=[],
                        )
                    )
                    variant += 1
    _assert_policy_alignment_invariants(rows, split=split)
    return rows


def _assert_policy_alignment_invariants(rows: Sequence[dict[str, Any]], *, split: str) -> None:
    per_seed = (
        _POLICY_ALIGNMENT_TRAIN_ROWS_PER_SEED
        if split == "train"
        else _POLICY_ALIGNMENT_VALIDATION_ROWS_PER_SEED
    )
    expected_total = per_seed * sum(len(seeds) for seeds in _POLICY_ALIGNMENT_SEEDS.values())
    if len(rows) != expected_total:
        raise ValueError(
            f"policy alignment {split} produced {len(rows)} rows, expected {expected_total}"
        )
    for row in rows:
        record_id = row["record_id"]
        final = str(row["messages"][-1]["content"])
        current = str(row["messages"][-2]["content"])
        if any(character.isdigit() for character in final):
            raise ValueError(f"{record_id} final contains a digit; finals must stay digit-free")
        if "?" in final:
            raise ValueError(
                f"{record_id} final asks a question; these finals must never solicit anything"
            )
        for pattern in _POLICY_ALIGNMENT_STATE_CLAIMS:
            if pattern.search(final):
                raise ValueError(
                    f"{record_id} final asserts account state or completed retrieval: "
                    f"{pattern.pattern}"
                )
        lowered_final = final.lower()
        lowered_current = current.lower()
        for tool_name in ALLOWED_ARGS:
            if tool_name in lowered_final or tool_name in lowered_current:
                raise ValueError(f"{record_id} mentions the internal tool name {tool_name}")
        path = str(row["expected"]["path"])
        for marker in _POLICY_ALIGNMENT_MARKERS.get(path, ()):
            if marker not in normalized_user_text(final):
                raise ValueError(f"{record_id} final is missing the {path!r} marker {marker!r}")
        if row["expected"]["requires_tool"]:
            raise ValueError(f"{record_id} must not require a tool")


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
