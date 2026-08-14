from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from hello_slm.banking_tool_sft_data import (
    BANKING_TOOL_SFT_CONTRACT,
    NO_TOOL_OOD_RESPONSE,
    POLICY_CHUNKS,
    SYSTEM_PROMPT,
    load_canonical_policy_corpus,
    validate_banking_tool_sft_manifest,
    validate_records,
)
from hello_slm.config import file_sha256

SPLITS = ("train", "validation", "test")
CREATED_AT = "2026-07-31T00:00:00Z"
GENERATOR_VERSION = "banking-servicing-alignment-sft/v5.2-coreference"
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
        "I am checking this in the mobile app",
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


def build_servicing_alignment_splits() -> tuple[
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


def write_servicing_alignment_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    base_sft_dir: Path = DEFAULT_BASE_SFT_DIR,
    synthetic_bank_path: Path = DEFAULT_SYNTHETIC_BANK_PATH,
) -> dict[str, Any]:
    alignment_splits, alignment_report = build_servicing_alignment_splits()
    base_manifest, base_splits = load_base_sft_splits(base_sft_dir)
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
        "base_manifest_sha256": file_sha256(base_sft_dir / "manifest.json"),
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
    manifest = {
        "format_version": 1,
        "name": "retail-bank-servicing-alignment-v5",
        "created_at": CREATED_AT,
        "contract": "banking-tool-sft-manifest",
        "schema_version": BANKING_TOOL_SFT_CONTRACT,
        "generator_version": GENERATOR_VERSION,
        "policy_corpus_revision": policy_revision,
        "tool_sft": entries,
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
    records.extend(_deictic_replace_curriculum("train"))
    return _expand_records(records, split="train")


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
    records.extend(_deictic_replace_curriculum("validation"))
    return _expand_records(records, split="validation")


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
                    "Which card should I replace? Please share the last four "
                    "digits shown in the app.",
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
                "action_prompt": "Replace that card",
                "ambiguity_prompt": "Please replace that card",
                "card_name": "Harbor Debit",
                "card_last4": "6107",
                "other_card_name": "Harbor Credit",
                "other_card_last4": "8130",
            },
            {
                "phrase_family": "that-one",
                "action_prompt": "Replace that one",
                "ambiguity_prompt": "Could you replace that one",
                "card_name": "Rewards Debit",
                "card_last4": "6249",
                "other_card_name": "Rewards Credit",
                "other_card_last4": "8241",
            },
            {
                "phrase_family": "same-card",
                "action_prompt": "Use the same card for the replacement",
                "ambiguity_prompt": "Can you use the same card for the replacement",
                "card_name": "Travel Debit",
                "card_last4": "6381",
                "other_card_name": "Travel Credit",
                "other_card_last4": "8352",
            },
            {
                "phrase_family": "just-listed",
                "action_prompt": "Replace the card you just listed",
                "ambiguity_prompt": "Please replace the card you just listed",
                "card_name": "Everyday Debit",
                "card_last4": "6473",
                "other_card_name": "Everyday Credit",
                "other_card_last4": "8463",
            },
            {
                "phrase_family": "it",
                "action_prompt": "Replace it",
                "ambiguity_prompt": "Would you replace it",
                "card_name": "Cashback Debit",
                "card_last4": "6592",
                "other_card_name": "Cashback Credit",
                "other_card_last4": "8574",
            },
            {
                "phrase_family": "yes-proceed",
                "action_prompt": "Yes, proceed with the replacement",
                "ambiguity_prompt": "Yes, please proceed with the replacement",
                "card_name": "Student Debit",
                "card_last4": "6714",
                "other_card_name": "Student Credit",
                "other_card_last4": "8685",
            },
            {
                "phrase_family": "selection",
                "action_prompt": "Proceed with that selection",
                "ambiguity_prompt": "Can you proceed with that selection",
                "card_name": "Market Debit",
                "card_last4": "6835",
                "other_card_name": "Market Credit",
                "other_card_last4": "8796",
            },
            {
                "phrase_family": "mentioned-above",
                "action_prompt": "Order a replacement for the card mentioned above",
                "ambiguity_prompt": "Please order a replacement for the card mentioned above",
                "card_name": "Premier Debit",
                "card_last4": "6947",
                "other_card_name": "Premier Credit",
                "other_card_last4": "8907",
            },
            {
                "phrase_family": "summary-reference",
                "action_prompt": "Request a replacement for the card from your summary",
                "ambiguity_prompt": "Please request a replacement for the card from your summary",
                "card_name": "Community Debit",
                "card_last4": "7059",
                "other_card_name": "Community Credit",
                "other_card_last4": "9018",
            },
            {
                "phrase_family": "continue-selection",
                "action_prompt": "Continue with replacement for that selection",
                "ambiguity_prompt": "Please continue with replacement for that selection",
                "card_name": "Secure Debit",
                "card_last4": "7183",
                "other_card_name": "Secure Credit",
                "other_card_last4": "9129",
            },
            {
                "phrase_family": "answer-reference",
                "action_prompt": "Go ahead and replace the debit card from your answer",
                "ambiguity_prompt": "Can you replace the debit card from your answer",
                "card_name": "Daily Debit",
                "card_last4": "7296",
                "other_card_name": "Daily Credit",
                "other_card_last4": "9230",
            },
            {
                "phrase_family": "same-one",
                "action_prompt": "Use the same one for replacement",
                "ambiguity_prompt": "Please use the same one for replacement",
                "card_name": "Classic Debit",
                "card_last4": "7418",
                "other_card_name": "Classic Credit",
                "other_card_last4": "9341",
            },
        )
    if split == "validation":
        return (
            {
                "phrase_family": "prior-answer",
                "action_prompt": "Swap out the card from your prior answer",
                "ambiguity_prompt": "Please swap out the card from your prior answer",
                "card_name": "Home Debit",
                "card_last4": "1526",
                "other_card_name": "Home Credit",
                "other_card_last4": "2516",
            },
            {
                "phrase_family": "identified-card",
                "action_prompt": "Order a new copy of the previously identified card",
                "ambiguity_prompt": "Please order a new copy of the previously identified card",
                "card_name": "Choice Debit",
                "card_last4": "1678",
                "other_card_name": "Choice Credit",
                "other_card_last4": "2671",
            },
            {
                "phrase_family": "previously-mentioned",
                "action_prompt": "Use the previously mentioned debit card for replacement",
                "ambiguity_prompt": "Please use the previously mentioned card for replacement",
                "card_name": "Compass Debit",
                "card_last4": "1794",
                "other_card_name": "Compass Credit",
                "other_card_last4": "2795",
            },
            {
                "phrase_family": "carry-forward",
                "action_prompt": "Carry forward the selected card and request a replacement",
                "ambiguity_prompt": "Please carry forward the selected card for replacement",
                "card_name": "Select Debit",
                "card_last4": "1836",
                "other_card_name": "Select Credit",
                "other_card_last4": "2837",
            },
        )
    raise ValueError(f"unsupported coreference curriculum split: {split}")


def _deictic_replace_curriculum(split: str) -> list[dict[str, Any]]:
    suffix = _suffix(split)
    records: list[dict[str, Any]] = []
    for spec in _coreference_curriculum_specs(split):
        family = spec["phrase_family"]
        entity_key = f"{spec['card_name']}|{spec['card_last4']}"
        action = _record(
            record_id=f"deictic_replace_{family}_{split}",
            split=split,
            scenario_family="deictic_replace_action",
            current=f"{spec['action_prompt']} {suffix}.",
            final=(
                f"Replacement is pending for your {spec['card_name']} ending in "
                f"{spec['card_last4']}."
            ),
            tool_plan=[("replace_card", {"last4": spec["card_last4"]})],
            grounding_facts=[
                f"card.last4={spec['card_last4']}",
                "card.status=replacement_pending",
            ],
            path="multi_turn",
            pre_messages=[
                _user(f"Show the card currently selected for replacement {suffix}."),
                _assistant(
                    f"The selected card is your {spec['card_name']} ending in "
                    f"{spec['card_last4']}.",
                    loss=False,
                ),
            ],
            tool_envelopes=[
                _success_envelope(
                    card={
                        "name": spec["card_name"],
                        "last4": spec["card_last4"],
                        "status": "replacement_pending",
                    }
                )
            ],
        )
        ambiguity = _record(
            record_id=f"deictic_ambiguous_{family}_{split}",
            split=split,
            scenario_family="deictic_replace_ambiguity",
            current=f"{spec['ambiguity_prompt']} {suffix}.",
            final=(
                f"I found {spec['card_name']} ending in {spec['card_last4']} and "
                f"{spec['other_card_name']} ending in {spec['other_card_last4']} in our "
                "conversation. Which should I replace? Please share the last four digits "
                "shown in the app."
            ),
            tool_plan=[],
            grounding_facts=[],
            path="clarification",
            pre_messages=[
                _user(f"Show the cards I could replace {suffix}."),
                _assistant(
                    f"I found {spec['card_name']} ending in {spec['card_last4']} and "
                    f"{spec['other_card_name']} ending in {spec['other_card_last4']}.",
                    loss=False,
                ),
            ],
        )
        for record, prompt, target in (
            (action, spec["action_prompt"], "replace_card"),
            (ambiguity, spec["ambiguity_prompt"], "clarification"),
        ):
            record["metadata"].update(
                {
                    "coreference_phrase_family": family,
                    "coreference_prompt": prompt,
                    "coreference_entity_key": entity_key,
                    "coreference_target": target,
                }
            )
        records.extend((action, ambiguity))
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
                "the last four digits shown in the app."
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
            record["validation"]["replay_hash"] = f"sha256:{_sha256_text(new_id + '|replay')}"
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
    return {
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
            "replay_hash": f"sha256:{_sha256_text(record_id + '|replay')}",
            "accepted": True,
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
        "train": "in the app",
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
    seen: set[str] = set()
    for record in records:
        normalized = _normalize(_last_user_text(record))
        if normalized in seen:
            raise ValueError(f"duplicate current user text in {split}: {normalized}")
        seen.add(normalized)


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
    if _count_pii_matches(splits):
        raise ValueError("servicing alignment splits contain PII-like text")
