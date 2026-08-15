from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from hello_slm.banking_domain_taxonomy import (
    ACTION_LABELS,
    DOMAIN_LABELS,
    ENTITY_RESOLUTION_LABELS,
    FAMILY_LABELS,
    INTENT_LABELS,
    LANE_LABELS,
    labels_for_example,
    lane_for_intent,
)

ROUTER_SPLITS = ("train", "validation", "test")

RELATION_LABELS = (
    "context_dependent",
    "agent_repair",
    "topic_shift",
    "clarification_answer",
    "resume_previous_service",
)

CLINC_EXTERNAL_OOD_LABELS = frozenset(
    {
        "alarm",
        "calendar",
        "calendar_update",
        "cook_time",
        "date",
        "definition",
        "directions",
        "distance",
        "email",
        "email_query",
        "events",
        "find_phone",
        "flight_status",
        "ingredient_substitution",
        "ingredients_list",
        "lost_luggage",
        "measurement_conversion",
        "meeting_schedule",
        "next_song",
        "nutrition_info",
        "oos",
        "play_music",
        "plug_type",
        "pto_request",
        "recipe",
        "reminder",
        "reminder_update",
        "shopping_list",
        "shopping_list_update",
        "spelling",
        "tell_joke",
        "text",
        "timer",
        "timezone",
        "todo_list",
        "todo_list_update",
        "traffic",
        "translate",
        "travel_alert",
        "travel_notification",
        "weather",
        "what_song",
    }
)

SCREENSHOT_REGRESSION_CURRENTS = frozenset(
    {
        "i didn t ask about mortgage",
        "ok thats the one i want to replace",
        "was the mailing address updated recently",
        "when was that created",
        "what is that all about when was it created",
        "what about the weather there",
        "why are you repeating yourself",
    }
)

PII_PATTERNS = (
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]?){12,19}\b"),
    re.compile(r"\b(?:\+?1[ -.]?)?(?:\(?\d{3}\)?[ -.]?)\d{3}[ -.]\d{4}\b"),
)

_INTENT_INDEX = {label: index for index, label in enumerate(INTENT_LABELS)}
_RELATION_INDEX = {label: index for index, label in enumerate(RELATION_LABELS)}


def render_router_input(
    current: str,
    history: Sequence[dict[str, Any]] | None = None,
    max_exchanges: int = 3,
    prior_dialogue_state: Mapping[str, Any] | None = None,
) -> str:
    rendered, _ = render_router_input_with_context(
        current,
        history,
        max_exchanges,
        prior_dialogue_state,
    )
    return rendered


def render_router_input_with_context(
    current: str,
    history: Sequence[dict[str, Any]] | None = None,
    max_exchanges: int = 3,
    prior_dialogue_state: Mapping[str, Any] | None = None,
) -> tuple[str, bool]:
    if not isinstance(current, str) or not current.strip():
        raise ValueError("current must be a non-empty string")
    if max_exchanges < 0:
        raise ValueError("max_exchanges must be non-negative")

    parts = []
    if prior_dialogue_state:
        parts.append(
            "[PRIOR_DIALOGUE_STATE]\n"
            + json.dumps(prior_dialogue_state, sort_keys=True, separators=(",", ":"))
        )
    parts.append(f"[CURRENT_USER]\n{current.strip()}")
    complete_exchanges = _visible_complete_exchanges(history or [])
    selected = complete_exchanges[-max_exchanges:] if max_exchanges else []
    for previous_user, previous_assistant in reversed(selected):
        parts.append(f"[PREVIOUS_ASSISTANT]\n{previous_assistant}")
        parts.append(f"[PREVIOUS_USER]\n{previous_user}")
    return "\n".join(parts), bool(selected or prior_dialogue_state)


def build_conversation_router_splits(
    sft_records_by_split: Mapping[str, Sequence[dict[str, Any]]],
    clinc_payload: dict[str, list[list[str]]],
    *,
    seed: int = 7404,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    splits: dict[str, list[dict[str, Any]]] = {split: [] for split in ROUTER_SPLITS}
    group_to_split: dict[str, str] = {}
    trajectory_to_split: dict[str, str] = {}
    counterfactual_pair_to_split: dict[str, str] = {}

    for split in ROUTER_SPLITS:
        for record in sft_records_by_split.get(split, []):
            row = _row_from_sft_record(record, split)
            if row is None:
                continue
            group_id = str(row["group_id"])
            previous_split = group_to_split.setdefault(group_id, split)
            if previous_split != split:
                raise ValueError(f"group {group_id!r} appears in both {previous_split} and {split}")
            trajectory_id = str(row["trajectory_id"])
            previous_split = trajectory_to_split.setdefault(trajectory_id, split)
            if previous_split != split:
                raise ValueError(
                    f"trajectory {trajectory_id!r} appears in both {previous_split} and {split}"
                )
            pair_id = row["counterfactual_pair_id"]
            if pair_id:
                previous_split = counterfactual_pair_to_split.setdefault(str(pair_id), split)
                if previous_split != split:
                    raise ValueError(
                        f"counterfactual pair {pair_id!r} appears in both "
                        f"{previous_split} and {split}"
                    )
            splits[split].append(row)

    for split in ROUTER_SPLITS:
        source_key = {"train": "train", "validation": "val", "test": "test"}[split]
        splits[split].extend(_external_clinc_rows(clinc_payload, source_key))
        oos_key = {"train": "oos_train", "validation": "oos_val", "test": "oos_test"}[split]
        splits[split].extend(_external_clinc_rows(clinc_payload, oos_key))

    for split in ROUTER_SPLITS:
        splits[split].extend(_synthetic_generalization_rows(splits[split], split, seed))
        splits[split].extend(_targeted_use_case_rows(split))
        splits[split].extend(_resume_trajectory_rows(split))
        splits[split].extend(_state_conditioned_negative_rows(split))
        splits[split].extend(_ineligible_entity_rows(split))
    splits["test"].extend(_held_out_regression_rows())

    deduplicated, duplicates_removed = _deduplicate_across_splits(splits)
    report = {
        "contract": "banking-conversation-router-data-report",
        "seed": seed,
        "intent_labels": INTENT_LABELS,
        "relation_labels": RELATION_LABELS,
        "domain_labels": DOMAIN_LABELS,
        "lane_labels": LANE_LABELS,
        "family_labels": FAMILY_LABELS,
        "action_labels": ACTION_LABELS,
        "entity_resolution_labels": ENTITY_RESOLUTION_LABELS,
        "split_counts": {split: len(rows) for split, rows in deduplicated.items()},
        "kind_counts": {
            split: dict(Counter(str(row["example_kind"]) for row in rows))
            for split, rows in deduplicated.items()
        },
        "domain_counts": {
            split: dict(Counter(int(row["domain_label"]) for row in rows))
            for split, rows in deduplicated.items()
        },
        "hierarchical_domain_counts": {
            split: dict(Counter(str(row["domain_name"]) for row in rows))
            for split, rows in deduplicated.items()
        },
        "action_counts": {
            split: dict(Counter(str(row["action_name"]) for row in rows))
            for split, rows in deduplicated.items()
        },
        "intent_counts": {
            split: dict(Counter(str(row["intent"]) for row in rows))
            for split, rows in deduplicated.items()
        },
        "cross_split_duplicates_removed": duplicates_removed,
        "pii_matches": _count_pii_matches(
            str(row["text"]) for rows in deduplicated.values() for row in rows
        ),
        "leakage": _leakage_report(deduplicated),
    }
    return deduplicated, report


def normalize_router_text(text: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in text).split()
    )


def rows_jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(rows_jsonl_bytes(rows)).hexdigest()


def _row_from_sft_record(record: dict[str, Any], split: str) -> dict[str, Any] | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    current = _last_visible_user(messages)
    if current is None:
        return None
    if _is_screenshot_regression_text(current):
        return None
    history = _visible_history_before_current(messages)
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    source = (
        "self-authored-coreference-shadow"
        if metadata.get("trainable") is False and metadata.get("coreference_pair_id")
        else "self-authored-banking-tool-sft"
    )
    path = str(metadata.get("path", ""))
    domain_label = 0 if path == "ood" else 1
    intent = None if path == "ood" else _intent_for_record(record, current)
    prior_dialogue_state = record.get("prior_dialogue_state")
    if prior_dialogue_state is None:
        prior_dialogue_state = metadata.get("prior_dialogue_state")
    if prior_dialogue_state is not None and not isinstance(prior_dialogue_state, Mapping):
        raise ValueError("prior_dialogue_state must be an object")
    relations = _relations_for_current(current, history, prior_dialogue_state)
    return _make_row(
        current=current,
        history=history,
        domain_label=domain_label,
        intent=intent,
        relation_names=relations,
        example_kind=_example_kind_for_record(record, history, relations),
        source=source,
        source_split=split,
        group_id=_group_id(record),
        trajectory_id=_trajectory_id(record),
        prior_dialogue_state=prior_dialogue_state,
        path=path,
        tool_names=_expected_tool_names(record),
        coreference_target=str(metadata.get("coreference_target", "")),
        actionable_entity_count=_optional_int(metadata.get("actionable_card_count")),
        explicit_entity_resolution=str(metadata.get("entity_resolution", "")),
        counterfactual_pair_id=_optional_text(metadata.get("coreference_pair_id")),
        counterfactual_target=_optional_text(metadata.get("coreference_target")),
        counterfactual_phrase_family=_optional_text(metadata.get("coreference_phrase_family")),
    )


def _external_clinc_rows(
    clinc_payload: dict[str, list[list[str]]],
    source_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(clinc_payload.get(source_key, [])):
        if len(item) != 2:
            raise ValueError(f"invalid CLINC row in {source_key}: {item!r}")
        current, label = str(item[0]).strip(), str(item[1]).strip()
        if not current or label not in CLINC_EXTERNAL_OOD_LABELS:
            continue
        rows.append(
            _make_row(
                current=current,
                history=[],
                domain_label=0,
                intent=None,
                relation_names=[],
                example_kind="clinc_external_ood",
                source="UCI/clinc150",
                source_split=source_key,
                group_id=f"clinc150|{source_key}|{label}|{index}",
            )
        )
    return rows


def _synthetic_generalization_rows(
    rows: Sequence[dict[str, Any]],
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    banking = [
        row
        for row in rows
        if row["domain_label"] == 1
        and row["intent"] is not None
        and row["source"] != "self-authored-coreference-shadow"
    ]
    external = [row for row in rows if row["domain_label"] == 0]
    if not banking:
        return []
    ordered = sorted(banking, key=lambda row: _stable_rank(seed, split, str(row["text"])))
    generated = []
    pair_count = min(len(external), len(ordered))
    for index, external_row in enumerate(external[:pair_count]):
        anchor = ordered[index % len(ordered)]
        generated.append(
            _make_row(
                current=str(external_row["current_text"]),
                history=_history_from_anchor(anchor),
                domain_label=0,
                intent=None,
                relation_names=["topic_shift"],
                example_kind="external_topic_shift",
                source="self-authored-router-v5-synthetic+UCI/clinc150",
                source_split=split,
                group_id=f"{anchor['group_id']}|router-v5|topic-shift|{index}",
            )
        )
        generated.append(
            _make_row(
                current=str(anchor["current_text"]),
                history=[
                    *_history_from_anchor(anchor),
                    {
                        "role": "user",
                        "content": str(external_row["current_text"]),
                    },
                    {
                        "role": "assistant",
                        "content": "That topic is outside the banking services I support.",
                    },
                ],
                domain_label=1,
                intent=str(anchor["intent"]),
                relation_names=["topic_shift"],
                example_kind="banking_topic_shift",
                source="self-authored-router-v5-synthetic+UCI/clinc150",
                source_split=split,
                group_id=(f"{anchor['group_id']}|router-v5|banking-topic-shift|{index}"),
                action=str(anchor["action_name"]),
                entity_resolution=str(anchor["entity_resolution_name"]),
            )
        )
    return generated


def _targeted_use_case_rows(split: str) -> list[dict[str, Any]]:
    case_histories = (
        [
            {"role": "user", "content": "Show my recent service requests."},
            {
                "role": "assistant",
                "content": "Your mailing-address update case is closed.",
            },
        ],
        [
            {"role": "user", "content": "What support cases are on my profile?"},
            {
                "role": "assistant",
                "content": "I found an address-change case created last month.",
            },
        ],
        [
            {"role": "user", "content": "Check the status of my address request."},
            {
                "role": "assistant",
                "content": "The address request is recorded as a closed service case.",
            },
        ],
        [
            {"role": "user", "content": "List the customer-service items we discussed."},
            {
                "role": "assistant",
                "content": "The relevant item is a completed mailing-address case.",
            },
        ],
    )
    card_histories = (
        [
            {"role": "user", "content": "List the debit cards on my profile."},
            {
                "role": "assistant",
                "content": "The active debit card ends in 4821.",
            },
        ],
        [
            {"role": "user", "content": "Which of my cards is active?"},
            {
                "role": "assistant",
                "content": "Your Everyday Visa Debit ending in 4821 is active.",
            },
        ],
        [
            {"role": "user", "content": "Show the two cards you found."},
            {
                "role": "assistant",
                "content": "I found cards ending in 4821 and 7319; 4821 is active.",
            },
        ],
        [
            {"role": "user", "content": "Help me choose the debit card to replace."},
            {
                "role": "assistant",
                "content": "The first option is the active card ending in 4821.",
            },
        ],
    )
    wrong_answer_histories = (
        [
            {"role": "user", "content": "Tell me about my address service case."},
            {
                "role": "assistant",
                "content": "Mortgage applications usually require an eligibility review.",
            },
        ],
        [
            {"role": "user", "content": "When did my address request start?"},
            {
                "role": "assistant",
                "content": "Home-loan rates vary with the selected product.",
            },
        ],
        [
            {"role": "user", "content": "I asked about the mailing-address case."},
            {
                "role": "assistant",
                "content": "A mortgage account may have closing costs.",
            },
        ],
        [
            {"role": "user", "content": "Check that customer-service request."},
            {
                "role": "assistant",
                "content": "Loan applications are reviewed by a lender.",
            },
        ],
    )
    prompts = _targeted_prompts(split)
    modifiers = _targeted_modifiers(split)
    history_limit = {"train": 4, "validation": 2, "test": 2}[split]
    rows: list[dict[str, Any]] = []
    categories = (
        (
            "service_case_detail",
            prompts["service_case_detail"],
            case_histories[:history_limit],
            "view_service_cases",
            ["context_dependent"],
            "targeted_contextual_followup",
        ),
        (
            "card_selection_action",
            prompts["card_selection_action"],
            card_histories[:history_limit],
            "replace_card",
            ["context_dependent", "clarification_answer"],
            "targeted_clarification_answer",
        ),
        (
            "repetition_repair",
            prompts["repetition_repair"],
            wrong_answer_histories[:history_limit],
            "view_service_cases",
            ["context_dependent", "agent_repair"],
            "targeted_agent_repair",
        ),
        (
            "wrong_topic_repair",
            prompts["wrong_topic_repair"],
            wrong_answer_histories[:history_limit],
            "view_service_cases",
            ["context_dependent", "agent_repair", "topic_shift"],
            "targeted_wrong_topic_repair",
        ),
    )
    for (
        category,
        category_prompts,
        histories,
        intent,
        relations,
        example_kind,
    ) in categories:
        for prompt_index, prompt in enumerate(category_prompts):
            for modifier_index, modifier in enumerate(modifiers):
                for history_index, history in enumerate(histories):
                    rows.append(
                        _make_row(
                            current=f"{prompt}{modifier}",
                            history=history,
                            domain_label=1,
                            intent=intent,
                            relation_names=relations,
                            example_kind=example_kind,
                            source="self-authored-router-v5-use-case-alignment",
                            source_split=split,
                            group_id=(
                                f"targeted|{split}|{category}|{prompt_index}|"
                                f"{modifier_index}|{history_index}"
                            ),
                        )
                    )
    for prompt_index, prompt in enumerate(prompts["standalone_address_case"]):
        for modifier_index, modifier in enumerate(modifiers):
            rows.append(
                _make_row(
                    current=f"{prompt}{modifier}",
                    history=[],
                    domain_label=1,
                    intent="view_service_cases",
                    relation_names=[],
                    example_kind="targeted_service_case",
                    source="self-authored-router-v5-use-case-alignment",
                    source_split=split,
                    group_id=(
                        f"targeted|{split}|standalone-address|{prompt_index}|{modifier_index}"
                    ),
                )
            )
    return rows


def _resume_trajectory_rows(split: str) -> list[dict[str, Any]]:
    intent_frames = {
        "view_accounts": (
            "Show me my account balances.",
            "Which accounts should I include?",
            "How is available balance different from current balance?",
        ),
        "view_cards": (
            "Show me my debit cards.",
            "Would you like all active cards?",
            "What does an active card status mean?",
        ),
        "freeze_card": (
            "I need to freeze a card.",
            "Which card should I freeze?",
            "What happens to pending payments after a card is frozen?",
        ),
        "replace_card": (
            "I need to replace my debit card.",
            "Which card should I replace?",
            "Is there a fee for a replacement card?",
        ),
        "view_transactions": (
            "Show my recent purchases.",
            "How many transactions should I retrieve?",
            "How long do pending card purchases take to post?",
        ),
        "dispute_transaction": (
            "I need to dispute a purchase.",
            "Which transaction should I look for?",
            "How long does a purchase dispute review take?",
        ),
        "view_transfers": (
            "Show my recent transfers.",
            "Should I include completed transfers?",
            "When does a scheduled transfer become final?",
        ),
        "cancel_transfer": (
            "I need to cancel a scheduled transfer.",
            "Which transfer should I cancel?",
            "When can a scheduled transfer still be cancelled?",
        ),
        "view_service_cases": (
            "Show my recent support requests.",
            "Should I include closed service cases?",
            "How long are closed service cases retained?",
        ),
    }
    prompt_stems = {
        "train": (
            "Let's continue with the original request",
            "Go back to the banking task we paused",
            "Please resume what we were doing before",
            "Okay, return to that earlier request",
            "Thanks; carry on with the first task",
            "Now finish the service request we started",
        ),
        "validation": (
            "Pick up the request from before the policy question",
            "Return to the unfinished banking action",
            "Continue the task that was on hold",
            "Let's get back to my initial request",
            "Resume the service item we paused",
            "Proceed with what I originally asked for",
        ),
        "test": (
            "Reopen the task we were handling earlier",
            "Take me back to the service request in progress",
            "Carry on from where we stopped before that question",
            "Finish the banking request that is still pending",
            "Move back to the issue we had underway",
            "Continue with the original matter now",
        ),
    }[split]
    modifiers = (
        (".", " please.", " now.", " from where we left off.") if split == "train" else (".",)
    )
    intent_subjects = {
        "view_accounts": "account-balance",
        "view_cards": "card-list",
        "freeze_card": "card-freeze",
        "replace_card": "card-replacement",
        "view_transactions": "transaction-history",
        "dispute_transaction": "purchase-dispute",
        "view_transfers": "transfer-history",
        "cancel_transfer": "transfer-cancellation",
        "view_service_cases": "service-case",
    }
    explicit_resume_templates = {
        "train": (
            "Let's resume the {subject} request.",
            "Return to working on my {subject} task.",
        ),
        "validation": ("Continue handling the {subject} matter.",),
        "test": ("Go back to the {subject} issue that we paused.",),
    }[split]
    rows = []
    for intent, (anchor_user, anchor_assistant, policy_question) in intent_frames.items():
        prior_state = {
            "version": 1,
            "pending_servicing": {
                "intent": intent,
                "anchor_user_message": anchor_user,
                "anchor_assistant_message": anchor_assistant,
                "phase": "awaiting_user",
            },
            "knowledge_detour_active": True,
        }
        history = [
            {"role": "user", "content": policy_question},
            {
                "role": "assistant",
                "content": "I can explain the applicable policy.",
            },
        ]
        for prompt_index, stem in enumerate(prompt_stems):
            for modifier_index, modifier in enumerate(modifiers):
                trajectory_id = f"v5-resume|{split}|{intent}|{prompt_index}|{modifier_index}"
                rows.append(
                    _make_row(
                        current=f"{stem}{modifier}",
                        history=history,
                        prior_dialogue_state=prior_state,
                        domain_label=1,
                        intent=intent,
                        relation_names=[
                            "context_dependent",
                            "resume_previous_service",
                        ],
                        example_kind="resume_previous_service",
                        source="self-authored-router-v5-resume-trajectory",
                        source_split=split,
                        group_id=trajectory_id,
                        trajectory_id=trajectory_id,
                    )
                )
        for prompt_index, template in enumerate(explicit_resume_templates):
            trajectory_id = f"v5-resume-explicit|{split}|{intent}|{prompt_index}"
            rows.append(
                _make_row(
                    current=template.format(subject=intent_subjects[intent]),
                    history=history,
                    prior_dialogue_state=prior_state,
                    domain_label=1,
                    intent=intent,
                    relation_names=[
                        "context_dependent",
                        "resume_previous_service",
                    ],
                    example_kind="resume_previous_service",
                    source="self-authored-router-v5-resume-trajectory",
                    source_split=split,
                    group_id=trajectory_id,
                    trajectory_id=trajectory_id,
                )
            )
    return rows


def _state_conditioned_negative_rows(split: str) -> list[dict[str, Any]]:
    """Prevent prior state from overriding an explicit current-turn meaning."""

    frames = {
        "view_accounts": "Show my account balances.",
        "view_cards": "Show my debit cards.",
        "freeze_card": "Freeze my debit card.",
        "replace_card": "Replace my debit card.",
        "view_transactions": "Show my recent purchases.",
        "dispute_transaction": "Dispute a purchase I did not make.",
        "view_transfers": "Show my recent transfers.",
        "cancel_transfer": "Cancel my pending transfer.",
        "view_service_cases": "Show my support requests.",
    }
    detour_questions = {
        "view_accounts": "What are the rules for interest on these accounts?",
        "view_cards": "What protections apply to debit cards?",
        "freeze_card": "What happens to pending payments after a card is frozen?",
        "replace_card": "Is there a fee for a replacement card?",
        "view_transactions": "How long do pending card purchases take to post?",
        "dispute_transaction": "What is the policy for reviewing a purchase dispute?",
        "view_transfers": "When does a scheduled transfer become final?",
        "cancel_transfer": "When can a scheduled transfer still be cancelled?",
        "view_service_cases": "How long are closed service cases retained?",
    }
    switch_prompts = cast(
        Mapping[str, tuple[str, ...]],
        (
            {
                "train": {
                    "view_accounts": (
                        "Actually, show my account balances instead.",
                        "Change direction and bring up my accounts.",
                        "I want to check my balances now, not continue that.",
                    ),
                    "view_cards": (
                        "Change of plan: list my debit cards.",
                        "Switch over and show the cards on my profile.",
                        "Put that aside because I need my card list.",
                    ),
                    "freeze_card": (
                        "Never mind that—freeze my card instead.",
                        "Stop the old task and lock my debit card.",
                        "This is a new request: block my card now.",
                    ),
                    "replace_card": (
                        "Switch tasks and replace my debit card.",
                        "Forget the earlier request and order me a new card.",
                        "I need a replacement debit card instead of that.",
                    ),
                    "view_transactions": (
                        "Instead, show my recent purchases.",
                        "Move to a different task and list my transactions.",
                        "Pause that; I want to review recent charges.",
                    ),
                    "dispute_transaction": (
                        "Actually, I need to dispute a purchase.",
                        "Change tasks because I want to challenge a charge.",
                        "Leave that and open a dispute for an unfamiliar purchase.",
                    ),
                    "view_transfers": (
                        "Leave that for now and show my transfers.",
                        "Switch topics and list my recent money transfers.",
                        "Pause the old request; bring up my transfer history.",
                    ),
                    "cancel_transfer": (
                        "Switch to cancelling my pending transfer.",
                        "Stop that task and cancel the transfer instead.",
                        "I changed my request: revoke my pending transfer.",
                    ),
                    "view_service_cases": (
                        "Instead, show my open support requests.",
                        "Move away from that and list my service cases.",
                        "Change tasks; I want to see my customer-service requests.",
                    ),
                },
                "validation": {
                    "view_accounts": ("Pause that and display my balances.",),
                    "view_cards": ("I want to see my cards now instead.",),
                    "freeze_card": ("Change tasks: lock my debit card now.",),
                    "replace_card": ("Leave this and order a replacement card.",),
                    "view_transactions": ("Move over to my latest transactions.",),
                    "dispute_transaction": ("Stop here; I need to challenge a purchase.",),
                    "view_transfers": ("Put this aside and list my transfers.",),
                    "cancel_transfer": ("Forget that for now and cancel my transfer.",),
                    "view_service_cases": ("Switch over to my customer-service cases.",),
                },
                "test": {
                    "view_accounts": ("Let's do something else: check my balances.",),
                    "view_cards": ("Actually take me to my card list.",),
                    "freeze_card": ("Drop the prior task and freeze my card.",),
                    "replace_card": ("I changed my mind; replace the debit card.",),
                    "view_transactions": ("Leave the old request and show my purchases.",),
                    "dispute_transaction": ("New request: dispute an unfamiliar charge.",),
                    "view_transfers": ("Set that aside and bring up my transfers.",),
                    "cancel_transfer": ("Do not continue that; cancel my pending transfer.",),
                    "view_service_cases": ("Put that on hold and show my support cases.",),
                },
            }
        )[split],
    )
    ood_prompts = {
        "train": (
            "What is the weather tomorrow?",
            "Is rain expected near me this weekend?",
            "Give me the local forecast for this week.",
            "Write a Python sorting function.",
            "Give me a pasta recipe.",
            "Play some jazz music.",
            "Set a timer for ten minutes.",
            "Translate hello into Japanese.",
        ),
        "validation": (
            "Will it rain near me this weekend?",
            "Draft a JavaScript web scraper.",
            "How long should I roast potatoes?",
            "Queue my workout playlist.",
            "Remind me to call the dentist.",
            "Translate good evening into Italian.",
        ),
        "test": (
            "Do I need an umbrella on Tuesday?",
            "Explain how to write a Rust macro.",
            "What ingredients go in vegetable soup?",
            "Start the next song in my playlist.",
            "Add milk to my shopping list.",
            "How do you say thank you in Korean?",
        ),
    }[split]
    policy_detours = (
        (
            "mortgage_opening",
            "What are the general rules for opening a mortgage?",
            "Mortgage eligibility can depend on income, debt, credit history, and the "
            "property. Supporting documents, timing, and fees vary by application.",
        ),
        (
            "deposit_opening",
            "What are the requirements for opening a checking or savings account?",
            "Account opening generally requires identity, address, eligibility, and "
            "funding information. Required documents, timing, fees, and next steps vary "
            "by account.",
        ),
        (
            "deposit_overdraft",
            "How does the bank handle deposit account overdrafts?",
            "Overdraft treatment depends on account settings and transaction type. "
            "Eligibility, fees, review timing, and next steps can differ by account.",
        ),
        (
            "savings_interest",
            "How is savings interest calculated?",
            "Savings interest depends on the applicable rate, balance method, and "
            "posting schedule. Account eligibility, timing, and fees can vary.",
        ),
        (
            "card_dispute",
            "What is the policy for reviewing a card purchase dispute?",
            "A card dispute review may require transaction details and supporting "
            "documents. Eligibility, investigation timing, fees, and next steps depend "
            "on the claim.",
        ),
        (
            "card_replacement",
            "What is the policy for replacing a lost or damaged card?",
            "Card replacement can require identity and delivery confirmation. "
            "Eligibility, delivery timing, fees, and next steps vary by circumstances.",
        ),
        (
            "card_fraud",
            "What is the policy for reporting suspected card fraud?",
            "A fraud report may require transaction and identity details. Review timing, "
            "eligibility protections, fees, and next steps depend on the report.",
        ),
    )
    social_prompt_families = {
        "train": (
            ("gratitude_direct_train", ("Much appreciated.", "I appreciate that.")),
            (
                "gratitude_explanation_train",
                ("Thanks for the explanation.", "Thank you for clarifying."),
            ),
            ("acknowledgement_train", ("All right.", "Understood.")),
            ("comprehension_train", ("That makes sense.", "I understand now.")),
            (
                "helpfulness_train",
                (
                    "That was helpful.",
                    "This clears things up.",
                    "That is useful.",
                    "This really helps.",
                    "That was useful.",
                    "This is helpful.",
                ),
            ),
            ("greeting_train", ("Good morning.", "Hi there.")),
            (
                "wellbeing_train",
                (
                    "Hope you are well.",
                    "How is your day going?",
                    "How have you been?",
                    "Are you doing okay?",
                    "How are things with you?",
                    "Is everything going well for you?",
                ),
            ),
            ("disengagement_train", ("Leave it there for now.", "We can stop here.")),
        ),
        "validation": (
            ("gratitude_direct_validation", ("Appreciate it.",)),
            ("gratitude_explanation_validation", ("Thanks for walking me through that.",)),
            ("acknowledgement_validation", ("All clear.",)),
            ("comprehension_validation", ("I follow you.",)),
            ("helpfulness_validation", ("That clears it up.",)),
            ("greeting_validation", ("Good afternoon.",)),
            ("wellbeing_validation", ("Hope things are going well.",)),
            ("disengagement_validation", ("Let's leave it for now.",)),
        ),
        "test": (
            ("gratitude_direct_test", ("Thanks",)),
            ("gratitude_explicit_test", ("Thank you",)),
            ("gratitude_ack_test", ("Okay, thanks",)),
            ("acknowledgement_test", ("Got it",)),
            ("helpfulness_test", ("That helps",)),
            ("greeting_test", ("Hello",)),
            ("wellbeing_test", ("How are you?",)),
            ("disengagement_test", ("Never mind",)),
        ),
    }[split]
    policy_followup_families = {
        "train": (
            (
                "documents_train",
                (
                    "Which supporting records are usually required?",
                    "What paperwork is typically involved?",
                ),
            ),
            (
                "timing_train",
                ("What is the usual timeframe for that?", "How soon is that normally completed?"),
            ),
            (
                "eligibility_train",
                ("What makes someone qualify for that?", "Who generally meets those requirements?"),
            ),
            (
                "fees_train",
                ("What charges can apply to that?", "Does that normally cost anything?"),
            ),
            (
                "next_steps_train",
                ("What should the customer do after that?", "Which step comes next?"),
            ),
        ),
        "validation": (
            ("documents_validation", ("What evidence could be requested?",)),
            ("timing_validation", ("What sort of wait is typical?",)),
            ("eligibility_validation", ("How do they decide who qualifies?",)),
            ("fees_validation", ("Are charges ever involved?",)),
            ("next_steps_validation", ("Where would someone go from there?",)),
        ),
        "test": (
            ("documents_test", ("What documents might you need?",)),
            ("timing_test", ("How long could that take?",)),
            ("eligibility_test", ("Who is eligible for that?",)),
            ("fees_test", ("Would there be any fees?",)),
            ("next_steps_test", ("What happens next?",)),
        ),
    }[split]
    resume_policy_templates = {
        "train": (
            "Continue the original {subject} request.",
            "Return to the {subject} task we paused.",
        ),
        "validation": ("Pick up the pending {subject} request again.",),
        "test": ("Go back to the {subject} issue that we paused.",),
    }[split]
    intent_subjects = {
        "view_accounts": "account-balance",
        "view_cards": "card-list",
        "freeze_card": "card-freeze",
        "replace_card": "card-replacement",
        "view_transactions": "transaction-history",
        "dispute_transaction": "purchase-dispute",
        "view_transfers": "transfer-history",
        "cancel_transfer": "transfer-cancellation",
        "view_service_cases": "service-case",
    }
    orphan_resume_prompts = {
        "train": (
            "Let's continue the request from before.",
            "Go back to the task we paused.",
            "Resume the original issue.",
            "Continue the unfinished banking matter.",
            "Take me back to the earlier service request.",
            "Finish the task that was left open.",
            "Return to the work we stopped.",
            "Pick up the previous request again.",
            "Carry on with the unresolved issue.",
        ),
        "validation": (
            "Pick up where we stopped earlier.",
            "Return to my unfinished request.",
            "Carry on with the prior task.",
        ),
        "test": (
            "Continue the matter that was on hold.",
            "Take me back to what we were doing.",
            "Finish the earlier request now.",
        ),
    }[split]

    rows = []
    for active_intent, anchor_user in frames.items():
        prior_state = {
            "version": 1,
            "pending_servicing": {
                "intent": active_intent,
                "anchor_user_message": anchor_user,
                "anchor_assistant_message": "Which record should I use?",
                "phase": "awaiting_user",
            },
            "knowledge_detour_active": True,
        }
        history = [
            {"role": "user", "content": detour_questions[active_intent]},
            {"role": "assistant", "content": "I can explain the applicable policy."},
        ]
        for target_intent, prompts in switch_prompts.items():
            if target_intent == active_intent:
                continue
            for prompt_index, current in enumerate(prompts):
                key = f"state-switch|{split}|{active_intent}|{target_intent}|{prompt_index}"
                rows.append(
                    _make_row(
                        current=current,
                        history=history,
                        prior_dialogue_state=prior_state,
                        domain_label=1,
                        intent=target_intent,
                        relation_names=["topic_shift"],
                        example_kind="state_intent_switch",
                        source="self-authored-router-v5-state-negatives",
                        source_split=split,
                        group_id=key,
                        trajectory_id=key,
                    )
                )
        for prompt_index, current in enumerate(ood_prompts):
            key = f"state-ood|{split}|{active_intent}|{prompt_index}"
            rows.append(
                _make_row(
                    current=current,
                    history=history,
                    prior_dialogue_state=prior_state,
                    domain_label=0,
                    intent=None,
                    relation_names=["topic_shift"],
                    example_kind="state_ood_detour",
                    source="self-authored-router-v5-state-negatives",
                    source_split=split,
                    group_id=key,
                    trajectory_id=key,
                )
            )
        for policy_topic, policy_question, policy_answer in policy_detours:
            policy_history = [
                {"role": "user", "content": anchor_user},
                {"role": "assistant", "content": "Which record should I use?"},
                {"role": "user", "content": policy_question},
                {"role": "assistant", "content": policy_answer},
            ]
            for target_intent, prompts in switch_prompts.items():
                if target_intent == active_intent:
                    continue
                for prompt_index, current in enumerate(prompts):
                    key = (
                        f"state-switch-policy|{split}|{active_intent}|{policy_topic}|"
                        f"{target_intent}|{prompt_index}"
                    )
                    rows.append(
                        _make_row(
                            current=current,
                            history=policy_history,
                            prior_dialogue_state=prior_state,
                            domain_label=1,
                            intent=target_intent,
                            relation_names=["topic_shift"],
                            example_kind="state_intent_switch",
                            source="self-authored-router-v5-state-negatives",
                            source_split=split,
                            group_id=key,
                            trajectory_id=key,
                        )
                    )
            for prompt_index, current in enumerate(ood_prompts):
                key = f"state-ood-policy|{split}|{active_intent}|{policy_topic}|{prompt_index}"
                rows.append(
                    _make_row(
                        current=current,
                        history=policy_history,
                        prior_dialogue_state=prior_state,
                        domain_label=0,
                        intent=None,
                        relation_names=["topic_shift"],
                        example_kind="state_ood_detour",
                        source="self-authored-router-v5-state-negatives",
                        source_split=split,
                        group_id=key,
                        trajectory_id=key,
                    )
                )
            for prompt_index, template in enumerate(resume_policy_templates):
                key = f"state-resume-policy|{split}|{active_intent}|{policy_topic}|{prompt_index}"
                rows.append(
                    _make_row(
                        current=template.format(subject=intent_subjects[active_intent]),
                        history=policy_history,
                        prior_dialogue_state=prior_state,
                        domain_label=1,
                        intent=active_intent,
                        relation_names=["context_dependent", "resume_previous_service"],
                        example_kind="resume_previous_service",
                        source="self-authored-router-v5-resume-trajectory",
                        source_split=split,
                        group_id=key,
                        trajectory_id=key,
                    )
                )
            for family, prompts in policy_followup_families:
                for prompt_index, current in enumerate(prompts):
                    key = (
                        f"state-policy|{split}|{active_intent}|{policy_topic}|"
                        f"{family}|{prompt_index}"
                    )
                    rows.append(
                        _make_row(
                            current=current,
                            history=policy_history,
                            prior_dialogue_state=prior_state,
                            domain_label=1,
                            intent="policy_knowledge",
                            relation_names=["context_dependent"],
                            example_kind=(
                                "state_policy_followup"
                                if split == "train"
                                else "heldout_policy_followup_generalization"
                            ),
                            source="self-authored-router-v5-state-negatives",
                            source_split=split,
                            group_id=f"state-policy-family|{family}",
                            trajectory_id=key,
                        )
                    )
            for family, prompts in social_prompt_families:
                for prompt_index, current in enumerate(prompts):
                    key = (
                        f"state-social|{split}|{active_intent}|{policy_topic}|"
                        f"{family}|{prompt_index}"
                    )
                    rows.append(
                        _make_row(
                            current=current,
                            history=policy_history,
                            prior_dialogue_state=prior_state,
                            domain_label=1,
                            intent="conversation",
                            relation_names=[],
                            example_kind=(
                                "state_social_detour"
                                if split == "train"
                                else "heldout_social_generalization"
                            ),
                            source="self-authored-router-v5-state-negatives",
                            source_split=split,
                            group_id=f"state-social-family|{family}",
                            trajectory_id=key,
                        )
                    )
    for prompt_index, current in enumerate(orphan_resume_prompts):
        key = f"orphan-resume|{split}|{prompt_index}"
        rows.append(
            _make_row(
                current=current,
                history=[],
                prior_dialogue_state=None,
                domain_label=1,
                intent="conversation",
                relation_names=[],
                example_kind="state_orphan_resume",
                source="self-authored-router-v5-state-negatives",
                source_split=split,
                group_id=key,
                trajectory_id=key,
            )
        )
    return rows


def _ineligible_entity_rows(split: str) -> list[dict[str, Any]]:
    """Teach the router to preserve intent while blocking an ineligible entity."""

    examples = cast(
        Mapping[str, tuple[str, ...]],
        {
            "train": {
                "freeze_card": (
                    "Freeze that closed card anyway.",
                    "Try to lock the inactive card.",
                    "Block the card that is already closed.",
                    "Put a freeze on the permanently disabled card.",
                    "Lock the card that has already been replaced.",
                    "Try freezing the expired card shown above.",
                    "Block that unusable card once more.",
                    "Apply a freeze to the card marked closed.",
                ),
                "replace_card": (
                    "Replace that card even though a replacement is pending.",
                    "Order another copy of the card already being replaced.",
                    "Replace the card whose replacement request is still open.",
                    "Request another replacement for the card already replaced.",
                    "Send a new card for the closed card shown above.",
                    "Replace that card despite the active replacement order.",
                    "Order a copy of the card that is no longer replaceable.",
                    "Proceed with replacing the card marked ineligible.",
                ),
                "dispute_transaction": (
                    "Dispute that reversed transaction anyway.",
                    "Open a dispute on the charge that was already reversed.",
                    "Challenge the transaction marked as fully reversed.",
                    "Dispute the charge whose refund has already posted.",
                    "Challenge that purchase even though a dispute is already open.",
                    "Open another dispute for the refunded transaction.",
                    "Proceed against the charge marked ineligible for dispute.",
                    "Try disputing the transaction that was already resolved.",
                ),
                "cancel_transfer": (
                    "Cancel that completed transfer anyway.",
                    "Stop the transfer that has already completed.",
                    "Try to cancel the money transfer marked complete.",
                    "Cancel the transfer that was already cancelled.",
                    "Revoke that settled transfer once more.",
                    "Stop the transfer whose cancellation window has closed.",
                    "Proceed with cancelling the transfer marked ineligible.",
                    "Try to reverse that completed transfer from here.",
                ),
            },
            "validation": {
                "freeze_card": ("Lock the card shown as closed.",),
                "replace_card": ("Replace the card with an open replacement order.",),
                "dispute_transaction": ("Dispute the purchase shown as reversed.",),
                "cancel_transfer": ("Cancel the transfer shown as completed.",),
            },
            "test": {
                "freeze_card": ("Put a freeze on the inactive card.",),
                "replace_card": ("Request a replacement for the card already in replacement.",),
                "dispute_transaction": ("Challenge the transaction that has been reversed.",),
                "cancel_transfer": ("Revoke the transfer that is already complete.",),
            },
        }[split],
    )
    histories = {
        "freeze_card": (
            ("Show my cards.", "The card ending in 1846 is closed and cannot be used."),
            ("Check the old debit card.", "That card was permanently disabled."),
            ("What happened to my expiring card?", "It expired and is not eligible to freeze."),
            ("Show the replaced card.", "That prior card is closed after replacement."),
        ),
        "replace_card": (
            ("Check my card replacement.", "A replacement for card 2964 is already pending."),
            ("Show the old card.", "That card was already replaced and is closed."),
            ("Can this card be replaced?", "It is not eligible for another replacement."),
            ("Check the replacement order.", "An active order already covers that card."),
        ),
        "dispute_transaction": (
            ("Show the status of that purchase.", "It was fully reversed and is ineligible."),
            ("Check the charge again.", "A refund for that charge has already posted."),
            ("Do I have a dispute for it?", "A dispute on that transaction is already open."),
            (
                "Show that resolved purchase.",
                "The transaction was resolved and cannot be disputed.",
            ),
        ),
        "cancel_transfer": (
            ("Check that transfer.", "It completed and is no longer eligible for cancellation."),
            ("Show the cancelled transfer.", "That transfer was already cancelled."),
            ("Can the settled transfer stop?", "It settled after the cancellation window closed."),
            ("Review that transfer status.", "The completed transfer cannot be cancelled here."),
        ),
    }
    rows: list[dict[str, Any]] = []
    for intent, prompts in examples.items():
        for prompt_index, current in enumerate(prompts):
            prior_user, prior_assistant = histories[intent][prompt_index % len(histories[intent])]
            history = [
                {"role": "user", "content": prior_user},
                {"role": "assistant", "content": prior_assistant},
            ]
            key = f"state-ineligible|{split}|{intent}|{prompt_index}"
            rows.append(
                _make_row(
                    current=current,
                    history=history,
                    domain_label=1,
                    intent=intent,
                    relation_names=["context_dependent"],
                    example_kind="state_ineligible_entity",
                    source="self-authored-router-v6-hierarchical-entity-state",
                    source_split=split,
                    group_id=key,
                    trajectory_id=key,
                    explicit_entity_resolution="ineligible",
                )
            )
    return rows


def _targeted_prompts(split: str) -> dict[str, tuple[str, ...]]:
    prompt_sets = {
        "train": {
            "service_case_detail": (
                "Can you explain that service request and tell me when it opened?",
                "What was that customer-service item about, and when did it start?",
                "Give me the details for the address case you just found.",
                "When was the support case above first recorded?",
                "What is the background on that address-change request?",
                "Tell me more about the case from your previous answer.",
            ),
            "standalone_address_case": (
                "Did my mailing-address change finish recently?",
                "Check whether the address-update request was completed.",
                "What is the latest status of my address-change case?",
                "Was there a recent service request for my mailing address?",
                "Look up the case related to changing my address.",
                "When did the bank record my address-update request?",
            ),
            "card_selection_action": (
                "Yes, replace the card we selected.",
                "Go ahead and replace that debit card.",
                "The active one is the card I want replaced.",
                "Use the first card you listed and replace it.",
                "That is the right card; order its replacement.",
                "Please replace the one from your previous answer.",
            ),
            "repetition_repair": (
                "You repeated an unrelated answer; return to my service case.",
                "That did not answer me, so check the address request again.",
                "Stop repeating the loan information and inspect my case.",
                "Your last reply missed the question about my support request.",
                "Please correct the repeated response and look up the case.",
                "That answer is still off topic; continue with my address case.",
                "Why do you keep repeating the same response?",
                "Why are you saying the same unrelated thing again?",
                "You keep giving me that answer over and over.",
                "This is repetitive and still not about my service case.",
                "Why are you repeating the answer?",
                "Why do you repeat yourself instead of checking the case?",
                "Why are you repeating this unrelated response?",
                "You are repeating yourself rather than answering my question.",
                "Why does the same answer keep coming back?",
                "Please stop repeating yourself and handle my request.",
            ),
            "wrong_topic_repair": (
                "I was asking about the address case, not home loans.",
                "No, the question was about my service request rather than lending.",
                "You switched to mortgages, but I meant the mailing-address case.",
                "That loan answer is not what I requested; check the support case.",
                "Return to the address request instead of discussing a mortgage.",
                "I did not mean a loan product; I meant my customer-service case.",
                "I never asked for mortgage information.",
                "I was not asking about a mortgage.",
                "That is not the mortgage question I asked.",
                "I did not ask you about home financing.",
                "I didn't request mortgage advice.",
                "I didn't ask about home loans.",
                "I haven't asked about a mortgage.",
                "I didn't ask for lending information.",
                "I wasn't asking about mortgage products.",
                "I did not request any home-loan guidance.",
            ),
        },
        "validation": {
            "service_case_detail": (
                "What does the request above concern, and when was it logged?",
                "Can you expand on that case and give its opening time?",
                "When did the address item from the prior answer begin?",
            ),
            "standalone_address_case": (
                "Has my address correction request been closed lately?",
                "Find the recent case about updating my mailing details.",
                "Was an address-change support item completed this month?",
            ),
            "card_selection_action": (
                "Replace the debit card identified in the last response.",
                "Proceed with a replacement for that active card.",
                "I choose the card you just named; replace it.",
            ),
            "repetition_repair": (
                "The response repeated the wrong subject; answer about my case.",
                "Try again without the loan material and inspect the service item.",
                "Correct the prior reply and continue with the address request.",
                "Why does the reply keep repeating unrelated information?",
                "Why are you repeating the response rather than checking my case?",
            ),
            "wrong_topic_repair": (
                "My question concerned the service case, not mortgage guidance.",
                "That was a lending answer; I need the address-request details.",
                "Please switch back from loans to my customer-service case.",
                "I was never asking for mortgage details.",
                "I didn't ask for advice about a home loan.",
            ),
        },
        "test": {
            "service_case_detail": (
                "What is that request about, and on what date was it created?",
                "Describe the service item you found and say when it was filed.",
                "How old is the address case mentioned in your last answer?",
            ),
            "standalone_address_case": (
                "Did the mailing-details update happen in the recent past?",
                "Check for a recently completed address service case.",
                "Is there a new case concerning my postal address?",
            ),
            "card_selection_action": (
                "That card is my choice, so submit a replacement.",
                "Replace the one identified as active.",
                "I want a new copy of the debit card you just showed.",
            ),
            "repetition_repair": (
                "You are giving the wrong response again; investigate my case.",
                "Do not repeat the lending answer; handle the address request.",
                "The reply is looping on another subject; return to my service item.",
                "Why does your answer keep repeating itself?",
                "Why are you repeating that reply instead of handling my case?",
            ),
            "wrong_topic_repair": (
                "I asked about the address request rather than a housing loan.",
                "The mortgage subject is wrong; look at my service case.",
                "My issue is the customer-service item, not a loan application.",
                "I was not asking you for mortgage information.",
                "I didn't request any mortgage guidance.",
            ),
        },
    }
    return prompt_sets[split]


def _targeted_modifiers(split: str) -> tuple[str, ...]:
    return {
        "train": (
            "",
            " Please use the earlier messages.",
            " Keep the answer focused on this banking session.",
            " Use the item already shown in the chat.",
            " I am continuing the same request.",
            " Check the signed-in profile.",
            " Please give a concise response.",
            " I am referring to the previous result.",
        ),
        "validation": (
            "",
            " Use the conversation above.",
            " Stay with the same banking task.",
            " Base the answer on the prior response.",
        ),
        "test": (
            "",
            " Continue from the preceding turn.",
            " Use what was already established.",
            " Treat this as the same customer-service conversation.",
        ),
    }[split]


def _held_out_regression_rows() -> list[dict[str, Any]]:
    case_history = [
        {"role": "user", "content": "Show my recent service cases."},
        {
            "role": "assistant",
            "content": "You have a closed mailing-address update case.",
        },
    ]
    card_history = [
        {"role": "user", "content": "Show me the cards on my profile."},
        {
            "role": "assistant",
            "content": "Your active debit card ends in 4821. Is that the one to replace?",
        },
    ]
    wrong_answer_history = [
        {
            "role": "user",
            "content": "Tell me when my mailing-address case was opened.",
        },
        {
            "role": "assistant",
            "content": "Mortgage applicants are typically at least 18.",
        },
    ]
    return [
        _make_row(
            current="When was that created?",
            history=case_history,
            domain_label=1,
            intent="view_service_cases",
            relation_names=["context_dependent"],
            example_kind="heldout_screenshot_regression",
            source="self-authored-heldout-regression",
            source_split="test",
            group_id="heldout|service-case-created",
        ),
        _make_row(
            current="what is that all about? when was it created?",
            history=case_history,
            domain_label=1,
            intent="view_service_cases",
            relation_names=["context_dependent"],
            example_kind="heldout_screenshot_regression",
            source="self-authored-heldout-regression",
            source_split="test",
            group_id="heldout|service-case-details",
        ),
        _make_row(
            current="was the mailing address updated recently?",
            history=[],
            domain_label=1,
            intent="view_service_cases",
            relation_names=[],
            example_kind="heldout_screenshot_regression",
            source="self-authored-heldout-regression",
            source_split="test",
            group_id="heldout|mailing-address-standalone",
        ),
        _make_row(
            current="ok, thats the one i want to replace",
            history=card_history,
            domain_label=1,
            intent="replace_card",
            relation_names=["context_dependent", "clarification_answer"],
            example_kind="heldout_screenshot_regression",
            source="self-authored-heldout-regression",
            source_split="test",
            group_id="heldout|card-selection",
            tool_names=("replace_card",),
            coreference_target="replace_card",
            actionable_entity_count=1,
        ),
        _make_row(
            current="why are you repeating yourself",
            history=wrong_answer_history,
            domain_label=1,
            intent="view_service_cases",
            relation_names=["context_dependent", "agent_repair"],
            example_kind="heldout_screenshot_regression",
            source="self-authored-heldout-regression",
            source_split="test",
            group_id="heldout|agent-repetition-repair",
        ),
        _make_row(
            current="I didn't ask about mortgage",
            history=wrong_answer_history,
            domain_label=1,
            intent="view_service_cases",
            relation_names=["context_dependent", "agent_repair", "topic_shift"],
            example_kind="heldout_screenshot_regression",
            source="self-authored-heldout-regression",
            source_split="test",
            group_id="heldout|wrong-topic-repair",
        ),
        _make_row(
            current="what about the weather there?",
            history=case_history,
            domain_label=0,
            intent=None,
            relation_names=["topic_shift"],
            example_kind="heldout_screenshot_regression",
            source="self-authored-heldout-regression",
            source_split="test",
            group_id="heldout|weather-topic-shift",
        ),
    ]


def _make_row(
    *,
    current: str,
    history: Sequence[dict[str, Any]],
    domain_label: int,
    intent: str | None,
    relation_names: Sequence[str],
    example_kind: str,
    source: str,
    source_split: str,
    group_id: str,
    trajectory_id: str | None = None,
    prior_dialogue_state: Mapping[str, Any] | None = None,
    path: str = "",
    tool_names: Sequence[str] | None = None,
    coreference_target: str = "",
    actionable_entity_count: int | None = None,
    explicit_entity_resolution: str = "",
    action: str | None = None,
    entity_resolution: str | None = None,
    counterfactual_pair_id: str | None = None,
    counterfactual_target: str | None = None,
    counterfactual_phrase_family: str | None = None,
) -> dict[str, Any]:
    if intent is not None and intent not in _INTENT_INDEX:
        raise ValueError(f"unsupported intent: {intent}")
    relation_labels = [0 for _ in RELATION_LABELS]
    for relation in relation_names:
        relation_labels[_RELATION_INDEX[relation]] = 1
    visible_history = [
        {"role": str(item["role"]), "content": str(item["content"]).strip()}
        for item in history
        if item.get("role") in {"user", "assistant"} and str(item.get("content", "")).strip()
    ]
    hierarchical_labels = labels_for_example(
        intent=intent,
        action=action,
        entity_resolution=entity_resolution,
        tool_names=tool_names,
        path=path,
        coreference_target=coreference_target,
        actionable_entity_count=actionable_entity_count,
        explicit_entity_resolution=explicit_entity_resolution,
    )
    expected_legacy_domain_label = 0 if hierarchical_labels["domain_name"] == "out_of_domain" else 1
    if int(domain_label) != expected_legacy_domain_label:
        raise ValueError(
            f"legacy domain_label {domain_label} conflicts with "
            f"domain {hierarchical_labels['domain_name']}"
        )
    return {
        "text": render_router_input(
            current,
            visible_history,
            prior_dialogue_state=prior_dialogue_state,
        ),
        "current_text": current.strip(),
        "history": visible_history,
        "domain_label": int(domain_label),
        **hierarchical_labels,
        "intent_label": _INTENT_INDEX[intent] if intent else -100,
        "intent": intent,
        "lane": lane_for_intent(intent) if intent else None,
        "relation_labels": relation_labels,
        "example_kind": example_kind,
        "source": source,
        "source_split": source_split,
        "group_id": group_id,
        "trajectory_id": trajectory_id or group_id,
        "prior_dialogue_state": dict(prior_dialogue_state or {}),
        "counterfactual_pair_id": counterfactual_pair_id,
        "counterfactual_target": counterfactual_target,
        "counterfactual_phrase_family": counterfactual_phrase_family,
        "actionable_entity_count": actionable_entity_count,
    }


def _last_visible_user(messages: Sequence[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            content = str(message["content"]).strip()
            if content:
                return content
    return None


def _visible_history_before_current(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    user_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    cutoff = user_positions[-1] if user_positions else len(messages)
    for message in messages[:cutoff]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            visible.append({"role": role, "content": content.strip()})
    return visible


def _visible_complete_exchanges(
    history: Sequence[dict[str, Any]],
) -> list[tuple[str, str]]:
    exchanges: list[tuple[str, str]] = []
    pending_user: str | None = None
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            pending_user = content.strip()
        elif role == "assistant" and pending_user is not None:
            exchanges.append((pending_user, content.strip()))
            pending_user = None
    return exchanges


def _intent_for_record(record: dict[str, Any], current: str) -> str:
    scenario = str(record.get("metadata", {}).get("scenario_family", "")).casefold()
    path = str(record.get("metadata", {}).get("path", "")).casefold()
    if path == "hard_negative":
        return "other_banking"
    expected = record.get("expected", {})
    tool_names = []
    if isinstance(expected, dict):
        tool_names = [
            str(call.get("name", "")).casefold()
            for call in expected.get("tool_calls", [])
            if isinstance(call, dict)
        ]
    # Governed route metadata is authoritative.  A policy question may contain
    # mutation words such as "replace" or "dispute" without requesting a bank
    # operation; keyword routing must not override the annotated no-tool path.
    if path == "retrieval_grounded_policy" or scenario.startswith("faq_"):
        return "policy_knowledge"
    joined = " ".join((scenario, current.casefold(), " ".join(tool_names)))
    if "freeze_card" in joined or "freeze" in joined or "stolen" in joined:
        return "freeze_card"
    if "replace_card" in joined or "replace" in joined:
        return "replace_card"
    if "dispute_transaction" in joined or "dispute" in joined:
        return "dispute_transaction"
    if "cancel_transfer" in joined or ("cancel" in joined and "transfer" in joined):
        return "cancel_transfer"
    if "case" in joined or "address" in joined:
        return "view_service_cases"
    if "faq" in joined or "policy" in joined or "mortgage" in joined:
        return "policy_knowledge"
    if "card" in joined:
        return "view_cards"
    if "transaction" in joined or "purchase" in joined:
        return "view_transactions"
    if "transfer" in joined:
        return "view_transfers"
    if "account" in joined or "balance" in joined:
        return "view_accounts"
    if "conversation" in joined or "greeting" in joined or "hello" in joined:
        return "conversation"
    return "other_banking"


def _expected_tool_names(record: Mapping[str, Any]) -> tuple[str, ...]:
    expected = record.get("expected")
    if not isinstance(expected, Mapping):
        return ()
    tool_calls = expected.get("tool_calls")
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, str | bytes):
        return ()
    return tuple(
        str(call["name"])
        for call in tool_calls
        if isinstance(call, Mapping) and isinstance(call.get("name"), str)
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("actionable entity count must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("actionable entity count must be an integer") from error
    if converted < 0:
        raise ValueError("actionable entity count must be non-negative")
    return converted


def _optional_text(value: Any) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None


def _relations_for_current(
    current: str,
    history: Sequence[dict[str, Any]],
    prior_dialogue_state: Mapping[str, Any] | None = None,
) -> list[str]:
    relations = []
    normalized = normalize_router_text(current)
    words = set(normalized.split())
    if history and (
        len(words) <= 6 or words & {"it", "that", "this", "those", "them", "one", "again", "there"}
    ):
        relations.append("context_dependent")
    if words & {"no", "not", "wrong", "instead", "asked"}:
        relations.append("agent_repair")
    if history and words & {"yes", "yeah", "yep", "4821", "that", "one"} and len(words) <= 6:
        relations.append("clarification_answer")
    if (
        prior_dialogue_state
        and prior_dialogue_state.get("knowledge_detour_active") is True
        and prior_dialogue_state.get("pending_servicing")
        and words & {"continue", "resume", "back", "return"}
    ):
        relations.append("resume_previous_service")
    return relations


def _example_kind_for_record(
    record: dict[str, Any],
    history: Sequence[dict[str, Any]],
    relations: Sequence[str],
) -> str:
    path = str(record.get("metadata", {}).get("path", ""))
    if "agent_repair" in relations:
        return "agent_repair"
    if "clarification_answer" in relations:
        return "clarification_answer"
    if "context_dependent" in relations:
        return "contextual_followup"
    if history:
        return "visible_multiturn"
    return path or "sft_single_turn"


def _history_from_anchor(anchor: dict[str, Any]) -> list[dict[str, Any]]:
    history = anchor.get("history")
    if isinstance(history, list) and history:
        return history
    return [
        {"role": "user", "content": str(anchor["current_text"])},
        {
            "role": "assistant",
            "content": _assistant_stub(
                str(anchor["intent"]),
                action=str(anchor["action_name"]),
                entity_resolution=str(anchor["entity_resolution_name"]),
            ),
        },
    ]


def _assistant_stub(
    intent: str,
    *,
    action: str = "execute_tool",
    entity_resolution: str = "not_required",
) -> str:
    if action == "clarify":
        return {
            "freeze_card": "Which card should I freeze?",
            "replace_card": "Which card should I replace?",
            "dispute_transaction": "Which transaction should I dispute?",
            "cancel_transfer": "Which transfer should I cancel?",
        }.get(intent, "Which banking item should I use?")
    if action in {"converse", "retrieve_policy"} or entity_resolution == "ineligible":
        return "I can explain that without performing a banking action."
    return {
        "view_accounts": "I found your account balances.",
        "view_cards": "I found your card details.",
        "freeze_card": "I can help freeze the selected card.",
        "replace_card": "I can help replace the selected card.",
        "view_transactions": "I found the matching transaction detail.",
        "dispute_transaction": "I can help dispute the selected transaction.",
        "view_transfers": "I found your recent transfers.",
        "cancel_transfer": "I can help cancel the selected transfer.",
        "view_service_cases": "I found your recent service cases.",
        "policy_knowledge": "I can answer that banking policy question.",
        "conversation": "I can help with your banking questions.",
        "other_banking": "I can help with that banking request.",
    }[intent]


def _split_variant(current: str, split: str) -> str:
    if split == "validation":
        return f"{current}, please"
    if split == "test":
        return f"{current}, thanks"
    return current


def _group_id(record: dict[str, Any]) -> str:
    split_keys = record.get("split_keys")
    if isinstance(split_keys, dict):
        parts = [
            str(split_keys.get("scenario_family", "")),
            str(split_keys.get("state_seed", "")),
            str(split_keys.get("customer_id", "")),
            str(split_keys.get("template_id", "")),
        ]
        return "|".join(part for part in parts if part) or str(record.get("record_id", "unknown"))
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("split_group"):
        return str(metadata["split_group"])
    return str(record.get("record_id", "unknown"))


def _trajectory_id(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")
    split_keys = record.get("split_keys")
    for container in (metadata, split_keys, record):
        if not isinstance(container, Mapping):
            continue
        for key in ("trajectory_id", "conversation_id"):
            value = container.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return _group_id(record)


def _deduplicate_across_splits(
    splits: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    seen: set[str] = set()
    kept: dict[str, list[dict[str, Any]]] = {split: [] for split in ROUTER_SPLITS}
    removed = 0
    for split in ("test", "validation", "train"):
        split_seen: set[str] = set()
        for row in splits[split]:
            normalized = normalize_router_text(str(row["text"]))
            if normalized in seen:
                removed += 1
                continue
            if normalized in split_seen:
                removed += 1
                continue
            kept[split].append(row)
            split_seen.add(normalized)
        seen.update(split_seen)
    return kept, removed


def _leakage_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    groups: dict[str, set[str]] = defaultdict(set)
    trajectories: dict[str, set[str]] = defaultdict(set)
    state_current_texts: dict[str, set[str]] = defaultdict(set)
    state_families: dict[str, set[str]] = defaultdict(set)
    counterfactual_pairs: dict[str, set[str]] = defaultdict(set)
    generalization_kinds = {
        "state_policy_followup",
        "state_social_detour",
        "heldout_policy_followup_generalization",
        "heldout_social_generalization",
    }
    for split, rows in splits.items():
        for row in rows:
            group_id = str(row["group_id"])
            groups[group_id].add(split)
            trajectories[str(row["trajectory_id"])].add(split)
            pair_id = row.get("counterfactual_pair_id")
            if pair_id:
                counterfactual_pairs[str(pair_id)].add(split)
            if str(row["example_kind"]) in generalization_kinds:
                state_current_texts[normalize_router_text(str(row["current_text"]))].add(split)
                if group_id.startswith(("state-policy-family|", "state-social-family|")):
                    state_families[group_id].add(split)
    leaking = {group: sorted(values) for group, values in groups.items() if len(values) > 1}
    trajectory_leaks = {
        trajectory: sorted(values) for trajectory, values in trajectories.items() if len(values) > 1
    }
    state_current_text_leaks = {
        current: sorted(values)
        for current, values in state_current_texts.items()
        if len(values) > 1
    }
    state_family_leaks = {
        family: sorted(values) for family, values in state_families.items() if len(values) > 1
    }
    counterfactual_pair_leaks = {
        pair_id: sorted(values)
        for pair_id, values in counterfactual_pairs.items()
        if len(values) > 1
    }
    return {
        "group_split_leaks": leaking,
        "group_split_leak_count": len(leaking),
        "trajectory_split_leaks": trajectory_leaks,
        "trajectory_split_leak_count": len(trajectory_leaks),
        "state_current_text_split_leaks": state_current_text_leaks,
        "state_current_text_split_leak_count": len(state_current_text_leaks),
        "state_paraphrase_family_split_leaks": state_family_leaks,
        "state_paraphrase_family_split_leak_count": len(state_family_leaks),
        "counterfactual_pair_split_leaks": counterfactual_pair_leaks,
        "counterfactual_pair_split_leak_count": len(counterfactual_pair_leaks),
    }


def _count_pii_matches(texts: Iterable[str]) -> int:
    return sum(1 for text in texts for pattern in PII_PATTERNS if pattern.search(text))


def _is_screenshot_regression_text(text: str) -> bool:
    return normalize_router_text(text) in SCREENSHOT_REGRESSION_CURRENTS


def _stable_rank(seed: int, *parts: str) -> str:
    return hashlib.sha256("\0".join((str(seed), *parts)).encode()).hexdigest()
