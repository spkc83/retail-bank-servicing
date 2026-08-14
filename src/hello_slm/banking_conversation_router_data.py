from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

ROUTER_SPLITS = ("train", "validation", "test")

INTENT_LABELS = (
    "view_accounts",
    "view_cards",
    "freeze_card",
    "replace_card",
    "view_transactions",
    "dispute_transaction",
    "view_transfers",
    "cancel_transfer",
    "view_service_cases",
    "policy_knowledge",
    "conversation",
    "other_banking",
)
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

_SERVICING_INTENTS = frozenset(
    {
        "view_accounts",
        "view_cards",
        "freeze_card",
        "replace_card",
        "view_transactions",
        "dispute_transaction",
        "view_transfers",
        "cancel_transfer",
        "view_service_cases",
    }
)
_V2_SERVICING_CAPABILITIES = frozenset(
    {"accounts", "cards", "card_actions", "transactions", "transfers", "service_cases"}
)


def lane_for_intent(intent: str) -> str:
    if intent in _SERVICING_INTENTS or intent in _V2_SERVICING_CAPABILITIES:
        return "servicing"
    if intent in {"policy_knowledge", "faq"}:
        return "policy"
    if intent == "conversation":
        return "conversation"
    if intent == "other_banking":
        return "other_banking"
    raise ValueError(f"unsupported intent: {intent}")


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
    splits["test"].extend(_held_out_regression_rows())

    deduplicated, duplicates_removed = _deduplicate_across_splits(splits)
    report = {
        "contract": "banking-conversation-router-data-report",
        "seed": seed,
        "intent_labels": INTENT_LABELS,
        "relation_labels": RELATION_LABELS,
        "split_counts": {split: len(rows) for split, rows in deduplicated.items()},
        "kind_counts": {
            split: dict(Counter(str(row["example_kind"]) for row in rows))
            for split, rows in deduplicated.items()
        },
        "domain_counts": {
            split: dict(Counter(int(row["domain_label"]) for row in rows))
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
    path = str(record.get("metadata", {}).get("path", ""))
    domain_label = 0 if path == "ood" else 1
    intent = None if path == "ood" else _intent_for_record(record, current)
    prior_dialogue_state = record.get("prior_dialogue_state")
    if prior_dialogue_state is None:
        prior_dialogue_state = record.get("metadata", {}).get("prior_dialogue_state")
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
        source="self-authored-banking-tool-sft",
        source_split=split,
        group_id=_group_id(record),
        trajectory_id=_trajectory_id(record),
        prior_dialogue_state=prior_dialogue_state,
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
    banking = [row for row in rows if row["domain_label"] == 1 and row["intent"] is not None]
    external = [row for row in rows if row["domain_label"] == 0]
    if not banking:
        return []
    ordered = sorted(banking, key=lambda row: _stable_rank(seed, split, str(row["text"])))
    prompts = (
        (
            "Could you tell me when that item was recorded",
            "contextual_followup",
            ["context_dependent"],
        ),
        (
            "What happened with the other one you mentioned",
            "contextual_followup",
            ["context_dependent"],
        ),
        (
            "Actually I meant the banking item from before",
            "agent_repair",
            ["context_dependent", "agent_repair"],
        ),
        (
            "That answer missed my question, please try it again",
            "agent_repair",
            ["context_dependent", "agent_repair"],
        ),
        (
            "Yes, use the first option you listed",
            "clarification_answer",
            ["context_dependent", "clarification_answer"],
        ),
        (
            "The second one, please",
            "clarification_answer",
            ["context_dependent", "clarification_answer"],
        ),
        (
            "Can yu show that agian",
            "typo_contextual_followup",
            ["context_dependent"],
        ),
        (
            "wht happened with it",
            "typo_contextual_followup",
            ["context_dependent"],
        ),
    )
    generated = []
    for anchor_index, anchor in enumerate(ordered):
        history = _history_from_anchor(anchor)
        for prompt_index, (current, kind, relations) in enumerate(prompts):
            generated.append(
                _make_row(
                    current=_split_variant(current, split),
                    history=history,
                    domain_label=1,
                    intent=str(anchor["intent"]),
                    relation_names=relations,
                    example_kind=kind,
                    source="self-authored-router-v5-synthetic",
                    source_split=split,
                    group_id=(
                        f"{anchor['group_id']}|router-v5|{kind}|{anchor_index}-{prompt_index}"
                    ),
                )
            )
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
            "content": "You have a debit card ending in 4821 and another ending in 7319.",
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
    return {
        "text": render_router_input(
            current,
            visible_history,
            prior_dialogue_state=prior_dialogue_state,
        ),
        "current_text": current.strip(),
        "history": visible_history,
        "domain_label": int(domain_label),
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
        {"role": "assistant", "content": _assistant_stub(str(anchor["intent"]))},
    ]


def _assistant_stub(intent: str) -> str:
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
    for split, rows in splits.items():
        for row in rows:
            groups[str(row["group_id"])].add(split)
            trajectories[str(row["trajectory_id"])].add(split)
    leaking = {group: sorted(values) for group, values in groups.items() if len(values) > 1}
    trajectory_leaks = {
        trajectory: sorted(values) for trajectory, values in trajectories.items() if len(values) > 1
    }
    return {
        "group_split_leaks": leaking,
        "group_split_leak_count": len(leaking),
        "trajectory_split_leaks": trajectory_leaks,
        "trajectory_split_leak_count": len(trajectory_leaks),
    }


def _count_pii_matches(texts: Iterable[str]) -> int:
    return sum(1 for text in texts for pattern in PII_PATTERNS if pattern.search(text))


def _is_screenshot_regression_text(text: str) -> bool:
    return normalize_router_text(text) in SCREENSHOT_REGRESSION_CURRENTS


def _stable_rank(seed: int, *parts: str) -> str:
    return hashlib.sha256("\0".join((str(seed), *parts)).encode()).hexdigest()
