"""The canonical label sets, resolved once for both routing implementations.

The Space artifact is standalone -- it ships without ``src/hello_slm`` -- so the
labels have to exist locally. They were previously mirrored inline inside
``router.py``; a second copy in the model router would have made three places to
forget. The import is tried first so a developer checkout validates against the
real module rather than against a copy of it.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised by whichever environment is present
    from hello_slm.banking_conversation_router_data import RELATION_LABELS
    from hello_slm.banking_domain_taxonomy import (
        ACTION_LABELS,
        DOMAIN_LABELS,
        ENTITY_RESOLUTION_LABELS,
        FAMILY_LABELS,
        INTENT_LABELS,
        LANE_LABELS,
    )

    CANONICAL_SOURCE = "hello_slm.banking_domain_taxonomy"
except ModuleNotFoundError:  # pragma: no cover - the deployed Space takes this path
    DOMAIN_LABELS = ("out_of_domain", "banking", "social")
    LANE_LABELS = ("out_of_domain", "servicing", "policy", "conversation", "other_banking")
    FAMILY_LABELS = (
        "external",
        "accounts",
        "cards",
        "transactions",
        "transfers",
        "service_cases",
        "policy",
        "social",
        "other_banking",
    )
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
    ACTION_LABELS = ("refuse_ood", "execute_tool", "clarify", "retrieve_policy", "converse")
    ENTITY_RESOLUTION_LABELS = (
        "not_required",
        "resolved",
        "missing",
        "ambiguous",
        "ineligible",
    )
    CANONICAL_SOURCE = "poc.taxonomy (standalone mirror)"

__all__ = [
    "ACTION_LABELS",
    "CANONICAL_SOURCE",
    "DOMAIN_LABELS",
    "ENTITY_RESOLUTION_LABELS",
    "FAMILY_LABELS",
    "INTENT_LABELS",
    "LANE_LABELS",
    "RELATION_LABELS",
]
