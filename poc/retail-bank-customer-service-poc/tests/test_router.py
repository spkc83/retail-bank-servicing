from __future__ import annotations

import pytest
import torch
from hello_slm.banking_conversation_router import decode_v4_joint as decode_core_v4_joint
from hello_slm.banking_conversation_router_data import RELATION_LABELS
from hello_slm.banking_domain_taxonomy import (
    ACTION_LABELS,
    DOMAIN_LABELS,
    ENTITY_RESOLUTION_LABELS,
    FAMILY_LABELS,
    INTENT_LABELS,
    LANE_LABELS,
)

from router import (
    ROUTER_REPO_ID,
    ROUTER_REVISION,
    ConversationRouterOutput,
    LearnedBankingRouter,
)
from router import (
    decode_v4_joint as decode_poc_v4_joint,
)


def test_router_defaults_pin_the_published_hierarchical_artifact() -> None:
    assert ROUTER_REPO_ID == "spkc83/retail-bank-conversation-router"
    assert ROUTER_REVISION == "36920330d2502dfcf4d60572eadf1e3e71cd23fa"


class FakeTokenizer:
    def __init__(self) -> None:
        self.rendered = ""

    def __call__(self, text, **_kwargs):
        self.rendered = text
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


class FakeModel:
    def __init__(
        self,
        domain_logits,
        capability_logits,
        relation_logits,
    ) -> None:
        self.domain_logits = torch.tensor([domain_logits], dtype=torch.float32)
        self.capability_logits = torch.tensor(
            [capability_logits],
            dtype=torch.float32,
        )
        self.relation_logits = torch.tensor([relation_logits], dtype=torch.float32)

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, **_kwargs):
        return (
            self.domain_logits,
            self.capability_logits,
            self.relation_logits,
        )


def make_router(
    domain_logits,
    relation_logits,
) -> tuple[LearnedBankingRouter, FakeTokenizer]:
    tokenizer = FakeTokenizer()
    return (
        LearnedBankingRouter(
            tokenizer=tokenizer,
            model=FakeModel(
                domain_logits,
                [0.1, 3.0, 1.0],
                relation_logits,
            ),
            capability_labels=("accounts", "service_cases", "cards"),
            relation_labels=(
                "context_dependent",
                "agent_repair",
                "topic_shift",
                "clarification_answer",
            ),
            ood_banking_threshold=0.20,
            in_domain_threshold=0.50,
            relation_rescue_threshold=0.50,
            max_length=256,
            max_exchanges=3,
        ),
        tokenizer,
    )


@pytest.mark.parametrize(
    ("domain_logits", "relation_logits", "expected_route"),
    [
        ((-8.0, 8.0), (-8.0, -8.0, -8.0, -8.0), "in_domain"),
        ((0.4, 0.0), (-8.0, -8.0, -8.0, -8.0), "uncertain"),
        ((8.0, -8.0), (-8.0, -8.0, 8.0, -8.0), "out_of_domain"),
        ((8.0, -8.0), (8.0, -8.0, -8.0, -8.0), "uncertain"),
        ((8.0, -8.0), (-8.0, 8.0, -8.0, -8.0), "uncertain"),
    ],
)
def test_router_combines_domain_and_relation_heads(
    domain_logits,
    relation_logits,
    expected_route,
) -> None:
    router, _ = make_router(domain_logits, relation_logits)

    result = router.classify("hello", [])

    assert result["route"] == expected_route
    assert result["ood_probability"] == pytest.approx(1 - result["banking_probability"])
    assert result["relation_probabilities"]
    if expected_route == "in_domain":
        assert result["capability"] == "service_cases"
    else:
        assert result["capability"] is None


def test_router_always_cross_encodes_recent_visible_history() -> None:
    router, tokenizer = make_router(
        (-8.0, 8.0),
        (8.0, -8.0, -8.0, -8.0),
    )

    result = router.classify(
        "When was that created?",
        [
            {"role": "user", "content": "Show my service cases."},
            {
                "role": "assistant",
                "content": "You have a closed mailing-address update case.",
            },
        ],
    )

    assert result["route"] == "in_domain"
    assert result["context_applied"] is True
    assert tokenizer.rendered.startswith("[CURRENT_USER]\nWhen was that created?")
    assert "[PREVIOUS_ASSISTANT]\nYou have a closed mailing-address" in tokenizer.rendered
    assert "[PREVIOUS_USER]\nShow my service cases." in tokenizer.rendered


def test_only_three_recent_complete_exchanges_are_rendered() -> None:
    router, tokenizer = make_router(
        (-8.0, 8.0),
        (8.0, -8.0, -8.0, -8.0),
    )
    history = []
    for index in range(5):
        history.extend(
            [
                {"role": "user", "content": f"user-{index}"},
                {"role": "assistant", "content": f"assistant-{index}"},
            ]
        )

    router.classify("current", history)

    assert "user-0" not in tokenizer.rendered
    assert "user-1" not in tokenizer.rendered
    assert "user-2" in tokenizer.rendered
    assert "user-4" in tokenizer.rendered
    assert tokenizer.rendered.index("assistant-4") < tokenizer.rendered.index("assistant-3")


class FakeV4Model:
    def __init__(self, *, domain="banking", lane="servicing", rescue=False):
        self.domain = domain
        self.lane = lane
        self.rescue = rescue

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, **_kwargs):
        def selected(labels, label):
            values = [-5.0] * len(labels)
            values[labels.index(label)] = 5.0
            return torch.tensor([values])

        return ConversationRouterOutput(
            domain_logits=selected(DOMAIN_LABELS, self.domain),
            lane_logits=selected(LANE_LABELS, self.lane),
            family_logits=selected(FAMILY_LABELS, "cards"),
            intent_logits=selected(INTENT_LABELS, "replace_card"),
            relation_logits=torch.tensor(
                [
                    [
                        5.0 if self.rescue and index == 0 else -5.0
                        for index in range(len(RELATION_LABELS))
                    ]
                ]
            ),
            action_logits=selected(ACTION_LABELS, "execute_tool"),
            entity_resolution_logits=selected(ENTITY_RESOLUTION_LABELS, "ambiguous"),
        )


def test_v4_poc_router_exposes_hierarchy_and_clarifies_ambiguity() -> None:
    router = LearnedBankingRouter(
        tokenizer=FakeTokenizer(),
        model=FakeV4Model(),
        intent_labels=INTENT_LABELS,
        relation_labels=RELATION_LABELS,
        domain_labels=DOMAIN_LABELS,
        lane_labels=LANE_LABELS,
        family_labels=FAMILY_LABELS,
        action_labels=ACTION_LABELS,
        entity_resolution_labels=ENTITY_RESOLUTION_LABELS,
        format_version=4,
        ood_banking_threshold=0.2,
        in_domain_threshold=0.5,
        relation_rescue_threshold=0.5,
        max_length=256,
        max_exchanges=3,
    )

    result = router.classify("Replace that card.", [])

    assert result["route"] == "in_domain"
    assert result["domain"] == "banking"
    assert result["lane"] == "servicing"
    assert result["family"] == "cards"
    assert result["action"] == "clarify"
    assert result["entity_resolution"] == "ambiguous"
    assert result["banking_probability"] == pytest.approx(
        1.0 - result["domain_probabilities"]["out_of_domain"]
    )
    assert result["support_probability"] == result["banking_probability"]


@pytest.mark.parametrize(
    ("domain", "lane", "rescue", "expected_route", "expected_action", "expected_entity"),
    [
        (
            "out_of_domain",
            "servicing",
            False,
            "out_of_domain",
            "refuse_ood",
            "not_required",
        ),
        ("out_of_domain", "servicing", True, "uncertain", None, None),
    ],
)
def test_v4_poc_suppresses_unsafe_operational_decisions(
    domain, lane, rescue, expected_route, expected_action, expected_entity
) -> None:
    router = LearnedBankingRouter(
        tokenizer=FakeTokenizer(),
        model=FakeV4Model(domain=domain, lane=lane, rescue=rescue),
        intent_labels=INTENT_LABELS,
        relation_labels=RELATION_LABELS,
        domain_labels=DOMAIN_LABELS,
        lane_labels=LANE_LABELS,
        family_labels=FAMILY_LABELS,
        action_labels=ACTION_LABELS,
        entity_resolution_labels=ENTITY_RESOLUTION_LABELS,
        format_version=4,
        ood_banking_threshold=0.2,
        in_domain_threshold=0.5,
        relation_rescue_threshold=0.5,
        max_length=256,
        max_exchanges=3,
    )

    result = router.classify("Replace that card.", [])

    assert result["route"] == expected_route
    assert result["intent"] is None
    assert result["lane"] is None
    assert result["family"] is None
    assert result["action"] == expected_action
    assert result["entity_resolution"] == expected_entity
    assert result["action_candidates"][0]["action"] == "execute_tool"


def test_v4_poc_joint_decoder_matches_core_conflict_resolution() -> None:
    router = LearnedBankingRouter(
        tokenizer=FakeTokenizer(),
        model=FakeV4Model(lane="policy"),
        intent_labels=INTENT_LABELS,
        relation_labels=RELATION_LABELS,
        domain_labels=DOMAIN_LABELS,
        lane_labels=LANE_LABELS,
        family_labels=FAMILY_LABELS,
        action_labels=ACTION_LABELS,
        entity_resolution_labels=ENTITY_RESOLUTION_LABELS,
        format_version=4,
        ood_banking_threshold=0.2,
        in_domain_threshold=0.5,
        relation_rescue_threshold=0.5,
        max_length=256,
        max_exchanges=3,
    )

    result = router.classify("Replace that card.", [])

    assert result["route"] == "in_domain"
    assert result["intent"] == "replace_card"
    assert result["lane"] == "servicing"
    assert result["lane_candidates"][0]["lane"] == "policy"
    assert (
        "constraint:joint-decoder-resolved-independent-head-conflict"
        in result["constraint_diagnostics"]
    )


def test_v4_joint_decoder_core_and_poc_parity() -> None:
    def scores(labels, selected):
        return [4.0 if label == selected else -3.0 for label in labels]

    kwargs = {
        "domain_scores": scores(DOMAIN_LABELS, "banking"),
        "lane_scores": scores(LANE_LABELS, "policy"),
        "family_scores": scores(FAMILY_LABELS, "cards"),
        "intent_scores": scores(INTENT_LABELS, "replace_card"),
        "action_scores": scores(ACTION_LABELS, "execute_tool"),
        "entity_resolution_scores": scores(ENTITY_RESOLUTION_LABELS, "resolved"),
        "domain_labels": DOMAIN_LABELS,
        "lane_labels": LANE_LABELS,
        "family_labels": FAMILY_LABELS,
        "intent_labels": INTENT_LABELS,
        "action_labels": ACTION_LABELS,
        "entity_resolution_labels": ENTITY_RESOLUTION_LABELS,
    }

    core = decode_core_v4_joint(**kwargs)
    poc = decode_poc_v4_joint(**kwargs)

    assert (
        core.domain,
        core.lane,
        core.family,
        core.intent,
        core.action,
        core.entity_resolution,
        core.score,
    ) == (
        poc.domain,
        poc.lane,
        poc.family,
        poc.intent,
        poc.action,
        poc.entity_resolution,
        poc.score,
    )
