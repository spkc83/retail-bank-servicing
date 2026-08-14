from __future__ import annotations

import pytest
import torch

from router import ROUTER_REPO_ID, ROUTER_REVISION, LearnedBankingRouter


def test_router_defaults_pin_the_published_v5_artifact() -> None:
    assert ROUTER_REPO_ID == "spkc83/retail-bank-conversation-router"
    assert ROUTER_REVISION == "bf6abca1c3982e35b23239de13ba9fcfed3f7920"


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
    assert result["ood_probability"] == pytest.approx(
        1 - result["banking_probability"]
    )
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
    assert tokenizer.rendered.startswith(
        "[CURRENT_USER]\nWhen was that created?"
    )
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
    assert tokenizer.rendered.index("assistant-4") < tokenizer.rendered.index(
        "assistant-3"
    )
