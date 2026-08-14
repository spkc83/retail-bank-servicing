from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hello_slm.banking_conversation_router import (
    ChatMessage,
    ConversationRouterModel,
    LearnedConversationRouter,
    verify_router_artifact,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.rendered = ""

    def __call__(self, text: str, **_kwargs: object) -> dict[str, torch.Tensor]:
        self.rendered = text
        return {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }


class FakeEncoder(nn.Module):
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        value = input_ids.float().unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=torch.cat((value, 1.0 - value), dim=-1))


def make_router(
    *,
    domain_logits: tuple[float, float],
    relation_logits: tuple[float, float, float, float, float],
) -> tuple[LearnedConversationRouter, RecordingTokenizer]:
    tokenizer = RecordingTokenizer()
    model = ConversationRouterModel(
        FakeEncoder(),
        hidden_size=2,
        num_intents=2,
        num_relations=5,
    )

    def fixed_forward(
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del input_ids, attention_mask
        return (
            torch.tensor([domain_logits]),
            torch.tensor([[4.0, 1.0]]),
            torch.tensor([relation_logits]),
        )

    model.forward = fixed_forward  # type: ignore[method-assign]
    return (
        LearnedConversationRouter(
            tokenizer=tokenizer,
            model=model,
            intent_labels=("view_service_cases", "view_cards"),
            relation_labels=(
                "context_dependent",
                "agent_repair",
                "topic_shift",
                "clarification_answer",
                "resume_previous_service",
            ),
            ood_banking_threshold=0.20,
            in_domain_threshold=0.50,
            relation_rescue_threshold=0.50,
            max_length=256,
        ),
        tokenizer,
    )


def banking_history() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content="Show my recent service cases."),
        ChatMessage(
            role="assistant",
            content="You have a closed mailing-address update case.",
        ),
    ]


def test_cross_encoder_always_receives_history_and_current_together() -> None:
    router, tokenizer = make_router(
        domain_logits=(-4.0, 4.0),
        relation_logits=(4.0, -4.0, -4.0, -4.0, -4.0),
    )

    result = router.classify(
        [
            *banking_history(),
            ChatMessage(role="user", content="When was that created?"),
        ]
    )

    assert result.route == "in_domain"
    assert result.intent == "view_service_cases"
    assert result.lane == "servicing"
    assert tokenizer.rendered.startswith("[CURRENT_USER]\nWhen was that created?")
    assert "[PREVIOUS_ASSISTANT]\nYou have a closed mailing-address" in tokenizer.rendered
    assert "[PREVIOUS_USER]\nShow my recent service cases." in tokenizer.rendered
    assert result.context_applied is True


def test_incomplete_history_is_not_reported_as_applied_context() -> None:
    router, tokenizer = make_router(
        domain_logits=(-4.0, 4.0),
        relation_logits=(-4.0, -4.0, -4.0, -4.0, -4.0),
    )

    result = router.classify(
        [
            ChatMessage(role="assistant", content="An orphan assistant reply."),
            ChatMessage(role="user", content="Show my balances."),
        ]
    )

    assert "[PREVIOUS_ASSISTANT]" not in tokenizer.rendered
    assert result.context_applied is False


def test_high_confidence_external_ood_is_blocked_without_context_signal() -> None:
    router, _ = make_router(
        domain_logits=(5.0, -5.0),
        relation_logits=(-5.0, -5.0, 5.0, -5.0, -5.0),
    )

    result = router.classify(
        [
            *banking_history(),
            ChatMessage(role="user", content="What is the weather tomorrow?"),
        ]
    )

    assert result.route == "out_of_domain"
    assert result.ood_probability > 0.99
    assert result.relation_probabilities["topic_shift"] > 0.99


@pytest.mark.parametrize("rescue_index", [0, 1, 3])
def test_context_repair_and_clarification_rescue_an_ood_score(
    rescue_index: int,
) -> None:
    relation_logits = [-5.0, -5.0, -5.0, -5.0, -5.0]
    relation_logits[rescue_index] = 5.0
    router, _ = make_router(
        domain_logits=(5.0, -5.0),
        relation_logits=tuple(relation_logits),  # type: ignore[arg-type]
    )

    result = router.classify(
        [
            *banking_history(),
            ChatMessage(role="user", content="No, that was not what I asked."),
        ]
    )

    assert result.route == "uncertain"
    assert result.intent is None


def test_prior_dialogue_state_is_rendered_before_history_and_enables_resume() -> None:
    router, tokenizer = make_router(
        domain_logits=(-4.0, 4.0),
        relation_logits=(4.0, -4.0, -4.0, -4.0, 4.0),
    )

    result = router.classify(
        [ChatMessage(role="user", content="Let's continue with that.")],
        prior_dialogue_state={
            "version": 1,
            "pending_servicing": {
                "intent": "dispute_transaction",
                "phase": "awaiting_user",
            },
            "knowledge_detour_active": True,
        },
    )

    assert tokenizer.rendered.index("[PRIOR_DIALOGUE_STATE]") < tokenizer.rendered.index(
        "[CURRENT_USER]"
    )
    assert "dispute_transaction" in tokenizer.rendered
    assert result.active_relations == (
        "context_dependent",
        "resume_previous_service",
    )


def test_manifest_verification_accepts_v2_and_rejects_corrupt_artifact(
    tmp_path: Path,
) -> None:
    config = {
        "contract": "banking-conversation-router",
        "format_version": 2,
        "capability_labels": ["service_cases"],
        "relation_labels": ["context_dependent"],
        "ood_banking_threshold": 0.2,
        "in_domain_threshold": 0.5,
        "relation_rescue_threshold": 0.5,
        "max_length": 256,
    }
    config_path = tmp_path / "router_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest = {
        "contract": "banking-conversation-router-artifact",
        "release_eligible": True,
        "files": [
            {
                "path": "router_config.json",
                "bytes": config_path.stat().st_size,
                "sha256": digest,
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_router_artifact(tmp_path) == config
    config_path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|digest mismatch"):
        verify_router_artifact(tmp_path)


def test_manifest_verification_accepts_explicit_v3_intent_contract(
    tmp_path: Path,
) -> None:
    config = {
        "contract": "banking-conversation-router",
        "format_version": 3,
        "intent_labels": ["view_accounts"],
        "relation_labels": ["resume_previous_service"],
    }
    config_path = tmp_path / "router_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = {
        "contract": "banking-conversation-router-artifact",
        "release_eligible": True,
        "files": [
            {
                "path": "router_config.json",
                "bytes": config_path.stat().st_size,
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_router_artifact(tmp_path)["intent_labels"] == ["view_accounts"]
