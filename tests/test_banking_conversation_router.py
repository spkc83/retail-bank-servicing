from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from hello_slm.banking_conversation_router import (
    ChatMessage,
    ConversationRouterModel,
    ConversationRouterOutput,
    LearnedConversationRouter,
    stabilize_active_relations,
    verify_router_artifact,
)
from hello_slm.banking_conversation_router_data import RELATION_LABELS
from hello_slm.banking_domain_taxonomy import (
    ACTION_LABELS,
    DOMAIN_LABELS,
    ENTITY_RESOLUTION_LABELS,
    FAMILY_LABELS,
    INTENT_LABELS,
    LANE_LABELS,
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


def test_explicit_topic_repair_closes_relation_set_only_with_history() -> None:
    assert stabilize_active_relations(
        ("topic_shift",),
        current_text="I didn't ask about mortgage",
        context_applied=True,
    ) == ("context_dependent", "agent_repair", "topic_shift")
    assert stabilize_active_relations(
        ("topic_shift",),
        current_text="Tell me about mortgage rates",
        context_applied=True,
    ) == ("topic_shift",)
    assert stabilize_active_relations(
        ("topic_shift",),
        current_text="I didn't ask about mortgage",
        context_applied=False,
    ) == ("topic_shift",)


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


class FakeV4Model:
    def __init__(
        self,
        *,
        domain: str = "banking",
        lane: str = "servicing",
        entity_resolution: str = "resolved",
        rescue: bool = False,
        action: str = "execute_tool",
    ) -> None:
        self.domain = domain
        self.lane = lane
        self.entity_resolution = entity_resolution
        self.rescue = rescue
        self.action = action

    def to(self, _device: object) -> FakeV4Model:
        return self

    def eval(self) -> FakeV4Model:
        return self

    def __call__(self, **_kwargs: object) -> ConversationRouterOutput:
        def selected(labels: tuple[str, ...], label: str) -> torch.Tensor:
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
            action_logits=selected(ACTION_LABELS, self.action),
            entity_resolution_logits=selected(ENTITY_RESOLUTION_LABELS, self.entity_resolution),
        )


def make_v4_router(
    *,
    domain: str = "banking",
    lane: str = "servicing",
    entity_resolution: str = "resolved",
    rescue: bool = False,
    action: str = "execute_tool",
) -> LearnedConversationRouter:
    return LearnedConversationRouter(
        tokenizer=RecordingTokenizer(),
        model=FakeV4Model(  # type: ignore[arg-type]
            domain=domain,
            lane=lane,
            entity_resolution=entity_resolution,
            rescue=rescue,
            action=action,
        ),
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
    )


def test_v4_returns_joint_hierarchy_and_compatibility_alias() -> None:
    result = make_v4_router().classify(
        [ChatMessage(role="user", content="Replace my card ending 2105.")]
    )

    assert result.route == "in_domain"
    assert result.domain == "banking"
    assert result.lane == "servicing"
    assert result.family == "cards"
    assert result.intent == "replace_card"
    assert result.action == "execute_tool"
    assert result.entity_resolution == "resolved"
    assert result.banking_probability == pytest.approx(
        1.0 - result.domain_probabilities["out_of_domain"]  # type: ignore[index]
    )
    assert result.support_probability == result.banking_probability
    assert result.domain_candidates[0]["domain"] == "banking"
    assert result.constraint_diagnostics == ()
    assert result.joint_decision_contract == "hierarchical-router-joint-decision/v1"
    assert result.joint_decision_accepted is True
    selected = next(item for item in result.intent_candidates if item["intent"] == result.intent)
    assert result.selected_intent_probability == selected["probability"]


def test_v4_ambiguous_entity_downgrades_execution_to_clarification() -> None:
    result = make_v4_router(entity_resolution="ambiguous").classify(
        [ChatMessage(role="user", content="Replace that card.")]
    )

    assert result.route == "in_domain"
    assert result.action == "clarify"
    assert result.entity_resolution == "ambiguous"
    assert "constraint:joint-decoder-resolved-independent-head-conflict" in (
        result.constraint_diagnostics
    )


def test_v4_ood_exposes_only_safe_refusal_disposition() -> None:
    result = make_v4_router(domain="out_of_domain").classify(
        [ChatMessage(role="user", content="Will it rain tomorrow?")]
    )

    assert result.route == "out_of_domain"
    assert result.intent is None
    assert result.lane is None
    assert result.family is None
    assert result.action == "refuse_ood"
    assert result.entity_resolution == "not_required"
    assert result.action_candidates[0]["action"] == "execute_tool"
    assert "constraint:ood-suppressed-downstream" in result.constraint_diagnostics


def test_v4_joint_decoder_resolves_independent_head_conflict() -> None:
    result = make_v4_router(lane="policy").classify(
        [ChatMessage(role="user", content="Replace my card.")]
    )

    assert result.route == "in_domain"
    assert result.intent == "replace_card"
    assert result.lane == "servicing"
    assert result.lane_candidates[0]["lane"] == "policy"
    assert "constraint:joint-decoder-resolved-independent-head-conflict" in (
        result.constraint_diagnostics
    )


def test_v4_uncertain_route_suppresses_operational_decisions() -> None:
    result = make_v4_router(domain="out_of_domain", rescue=True).classify(
        [ChatMessage(role="user", content="Continue with that.")]
    )

    assert result.route == "uncertain"
    assert result.intent is None
    assert result.lane is None
    assert result.family is None
    assert result.action is None
    assert result.entity_resolution is None
    assert result.intent_candidates[0]["intent"] == "replace_card"


def test_v4_live_path_downgrades_mutation_converse_to_clarify() -> None:
    result = make_v4_router(action="converse", entity_resolution="not_required").classify(
        [ChatMessage(role="user", content="Replace that card.")]
    )

    assert result.route == "in_domain"
    assert result.intent == "replace_card"
    assert result.action == "clarify"
    assert "constraint:mutation-intent-cannot-converse" in result.constraint_diagnostics


def _write_v4_artifact(tmp_path: Path, *, include_heads: bool) -> None:
    config = {
        "contract": "banking-conversation-router",
        "format_version": 4,
        "domain_labels": list(DOMAIN_LABELS),
        "lane_labels": list(LANE_LABELS),
        "family_labels": list(FAMILY_LABELS),
        "intent_labels": list(INTENT_LABELS),
        "relation_labels": list(RELATION_LABELS),
        "action_labels": list(ACTION_LABELS),
        "entity_resolution_labels": list(ENTITY_RESOLUTION_LABELS),
    }
    config_path = tmp_path / "router_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    if include_heads:
        save_file(
            {
                "domain_head.weight": torch.zeros(len(DOMAIN_LABELS), 2),
                "domain_head.bias": torch.zeros(len(DOMAIN_LABELS)),
            },
            tmp_path / "classifier_heads.safetensors",
        )


@pytest.mark.parametrize("include_heads", [False, True])
def test_v4_rejects_missing_or_incomplete_heads(tmp_path: Path, include_heads: bool) -> None:
    _write_v4_artifact(tmp_path, include_heads=include_heads)

    with pytest.raises(ValueError, match="classifier_heads|lane_head"):
        verify_router_artifact(tmp_path)


def test_v3_artifact_still_loads_and_infers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers

    config = {
        "contract": "banking-conversation-router",
        "format_version": 3,
        "intent_labels": ["view_accounts"],
        "relation_labels": ["context_dependent"],
        "ood_banking_threshold": 0.2,
        "in_domain_threshold": 0.5,
        "relation_rescue_threshold": 0.5,
        "max_length": 256,
    }
    config_path = tmp_path / "router_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "domain_head.weight": torch.tensor([[-2.0, 0.0], [2.0, 0.0]]),
            "domain_head.bias": torch.zeros(2),
            "intent_head.weight": torch.zeros(1, 2),
            "intent_head.bias": torch.zeros(1),
            "relation_head.weight": torch.zeros(1, 2),
            "relation_head.bias": torch.full((1,), -5.0),
        },
        tmp_path / "classifier_heads.safetensors",
    )
    encoder = FakeEncoder()
    encoder.config = SimpleNamespace(hidden_size=2)  # type: ignore[attr-defined]
    tokenizer = RecordingTokenizer()
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *_a, **_k: encoder)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tokenizer)

    router = LearnedConversationRouter.from_artifact_dir(tmp_path)
    result = router.classify([ChatMessage(role="user", content="Show my accounts.")])

    assert result.route == "in_domain"
    assert result.intent == "view_accounts"
    assert result.capability == "view_accounts"
    assert result.domain is None
