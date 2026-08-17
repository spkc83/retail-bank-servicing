from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file
from torch import nn

from hello_slm.banking_conversation_router_data import (
    RELATION_LABELS,
    lane_for_intent,
    render_router_input_with_context,
)

MessageRole = Literal["system", "user", "assistant"]
RouteDecision = Literal["in_domain", "out_of_domain", "uncertain"]
MUTATION_FINE_INTENTS = frozenset(
    {"freeze_card", "replace_card", "dispute_transaction", "cancel_transfer"}
)


@dataclass(frozen=True)
class ChatMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True)
class ConversationRouteResult:
    route: RouteDecision
    confidence: float
    banking_probability: float
    ood_probability: float
    intent: str | None
    lane: str | None
    intent_confidence: float
    intent_candidates: tuple[dict[str, float | str], ...]
    relation_probabilities: dict[str, float]
    active_relations: tuple[str, ...]
    context_applied: bool
    reason: str
    support_probability: float = 0.0
    domain: str | None = None
    domain_probabilities: dict[str, float] | None = None
    domain_confidence: float = 0.0
    domain_candidates: tuple[dict[str, float | str], ...] = ()
    lane_confidence: float = 0.0
    lane_candidates: tuple[dict[str, float | str], ...] = ()
    family: str | None = None
    family_confidence: float = 0.0
    family_candidates: tuple[dict[str, float | str], ...] = ()
    action: str | None = None
    action_confidence: float = 0.0
    action_candidates: tuple[dict[str, float | str], ...] = ()
    entity_resolution: str | None = None
    entity_resolution_confidence: float = 0.0
    entity_resolution_candidates: tuple[dict[str, float | str], ...] = ()
    constraint_diagnostics: tuple[str, ...] = ()
    selected_intent_probability: float = 0.0
    joint_decision_contract: str | None = None
    joint_decision_accepted: bool = False
    topic_shift_recheck_applied: bool = False
    contextual_decision: dict[str, str | None] | None = None

    @property
    def capability(self) -> str | None:
        """V2 compatibility alias; V4 artifact callers should use ``intent``."""
        return self.intent

    @property
    def capability_confidence(self) -> float:
        return self.intent_confidence

    @property
    def capability_candidates(self) -> tuple[dict[str, float | str], ...]:
        return tuple(
            {
                "capability": candidate["intent"],
                "probability": candidate["probability"],
            }
            for candidate in self.intent_candidates
        )


@dataclass(frozen=True)
class ConversationRouterOutput:
    """Named V4 head output shared by training and inference."""

    domain_logits: torch.Tensor
    lane_logits: torch.Tensor
    family_logits: torch.Tensor
    intent_logits: torch.Tensor
    relation_logits: torch.Tensor
    action_logits: torch.Tensor
    entity_resolution_logits: torch.Tensor


@dataclass(frozen=True)
class V4JointDecision:
    """Highest-scoring legal tuple under the canonical banking ontology."""

    domain: str
    lane: str
    family: str
    intent: str | None
    action: str
    entity_resolution: str
    score: float


class ConversationRouterModel(nn.Module):
    """One shared encoder with legacy V2/V3 or hierarchical V4 heads."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int,
        num_relations: int,
        num_intents: int | None = None,
        num_capabilities: int | None = None,
        num_domains: int = 2,
        num_lanes: int | None = None,
        num_families: int | None = None,
        num_actions: int | None = None,
        num_entity_resolutions: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.domain_head = nn.Linear(hidden_size, num_domains)
        intent_count = num_intents if num_intents is not None else num_capabilities
        if intent_count is None:
            raise ValueError("num_intents is required")
        self.intent_head = nn.Linear(hidden_size, intent_count)
        self.relation_head = nn.Linear(hidden_size, num_relations)
        hierarchy_counts = (
            num_lanes,
            num_families,
            num_actions,
            num_entity_resolutions,
        )
        if any(count is not None for count in hierarchy_counts) and not all(
            count is not None for count in hierarchy_counts
        ):
            raise ValueError("V4 hierarchy head counts must be provided together")
        self.lane_head = nn.Linear(hidden_size, num_lanes) if num_lanes is not None else None
        self.family_head = (
            nn.Linear(hidden_size, num_families) if num_families is not None else None
        )
        self.action_head = nn.Linear(hidden_size, num_actions) if num_actions is not None else None
        self.entity_resolution_head = (
            nn.Linear(hidden_size, num_entity_resolutions)
            if num_entity_resolutions is not None
            else None
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | ConversationRouterOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        domain_logits = self.domain_head(pooled)
        intent_logits = self.intent_head(pooled)
        relation_logits = self.relation_head(pooled)
        if self.lane_head is None:
            return domain_logits, intent_logits, relation_logits
        assert self.family_head is not None
        assert self.action_head is not None
        assert self.entity_resolution_head is not None
        return ConversationRouterOutput(
            domain_logits=domain_logits,
            lane_logits=self.lane_head(pooled),
            family_logits=self.family_head(pooled),
            intent_logits=intent_logits,
            relation_logits=relation_logits,
            action_logits=self.action_head(pooled),
            entity_resolution_logits=self.entity_resolution_head(pooled),
        )


class LearnedConversationRouter:
    """History-aware OOD/continuity gate with diagnostic fine intents."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: ConversationRouterModel,
        intent_labels: Sequence[str] | None = None,
        relation_labels: Sequence[str],
        ood_banking_threshold: float,
        in_domain_threshold: float,
        relation_rescue_threshold: float,
        relation_thresholds: Mapping[str, float] | None = None,
        max_length: int,
        max_exchanges: int = 3,
        device: torch.device | str = "cpu",
        capability_labels: Sequence[str] | None = None,
        format_version: int = 3,
        domain_labels: Sequence[str] | None = None,
        lane_labels: Sequence[str] | None = None,
        family_labels: Sequence[str] | None = None,
        action_labels: Sequence[str] | None = None,
        entity_resolution_labels: Sequence[str] | None = None,
    ) -> None:
        labels = intent_labels if intent_labels is not None else capability_labels
        if not labels:
            raise ValueError("intent_labels must not be empty")
        if not relation_labels:
            raise ValueError("relation_labels must not be empty")
        if not 0.0 < ood_banking_threshold < in_domain_threshold < 1.0:
            raise ValueError("domain thresholds must satisfy 0 < OOD < in-domain < 1")
        if not 0.0 < relation_rescue_threshold < 1.0:
            raise ValueError("relation_rescue_threshold must be between zero and one")
        if max_length < 32:
            raise ValueError("max_length must be at least 32")
        if max_exchanges < 1:
            raise ValueError("max_exchanges must be positive")

        self.tokenizer = tokenizer
        self.model = model
        self.intent_labels = tuple(labels)
        self.relation_labels = tuple(relation_labels)
        self.format_version = format_version
        self.domain_labels = tuple(domain_labels or ())
        self.lane_labels = tuple(lane_labels or ())
        self.family_labels = tuple(family_labels or ())
        self.action_labels = tuple(action_labels or ())
        self.entity_resolution_labels = tuple(entity_resolution_labels or ())
        if self.format_version == 4 and not all(
            (
                self.domain_labels,
                self.lane_labels,
                self.family_labels,
                self.action_labels,
                self.entity_resolution_labels,
            )
        ):
            raise ValueError("V4 hierarchy labels must not be empty")
        self.ood_banking_threshold = ood_banking_threshold
        self.in_domain_threshold = in_domain_threshold
        self.relation_rescue_threshold = relation_rescue_threshold
        self.relation_thresholds = {
            label: float((relation_thresholds or {}).get(label, 0.5))
            for label in self.relation_labels
        }
        self.max_length = max_length
        self.max_exchanges = max_exchanges
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> LearnedConversationRouter:
        from transformers import AutoModel, AutoTokenizer

        root = Path(artifact_dir)
        config = verify_router_artifact(root)
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        encoder = AutoModel.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
        )
        format_version = int(config["format_version"])
        label_key = "capability_labels" if format_version == 2 else "intent_labels"
        intents = tuple(str(label) for label in config[label_key])
        relations = tuple(str(label) for label in config["relation_labels"])
        domain_labels = tuple(str(label) for label in config.get("domain_labels", ()))
        lane_labels = tuple(str(label) for label in config.get("lane_labels", ()))
        family_labels = tuple(str(label) for label in config.get("family_labels", ()))
        action_labels = tuple(str(label) for label in config.get("action_labels", ()))
        entity_resolution_labels = tuple(
            str(label) for label in config.get("entity_resolution_labels", ())
        )
        model = ConversationRouterModel(
            encoder,
            hidden_size=int(encoder.config.hidden_size),
            num_intents=len(intents),
            num_relations=len(relations),
            num_domains=len(domain_labels) if format_version == 4 else 2,
            num_lanes=len(lane_labels) if format_version == 4 else None,
            num_families=len(family_labels) if format_version == 4 else None,
            num_actions=len(action_labels) if format_version == 4 else None,
            num_entity_resolutions=(len(entity_resolution_labels) if format_version == 4 else None),
        )
        heads = load_file(root / "classifier_heads.safetensors", device="cpu")
        head_modules: list[tuple[str, nn.Linear | None]] = [
            ("domain_head", model.domain_head),
            (
                "capability_head" if format_version == 2 else "intent_head",
                model.intent_head,
            ),
            ("relation_head", model.relation_head),
        ]
        if format_version == 4:
            head_modules.extend(
                (
                    ("lane_head", model.lane_head),
                    ("family_head", model.family_head),
                    ("action_head", model.action_head),
                    ("entity_resolution_head", model.entity_resolution_head),
                )
            )
        _load_head_modules(heads, head_modules)
        return cls(
            tokenizer=tokenizer,
            model=model,
            intent_labels=intents,
            relation_labels=relations,
            ood_banking_threshold=float(config["ood_banking_threshold"]),
            in_domain_threshold=float(config["in_domain_threshold"]),
            relation_rescue_threshold=float(config["relation_rescue_threshold"]),
            relation_thresholds=config.get("relation_thresholds"),
            max_length=int(config["max_length"]),
            max_exchanges=int(config.get("max_exchanges", 3)),
            device=device,
            format_version=format_version,
            domain_labels=domain_labels,
            lane_labels=lane_labels,
            family_labels=family_labels,
            action_labels=action_labels,
            entity_resolution_labels=entity_resolution_labels,
        )

    @classmethod
    def from_hub(
        cls,
        *,
        repo_id: str,
        revision: str,
        device: torch.device | str = "cpu",
        token: str | None = None,
    ) -> LearnedConversationRouter:
        from huggingface_hub import snapshot_download

        invalid_character = any(character not in "0123456789abcdef" for character in revision)
        if len(revision) != 40 or invalid_character:
            raise ValueError("revision must be an immutable 40-character commit")
        root = snapshot_download(repo_id=repo_id, revision=revision, token=token)
        return cls.from_artifact_dir(root, device=device)

    def classify(
        self,
        messages: Sequence[ChatMessage],
        *,
        prior_dialogue_state: Mapping[str, Any] | None = None,
    ) -> ConversationRouteResult:
        user_indices = [
            index
            for index, message in enumerate(messages)
            if message.role == "user" and message.content.strip()
        ]
        if not user_indices:
            raise ValueError("at least one non-empty user message is required")
        current_index = user_indices[-1]
        current = messages[current_index].content.strip()
        history = [
            {"role": message.role, "content": message.content.strip()}
            for message in messages[:current_index]
            if message.role in {"user", "assistant"} and message.content.strip()
        ]
        rendered, context_applied = render_router_input_with_context(
            current,
            history,
            max_exchanges=self.max_exchanges,
            prior_dialogue_state=prior_dialogue_state,
        )
        result = self._predict(
            rendered,
            context_applied=context_applied,
            current_text=current,
        )
        if self.format_version == 4 and _needs_topic_shift_recheck(result):
            current_only_rendered, _ = render_router_input_with_context(
                current,
                [],
                max_exchanges=self.max_exchanges,
                prior_dialogue_state=None,
            )
            current_result = self._predict(
                current_only_rendered,
                context_applied=False,
                current_text=current,
            )
            if _prefer_current_only_topic_shift(result, current_result):
                result = replace(
                    current_result,
                    context_applied=context_applied,
                    topic_shift_recheck_applied=True,
                    contextual_decision={
                        "domain": result.domain,
                        "lane": result.lane,
                        "family": result.family,
                        "intent": result.intent,
                        "action": result.action,
                    },
                    constraint_diagnostics=(
                        *current_result.constraint_diagnostics,
                        "constraint:topic-shift-current-only-recheck",
                    ),
                )
        return result

    def _predict(
        self,
        rendered: str,
        *,
        context_applied: bool,
        current_text: str = "",
    ) -> ConversationRouteResult:
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {
            name: tensor.to(self.device)
            for name, tensor in encoded.items()
            if name in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            output = self.model(**inputs)
        if self.format_version == 4:
            if not isinstance(output, ConversationRouterOutput):
                raise TypeError("V4 router model must return ConversationRouterOutput")
            return self._predict_v4(
                output,
                context_applied=context_applied,
                current_text=current_text,
            )
        if not isinstance(output, tuple) or len(output) != 3:
            raise TypeError("V2/V3 router model must return three logits tensors")
        domain_logits, intent_logits, relation_logits = output
        return self._predict_legacy(
            domain_logits,
            intent_logits,
            relation_logits,
            context_applied=context_applied,
        )

    def _predict_legacy(
        self,
        domain_logits: torch.Tensor,
        intent_logits: torch.Tensor,
        relation_logits: torch.Tensor,
        *,
        context_applied: bool,
    ) -> ConversationRouteResult:
        banking_probability = float(torch.softmax(domain_logits.float(), dim=-1)[0, 1].cpu())
        ood_probability = 1.0 - banking_probability
        relations, active_relations, rescue_probability = self._relations(relation_logits)
        route = self._route(banking_probability, rescue_probability)
        candidates, intent_confidence = _top_candidates(
            intent_logits,
            self.intent_labels,
            key="intent",
        )
        intent = str(candidates[0]["intent"]) if route == "in_domain" else None
        confidence = (
            banking_probability
            if route == "in_domain"
            else ood_probability
            if route == "out_of_domain"
            else max(banking_probability, rescue_probability)
        )
        reason = (
            f"banking={banking_probability:.6f}, "
            f"rescue={rescue_probability:.6f}, "
            f"ood_boundary={self.ood_banking_threshold:.6f}, "
            f"in_domain_boundary={self.in_domain_threshold:.6f}"
        )
        return ConversationRouteResult(
            route=route,
            confidence=confidence,
            banking_probability=banking_probability,
            ood_probability=ood_probability,
            intent=intent,
            lane=lane_for_intent(intent) if intent is not None else None,
            intent_confidence=intent_confidence,
            intent_candidates=candidates,
            relation_probabilities=relations,
            active_relations=active_relations,
            context_applied=context_applied,
            reason=reason,
            support_probability=banking_probability,
        )

    def _predict_v4(
        self,
        output: ConversationRouterOutput,
        *,
        context_applied: bool,
        current_text: str = "",
    ) -> ConversationRouteResult:
        domain_candidates, domain_confidence = _top_candidates(
            output.domain_logits,
            self.domain_labels,
            key="domain",
        )
        domain_probabilities = {
            str(item["domain"]): float(item["probability"])
            for item in _all_candidates(
                output.domain_logits,
                self.domain_labels,
                key="domain",
            )
        }
        ood_probability = domain_probabilities["out_of_domain"]
        banking_probability = 1.0 - ood_probability
        relations, active_relations, rescue_probability = self._relations(output.relation_logits)
        active_relations = stabilize_active_relations(
            active_relations,
            current_text=current_text,
            context_applied=context_applied,
        )
        route = self._route(banking_probability, rescue_probability)
        intent_candidates, intent_confidence = _top_candidates(
            output.intent_logits,
            self.intent_labels,
            key="intent",
        )
        lane_candidates, lane_confidence = _top_candidates(
            output.lane_logits,
            self.lane_labels,
            key="lane",
        )
        family_candidates, family_confidence = _top_candidates(
            output.family_logits,
            self.family_labels,
            key="family",
        )
        action_candidates, action_confidence = _top_candidates(
            output.action_logits,
            self.action_labels,
            key="action",
        )
        entity_candidates, entity_confidence = _top_candidates(
            output.entity_resolution_logits,
            self.entity_resolution_labels,
            key="entity_resolution",
        )
        raw_tuple = (
            str(domain_candidates[0]["domain"]),
            str(lane_candidates[0]["lane"]),
            str(family_candidates[0]["family"]),
            str(intent_candidates[0]["intent"]),
            str(action_candidates[0]["action"]),
            str(entity_candidates[0]["entity_resolution"]),
        )
        decision = decode_v4_joint(
            domain_scores=output.domain_logits.float()[0].cpu().tolist(),
            lane_scores=output.lane_logits.float()[0].cpu().tolist(),
            family_scores=output.family_logits.float()[0].cpu().tolist(),
            intent_scores=output.intent_logits.float()[0].cpu().tolist(),
            action_scores=output.action_logits.float()[0].cpu().tolist(),
            entity_resolution_scores=(output.entity_resolution_logits.float()[0].cpu().tolist()),
            domain_labels=self.domain_labels,
            lane_labels=self.lane_labels,
            family_labels=self.family_labels,
            intent_labels=self.intent_labels,
            action_labels=self.action_labels,
            entity_resolution_labels=self.entity_resolution_labels,
        )
        domain = decision.domain
        lane: str | None = decision.lane
        family: str | None = decision.family
        intent: str | None = decision.intent
        action: str | None = decision.action
        entity_resolution: str | None = decision.entity_resolution
        decoded_tuple = (domain, lane, family, intent, action, entity_resolution)
        diagnostics: tuple[str, ...] = (
            ("constraint:joint-decoder-resolved-independent-head-conflict",)
            if decoded_tuple != raw_tuple
            else ()
        )
        if route == "out_of_domain":
            if domain != "out_of_domain":
                diagnostics += ("constraint:route-ood-overrode-joint-decision",)
            diagnostics += ("constraint:ood-suppressed-downstream",)
            domain = "out_of_domain"
            intent = None
            lane = None
            family = None
            action = "refuse_ood"
            entity_resolution = "not_required"
        elif route == "uncertain":
            intent = None
            lane = None
            family = None
            action = None
            entity_resolution = None
        elif domain == "out_of_domain":
            route = "uncertain"
            intent = None
            lane = None
            family = None
            action = None
            entity_resolution = None
            diagnostics += ("constraint:domain-route-conflict",)
        elif action == "converse" and intent in MUTATION_FINE_INTENTS:
            diagnostics += ("constraint:mutation-intent-cannot-converse",)
            if "clarify" in self.action_labels:
                action = "clarify"
                if entity_resolution not in {"missing", "ambiguous", "ineligible"}:
                    entity_resolution = "missing"
            else:
                route = "uncertain"
                intent = None
                lane = None
                family = None
                action = None
                entity_resolution = None
        confidence = (
            banking_probability
            if route == "in_domain"
            else ood_probability
            if route == "out_of_domain"
            else max(banking_probability, rescue_probability)
        )
        reason_parts = [
            f"support={banking_probability:.6f}",
            f"ood={ood_probability:.6f}",
            f"rescue={rescue_probability:.6f}",
            f"ood_boundary={self.ood_banking_threshold:.6f}",
            f"in_domain_boundary={self.in_domain_threshold:.6f}",
        ]
        reason_parts.extend(diagnostics)
        return ConversationRouteResult(
            route=route,
            confidence=confidence,
            banking_probability=banking_probability,
            ood_probability=ood_probability,
            intent=intent,
            lane=lane,
            intent_confidence=intent_confidence,
            intent_candidates=intent_candidates,
            relation_probabilities=relations,
            active_relations=active_relations,
            context_applied=context_applied,
            reason=", ".join(reason_parts),
            support_probability=banking_probability,
            domain=domain,
            domain_probabilities=domain_probabilities,
            domain_confidence=domain_confidence,
            domain_candidates=domain_candidates,
            lane_confidence=lane_confidence,
            lane_candidates=lane_candidates,
            family=family,
            family_confidence=family_confidence,
            family_candidates=family_candidates,
            action=action,
            action_confidence=action_confidence,
            action_candidates=action_candidates,
            entity_resolution=entity_resolution,
            entity_resolution_confidence=entity_confidence,
            entity_resolution_candidates=entity_candidates,
            constraint_diagnostics=diagnostics,
            selected_intent_probability=_candidate_probability(
                intent_candidates,
                "intent",
                intent,
            ),
            joint_decision_contract="hierarchical-router-joint-decision/v1",
            joint_decision_accepted=route == "in_domain",
        )

    def _relations(
        self,
        relation_logits: torch.Tensor,
    ) -> tuple[dict[str, float], tuple[str, ...], float]:
        relation_values = torch.sigmoid(relation_logits.float())[0].cpu().tolist()
        relations = dict(zip(self.relation_labels, relation_values, strict=True))
        active_relations = tuple(
            label
            for label in self.relation_labels
            if relations[label] >= self.relation_thresholds[label]
        )
        rescue_probability = max(
            relations.get("context_dependent", 0.0),
            relations.get("agent_repair", 0.0),
            relations.get("clarification_answer", 0.0),
            relations.get("resume_previous_service", 0.0),
        )
        return relations, active_relations, rescue_probability

    def _route(
        self,
        support_probability: float,
        rescue_probability: float,
    ) -> RouteDecision:
        if support_probability >= self.in_domain_threshold:
            return "in_domain"
        if (
            support_probability < self.ood_banking_threshold
            and rescue_probability < self.relation_rescue_threshold
        ):
            return "out_of_domain"
        return "uncertain"


def verify_router_artifact(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    config_path = root / "router_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != "banking-conversation-router-artifact":
        raise ValueError("unexpected router artifact contract")
    if manifest.get("release_eligible") is not True:
        raise ValueError("router artifact did not pass release gates")
    for entry in manifest.get("files", []):
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe router artifact path: {relative}")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing router artifact file: {relative}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"router artifact size mismatch: {relative}")
        if _file_sha256(path) != str(entry["sha256"]):
            raise ValueError(f"router artifact digest mismatch: {relative}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("contract") != "banking-conversation-router":
        raise ValueError("unexpected router configuration contract")
    format_version = int(config.get("format_version", 0))
    if format_version not in {2, 3, 4}:
        raise ValueError("unsupported router configuration version")
    label_key = "capability_labels" if format_version == 2 else "intent_labels"
    if not isinstance(config.get(label_key), list) or not config[label_key]:
        raise ValueError(f"router configuration is missing {label_key}")
    if format_version == 4:
        _verify_v4_config(config)
        _verify_v4_head_artifact(root, config)
    return config


def _verify_v4_config(config: Mapping[str, Any]) -> None:
    from hello_slm.banking_conversation_router_data import RELATION_LABELS
    from hello_slm.banking_domain_taxonomy import (
        ACTION_LABELS,
        DOMAIN_LABELS,
        ENTITY_RESOLUTION_LABELS,
        FAMILY_LABELS,
        INTENT_LABELS,
        LANE_LABELS,
    )

    expected_labels = {
        "domain_labels": DOMAIN_LABELS,
        "lane_labels": LANE_LABELS,
        "family_labels": FAMILY_LABELS,
        "intent_labels": INTENT_LABELS,
        "relation_labels": RELATION_LABELS,
        "action_labels": ACTION_LABELS,
        "entity_resolution_labels": ENTITY_RESOLUTION_LABELS,
    }
    for key, expected in expected_labels.items():
        actual = config.get(key)
        if not isinstance(actual, list) or tuple(actual) != tuple(expected):
            raise ValueError(f"V4 router configuration has non-canonical {key}")


def _verify_v4_head_artifact(root: Path, config: Mapping[str, Any]) -> None:
    path = root / "classifier_heads.safetensors"
    if not path.is_file():
        raise ValueError("V4 router artifact is missing classifier_heads.safetensors")
    try:
        heads = load_file(path, device="cpu")
    except Exception as error:
        raise ValueError("V4 router classifier heads are corrupt") from error
    label_keys = {
        "domain_head": "domain_labels",
        "lane_head": "lane_labels",
        "family_head": "family_labels",
        "intent_head": "intent_labels",
        "relation_head": "relation_labels",
        "action_head": "action_labels",
        "entity_resolution_head": "entity_resolution_labels",
    }
    hidden_sizes: set[int] = set()
    for head_name, label_key in label_keys.items():
        weight_key = f"{head_name}.weight"
        bias_key = f"{head_name}.bias"
        if weight_key not in heads or bias_key not in heads:
            raise ValueError(f"V4 router artifact is missing {head_name} tensors")
        weight = heads[weight_key]
        bias = heads[bias_key]
        label_count = len(config[label_key])
        if (
            weight.ndim != 2
            or bias.ndim != 1
            or weight.shape[0] != label_count
            or bias.shape[0] != label_count
        ):
            raise ValueError(f"V4 router artifact has invalid {head_name} tensor shape")
        hidden_sizes.add(int(weight.shape[1]))
    if len(hidden_sizes) != 1:
        raise ValueError("V4 router classifier heads do not share an encoder width")


def _load_head_modules(
    tensors: Mapping[str, torch.Tensor],
    modules: Sequence[tuple[str, nn.Linear | None]],
) -> None:
    for name, head in modules:
        if head is None:
            raise ValueError(f"router model is missing {name}")
        try:
            state = {
                "weight": tensors[f"{name}.weight"],
                "bias": tensors[f"{name}.bias"],
            }
        except KeyError as error:
            raise ValueError(f"router artifact is missing {name} tensors") from error
        try:
            head.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError(f"router artifact has invalid {name} tensors") from error


def _all_candidates(
    logits: torch.Tensor,
    labels: Sequence[str],
    *,
    key: str,
) -> tuple[dict[str, float | str], ...]:
    probabilities = torch.softmax(logits.float(), dim=-1)[0]
    if probabilities.numel() != len(labels):
        raise ValueError(f"{key} logits do not match configured labels")
    return tuple(
        {key: label, "probability": float(probability.cpu())}
        for label, probability in zip(labels, probabilities, strict=True)
    )


def _candidate_probability(
    candidates: Sequence[Mapping[str, float | str]],
    key: str,
    selected: str | None,
) -> float:
    if selected is None:
        return 0.0
    return next(
        (
            float(candidate["probability"])
            for candidate in candidates
            if candidate.get(key) == selected
        ),
        0.0,
    )


def _needs_topic_shift_recheck(result: ConversationRouteResult) -> bool:
    return (
        result.route == "in_domain"
        and result.context_applied
        and set(result.active_relations) == {"topic_shift"}
    )


def _prefer_current_only_topic_shift(
    contextual: ConversationRouteResult,
    current_only: ConversationRouteResult,
) -> bool:
    return (
        current_only.route == "in_domain"
        and current_only.joint_decision_accepted
        and current_only.selected_intent_probability >= 0.75
        and current_only.intent != contextual.intent
    )


def stabilize_active_relations(
    active_relations: Sequence[str],
    *,
    current_text: str,
    context_applied: bool,
) -> tuple[str, ...]:
    """Complete explicit prior-answer repairs without changing the learned intent."""

    active = set(active_relations)
    normalized = " ".join(current_text.casefold().replace("’", "'").split())
    repair_markers = (
        "didn't ask",
        "did not ask",
        "wasn't asking",
        "was not asking",
        "never asked",
        "not what i asked",
        "not my request",
        "wrong subject",
        "wrong topic",
    )
    if (
        context_applied
        and "topic_shift" in active
        and any(marker in normalized for marker in repair_markers)
    ):
        active.update(("agent_repair", "context_dependent"))
    return tuple(label for label in RELATION_LABELS if label in active)


def _top_candidates(
    logits: torch.Tensor,
    labels: Sequence[str],
    *,
    key: str,
) -> tuple[tuple[dict[str, float | str], ...], float]:
    candidates = _all_candidates(logits, labels, key=key)
    ranked = tuple(
        sorted(candidates, key=lambda item: float(item["probability"]), reverse=True)[:3]
    )
    return ranked, float(ranked[0]["probability"])


def decode_v4_joint(
    *,
    domain_scores: Sequence[float],
    lane_scores: Sequence[float],
    family_scores: Sequence[float],
    intent_scores: Sequence[float],
    action_scores: Sequence[float],
    entity_resolution_scores: Sequence[float],
    domain_labels: Sequence[str],
    lane_labels: Sequence[str],
    family_labels: Sequence[str],
    intent_labels: Sequence[str],
    action_labels: Sequence[str],
    entity_resolution_labels: Sequence[str],
) -> V4JointDecision:
    """Decode the maximum-score legal tuple without cascading head decisions."""
    dimensions = (
        ("domain", domain_scores, domain_labels),
        ("lane", lane_scores, lane_labels),
        ("family", family_scores, family_labels),
        ("intent", intent_scores, intent_labels),
        ("action", action_scores, action_labels),
        ("entity_resolution", entity_resolution_scores, entity_resolution_labels),
    )
    for name, scores, labels in dimensions:
        if len(scores) != len(labels) or not labels:
            raise ValueError(f"{name} scores do not match configured labels")
    score_maps = {
        name: dict(zip(labels, (float(score) for score in scores), strict=True))
        for name, scores, labels in dimensions
    }
    candidates: list[V4JointDecision] = []
    ood_score = (
        score_maps["domain"]["out_of_domain"]
        + score_maps["lane"]["out_of_domain"]
        + score_maps["family"]["external"]
        + max(score_maps["intent"].values())
        + score_maps["action"]["refuse_ood"]
        + score_maps["entity_resolution"]["not_required"]
    )
    candidates.append(
        V4JointDecision(
            domain="out_of_domain",
            lane="out_of_domain",
            family="external",
            intent=None,
            action="refuse_ood",
            entity_resolution="not_required",
            score=ood_score,
        )
    )
    for intent in intent_labels:
        domain, lane, family = _intent_hierarchy(intent)
        for action, entity_resolution in _legal_action_entities(intent, lane):
            score = (
                score_maps["domain"][domain]
                + score_maps["lane"][lane]
                + score_maps["family"][family]
                + score_maps["intent"][intent]
                + score_maps["action"][action]
                + score_maps["entity_resolution"][entity_resolution]
            )
            candidates.append(
                V4JointDecision(
                    domain=domain,
                    lane=lane,
                    family=family,
                    intent=intent,
                    action=action,
                    entity_resolution=entity_resolution,
                    score=score,
                )
            )
    safety_priority = {"refuse_ood": 3, "clarify": 3, "converse": 2, "retrieve_policy": 2}
    return max(
        candidates,
        key=lambda item: (item.score, safety_priority.get(item.action, 1)),
    )


def _legal_action_entities(intent: str, lane: str) -> tuple[tuple[str, str], ...]:
    if lane == "policy":
        return (("retrieve_policy", "not_required"),)
    if lane in {"conversation", "other_banking"}:
        return (("converse", "not_required"),)
    entity_required = intent in {
        "freeze_card",
        "replace_card",
        "dispute_transaction",
        "cancel_transfer",
    }
    execute_entity = "resolved" if entity_required else "not_required"
    return (
        ("execute_tool", execute_entity),
        ("clarify", "missing"),
        ("clarify", "ambiguous"),
        ("converse", "not_required"),
        ("converse", "ineligible"),
    )


def _apply_v4_constraints(
    *,
    route: RouteDecision,
    domain: str,
    lane: str,
    family: str,
    intent: str,
    action: str,
    entity_resolution: str,
    action_labels: Sequence[str],
) -> tuple[RouteDecision, str, tuple[str, ...]]:
    diagnostics: list[str] = []
    if domain == "out_of_domain":
        if route != "out_of_domain":
            route = "uncertain"
            diagnostics.append("constraint:domain-route-conflict")
        diagnostics.append("constraint:ood-suppressed-downstream")
        return route, action, tuple(diagnostics)

    expected_domain, expected_lane, expected_family = _intent_hierarchy(intent)
    if domain != expected_domain:
        route = "uncertain"
        diagnostics.append("constraint:intent-domain-incompatible")
    if expected_lane is not None and lane != expected_lane:
        route = "uncertain"
        diagnostics.append("constraint:intent-lane-incompatible")
    if expected_family is not None and family != expected_family:
        route = "uncertain"
        diagnostics.append("constraint:intent-family-incompatible")

    if action == "execute_tool" and entity_resolution in {"missing", "ambiguous"}:
        if "clarify" in action_labels:
            action = "clarify"
            diagnostics.append(f"constraint:{entity_resolution}-entity-requires-clarification")
        else:
            route = "uncertain"
            diagnostics.append(f"constraint:{entity_resolution}-entity-blocked-execution")
    elif action == "execute_tool" and entity_resolution == "ineligible":
        route = "uncertain"
        diagnostics.append("constraint:ineligible-entity-blocked-execution")
    if action == "clarify" and entity_resolution not in {"missing", "ambiguous"}:
        route = "uncertain"
        diagnostics.append("constraint:clarify-requires-unresolved-entity")
    if action == "converse" and intent in MUTATION_FINE_INTENTS:
        if "clarify" in action_labels:
            action = "clarify"
            if entity_resolution not in {"missing", "ambiguous", "ineligible"}:
                entity_resolution = "missing"
            diagnostics.append("constraint:mutation-intent-cannot-converse")
        else:
            route = "uncertain"
            diagnostics.append("constraint:mutation-intent-cannot-converse")

    allowed_actions = _allowed_actions(lane)
    if allowed_actions is not None and action not in allowed_actions:
        route = "uncertain"
        diagnostics.append("constraint:lane-action-incompatible")
    return route, action, tuple(diagnostics)


def _intent_hierarchy(intent: str) -> tuple[str, str, str]:
    from hello_slm.banking_domain_taxonomy import hierarchy_for_intent

    return hierarchy_for_intent(intent)


def _allowed_actions(lane: str) -> frozenset[str] | None:
    return {
        "servicing": frozenset({"execute_tool", "clarify", "converse"}),
        "policy": frozenset({"retrieve_policy", "clarify"}),
        "conversation": frozenset({"converse"}),
        "other_banking": frozenset({"converse", "clarify"}),
    }.get(lane)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
