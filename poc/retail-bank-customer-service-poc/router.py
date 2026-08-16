from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROUTER_REPO_ID = os.environ.get(
    "RETAIL_BANK_ROUTER_ID",
    "spkc83/retail-bank-conversation-router",
)
ROUTER_REVISION = os.environ.get(
    "RETAIL_BANK_ROUTER_REVISION",
    "25176d6b7f46d10812443cb0f8f043e3dbd36f48",
)
RELATION_LABELS = (
    "context_dependent",
    "agent_repair",
    "topic_shift",
    "clarification_answer",
    "resume_previous_service",
)


@dataclass(frozen=True)
class ConversationRouterOutput:
    domain_logits: torch.Tensor
    lane_logits: torch.Tensor
    family_logits: torch.Tensor
    intent_logits: torch.Tensor
    relation_logits: torch.Tensor
    action_logits: torch.Tensor
    entity_resolution_logits: torch.Tensor


@dataclass(frozen=True)
class V4JointDecision:
    domain: str
    lane: str
    family: str
    intent: str | None
    action: str
    entity_resolution: str
    score: float


class ConversationRouterModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int,
        intent_count: int | None = None,
        capability_count: int | None = None,
        relation_count: int,
        domain_count: int = 2,
        lane_count: int | None = None,
        family_count: int | None = None,
        action_count: int | None = None,
        entity_resolution_count: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.domain_head = nn.Linear(hidden_size, domain_count)
        label_count = intent_count if intent_count is not None else capability_count
        if label_count is None:
            raise ValueError("intent_count is required")
        self.intent_head = nn.Linear(hidden_size, label_count)
        self.relation_head = nn.Linear(hidden_size, relation_count)
        hierarchy_counts = (
            lane_count,
            family_count,
            action_count,
            entity_resolution_count,
        )
        if any(count is not None for count in hierarchy_counts) and not all(
            count is not None for count in hierarchy_counts
        ):
            raise ValueError("V4 hierarchy head counts must be provided together")
        self.lane_head = nn.Linear(hidden_size, lane_count) if lane_count is not None else None
        self.family_head = (
            nn.Linear(hidden_size, family_count) if family_count is not None else None
        )
        self.action_head = (
            nn.Linear(hidden_size, action_count) if action_count is not None else None
        )
        self.entity_resolution_head = (
            nn.Linear(hidden_size, entity_resolution_count)
            if entity_resolution_count is not None
            else None
        )

    @property
    def capability_head(self) -> nn.Linear:
        """Compatibility alias for V2 artifacts and tests."""
        return self.intent_head

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | ConversationRouterOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
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


class LearnedBankingRouter:
    """History- and state-aware CPU gate with calibrated fine intents."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        intent_labels: tuple[str, ...] | None = None,
        capability_labels: tuple[str, ...] | None = None,
        relation_labels: tuple[str, ...],
        ood_banking_threshold: float,
        in_domain_threshold: float,
        relation_rescue_threshold: float,
        relation_thresholds: Mapping[str, float] | None = None,
        max_length: int,
        max_exchanges: int,
        artifact_identity: Mapping[str, str] | None = None,
        format_version: int = 3,
        domain_labels: tuple[str, ...] = (),
        lane_labels: tuple[str, ...] = (),
        family_labels: tuple[str, ...] = (),
        action_labels: tuple[str, ...] = (),
        entity_resolution_labels: tuple[str, ...] = (),
    ) -> None:
        labels = intent_labels if intent_labels is not None else capability_labels
        if not labels or not relation_labels:
            raise ValueError("router labels must not be empty")
        if not 0.0 < ood_banking_threshold < in_domain_threshold < 1.0:
            raise ValueError("invalid domain thresholds")
        if not 0.0 < relation_rescue_threshold < 1.0:
            raise ValueError("invalid relation rescue threshold")
        self.tokenizer = tokenizer
        self.model = model.to("cpu").eval()
        self.intent_labels = tuple(labels)
        self.relation_labels = relation_labels
        self.format_version = format_version
        self.domain_labels = domain_labels
        self.lane_labels = lane_labels
        self.family_labels = family_labels
        self.action_labels = action_labels
        self.entity_resolution_labels = entity_resolution_labels
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
        self.artifact_identity = dict(artifact_identity or {})

    @classmethod
    def from_hub(cls) -> LearnedBankingRouter:
        if not _is_commit(ROUTER_REVISION):
            raise RuntimeError(
                "RETAIL_BANK_ROUTER_REVISION must pin the published V7 router commit"
            )
        root = Path(snapshot_download(ROUTER_REPO_ID, revision=ROUTER_REVISION))
        return cls.from_artifact_dir(
            root,
            artifact_identity={
                "router_source": "hub",
                "router_repo_id": ROUTER_REPO_ID,
                "router_revision": ROUTER_REVISION,
            },
        )

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        *,
        artifact_identity: Mapping[str, str] | None = None,
    ) -> LearnedBankingRouter:
        root = Path(artifact_dir)
        config = verify_artifact(root)
        manifest_sha256 = _sha256(root / "manifest.json")
        identity = {
            "router_source": "local",
            "router_artifact_path": str(root.resolve()),
            "router_manifest_sha256": manifest_sha256,
            "router_revision": f"local-sha256:{manifest_sha256}",
            "router_config_sha256": _sha256(root / "router_config.json"),
            "router_data_manifest_sha256": str(config.get("data_manifest_sha256", "unavailable")),
            **dict(artifact_identity or {}),
        }
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
            intent_count=len(intents),
            relation_count=len(relations),
            domain_count=len(domain_labels) if format_version == 4 else 2,
            lane_count=len(lane_labels) if format_version == 4 else None,
            family_count=len(family_labels) if format_version == 4 else None,
            action_count=len(action_labels) if format_version == 4 else None,
            entity_resolution_count=(
                len(entity_resolution_labels) if format_version == 4 else None
            ),
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
            artifact_identity=identity,
            format_version=format_version,
            domain_labels=domain_labels,
            lane_labels=lane_labels,
            family_labels=family_labels,
            action_labels=action_labels,
            entity_resolution_labels=entity_resolution_labels,
        )

    def artifact_metadata(self) -> dict[str, str]:
        return dict(self.artifact_identity)

    def classify(
        self,
        message: str,
        history: list[dict[str, Any]] | None,
        *,
        dialogue_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        rendered, context_applied = render_router_input(
            message.strip(),
            history,
            max_exchanges=self.max_exchanges,
            dialogue_state=dialogue_state,
        )
        current = message.strip()
        result = self._predict(
            rendered,
            context_applied=context_applied,
            current_text=current,
        )
        if self.format_version == 4 and _needs_topic_shift_recheck(result):
            current_only, _ = render_router_input(
                current,
                [],
                max_exchanges=self.max_exchanges,
                dialogue_state=None,
            )
            current_result = self._predict(
                current_only,
                context_applied=False,
                current_text=current,
            )
            if _prefer_current_only_topic_shift(result, current_result):
                result = {
                    **current_result,
                    "context_applied": context_applied,
                    "topic_shift_recheck_applied": True,
                    "contextual_decision": {
                        key: result.get(key)
                        for key in ("domain", "lane", "family", "intent", "action")
                    },
                    "constraint_diagnostics": tuple(result.get("constraint_diagnostics", ()))
                    + ("constraint:topic-shift-current-only-recheck",),
                }
        return result

    def _predict(
        self,
        rendered: str,
        *,
        context_applied: bool,
        current_text: str = "",
    ) -> dict[str, Any]:
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.inference_mode():
            output = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        if self.format_version == 4:
            if not isinstance(output, ConversationRouterOutput):
                raise TypeError("V4 router model must return ConversationRouterOutput")
            return self._v4_result(
                output,
                context_applied=context_applied,
                current_text=current_text,
            )
        if not isinstance(output, tuple) or len(output) != 3:
            raise TypeError("V2/V3 router model must return three logits tensors")
        domain_logits, intent_logits, relation_logits = output
        banking_probability = float(torch.softmax(domain_logits.float(), dim=-1)[0, 1])
        ood_probability = 1.0 - banking_probability
        relations, active_relations, rescue_probability = self._relations(relation_logits)
        route = self._route(banking_probability, rescue_probability)
        ranked_intents, intent_confidence = _top_candidates(
            intent_logits, self.intent_labels, key="intent"
        )
        intent_candidates = list(ranked_intents)
        intent = cast(str, intent_candidates[0]["intent"]) if route == "in_domain" else None
        capability_candidates = [
            {"capability": item["intent"], "probability": item["probability"]}
            for item in intent_candidates
        ]
        return {
            "route": route,
            "banking_probability": banking_probability,
            "ood_probability": ood_probability,
            "confidence": (
                banking_probability
                if route == "in_domain"
                else ood_probability
                if route == "out_of_domain"
                else max(banking_probability, rescue_probability)
            ),
            "intent": intent,
            "lane": _lane_for_intent(intent) if intent is not None else None,
            "intent_confidence": intent_confidence,
            "intent_candidates": intent_candidates,
            # V2 diagnostic aliases stay available during artifact migration.
            "capability": intent,
            "capability_confidence": intent_confidence,
            "capability_candidates": capability_candidates,
            "relation_probabilities": relations,
            "relation_thresholds": self.relation_thresholds,
            "active_relations": active_relations,
            "context_applied": context_applied,
            "ood_banking_threshold": self.ood_banking_threshold,
            "in_domain_threshold": self.in_domain_threshold,
            "relation_rescue_threshold": self.relation_rescue_threshold,
            "router_revision": self.artifact_identity.get(
                "router_revision",
                ROUTER_REVISION,
            ),
            "router_architecture": "hierarchical-cross-encoder-v6",
            "router_artifact": dict(self.artifact_identity),
        }

    def _v4_result(
        self,
        output: ConversationRouterOutput,
        *,
        context_applied: bool,
        current_text: str = "",
    ) -> dict[str, Any]:
        domain_candidates, domain_confidence = _top_candidates(
            output.domain_logits, self.domain_labels, key="domain"
        )
        domain_probabilities = {
            str(item["domain"]): float(item["probability"])
            for item in _all_candidates(output.domain_logits, self.domain_labels, key="domain")
        }
        ood_probability = domain_probabilities["out_of_domain"]
        banking_probability = 1.0 - ood_probability
        relations, active_relations, rescue_probability = self._relations(output.relation_logits)
        active_relations = _stabilize_active_relations(
            active_relations,
            current_text=current_text,
            context_applied=context_applied,
        )
        route = self._route(banking_probability, rescue_probability)
        intent_candidates, intent_confidence = _top_candidates(
            output.intent_logits, self.intent_labels, key="intent"
        )
        lane_candidates, lane_confidence = _top_candidates(
            output.lane_logits, self.lane_labels, key="lane"
        )
        family_candidates, family_confidence = _top_candidates(
            output.family_logits, self.family_labels, key="family"
        )
        action_candidates, action_confidence = _top_candidates(
            output.action_logits, self.action_labels, key="action"
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
            domain_scores=output.domain_logits.float()[0].tolist(),
            lane_scores=output.lane_logits.float()[0].tolist(),
            family_scores=output.family_logits.float()[0].tolist(),
            intent_scores=output.intent_logits.float()[0].tolist(),
            action_scores=output.action_logits.float()[0].tolist(),
            entity_resolution_scores=output.entity_resolution_logits.float()[0].tolist(),
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
        diagnostics: tuple[str, ...] = (
            ("constraint:joint-decoder-resolved-independent-head-conflict",)
            if (domain, lane, family, intent, action, entity_resolution) != raw_tuple
            else ()
        )
        if route == "out_of_domain":
            if domain != "out_of_domain":
                diagnostics += ("constraint:route-ood-overrode-joint-decision",)
            diagnostics += ("constraint:ood-suppressed-downstream",)
            domain = "out_of_domain"
            intent = lane = family = None
            action = "refuse_ood"
            entity_resolution = "not_required"
        elif route == "uncertain":
            intent = lane = family = action = entity_resolution = None
        elif domain == "out_of_domain":
            route = "uncertain"
            intent = lane = family = action = entity_resolution = None
            diagnostics += ("constraint:domain-route-conflict",)
        return {
            "route": route,
            "domain": domain,
            "domain_probabilities": domain_probabilities,
            "domain_confidence": domain_confidence,
            "domain_candidates": domain_candidates,
            "support_probability": banking_probability,
            # Compatibility alias retained for deployed V2/V3 consumers.
            "banking_probability": banking_probability,
            "ood_probability": ood_probability,
            "confidence": (
                banking_probability
                if route == "in_domain"
                else ood_probability
                if route == "out_of_domain"
                else max(banking_probability, rescue_probability)
            ),
            "intent": intent,
            "intent_confidence": intent_confidence,
            "selected_intent_probability": _candidate_probability(
                intent_candidates,
                "intent",
                intent,
            ),
            "intent_candidates": intent_candidates,
            "capability": intent,
            "capability_confidence": intent_confidence,
            "capability_candidates": [
                {"capability": item["intent"], "probability": item["probability"]}
                for item in intent_candidates
            ],
            "lane": lane,
            "lane_confidence": lane_confidence,
            "lane_candidates": lane_candidates,
            "family": family,
            "family_confidence": family_confidence,
            "family_candidates": family_candidates,
            "action": action,
            "action_confidence": action_confidence,
            "action_candidates": action_candidates,
            "entity_resolution": entity_resolution,
            "entity_resolution_confidence": entity_confidence,
            "entity_resolution_candidates": entity_candidates,
            "relation_probabilities": relations,
            "relation_thresholds": self.relation_thresholds,
            "active_relations": active_relations,
            "constraint_diagnostics": diagnostics,
            "joint_decision_contract": "hierarchical-router-joint-decision/v1",
            "joint_decision_accepted": route == "in_domain",
            "context_applied": context_applied,
            "ood_banking_threshold": self.ood_banking_threshold,
            "in_domain_threshold": self.in_domain_threshold,
            "relation_rescue_threshold": self.relation_rescue_threshold,
            "router_revision": self.artifact_identity.get("router_revision", ROUTER_REVISION),
            "router_architecture": "hierarchical-cross-encoder-v4",
            "router_artifact": dict(self.artifact_identity),
        }

    def _relations(
        self, relation_logits: torch.Tensor
    ) -> tuple[dict[str, float], list[str], float]:
        values = torch.sigmoid(relation_logits.float())[0].tolist()
        relations = dict(zip(self.relation_labels, values, strict=True))
        active = [
            label
            for label in self.relation_labels
            if relations[label] >= self.relation_thresholds[label]
        ]
        rescue = max(
            relations.get("context_dependent", 0.0),
            relations.get("agent_repair", 0.0),
            relations.get("clarification_answer", 0.0),
            relations.get("resume_previous_service", 0.0),
        )
        return relations, active, rescue

    def _route(self, support_probability: float, rescue_probability: float) -> str:
        if support_probability >= self.in_domain_threshold:
            return "in_domain"
        if (
            support_probability < self.ood_banking_threshold
            and rescue_probability < self.relation_rescue_threshold
        ):
            return "out_of_domain"
        return "uncertain"


def render_router_input(
    current: str,
    history: list[dict[str, Any]] | None,
    *,
    max_exchanges: int,
    dialogue_state: Mapping[str, Any] | None = None,
) -> tuple[str, bool]:
    exchanges = _recent_exchanges(history)[-max_exchanges:]
    parts = []
    meaningful_state = _meaningful_dialogue_state(dialogue_state)
    if meaningful_state is not None:
        parts.append(
            "[PRIOR_DIALOGUE_STATE]\n"
            + json.dumps(meaningful_state, sort_keys=True, separators=(",", ":"))
        )
    parts.append(f"[CURRENT_USER]\n{current.strip()}")
    for previous_user, previous_assistant in reversed(exchanges):
        parts.append(f"[PREVIOUS_ASSISTANT]\n{previous_assistant}")
        parts.append(f"[PREVIOUS_USER]\n{previous_user}")
    return "\n".join(parts), bool(exchanges or meaningful_state)


def _meaningful_dialogue_state(
    state: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not state:
        return None
    if state.get("pending_servicing") or state.get("knowledge_detour_active") is True:
        return state
    return None


def verify_artifact(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("contract") != "banking-conversation-router-artifact"
        or manifest.get("release_eligible") is not True
    ):
        raise ValueError("router artifact is not release eligible")
    for entry in manifest["files"]:
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe router artifact path")
        path = root / relative
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"router artifact file mismatch: {relative}")
        if _sha256(path) != str(entry["sha256"]):
            raise ValueError(f"router artifact digest mismatch: {relative}")
    config = json.loads((root / "router_config.json").read_text(encoding="utf-8"))
    if config.get("contract") != "banking-conversation-router" or int(
        config.get("format_version", 0)
    ) not in {2, 3, 4}:
        raise ValueError("unexpected router configuration")
    if int(config["format_version"]) == 4:
        _verify_v4_config(config)
        _verify_v4_heads(root, config)
    if int(config["format_version"]) == 4:
        guidance_contract = config.get("generation_guidance_contract")
        if guidance_contract not in {
            "intent-selects-tool-schema-no-arguments-v1",
            "intent-selects-tool-schema-with-grounded-public-selector-v2",
        }:
            raise ValueError("V4 router generation-guidance contract is missing")
        if (
            guidance_contract == "intent-selects-tool-schema-with-grounded-public-selector-v2"
            and config.get("effective_decision_contract")
            != "retail-bank-effective-turn-decision/v1"
        ):
            raise ValueError("V4 router effective-decision contract is missing")
    else:
        prompt_flag = config.get(
            "intent_enters_generation_prompt",
            config.get("capability_enters_generation_prompt"),
        )
        if prompt_flag is not False:
            raise ValueError("router intent must remain outside the generation prompt")
    return config


def _verify_v4_config(config: Mapping[str, Any]) -> None:
    try:
        from hello_slm.banking_conversation_router_data import RELATION_LABELS
        from hello_slm.banking_domain_taxonomy import (
            ACTION_LABELS,
            DOMAIN_LABELS,
            ENTITY_RESOLUTION_LABELS,
            FAMILY_LABELS,
            INTENT_LABELS,
            LANE_LABELS,
        )
    except ModuleNotFoundError:
        # The Space artifact is standalone; this fallback mirrors the canonical module.
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

    expected = {
        "domain_labels": DOMAIN_LABELS,
        "lane_labels": LANE_LABELS,
        "family_labels": FAMILY_LABELS,
        "intent_labels": INTENT_LABELS,
        "relation_labels": RELATION_LABELS,
        "action_labels": ACTION_LABELS,
        "entity_resolution_labels": ENTITY_RESOLUTION_LABELS,
    }
    for key, labels in expected.items():
        if tuple(config.get(key, ())) != tuple(labels):
            raise ValueError(f"V4 router configuration has non-canonical {key}")


def _verify_v4_heads(root: Path, config: Mapping[str, Any]) -> None:
    path = root / "classifier_heads.safetensors"
    if not path.is_file():
        raise ValueError("V4 router artifact is missing classifier_heads.safetensors")
    try:
        tensors = load_file(path, device="cpu")
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
    widths: set[int] = set()
    for head_name, label_key in label_keys.items():
        weight = tensors.get(f"{head_name}.weight")
        bias = tensors.get(f"{head_name}.bias")
        count = len(config[label_key])
        if (
            weight is None
            or bias is None
            or weight.ndim != 2
            or bias.ndim != 1
            or weight.shape[0] != count
            or bias.shape[0] != count
        ):
            raise ValueError(f"V4 router artifact has invalid {head_name} tensors")
        widths.add(int(weight.shape[1]))
    if len(widths) != 1:
        raise ValueError("V4 router classifier heads do not share an encoder width")


def _load_head_modules(
    tensors: Mapping[str, torch.Tensor],
    modules: list[tuple[str, nn.Linear | None]],
) -> None:
    for name, module in modules:
        if module is None:
            raise ValueError(f"router model is missing {name}")
        try:
            module.load_state_dict(
                {
                    "weight": tensors[f"{name}.weight"],
                    "bias": tensors[f"{name}.bias"],
                },
                strict=True,
            )
        except (KeyError, RuntimeError) as error:
            raise ValueError(f"router artifact has invalid {name} tensors") from error


def _all_candidates(
    logits: torch.Tensor, labels: tuple[str, ...], *, key: str
) -> tuple[dict[str, float | str], ...]:
    probabilities = torch.softmax(logits.float(), dim=-1)[0]
    if probabilities.numel() != len(labels):
        raise ValueError(f"{key} logits do not match configured labels")
    return tuple(
        {key: label, "probability": float(probability)}
        for label, probability in zip(labels, probabilities, strict=True)
    )


def _candidate_probability(
    candidates: tuple[dict[str, float | str], ...],
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


def _needs_topic_shift_recheck(result: Mapping[str, Any]) -> bool:
    return (
        result.get("route") == "in_domain"
        and result.get("context_applied") is True
        and set(result.get("active_relations", ())) == {"topic_shift"}
    )


def _prefer_current_only_topic_shift(
    contextual: Mapping[str, Any],
    current_only: Mapping[str, Any],
) -> bool:
    return (
        current_only.get("route") == "in_domain"
        and current_only.get("joint_decision_accepted") is True
        and isinstance(current_only.get("selected_intent_probability"), int | float)
        and float(current_only["selected_intent_probability"]) >= 0.75
        and current_only.get("intent") != contextual.get("intent")
    )


def _stabilize_active_relations(
    active_relations: Sequence[str],
    *,
    current_text: str,
    context_applied: bool,
) -> list[str]:
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
    return [label for label in RELATION_LABELS if label in active]


def _top_candidates(
    logits: torch.Tensor, labels: tuple[str, ...], *, key: str
) -> tuple[tuple[dict[str, float | str], ...], float]:
    candidates = _all_candidates(logits, labels, key=key)
    ranked = tuple(
        sorted(candidates, key=lambda item: float(item["probability"]), reverse=True)[:3]
    )
    return ranked, float(ranked[0]["probability"])


def decode_v4_joint(
    *,
    domain_scores: tuple[float, ...] | list[float],
    lane_scores: tuple[float, ...] | list[float],
    family_scores: tuple[float, ...] | list[float],
    intent_scores: tuple[float, ...] | list[float],
    action_scores: tuple[float, ...] | list[float],
    entity_resolution_scores: tuple[float, ...] | list[float],
    domain_labels: tuple[str, ...],
    lane_labels: tuple[str, ...],
    family_labels: tuple[str, ...],
    intent_labels: tuple[str, ...],
    action_labels: tuple[str, ...],
    entity_resolution_labels: tuple[str, ...],
) -> V4JointDecision:
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
    candidates = [
        V4JointDecision(
            domain="out_of_domain",
            lane="out_of_domain",
            family="external",
            intent=None,
            action="refuse_ood",
            entity_resolution="not_required",
            score=(
                score_maps["domain"]["out_of_domain"]
                + score_maps["lane"]["out_of_domain"]
                + score_maps["family"]["external"]
                + max(score_maps["intent"].values())
                + score_maps["action"]["refuse_ood"]
                + score_maps["entity_resolution"]["not_required"]
            ),
        )
    ]
    for intent in intent_labels:
        domain, lane, family = _intent_hierarchy(intent)
        for action, entity_resolution in _legal_action_entities(intent, lane):
            candidates.append(
                V4JointDecision(
                    domain=domain,
                    lane=lane,
                    family=family,
                    intent=intent,
                    action=action,
                    entity_resolution=entity_resolution,
                    score=(
                        score_maps["domain"][domain]
                        + score_maps["lane"][lane]
                        + score_maps["family"][family]
                        + score_maps["intent"][intent]
                        + score_maps["action"][action]
                        + score_maps["entity_resolution"][entity_resolution]
                    ),
                )
            )
    safety_priority = {"refuse_ood": 3, "clarify": 3, "converse": 2, "retrieve_policy": 2}
    return max(candidates, key=lambda item: (item.score, safety_priority.get(item.action, 1)))


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
    return (
        ("execute_tool", "resolved" if entity_required else "not_required"),
        ("clarify", "missing"),
        ("clarify", "ambiguous"),
        ("converse", "not_required"),
        ("converse", "ineligible"),
    )


def _apply_v4_constraints(
    *,
    route: str,
    domain: str,
    lane: str,
    family: str,
    intent: str,
    action: str,
    entity_resolution: str,
    action_labels: tuple[str, ...],
) -> tuple[str, str, tuple[str, ...]]:
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
    elif action == "execute_tool" and entity_resolution == "ineligible":
        route = "uncertain"
        diagnostics.append("constraint:ineligible-entity-blocked-execution")
    if action == "clarify" and entity_resolution not in {"missing", "ambiguous"}:
        route = "uncertain"
        diagnostics.append("constraint:clarify-requires-unresolved-entity")
    allowed = _allowed_actions(lane)
    if allowed is not None and action not in allowed:
        route = "uncertain"
        diagnostics.append("constraint:lane-action-incompatible")
    return route, action, tuple(diagnostics)


def _intent_hierarchy(intent: str) -> tuple[str, str, str]:
    try:
        from hello_slm.banking_domain_taxonomy import hierarchy_for_intent

        return hierarchy_for_intent(intent)
    except ModuleNotFoundError:
        mapping = {
            "view_accounts": ("banking", "servicing", "accounts"),
            "view_cards": ("banking", "servicing", "cards"),
            "freeze_card": ("banking", "servicing", "cards"),
            "replace_card": ("banking", "servicing", "cards"),
            "view_transactions": ("banking", "servicing", "transactions"),
            "dispute_transaction": ("banking", "servicing", "transactions"),
            "view_transfers": ("banking", "servicing", "transfers"),
            "cancel_transfer": ("banking", "servicing", "transfers"),
            "view_service_cases": ("banking", "servicing", "service_cases"),
            "policy_knowledge": ("banking", "policy", "policy"),
            "conversation": ("social", "conversation", "social"),
            "other_banking": ("banking", "other_banking", "other_banking"),
        }
        return mapping[intent]


def _allowed_actions(lane: str) -> frozenset[str] | None:
    return {
        "servicing": frozenset({"execute_tool", "clarify", "converse"}),
        "policy": frozenset({"retrieve_policy", "clarify"}),
        "conversation": frozenset({"converse"}),
        "other_banking": frozenset({"converse", "clarify"}),
    }.get(lane)


def _lane_for_intent(intent: str) -> str:
    if intent in {
        "view_accounts",
        "view_cards",
        "freeze_card",
        "replace_card",
        "view_transactions",
        "dispute_transaction",
        "view_transfers",
        "cancel_transfer",
        "view_service_cases",
        "accounts",
        "cards",
        "card_actions",
        "transactions",
        "transfers",
        "service_cases",
    }:
        return "servicing"
    if intent in {"policy_knowledge", "faq"}:
        return "policy"
    if intent == "conversation":
        return "conversation"
    return "other_banking"


def _recent_exchanges(
    history: list[dict[str, Any]] | None,
) -> list[tuple[str, str]]:
    messages = [
        (str(item.get("role")), str(item.get("content")).strip())
        for item in (history or [])
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and str(item["content"]).strip()
    ]
    exchanges: list[tuple[str, str]] = []
    active_user: str | None = None
    for role, content in messages:
        if role == "user":
            active_user = content
        elif active_user is not None:
            exchanges.append((active_user, content))
            active_user = None
    return exchanges


def _is_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
