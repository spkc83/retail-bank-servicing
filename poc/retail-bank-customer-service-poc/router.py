from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    "4c5bc613409e49a38bb29463adbd0755e9382ec9",
)


class ConversationRouterModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int,
        intent_count: int | None = None,
        capability_count: int | None = None,
        relation_count: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.domain_head = nn.Linear(hidden_size, 2)
        label_count = intent_count if intent_count is not None else capability_count
        if label_count is None:
            raise ValueError("intent_count is required")
        self.intent_head = nn.Linear(hidden_size, label_count)
        self.relation_head = nn.Linear(hidden_size, relation_count)

    @property
    def capability_head(self) -> nn.Linear:
        """Compatibility alias for V2 artifacts and tests."""
        return self.intent_head

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return (
            self.domain_head(pooled),
            self.intent_head(pooled),
            self.relation_head(pooled),
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
        self.ood_banking_threshold = ood_banking_threshold
        self.in_domain_threshold = in_domain_threshold
        self.relation_rescue_threshold = relation_rescue_threshold
        self.relation_thresholds = {
            label: float((relation_thresholds or {}).get(label, 0.5))
            for label in self.relation_labels
        }
        self.max_length = max_length
        self.max_exchanges = max_exchanges

    @classmethod
    def from_hub(cls) -> LearnedBankingRouter:
        if not _is_commit(ROUTER_REVISION):
            raise RuntimeError(
                "RETAIL_BANK_ROUTER_REVISION must pin the published V5 router commit"
            )
        root = Path(snapshot_download(ROUTER_REPO_ID, revision=ROUTER_REVISION))
        return cls.from_artifact_dir(root)

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
    ) -> LearnedBankingRouter:
        root = Path(artifact_dir)
        config = verify_artifact(root)
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
        label_key = "intent_labels" if format_version == 3 else "capability_labels"
        intents = tuple(str(label) for label in config[label_key])
        relations = tuple(str(label) for label in config["relation_labels"])
        model = ConversationRouterModel(
            encoder,
            hidden_size=int(encoder.config.hidden_size),
            intent_count=len(intents),
            relation_count=len(relations),
        )
        heads = load_file(root / "classifier_heads.safetensors", device="cpu")
        for name, head in (
            ("domain_head", model.domain_head),
            (
                "intent_head" if format_version == 3 else "capability_head",
                model.intent_head,
            ),
            ("relation_head", model.relation_head),
        ):
            head.load_state_dict(
                {
                    "weight": heads[f"{name}.weight"],
                    "bias": heads[f"{name}.bias"],
                },
                strict=True,
            )
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
        )

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
        return self._predict(rendered, context_applied=context_applied)

    def _predict(
        self,
        rendered: str,
        *,
        context_applied: bool,
    ) -> dict[str, Any]:
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.inference_mode():
            domain_logits, intent_logits, relation_logits = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        banking_probability = float(torch.softmax(domain_logits.float(), dim=-1)[0, 1])
        ood_probability = 1.0 - banking_probability
        relation_values = torch.sigmoid(relation_logits.float())[0].tolist()
        relations = dict(zip(self.relation_labels, relation_values, strict=True))
        active_relations = [
            label
            for label in self.relation_labels
            if relations[label] >= self.relation_thresholds[label]
        ]
        rescue_probability = max(
            relations.get("context_dependent", 0.0),
            relations.get("agent_repair", 0.0),
            relations.get("clarification_answer", 0.0),
            relations.get("resume_previous_service", 0.0),
        )
        if banking_probability >= self.in_domain_threshold:
            route = "in_domain"
        elif (
            banking_probability < self.ood_banking_threshold
            and rescue_probability < self.relation_rescue_threshold
        ):
            route = "out_of_domain"
        else:
            route = "uncertain"

        intent_probabilities = torch.softmax(
            intent_logits.float(),
            dim=-1,
        )[0]
        candidate_count = min(3, len(self.intent_labels))
        candidate_probabilities, candidate_indices = torch.topk(
            intent_probabilities,
            k=candidate_count,
        )
        intent_candidates = [
            {
                "intent": self.intent_labels[int(index)],
                "probability": float(probability),
            }
            for probability, index in zip(
                candidate_probabilities,
                candidate_indices,
                strict=True,
            )
        ]
        intent = intent_candidates[0]["intent"] if route == "in_domain" else None
        intent_confidence = float(candidate_probabilities[0])
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
            "router_revision": ROUTER_REVISION,
            "router_architecture": "history-and-state-aware-cross-encoder-v5",
        }


def render_router_input(
    current: str,
    history: list[dict[str, Any]] | None,
    *,
    max_exchanges: int,
    dialogue_state: Mapping[str, Any] | None = None,
) -> tuple[str, bool]:
    exchanges = _recent_exchanges(history)[-max_exchanges:]
    parts = []
    if dialogue_state:
        parts.append(
            "[PRIOR_DIALOGUE_STATE]\n"
            + json.dumps(dialogue_state, sort_keys=True, separators=(",", ":"))
        )
    parts.append(f"[CURRENT_USER]\n{current.strip()}")
    for previous_user, previous_assistant in reversed(exchanges):
        parts.append(f"[PREVIOUS_ASSISTANT]\n{previous_assistant}")
        parts.append(f"[PREVIOUS_USER]\n{previous_user}")
    return "\n".join(parts), bool(exchanges or dialogue_state)


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
    ) not in {2, 3}:
        raise ValueError("unexpected router configuration")
    prompt_flag = config.get(
        "intent_enters_generation_prompt",
        config.get("capability_enters_generation_prompt"),
    )
    if prompt_flag is not False:
        raise ValueError("router intent must remain outside the generation prompt")
    return config


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
