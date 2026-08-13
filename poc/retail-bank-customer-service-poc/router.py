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
    "9e090c0fa21cebbaa03a431a7ce61e656c0739fe",
)


class ConversationRouterModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int,
        capability_count: int,
        relation_count: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.domain_head = nn.Linear(hidden_size, 2)
        self.capability_head = nn.Linear(hidden_size, capability_count)
        self.relation_head = nn.Linear(hidden_size, relation_count)

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
            self.capability_head(pooled),
            self.relation_head(pooled),
        )


class LearnedBankingRouter:
    """History-aware CPU gate; capability predictions are diagnostics only."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        capability_labels: tuple[str, ...],
        relation_labels: tuple[str, ...],
        ood_banking_threshold: float,
        in_domain_threshold: float,
        relation_rescue_threshold: float,
        relation_thresholds: Mapping[str, float] | None = None,
        max_length: int,
        max_exchanges: int,
    ) -> None:
        if not capability_labels or not relation_labels:
            raise ValueError("router labels must not be empty")
        if not 0.0 < ood_banking_threshold < in_domain_threshold < 1.0:
            raise ValueError("invalid domain thresholds")
        if not 0.0 < relation_rescue_threshold < 1.0:
            raise ValueError("invalid relation rescue threshold")
        self.tokenizer = tokenizer
        self.model = model.to("cpu").eval()
        self.capability_labels = capability_labels
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
                "RETAIL_BANK_ROUTER_REVISION must pin the published v4 router commit"
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
        capabilities = tuple(str(label) for label in config["capability_labels"])
        relations = tuple(str(label) for label in config["relation_labels"])
        model = ConversationRouterModel(
            encoder,
            hidden_size=int(encoder.config.hidden_size),
            capability_count=len(capabilities),
            relation_count=len(relations),
        )
        heads = load_file(root / "classifier_heads.safetensors", device="cpu")
        for name, head in (
            ("domain_head", model.domain_head),
            ("capability_head", model.capability_head),
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
            capability_labels=capabilities,
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
    ) -> dict[str, Any]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        rendered, context_applied = render_router_input(
            message.strip(),
            history,
            max_exchanges=self.max_exchanges,
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
            domain_logits, capability_logits, relation_logits = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        banking_probability = float(
            torch.softmax(domain_logits.float(), dim=-1)[0, 1]
        )
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

        capability_probabilities = torch.softmax(
            capability_logits.float(),
            dim=-1,
        )[0]
        candidate_count = min(3, len(self.capability_labels))
        candidate_probabilities, candidate_indices = torch.topk(
            capability_probabilities,
            k=candidate_count,
        )
        capability_candidates = [
            {
                "capability": self.capability_labels[int(index)],
                "probability": float(probability),
            }
            for probability, index in zip(
                candidate_probabilities,
                candidate_indices,
                strict=True,
            )
        ]
        capability = (
            capability_candidates[0]["capability"] if route == "in_domain" else None
        )
        capability_confidence = float(candidate_probabilities[0])
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
            "capability": capability,
            "capability_confidence": capability_confidence,
            "capability_candidates": capability_candidates,
            "relation_probabilities": relations,
            "relation_thresholds": self.relation_thresholds,
            "active_relations": active_relations,
            "context_applied": context_applied,
            "ood_banking_threshold": self.ood_banking_threshold,
            "in_domain_threshold": self.in_domain_threshold,
            "relation_rescue_threshold": self.relation_rescue_threshold,
            "router_revision": ROUTER_REVISION,
            "router_architecture": "history-aware-cross-encoder-v4",
        }


def render_router_input(
    current: str,
    history: list[dict[str, Any]] | None,
    *,
    max_exchanges: int,
) -> tuple[str, bool]:
    exchanges = _recent_exchanges(history)[-max_exchanges:]
    parts = [f"[CURRENT_USER]\n{current.strip()}"]
    for previous_user, previous_assistant in reversed(exchanges):
        parts.append(f"[PREVIOUS_ASSISTANT]\n{previous_assistant}")
        parts.append(f"[PREVIOUS_USER]\n{previous_user}")
    return "\n".join(parts), bool(exchanges)


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
    if (
        config.get("contract") != "banking-conversation-router"
        or int(config.get("format_version", 0)) != 2
    ):
        raise ValueError("unexpected router configuration")
    if config.get("capability_enters_generation_prompt") is not False:
        raise ValueError("router capability must remain diagnostic")
    return config


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
