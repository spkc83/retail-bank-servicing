# /// script
# dependencies = [
#   "huggingface-hub>=1.4,<2",
#   "numpy>=2.2,<3",
#   "safetensors>=0.6,<1",
#   "torch==2.12.1",
#   "transformers>=5.13,<5.14",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# ///
"""Train and evaluate the history-aware retail-bank conversation router."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from hello_slm.banking_conversation_router import ConversationRouterModel
from hello_slm.banking_conversation_router_data import (
    INTENT_LABELS,
    RELATION_LABELS,
)

BASE_MODEL_ID = "distilbert/distilbert-base-uncased"
BASE_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DEFAULT_DATASET_DIR = Path("data/banking-conversation-router-v5")
DEFAULT_OUTPUT_DIR = Path("artifacts/banking-conversation-router-v5")
DEFAULT_DESTINATION_ID = "spkc83/retail-bank-conversation-router"
SEED = 7401
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
INTENT_LOSS_WEIGHT = 0.7
RELATION_LOSS_WEIGHT = 0.6
MAX_RELATION_POSITIVE_WEIGHT = 12.0
TARGETED_ROW_WEIGHT = 6.0
TARGETED_SOURCES = frozenset(
    {
        "self-authored-router-v5-use-case-alignment",
        "self-authored-router-v5-state-negatives",
        "self-authored-router-v5-resume-trajectory",
    }
)
RELATION_F1_CALIBRATION_TOLERANCE = 0.005
RELATION_THRESHOLD_CEILINGS = {"agent_repair": 0.75}
IN_DOMAIN_THRESHOLD = 0.50


class RouterDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--destination-id", default=DEFAULT_DESTINATION_ID)
    parser.add_argument(
        "--data-revision",
        help="Immutable published dataset revision; required with --publish.",
    )
    return parser.parse_args()


def route_predictions(
    *,
    domain_probabilities: Sequence[float],
    relation_probabilities: Sequence[Sequence[float]],
    ood_banking_threshold: float,
    in_domain_threshold: float,
    relation_rescue_threshold: float,
) -> list[str]:
    if len(domain_probabilities) != len(relation_probabilities):
        raise ValueError("domain and relation probabilities must have equal lengths")
    routes: list[str] = []
    for banking_probability, relations in zip(
        domain_probabilities,
        relation_probabilities,
        strict=True,
    ):
        if len(relations) != len(RELATION_LABELS):
            raise ValueError("relation probability width does not match labels")
        relation_by_name = dict(zip(RELATION_LABELS, relations, strict=True))
        rescue_probability = max(
            relation_by_name["context_dependent"],
            relation_by_name["agent_repair"],
            relation_by_name["clarification_answer"],
            relation_by_name["resume_previous_service"],
        )
        if banking_probability >= in_domain_threshold:
            routes.append("in_domain")
        elif (
            banking_probability < ood_banking_threshold
            and rescue_probability < relation_rescue_threshold
        ):
            routes.append("out_of_domain")
        else:
            routes.append("uncertain")
    return routes


def calibrate_policy(
    *,
    domain_probabilities: Sequence[float],
    domain_labels: Sequence[int],
    relation_probabilities: Sequence[Sequence[float]],
    relation_labels: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if not domain_labels or len(domain_probabilities) != len(domain_labels):
        raise ValueError("domain probabilities and labels must be non-empty")
    candidates: list[dict[str, float]] = []
    for ood_boundary_int in range(10, 50, 5):
        ood_boundary = ood_boundary_int / 100
        for rescue_boundary in (0.40, 0.50, 0.60):
            routes = route_predictions(
                domain_probabilities=domain_probabilities,
                relation_probabilities=relation_probabilities,
                ood_banking_threshold=ood_boundary,
                in_domain_threshold=IN_DOMAIN_THRESHOLD,
                relation_rescue_threshold=rescue_boundary,
            )
            accepted = [route != "out_of_domain" for route in routes]
            in_domain_recall = _expected_acceptance_rate(
                accepted,
                domain_labels,
                expected_label=1,
            )
            ood_specificity = _expected_rejection_rate(
                accepted,
                domain_labels,
                expected_label=0,
            )
            contextual_recall = _relation_acceptance_rate(
                accepted,
                domain_labels,
                relation_labels,
                "context_dependent",
            )
            repair_recall = _relation_acceptance_rate(
                accepted,
                domain_labels,
                relation_labels,
                "agent_repair",
            )
            candidates.append(
                {
                    "ood_banking_threshold": ood_boundary,
                    "in_domain_threshold": IN_DOMAIN_THRESHOLD,
                    "relation_rescue_threshold": rescue_boundary,
                    "in_domain_recall": in_domain_recall,
                    "ood_specificity": ood_specificity,
                    "contextual_recall": contextual_recall,
                    "repair_recall": repair_recall,
                }
            )
    eligible = [
        candidate
        for candidate in candidates
        if candidate["in_domain_recall"] >= 0.98
        and candidate["contextual_recall"] >= 0.98
        and candidate["repair_recall"] >= 0.99
    ]
    pool = eligible or candidates
    selected = max(
        pool,
        key=lambda candidate: (
            candidate["ood_specificity"],
            candidate["in_domain_recall"],
            candidate["contextual_recall"],
            candidate["repair_recall"],
            candidate["ood_banking_threshold"],
        ),
    )
    return {
        **selected,
        "relation_thresholds": calibrate_relation_thresholds(
            relation_probabilities,
            relation_labels,
        ),
    }


def calibrate_relation_thresholds(
    relation_probabilities: Sequence[Sequence[float]],
    relation_labels: Sequence[Sequence[int]],
) -> dict[str, float]:
    if not relation_probabilities or (len(relation_probabilities) != len(relation_labels)):
        raise ValueError("relation calibration fields must be non-empty and equal")
    thresholds: dict[str, float] = {}
    for relation_index, relation_name in enumerate(RELATION_LABELS):
        candidates = []
        for threshold_int in range(5, 100, 5):
            threshold = threshold_int / 100
            true_positive = false_positive = false_negative = 0
            for probabilities, labels in zip(
                relation_probabilities,
                relation_labels,
                strict=True,
            ):
                predicted = probabilities[relation_index] >= threshold
                expected = bool(labels[relation_index])
                true_positive += int(predicted and expected)
                false_positive += int(predicted and not expected)
                false_negative += int(not predicted and expected)
            denominator = 2 * true_positive + false_positive + false_negative
            f1 = 2 * true_positive / denominator if denominator else 0.0
            candidates.append((f1, -threshold, threshold))
        best_f1 = max(candidate[0] for candidate in candidates)
        if relation_name == "context_dependent":
            eligible = [
                candidate
                for candidate in candidates
                if candidate[0] >= best_f1 - RELATION_F1_CALIBRATION_TOLERANCE
            ]
            thresholds[relation_name] = min(candidate[2] for candidate in eligible)
        else:
            optimal = [
                candidate
                for candidate in candidates
                if math.isclose(
                    candidate[0],
                    best_f1,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
            thresholds[relation_name] = min(candidate[2] for candidate in optimal)
        if relation_name in RELATION_THRESHOLD_CEILINGS:
            thresholds[relation_name] = min(
                thresholds[relation_name],
                RELATION_THRESHOLD_CEILINGS[relation_name],
            )
    return thresholds


def evaluate_predictions(
    *,
    domain_probabilities: Sequence[float],
    domain_labels: Sequence[int],
    intent_predictions: Sequence[int],
    intent_labels: Sequence[int],
    relation_probabilities: Sequence[Sequence[float]],
    relation_labels: Sequence[Sequence[int]],
    example_kinds: Sequence[str],
    current_texts: Sequence[str] | None = None,
    relation_thresholds: Mapping[str, float] | None = None,
    ood_banking_threshold: float,
    in_domain_threshold: float,
    relation_rescue_threshold: float,
    num_intents: int,
) -> dict[str, Any]:
    current_texts = current_texts or ["" for _ in domain_labels]
    relation_thresholds = relation_thresholds or {label: 0.5 for label in RELATION_LABELS}
    lengths = {
        len(domain_probabilities),
        len(domain_labels),
        len(intent_predictions),
        len(intent_labels),
        len(relation_probabilities),
        len(relation_labels),
        len(example_kinds),
        len(current_texts),
    }
    if len(lengths) != 1:
        raise ValueError("prediction fields must have equal lengths")
    routes = route_predictions(
        domain_probabilities=domain_probabilities,
        relation_probabilities=relation_probabilities,
        ood_banking_threshold=ood_banking_threshold,
        in_domain_threshold=in_domain_threshold,
        relation_rescue_threshold=relation_rescue_threshold,
    )
    accepted = [route != "out_of_domain" for route in routes]
    exposed_intents = [
        INTENT_LABELS[prediction] if route == "in_domain" else None
        for route, prediction in zip(routes, intent_predictions, strict=True)
    ]
    domain_counts = _domain_counts(accepted, domain_labels)
    intent_pairs = [
        (prediction, label)
        for prediction, label in zip(
            intent_predictions,
            intent_labels,
            strict=True,
        )
        if label >= 0
    ]
    relation_pairs = [
        (
            [
                probability >= relation_thresholds[relation_name]
                for relation_name, probability in zip(
                    RELATION_LABELS,
                    probabilities,
                    strict=True,
                )
            ],
            [bool(label) for label in labels],
        )
        for probabilities, labels in zip(
            relation_probabilities,
            relation_labels,
            strict=True,
        )
    ]
    metrics: dict[str, Any] = {
        "rows": len(domain_labels),
        "domain_confusion": domain_counts,
        "in_domain_false_refusal_rate": _safe_ratio(
            domain_counts["false_negative"],
            domain_counts["positive"],
        ),
        "ood_false_accept_rate": _safe_ratio(
            domain_counts["false_positive"],
            domain_counts["negative"],
        ),
        "intent_macro_f1": _macro_f1(
            intent_pairs,
            class_count=num_intents,
        ),
        "intent_rows": len(intent_pairs),
        "relation_macro_f1": _multilabel_macro_f1(relation_pairs),
    }
    metrics["contextual_false_refusal_rate"] = _relation_false_refusal_rate(
        accepted,
        domain_labels,
        relation_labels,
        "context_dependent",
    )
    metrics["repair_false_refusal_rate"] = _relation_false_refusal_rate(
        accepted,
        domain_labels,
        relation_labels,
        "agent_repair",
    )
    metrics["topic_shift_ood_false_accept_rate"] = _relation_ood_false_accept_rate(
        accepted,
        domain_labels,
        relation_labels,
        "topic_shift",
    )
    resume_indices = [
        index for index, kind in enumerate(example_kinds) if kind == "resume_previous_service"
    ]
    resume_relation_index = RELATION_LABELS.index("resume_previous_service")
    metrics["trajectory_resume_intent_error_rate"] = _safe_ratio(
        sum(
            exposed_intents[index] != INTENT_LABELS[intent_labels[index]]
            for index in resume_indices
            if intent_labels[index] >= 0
        ),
        sum(intent_labels[index] >= 0 for index in resume_indices),
    )
    metrics["trajectory_resume_relation_error_rate"] = _safe_ratio(
        sum(
            routes[index] != "in_domain"
            or (
                relation_probabilities[index][resume_relation_index]
                >= relation_thresholds["resume_previous_service"]
            )
            != bool(relation_labels[index][resume_relation_index])
            for index in resume_indices
        ),
        len(resume_indices),
    )
    metrics["trajectory_resume_rows"] = len(resume_indices)
    state_negative_kinds = {
        "state_intent_switch",
        "state_ood_detour",
        "state_policy_followup",
        "state_social_detour",
        "state_orphan_resume",
    }
    state_negative_indices = [
        index for index, kind in enumerate(example_kinds) if kind in state_negative_kinds
    ]
    state_negative_intent_indices = [
        index for index in state_negative_indices if intent_labels[index] >= 0
    ]
    metrics["trajectory_state_route_error_rate"] = _safe_ratio(
        sum(
            routes[index]
            != ("in_domain" if domain_labels[index] == 1 else "out_of_domain")
            for index in state_negative_indices
        ),
        len(state_negative_indices),
    )
    metrics["trajectory_state_intent_error_rate"] = _safe_ratio(
        sum(
            exposed_intents[index] != INTENT_LABELS[intent_labels[index]]
            for index in state_negative_intent_indices
        ),
        len(state_negative_intent_indices),
    )
    metrics["trajectory_non_resume_false_positive_rate"] = _safe_ratio(
        sum(
            relation_probabilities[index][resume_relation_index]
            >= relation_thresholds["resume_previous_service"]
            for index in state_negative_indices
        ),
        len(state_negative_indices),
    )
    metrics["trajectory_state_negative_rows"] = len(state_negative_indices)
    metrics["trajectory_state_metrics_by_kind"] = {
        kind: {
            "rows": len(kind_indices),
            "route_error_rate": _safe_ratio(
                sum(
                    routes[index]
                    != ("in_domain" if domain_labels[index] == 1 else "out_of_domain")
                    for index in kind_indices
                ),
                len(kind_indices),
            ),
            "intent_error_rate": _safe_ratio(
                sum(
                    exposed_intents[index] != INTENT_LABELS[intent_labels[index]]
                    for index in kind_indices
                    if intent_labels[index] >= 0
                ),
                sum(intent_labels[index] >= 0 for index in kind_indices),
            ),
            "resume_false_positive_rate": _safe_ratio(
                sum(
                    relation_probabilities[index][resume_relation_index]
                    >= relation_thresholds["resume_previous_service"]
                    for index in kind_indices
                ),
                len(kind_indices),
            ),
        }
        for kind in sorted(state_negative_kinds)
        if (
            kind_indices := [
                index for index in state_negative_indices if example_kinds[index] == kind
            ]
        )
    }
    trajectory_indices = [*resume_indices, *state_negative_indices]
    metrics["trajectory_runtime_transition_error_rate"] = _safe_ratio(
        sum(
            not _runtime_transition_matches(
                kind=example_kinds[index],
                route=routes[index],
                exposed_intent=exposed_intents[index],
                expected_intent=(
                    INTENT_LABELS[intent_labels[index]]
                    if intent_labels[index] >= 0
                    else None
                ),
                resume_active=(
                    relation_probabilities[index][resume_relation_index]
                    >= relation_thresholds["resume_previous_service"]
                ),
            )
            for index in trajectory_indices
        ),
        len(trajectory_indices),
    )
    heldout_indices = [
        index for index, kind in enumerate(example_kinds) if kind == "heldout_screenshot_regression"
    ]
    heldout_route_errors = sum(
        accepted[index] != bool(domain_labels[index]) for index in heldout_indices
    )
    heldout_intent_indices = [index for index in heldout_indices if intent_labels[index] >= 0]
    heldout_intent_errors = sum(
        intent_predictions[index] != intent_labels[index] for index in heldout_intent_indices
    )
    heldout_relation_errors = sum(
        any(
            (probability >= relation_thresholds[relation_name]) != bool(label)
            for relation_name, probability, label in zip(
                RELATION_LABELS,
                relation_probabilities[index],
                relation_labels[index],
                strict=True,
            )
        )
        for index in heldout_indices
    )
    metrics["heldout_regression_route_error_rate"] = _safe_ratio(
        heldout_route_errors,
        len(heldout_indices),
    )
    metrics["heldout_regression_intent_error_rate"] = _safe_ratio(
        heldout_intent_errors,
        len(heldout_intent_indices),
    )
    metrics["heldout_regression_relation_error_rate"] = _safe_ratio(
        heldout_relation_errors,
        len(heldout_indices),
    )
    metrics["heldout_regression_rows"] = len(heldout_indices)
    metrics["heldout_regression_predictions"] = [
        {
            "current_text": current_texts[index],
            "expected_domain": int(domain_labels[index]),
            "route": routes[index],
            "banking_probability": float(domain_probabilities[index]),
            "expected_intent": (
                INTENT_LABELS[intent_labels[index]] if intent_labels[index] >= 0 else None
            ),
            "predicted_intent": INTENT_LABELS[intent_predictions[index]],
            "expected_relations": [
                RELATION_LABELS[relation_index]
                for relation_index, label in enumerate(relation_labels[index])
                if label
            ],
            "relation_probabilities": dict(
                zip(
                    RELATION_LABELS,
                    relation_probabilities[index],
                    strict=True,
                )
            ),
            "relation_thresholds": dict(relation_thresholds),
        }
        for index in heldout_indices
    ]
    metrics["example_kind_counts"] = {
        kind: example_kinds.count(kind) for kind in sorted(set(example_kinds))
    }
    return metrics


def _runtime_transition_matches(
    *,
    kind: str,
    route: str,
    exposed_intent: str | None,
    expected_intent: str | None,
    resume_active: bool,
) -> bool:
    """Mirror the observations that DialogueState.begin_turn will act on."""
    if kind == "resume_previous_service":
        return (
            route == "in_domain"
            and exposed_intent == expected_intent
            and resume_active
        )
    if kind == "state_ood_detour":
        return route == "out_of_domain" and not resume_active
    return (
        route == "in_domain"
        and exposed_intent == expected_intent
        and not resume_active
    )


def release_gate_failures(metrics: dict[str, Any]) -> list[str]:
    gates = (
        ("intent_macro_f1", ">=", 0.85),
        ("relation_macro_f1", ">=", 0.85),
        ("in_domain_false_refusal_rate", "<=", 0.02),
        ("ood_false_accept_rate", "<=", 0.05),
        ("contextual_false_refusal_rate", "<=", 0.02),
        ("repair_false_refusal_rate", "<=", 0.01),
        ("topic_shift_ood_false_accept_rate", "<=", 0.05),
        ("trajectory_resume_intent_error_rate", "<=", 0.00),
        ("trajectory_resume_relation_error_rate", "<=", 0.00),
        ("trajectory_state_route_error_rate", "<=", 0.00),
        ("trajectory_state_intent_error_rate", "<=", 0.00),
        ("trajectory_non_resume_false_positive_rate", "<=", 0.00),
        ("trajectory_runtime_transition_error_rate", "<=", 0.00),
        ("heldout_regression_route_error_rate", "<=", 0.00),
        ("heldout_regression_intent_error_rate", "<=", 0.00),
        ("heldout_regression_relation_error_rate", "<=", 0.00),
    )
    failures: list[str] = []
    for name, operator, threshold in gates:
        value = float(metrics[name])
        equal = math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12)
        passed = value > threshold or equal if operator == ">=" else value < threshold or equal
        if not passed:
            failures.append(f"{name}={value:.6f} must be {operator} {threshold:.6f}")
    return failures


def relation_positive_weights(
    rows: Sequence[dict[str, Any]],
    *,
    max_weight: float = MAX_RELATION_POSITIVE_WEIGHT,
) -> list[float]:
    if max_weight < 1.0:
        raise ValueError("max_weight must be at least one")
    row_count = len(rows)
    if row_count == 0:
        raise ValueError("relation weight calibration requires training rows")
    positives = [0 for _ in RELATION_LABELS]
    for row in rows:
        labels = row["relation_labels"]
        if len(labels) != len(RELATION_LABELS):
            raise ValueError("relation label width does not match labels")
        for index, label in enumerate(labels):
            positives[index] += int(bool(label))
    return [
        max(
            1.0,
            min(
                max_weight,
                (row_count - positive_count) / max(positive_count, 1),
            ),
        )
        for positive_count in positives
    ]


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 32:
        raise ValueError("epochs/batch-size must be positive and max-length >= 32")
    if args.publish and not args.data_revision:
        raise ValueError("--data-revision is required with --publish")
    _set_seed(SEED)
    manifest, rows_by_split = load_governed_data(args.dataset_dir)

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        token=token,
    )
    encoder = AutoModel.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        token=token,
    )
    model = ConversationRouterModel(
        encoder,
        hidden_size=int(encoder.config.hidden_size),
        num_intents=len(INTENT_LABELS),
        num_relations=len(RELATION_LABELS),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(
        json.dumps(
            {
                "stage": "initialized",
                "device": str(device),
                "train_rows": len(rows_by_split["train"]),
                "validation_rows": len(rows_by_split["validation"]),
                "test_rows": len(rows_by_split["test"]),
            }
        ),
        flush=True,
    )

    collate = make_collate(tokenizer, max_length=args.max_length)
    train_loader = _loader(
        rows_by_split["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate=collate,
        device=device,
    )
    validation_loader = _loader(
        rows_by_split["validation"],
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate=collate,
        device=device,
    )
    test_loader = _loader(
        rows_by_split["test"],
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate=collate,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    relation_pos_weight_values = relation_positive_weights(rows_by_split["train"])
    relation_pos_weight = torch.tensor(
        relation_pos_weight_values,
        dtype=torch.float32,
        device=device,
    )
    history: list[dict[str, Any]] = []
    best_score = -math.inf

    with tempfile.TemporaryDirectory(prefix="retail-bank-conversation-router-") as temp_dir:
        best_path = Path(temp_dir) / "best.safetensors"
        for epoch in range(1, args.epochs + 1):
            training_loss = train_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                relation_pos_weight=relation_pos_weight,
                device=device,
                epoch=epoch,
            )
            validation_predictions = predict(
                model,
                validation_loader,
                device=device,
            )
            calibration = calibrate_policy(
                domain_probabilities=validation_predictions["domain_probabilities"],
                domain_labels=validation_predictions["domain_labels"],
                relation_probabilities=validation_predictions["relation_probabilities"],
                relation_labels=validation_predictions["relation_labels"],
            )
            validation_metrics = evaluate_predictions(
                **validation_predictions,
                ood_banking_threshold=calibration["ood_banking_threshold"],
                in_domain_threshold=calibration["in_domain_threshold"],
                relation_rescue_threshold=calibration["relation_rescue_threshold"],
                relation_thresholds=calibration["relation_thresholds"],
                num_intents=len(INTENT_LABELS),
            )
            score = (
                float(validation_metrics["intent_macro_f1"])
                + float(validation_metrics["relation_macro_f1"])
                + 1.0
                - float(validation_metrics["in_domain_false_refusal_rate"])
                + 1.0
                - float(validation_metrics["ood_false_accept_rate"])
                + 1.0
                - float(validation_metrics["trajectory_state_route_error_rate"])
                + 1.0
                - float(validation_metrics["trajectory_state_intent_error_rate"])
                + 1.0
                - float(validation_metrics["trajectory_non_resume_false_positive_rate"])
            )
            epoch_result = {
                "epoch": epoch,
                "training_loss": training_loss,
                "calibration": calibration,
                "validation": validation_metrics,
                "selection_score": score,
            }
            history.append(epoch_result)
            print(json.dumps({"stage": "epoch_complete", **epoch_result}), flush=True)
            if score > best_score:
                best_score = score
                save_file(
                    {
                        name: tensor.detach().cpu().contiguous()
                        for name, tensor in model.state_dict().items()
                    },
                    best_path,
                )

        model.load_state_dict(load_file(best_path, device=str(device)), strict=True)
        best_epoch = max(history, key=lambda item: float(item["selection_score"]))
        calibration = best_epoch["calibration"]
        test_predictions = predict(model, test_loader, device=device)
        test_metrics = evaluate_predictions(
            **test_predictions,
            ood_banking_threshold=calibration["ood_banking_threshold"],
            in_domain_threshold=calibration["in_domain_threshold"],
            relation_rescue_threshold=calibration["relation_rescue_threshold"],
            relation_thresholds=calibration["relation_thresholds"],
            num_intents=len(INTENT_LABELS),
        )
        failures = release_gate_failures(test_metrics)
        metrics = {
            "base_model": BASE_MODEL_ID,
            "base_revision": BASE_MODEL_REVISION,
            "data_manifest_sha256": file_sha256(args.dataset_dir / "manifest.json"),
            "seed": SEED,
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "intent_loss_weight": INTENT_LOSS_WEIGHT,
                "relation_loss_weight": RELATION_LOSS_WEIGHT,
                "relation_positive_weights": relation_pos_weight_values,
                "targeted_row_weight": TARGETED_ROW_WEIGHT,
            },
            "history": history,
            "selected_epoch": best_epoch["epoch"],
            "policy": calibration,
            "test": test_metrics,
            "release_gate_failures": failures,
            "release_eligible": not failures,
        }
        save_artifact(
            model=model,
            tokenizer=tokenizer,
            output=args.output_dir,
            data_manifest=manifest,
            metrics=metrics,
            max_length=args.max_length,
            data_revision=args.data_revision,
        )
        if failures:
            print(
                json.dumps({"stage": "release_gate_failed", **metrics}, indent=2),
                flush=True,
            )
            return 2
        if args.publish:
            publish_artifact(
                output=args.output_dir,
                destination_id=args.destination_id,
                token=token,
            )
        print(json.dumps({"stage": "completed", **metrics}, indent=2), flush=True)
    return 0


def load_governed_data(
    dataset_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != "banking-conversation-router-data":
        raise ValueError("unexpected conversation-router dataset contract")
    if int(manifest.get("format_version", 0)) != 2:
        raise ValueError("unsupported conversation-router dataset version")
    if tuple(manifest.get("intent_labels", ())) != INTENT_LABELS:
        raise ValueError("dataset intent labels do not match the V5 contract")
    if tuple(manifest.get("relation_labels", ())) != RELATION_LABELS:
        raise ValueError("dataset relation labels do not match the V5 contract")
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest["splits"]:
        split = str(entry["name"])
        path = dataset_dir / str(entry["path"])
        if file_sha256(path) != str(entry["sha256"]):
            raise ValueError(f"{split} dataset digest mismatch")
        rows_by_split[split] = [
            json.loads(line) for line in path.open(encoding="utf-8") if line.strip()
        ]
    if set(rows_by_split) != {"train", "validation", "test"}:
        raise ValueError("dataset must contain train, validation, and test splits")
    trajectory_splits: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            trajectory_id = str(row["trajectory_id"])
            previous = trajectory_splits.setdefault(trajectory_id, split)
            if previous != split:
                raise ValueError(
                    f"trajectory {trajectory_id!r} appears in both {previous} and {split}"
                )
    return manifest, rows_by_split


def make_collate(tokenizer: Any, *, max_length: int) -> Any:
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [str(row["text"]) for row in rows],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            **encoded,
            "domain_labels": torch.tensor(
                [int(row["domain_label"]) for row in rows],
                dtype=torch.long,
            ),
            "intent_labels": torch.tensor(
                [int(row["intent_label"]) for row in rows],
                dtype=torch.long,
            ),
            "relation_labels": torch.tensor(
                [list(map(float, row["relation_labels"])) for row in rows],
                dtype=torch.float32,
            ),
            "row_weights": torch.tensor(
                [TARGETED_ROW_WEIGHT if row["source"] in TARGETED_SOURCES else 1.0 for row in rows],
                dtype=torch.float32,
            ),
            "example_kinds": [str(row["example_kind"]) for row in rows],
            "current_texts": [str(row["current_text"]) for row in rows],
        }

    return collate


def train_epoch(
    model: ConversationRouterModel,
    loader: DataLoader[Any],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    relation_pos_weight: torch.Tensor,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        domain_labels = batch["domain_labels"].to(device, non_blocking=True)
        intent_labels = batch["intent_labels"].to(
            device,
            non_blocking=True,
        )
        relation_labels = batch["relation_labels"].to(device, non_blocking=True)
        row_weights = batch["row_weights"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            domain_logits, intent_logits, relation_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            domain_losses = nn.functional.cross_entropy(
                domain_logits,
                domain_labels,
                reduction="none",
            )
            domain_loss = _weighted_mean(domain_losses, row_weights)
            active_intent = intent_labels >= 0
            intent_loss = (
                _weighted_mean(
                    nn.functional.cross_entropy(
                        intent_logits[active_intent],
                        intent_labels[active_intent],
                        reduction="none",
                    ),
                    row_weights[active_intent],
                )
                if active_intent.any()
                else domain_loss.new_zeros(())
            )
            relation_losses = nn.functional.binary_cross_entropy_with_logits(
                relation_logits,
                relation_labels,
                pos_weight=relation_pos_weight,
                reduction="none",
            ).mean(dim=1)
            relation_loss = _weighted_mean(
                relation_losses,
                row_weights,
            )
            loss = (
                domain_loss
                + INTENT_LOSS_WEIGHT * intent_loss
                + RELATION_LOSS_WEIGHT * relation_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before_step:
            scheduler.step()
        total_loss += float(loss.detach())
        if step % 100 == 0:
            print(
                json.dumps(
                    {
                        "stage": "training",
                        "epoch": epoch,
                        "step": step,
                        "steps_in_epoch": len(loader),
                        "loss": float(loss.detach()),
                    }
                ),
                flush=True,
            )
    return total_loss / max(len(loader), 1)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("weighted mean expects equal one-dimensional tensors")
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def predict(
    model: ConversationRouterModel,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> dict[str, list[Any]]:
    model.eval()
    result: dict[str, list[Any]] = {
        "domain_probabilities": [],
        "domain_labels": [],
        "intent_predictions": [],
        "intent_labels": [],
        "relation_probabilities": [],
        "relation_labels": [],
        "example_kinds": [],
        "current_texts": [],
    }
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                domain_logits, intent_logits, relation_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            result["domain_probabilities"].extend(
                torch.softmax(domain_logits.float(), dim=-1)[:, 1].cpu().tolist()
            )
            result["domain_labels"].extend(batch["domain_labels"].tolist())
            result["intent_predictions"].extend(intent_logits.argmax(dim=-1).cpu().tolist())
            result["intent_labels"].extend(batch["intent_labels"].tolist())
            result["relation_probabilities"].extend(
                torch.sigmoid(relation_logits.float()).cpu().tolist()
            )
            result["relation_labels"].extend(batch["relation_labels"].tolist())
            result["example_kinds"].extend(batch["example_kinds"])
            result["current_texts"].extend(batch["current_texts"])
    return result


def save_artifact(
    *,
    model: ConversationRouterModel,
    tokenizer: Any,
    output: Path,
    data_manifest: dict[str, Any],
    metrics: dict[str, Any],
    max_length: int,
    data_revision: str | None,
) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    encoder = cast(Any, model.encoder)
    encoder.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    save_file(
        {
            "domain_head.weight": model.domain_head.weight.detach().cpu().contiguous(),
            "domain_head.bias": model.domain_head.bias.detach().cpu().contiguous(),
            "intent_head.weight": (model.intent_head.weight.detach().cpu().contiguous()),
            "intent_head.bias": (model.intent_head.bias.detach().cpu().contiguous()),
            "relation_head.weight": (model.relation_head.weight.detach().cpu().contiguous()),
            "relation_head.bias": (model.relation_head.bias.detach().cpu().contiguous()),
        },
        output / "classifier_heads.safetensors",
    )
    policy = metrics["policy"]
    router_config = {
        "contract": "banking-conversation-router",
        "format_version": 3,
        "architecture": "distilbert-cross-encoder-multitask",
        "base_model": BASE_MODEL_ID,
        "base_revision": BASE_MODEL_REVISION,
        "data_revision": data_revision,
        "data_manifest_sha256": metrics["data_manifest_sha256"],
        "domain_labels": ["out_of_domain", "in_domain"],
        "intent_labels": list(INTENT_LABELS),
        "relation_labels": list(RELATION_LABELS),
        "ood_banking_threshold": policy["ood_banking_threshold"],
        "in_domain_threshold": policy["in_domain_threshold"],
        "relation_rescue_threshold": policy["relation_rescue_threshold"],
        "relation_thresholds": policy["relation_thresholds"],
        "max_length": max_length,
        "max_exchanges": int(data_manifest.get("max_exchanges", 3)),
        "input_format": (
            "[PRIOR_DIALOGUE_STATE]\\n{state_json}\\n"
            "[CURRENT_USER]\\n{text}\\n"
            "[PREVIOUS_ASSISTANT]\\n{assistant}\\n"
            "[PREVIOUS_USER]\\n{user}"
        ),
        "intent_is_advisory": True,
        "intent_enters_generation_prompt": False,
        "lane_is_derived_from_intent": True,
    }
    (output / "router_config.json").write_text(
        json.dumps(router_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        _model_card(metrics=metrics, data_manifest=data_manifest),
        encoding="utf-8",
    )
    files = [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "contract": "banking-conversation-router-artifact",
        "format_version": 3,
        "implementation_version": os.environ.get("SOURCE_COMMIT", "local"),
        "release_eligible": metrics["release_eligible"],
        "signed": False,
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def publish_artifact(
    *,
    output: Path,
    destination_id: str,
    token: str | None,
) -> None:
    if not token:
        raise RuntimeError("HF_TOKEN is required with --publish")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=destination_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    commit = api.upload_folder(
        folder_path=output,
        repo_id=destination_id,
        repo_type="model",
        commit_message="Publish history-aware conversation router V5",
    )
    print(json.dumps({"stage": "published", "commit": str(commit)}), flush=True)


def _loader(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    collate: Any,
    device: torch.device,
) -> DataLoader[Any]:
    return DataLoader(
        RouterDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )


def _domain_counts(
    accepted: Sequence[bool],
    labels: Sequence[int],
) -> dict[str, int]:
    true_positive = false_positive = true_negative = false_negative = 0
    for predicted, label in zip(accepted, labels, strict=True):
        if label == 1 and predicted:
            true_positive += 1
        elif label == 1:
            false_negative += 1
        elif predicted:
            false_positive += 1
        else:
            true_negative += 1
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "positive": true_positive + false_negative,
        "negative": true_negative + false_positive,
    }


def _macro_f1(
    pairs: Sequence[tuple[int, int]],
    *,
    class_count: int,
) -> float:
    if not pairs:
        return 0.0
    scores = []
    for label in range(class_count):
        true_positive = sum(prediction == label and truth == label for prediction, truth in pairs)
        false_positive = sum(prediction == label and truth != label for prediction, truth in pairs)
        false_negative = sum(prediction != label and truth == label for prediction, truth in pairs)
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator:
            scores.append(2 * true_positive / denominator)
    return sum(scores) / len(scores) if scores else 0.0


def _multilabel_macro_f1(
    pairs: Sequence[tuple[Sequence[bool], Sequence[bool]]],
) -> float:
    if not pairs:
        return 0.0
    scores = []
    for label_index in range(len(RELATION_LABELS)):
        true_positive = sum(
            prediction[label_index] and truth[label_index] for prediction, truth in pairs
        )
        false_positive = sum(
            prediction[label_index] and not truth[label_index] for prediction, truth in pairs
        )
        false_negative = sum(
            not prediction[label_index] and truth[label_index] for prediction, truth in pairs
        )
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator:
            scores.append(2 * true_positive / denominator)
    return sum(scores) / len(scores) if scores else 0.0


def _relation_false_refusal_rate(
    accepted: Sequence[bool],
    domain_labels: Sequence[int],
    relation_labels: Sequence[Sequence[int]],
    relation_name: str,
) -> float:
    index = RELATION_LABELS.index(relation_name)
    positions = [
        position
        for position, (domain, relations) in enumerate(
            zip(domain_labels, relation_labels, strict=True)
        )
        if domain == 1 and bool(relations[index])
    ]
    return _safe_ratio(
        sum(not accepted[position] for position in positions),
        len(positions),
    )


def _relation_ood_false_accept_rate(
    accepted: Sequence[bool],
    domain_labels: Sequence[int],
    relation_labels: Sequence[Sequence[int]],
    relation_name: str,
) -> float:
    index = RELATION_LABELS.index(relation_name)
    positions = [
        position
        for position, (domain, relations) in enumerate(
            zip(domain_labels, relation_labels, strict=True)
        )
        if domain == 0 and bool(relations[index])
    ]
    return _safe_ratio(
        sum(accepted[position] for position in positions),
        len(positions),
    )


def _expected_acceptance_rate(
    accepted: Sequence[bool],
    labels: Sequence[int],
    *,
    expected_label: int,
) -> float:
    positions = [index for index, label in enumerate(labels) if label == expected_label]
    return _safe_ratio(
        sum(accepted[index] for index in positions),
        len(positions),
    )


def _expected_rejection_rate(
    accepted: Sequence[bool],
    labels: Sequence[int],
    *,
    expected_label: int,
) -> float:
    positions = [index for index, label in enumerate(labels) if label == expected_label]
    return _safe_ratio(
        sum(not accepted[index] for index in positions),
        len(positions),
    )


def _relation_acceptance_rate(
    accepted: Sequence[bool],
    domain_labels: Sequence[int],
    relation_labels: Sequence[Sequence[int]],
    relation_name: str,
) -> float:
    index = RELATION_LABELS.index(relation_name)
    positions = [
        position
        for position, (domain, relations) in enumerate(
            zip(domain_labels, relation_labels, strict=True)
        )
        if domain == 1 and bool(relations[index])
    ]
    return (
        _safe_ratio(
            sum(accepted[position] for position in positions),
            len(positions),
        )
        if positions
        else 1.0
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_card(
    *,
    metrics: dict[str, Any],
    data_manifest: dict[str, Any],
) -> str:
    test = metrics["test"]
    return f"""---
base_model: {BASE_MODEL_ID}
library_name: transformers
license: apache-2.0
pipeline_tag: text-classification
tags:
  - banking
  - conversation-routing
  - out-of-domain-detection
---

# Retail Bank Conversation Router

History-aware DistilBERT cross-encoder with a supported-domain head, a
fine-grained intent head, and independent conversation-relation labels. The
broad lane is derived from the intent. Intents are diagnostic and never
authorize tools or enter the generation prompt.

## Held-out results

- Release eligible: `{metrics["release_eligible"]}`
- Intent macro F1: `{test["intent_macro_f1"]:.6f}`
- Relation macro F1: `{test["relation_macro_f1"]:.6f}`
- In-domain false-refusal rate: `{test["in_domain_false_refusal_rate"]:.6f}`
- OOD false-accept rate: `{test["ood_false_accept_rate"]:.6f}`
- Contextual false-refusal rate: `{test["contextual_false_refusal_rate"]:.6f}`
- Repair false-refusal rate: `{test["repair_false_refusal_rate"]:.6f}`
- Topic-shift OOD false-accept rate:
  `{test["topic_shift_ood_false_accept_rate"]:.6f}`

## Data

The governed corpus contains
{data_manifest["report"]["split_counts"]["train"]} train,
{data_manifest["report"]["split_counts"]["validation"]} validation, and
{data_manifest["report"]["split_counts"]["test"]} test rows. Inputs include
only prior visible user/assistant text, pre-turn dialogue state, and the current
user turn; current-turn tool plans, tool results, expected outputs, and final
answers are excluded.
"""


if __name__ == "__main__":
    raise SystemExit(main())
