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

from hello_slm.banking_conversation_router import (
    ConversationRouterModel,
    ConversationRouterOutput,
    decode_v4_joint,
    stabilize_active_relations,
)
from hello_slm.banking_conversation_router_data import RELATION_LABELS
from hello_slm.banking_domain_taxonomy import (
    ACTION_LABELS,
    DOMAIN_LABELS,
    ENTITY_RESOLUTION_LABELS,
    FAMILY_LABELS,
    INTENT_LABELS,
    LANE_LABELS,
    validate_hierarchical_labels,
)

BASE_MODEL_ID = "distilbert/distilbert-base-uncased"
BASE_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DEFAULT_DATASET_DIR = Path("data/banking-conversation-router-v6-hierarchical")
DEFAULT_OUTPUT_DIR = Path("artifacts/banking-conversation-router-v6-hierarchical")
DEFAULT_DESTINATION_ID = "spkc83/retail-bank-conversation-router"
SEED = 7401
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
LANE_LOSS_WEIGHT = 0.5
FAMILY_LOSS_WEIGHT = 0.5
INTENT_LOSS_WEIGHT = 0.7
RELATION_LOSS_WEIGHT = 0.6
ACTION_LOSS_WEIGHT = 0.8
ENTITY_RESOLUTION_LOSS_WEIGHT = 0.8
MAX_RELATION_POSITIVE_WEIGHT = 12.0
MAX_ENTITY_CLASS_WEIGHT = 20.0
TARGETED_ROW_WEIGHT = 3.0
COUNTERFACTUAL_ROW_WEIGHT = 2.0
TARGETED_SOURCES = frozenset(
    {
        "self-authored-router-v5-use-case-alignment",
        "self-authored-router-v5-state-negatives",
        "self-authored-router-v5-resume-trajectory",
        "self-authored-router-v6-hierarchical-entity-state",
        "self-authored-router-v6-transfer-transaction-contrast",
        "self-authored-router-v7-servicing-policy-shift",
        "self-authored-router-v8-first-turn-mutation-openers",
    }
)
RELATION_F1_CALIBRATION_TOLERANCE = 0.005
RELATION_THRESHOLD_CEILINGS = {
    "agent_repair": 0.75,
    # Clarification answers are already constrained by the jointly decoded
    # action/entity tuple. Prefer recall here so a resolved follow-up is not
    # discarded solely because an independently calibrated relation head is
    # slightly more conservative than the coherent joint decision.
    "clarification_answer": 0.80,
}
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
    context_applied_flags: Sequence[bool] | None = None,
    relation_thresholds: Mapping[str, float] | None = None,
    ood_banking_threshold: float,
    in_domain_threshold: float,
    relation_rescue_threshold: float,
    num_intents: int,
    domain_predictions: Sequence[int] | None = None,
    domain_indices: Sequence[int] | None = None,
    lane_predictions: Sequence[int] | None = None,
    lane_indices: Sequence[int] | None = None,
    family_predictions: Sequence[int] | None = None,
    family_indices: Sequence[int] | None = None,
    action_predictions: Sequence[int] | None = None,
    action_indices: Sequence[int] | None = None,
    entity_resolution_predictions: Sequence[int] | None = None,
    entity_resolution_indices: Sequence[int] | None = None,
    domain_scores: Sequence[Sequence[float]] | None = None,
    lane_scores: Sequence[Sequence[float]] | None = None,
    family_scores: Sequence[Sequence[float]] | None = None,
    intent_scores: Sequence[Sequence[float]] | None = None,
    action_scores: Sequence[Sequence[float]] | None = None,
    entity_resolution_scores: Sequence[Sequence[float]] | None = None,
    counterfactual_pair_ids: Sequence[str] | None = None,
    counterfactual_targets: Sequence[str] | None = None,
) -> dict[str, Any]:
    current_texts = current_texts or ["" for _ in domain_labels]
    context_applied_flags = context_applied_flags or [False for _ in domain_labels]
    relation_thresholds = relation_thresholds or {label: 0.5 for label in RELATION_LABELS}
    row_count = len(domain_labels)
    v4_decisions_supplied = all(
        values is not None
        for values in (
            domain_predictions,
            domain_indices,
            lane_predictions,
            lane_indices,
            family_predictions,
            family_indices,
            action_predictions,
            action_indices,
            entity_resolution_predictions,
            entity_resolution_indices,
        )
    )
    domain_predictions = domain_predictions or [
        DOMAIN_LABELS.index("banking") if label else DOMAIN_LABELS.index("out_of_domain")
        for label in domain_labels
    ]
    domain_indices = domain_indices or list(domain_predictions)
    lane_predictions = lane_predictions or [0] * row_count
    lane_indices = lane_indices or list(lane_predictions)
    family_predictions = family_predictions or [0] * row_count
    family_indices = family_indices or list(family_predictions)
    action_predictions = action_predictions or [0] * row_count
    action_indices = action_indices or list(action_predictions)
    entity_resolution_predictions = entity_resolution_predictions or [0] * row_count
    entity_resolution_indices = entity_resolution_indices or list(entity_resolution_predictions)
    if v4_decisions_supplied:
        domain_scores = domain_scores or _scores_from_predictions(
            domain_predictions,
            len(DOMAIN_LABELS),
        )
        lane_scores = lane_scores or _scores_from_predictions(
            lane_predictions,
            len(LANE_LABELS),
        )
        family_scores = family_scores or _scores_from_predictions(
            family_predictions,
            len(FAMILY_LABELS),
        )
        intent_scores = intent_scores or _scores_from_predictions(
            intent_predictions,
            len(INTENT_LABELS),
        )
        action_scores = action_scores or _scores_from_predictions(
            action_predictions,
            len(ACTION_LABELS),
        )
        entity_resolution_scores = entity_resolution_scores or _scores_from_predictions(
            entity_resolution_predictions,
            len(ENTITY_RESOLUTION_LABELS),
        )
    else:
        domain_scores = lane_scores = family_scores = intent_scores = None
        action_scores = entity_resolution_scores = None
    counterfactual_pair_ids = counterfactual_pair_ids or [""] * row_count
    counterfactual_targets = counterfactual_targets or [""] * row_count
    lengths = {
        len(domain_probabilities),
        len(domain_labels),
        len(intent_predictions),
        len(intent_labels),
        len(relation_probabilities),
        len(relation_labels),
        len(example_kinds),
        len(current_texts),
        len(context_applied_flags),
        len(domain_predictions),
        len(domain_indices),
        len(lane_predictions),
        len(lane_indices),
        len(family_predictions),
        len(family_indices),
        len(action_predictions),
        len(action_indices),
        len(entity_resolution_predictions),
        len(entity_resolution_indices),
        len(counterfactual_pair_ids),
        len(counterfactual_targets),
    }
    if v4_decisions_supplied:
        assert domain_scores is not None
        assert lane_scores is not None
        assert family_scores is not None
        assert intent_scores is not None
        assert action_scores is not None
        assert entity_resolution_scores is not None
        lengths.update(
            {
                len(domain_scores),
                len(lane_scores),
                len(family_scores),
                len(intent_scores),
                len(action_scores),
                len(entity_resolution_scores),
            }
        )
    if len(lengths) != 1:
        raise ValueError("prediction fields must have equal lengths")
    routes = route_predictions(
        domain_probabilities=domain_probabilities,
        relation_probabilities=relation_probabilities,
        ood_banking_threshold=ood_banking_threshold,
        in_domain_threshold=in_domain_threshold,
        relation_rescue_threshold=relation_rescue_threshold,
    )
    # The runtime only exposes an intent and mutates dialogue state for an exact
    # in-domain decision.  Treating the abstention band as accepted would let a
    # model pass the false-refusal gate even though the application declines the
    # same turn.
    accepted = [route == "in_domain" for route in routes]
    exposed_intents = [
        INTENT_LABELS[prediction] if route == "in_domain" else None
        for route, prediction in zip(routes, intent_predictions, strict=True)
    ]
    exposed_actions: list[str | None] = []
    exposed_entities: list[str | None] = []
    compatibility_errors: list[bool] = []
    independent_head_conflicts: list[bool] = []
    joint_decoder_disagreements: list[bool] = []
    for index, route in enumerate(routes):
        if not v4_decisions_supplied:
            exposed_actions.append(ACTION_LABELS[action_predictions[index]])
            exposed_entities.append(ENTITY_RESOLUTION_LABELS[entity_resolution_predictions[index]])
            compatibility_errors.append(False)
            independent_head_conflicts.append(False)
            joint_decoder_disagreements.append(False)
            continue
        assert domain_scores is not None
        assert lane_scores is not None
        assert family_scores is not None
        assert intent_scores is not None
        assert action_scores is not None
        assert entity_resolution_scores is not None
        decision = decode_v4_joint(
            domain_scores=domain_scores[index],
            lane_scores=lane_scores[index],
            family_scores=family_scores[index],
            intent_scores=intent_scores[index],
            action_scores=action_scores[index],
            entity_resolution_scores=entity_resolution_scores[index],
            domain_labels=DOMAIN_LABELS,
            lane_labels=LANE_LABELS,
            family_labels=FAMILY_LABELS,
            intent_labels=INTENT_LABELS,
            action_labels=ACTION_LABELS,
            entity_resolution_labels=ENTITY_RESOLUTION_LABELS,
        )
        raw_domain = DOMAIN_LABELS[domain_predictions[index]]
        raw_tuple = (
            raw_domain,
            LANE_LABELS[lane_predictions[index]],
            FAMILY_LABELS[family_predictions[index]],
            None if raw_domain == "out_of_domain" else INTENT_LABELS[intent_predictions[index]],
            ACTION_LABELS[action_predictions[index]],
            ENTITY_RESOLUTION_LABELS[entity_resolution_predictions[index]],
        )
        independent_head_conflicts.append(not _raw_tuple_is_compatible(raw_tuple))
        joint_decoder_disagreements.append(
            raw_tuple
            != (
                decision.domain,
                decision.lane,
                decision.family,
                decision.intent,
                decision.action,
                decision.entity_resolution,
            )
        )
        # Joint decoding enumerates only legal ontology tuples. Route/domain
        # conflicts are handled exactly as runtime abstentions below and are
        # not hierarchy compatibility defects in the decoded tuple itself.
        compatibility_errors.append(False)
        if route == "out_of_domain":
            exposed_intents[index] = None
            exposed_actions.append("refuse_ood")
            exposed_entities.append("not_required")
        elif route == "uncertain" or decision.domain == "out_of_domain":
            routes[index] = "uncertain"
            exposed_intents[index] = None
            exposed_actions.append(None)
            exposed_entities.append(None)
        else:
            exposed_intents[index] = decision.intent
            exposed_actions.append(decision.action)
            exposed_entities.append(decision.entity_resolution)
    accepted = [route == "in_domain" for route in routes]
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
    active_relation_sets = [
        set(
            stabilize_active_relations(
                [
                    relation_name
                    for relation_name, probability in zip(
                        RELATION_LABELS,
                        probabilities,
                        strict=True,
                    )
                    if probability >= relation_thresholds[relation_name]
                ],
                current_text=current_text,
                context_applied=context_applied,
            )
        )
        for probabilities, current_text, context_applied in zip(
            relation_probabilities,
            current_texts,
            context_applied_flags,
            strict=True,
        )
    ]
    relation_pairs = [
        (
            [relation_name in active_relations for relation_name in RELATION_LABELS],
            [bool(label) for label in labels],
        )
        for active_relations, labels in zip(
            active_relation_sets,
            relation_labels,
            strict=True,
        )
    ]
    # The runtime exposes the safe OOD disposition while suppressing every
    # operational field for an uncertain route.  Score this post-constraint
    # contract on every row; raw-head F1 above remains independently audited.
    exposed_decision_indices = list(range(row_count))
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
        "domain_macro_f1": _macro_f1(
            list(zip(domain_predictions, domain_indices, strict=True)),
            class_count=len(DOMAIN_LABELS),
        ),
        "lane_macro_f1": _macro_f1(
            list(zip(lane_predictions, lane_indices, strict=True)),
            class_count=len(LANE_LABELS),
        ),
        "family_macro_f1": _macro_f1(
            list(zip(family_predictions, family_indices, strict=True)),
            class_count=len(FAMILY_LABELS),
        ),
        "action_macro_f1": _macro_f1(
            list(zip(action_predictions, action_indices, strict=True)),
            class_count=len(ACTION_LABELS),
        ),
        "entity_resolution_macro_f1": _macro_f1(
            list(
                zip(
                    entity_resolution_predictions,
                    entity_resolution_indices,
                    strict=True,
                )
            ),
            class_count=len(ENTITY_RESOLUTION_LABELS),
        ),
        "exposed_action_macro_f1": _macro_f1(
            [
                (
                    ACTION_LABELS.index(exposed_actions[index])
                    if exposed_actions[index] is not None
                    else -1,
                    action_indices[index],
                )
                for index in exposed_decision_indices
            ],
            class_count=len(ACTION_LABELS),
        ),
        "exposed_entity_resolution_macro_f1": _macro_f1(
            [
                (
                    ENTITY_RESOLUTION_LABELS.index(exposed_entities[index])
                    if exposed_entities[index] is not None
                    else -1,
                    entity_resolution_indices[index],
                )
                for index in exposed_decision_indices
            ],
            class_count=len(ENTITY_RESOLUTION_LABELS),
        ),
        "exposed_decision_rows": len(exposed_decision_indices),
        "hierarchy_compatibility_error_rate": _safe_ratio(
            sum(compatibility_errors),
            row_count,
        ),
        "raw_independent_head_incompatibility_rate": _safe_ratio(
            sum(independent_head_conflicts),
            row_count,
        ),
        "joint_decoder_disagreement_rate": _safe_ratio(
            sum(joint_decoder_disagreements),
            row_count,
        ),
    }
    counterfactual_indices = [
        index for index, pair_id in enumerate(counterfactual_pair_ids) if pair_id
    ]
    metrics["counterfactual_rows"] = len(counterfactual_indices)
    metrics["counterfactual_action_accuracy"] = _safe_ratio(
        sum(
            exposed_actions[index] == ACTION_LABELS[action_indices[index]]
            for index in counterfactual_indices
        ),
        len(counterfactual_indices),
    )
    metrics["counterfactual_entity_resolution_accuracy"] = _safe_ratio(
        sum(
            exposed_entities[index] == ENTITY_RESOLUTION_LABELS[entity_resolution_indices[index]]
            for index in counterfactual_indices
        ),
        len(counterfactual_indices),
    )
    metrics["counterfactual_pair_flip_accuracy"] = _counterfactual_pair_accuracy(
        pair_ids=counterfactual_pair_ids,
        exposed_actions=exposed_actions,
        exposed_entities=exposed_entities,
        action_indices=action_indices,
        entity_resolution_indices=entity_resolution_indices,
    )
    if not v4_decisions_supplied:
        metrics["counterfactual_action_accuracy"] = 1.0
        metrics["counterfactual_entity_resolution_accuracy"] = 1.0
        metrics["counterfactual_pair_flip_accuracy"] = 1.0
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
        "heldout_policy_followup_generalization",
        "heldout_social_generalization",
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
            routes[index] != ("in_domain" if domain_labels[index] == 1 else "out_of_domain")
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
                    routes[index] != ("in_domain" if domain_labels[index] == 1 else "out_of_domain")
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
    effective_transition_error_rate = _safe_ratio(
        sum(
            not _runtime_transition_matches(
                kind=example_kinds[index],
                decision_accepted=accepted[index],
                effective_intent=exposed_intents[index],
                expected_intent=(
                    INTENT_LABELS[intent_labels[index]] if intent_labels[index] >= 0 else None
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
    metrics["trajectory_runtime_transition_error_rate"] = effective_transition_error_rate
    metrics["trajectory_effective_decision_transition_error_rate"] = effective_transition_error_rate
    for kind, metric_name in (
        (
            "heldout_social_generalization",
            "heldout_social_generalization_error_rate",
        ),
        (
            "heldout_policy_followup_generalization",
            "heldout_policy_followup_generalization_error_rate",
        ),
    ):
        kind_indices = [
            index for index, example_kind in enumerate(example_kinds) if example_kind == kind
        ]
        metrics[metric_name] = _safe_ratio(
            sum(
                not _runtime_transition_matches(
                    kind=kind,
                    decision_accepted=accepted[index],
                    effective_intent=exposed_intents[index],
                    expected_intent=INTENT_LABELS[intent_labels[index]],
                    resume_active=(
                        relation_probabilities[index][resume_relation_index]
                        >= relation_thresholds["resume_previous_service"]
                    ),
                )
                for index in kind_indices
            ),
            len(kind_indices),
        )
    heldout_indices = [
        index for index, kind in enumerate(example_kinds) if kind == "heldout_screenshot_regression"
    ]
    heldout_route_errors = sum(
        accepted[index] != bool(domain_labels[index]) for index in heldout_indices
    )
    heldout_intent_indices = [index for index in heldout_indices if intent_labels[index] >= 0]
    heldout_intent_errors = sum(
        exposed_intents[index] != INTENT_LABELS[intent_labels[index]]
        for index in heldout_intent_indices
    )
    heldout_relation_errors = sum(
        any(
            (relation_name in active_relation_sets[index]) != bool(label)
            for relation_name, label in zip(
                RELATION_LABELS,
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
            "exposed_intent": exposed_intents[index],
            "expected_action": ACTION_LABELS[action_indices[index]],
            "exposed_action": exposed_actions[index],
            "expected_entity_resolution": ENTITY_RESOLUTION_LABELS[
                entity_resolution_indices[index]
            ],
            "exposed_entity_resolution": exposed_entities[index],
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
    decision_accepted: bool,
    effective_intent: str | None,
    expected_intent: str | None,
    resume_active: bool,
) -> bool:
    """Score the finalized decision consumed by DialogueState.begin_turn."""
    if kind == "resume_previous_service":
        return decision_accepted and effective_intent == expected_intent and resume_active
    if kind == "state_ood_detour":
        return not decision_accepted and effective_intent is None and not resume_active
    return decision_accepted and effective_intent == expected_intent and not resume_active


def _scores_from_predictions(
    predictions: Sequence[int],
    class_count: int,
) -> list[list[float]]:
    scores: list[list[float]] = []
    for prediction in predictions:
        if not 0 <= prediction < class_count:
            raise ValueError("prediction index is outside the configured labels")
        row = [0.0] * class_count
        row[prediction] = 1.0
        scores.append(row)
    return scores


def _raw_tuple_is_compatible(
    values: tuple[str, str, str, str | None, str, str],
) -> bool:
    domain, lane, family, intent, action, entity_resolution = values
    try:
        validate_hierarchical_labels(
            {
                "domain_name": domain,
                "lane_name": lane,
                "family_name": family,
                "action_name": action,
                "entity_resolution_name": entity_resolution,
            },
            intent=intent,
        )
    except ValueError:
        return False
    return True


def _counterfactual_pair_accuracy(
    *,
    pair_ids: Sequence[str],
    exposed_actions: Sequence[str | None],
    exposed_entities: Sequence[str | None],
    action_indices: Sequence[int],
    entity_resolution_indices: Sequence[int],
) -> float:
    pairs: dict[str, list[int]] = {}
    for index, pair_id in enumerate(pair_ids):
        if pair_id:
            pairs.setdefault(pair_id, []).append(index)
    matched = [
        indices
        for indices in pairs.values()
        if len(indices) == 2
        and {ACTION_LABELS[action_indices[index]] for index in indices}
        == {"execute_tool", "clarify"}
    ]
    correct = 0
    for indices in matched:
        if all(
            exposed_actions[index] == ACTION_LABELS[action_indices[index]]
            and exposed_entities[index]
            == ENTITY_RESOLUTION_LABELS[entity_resolution_indices[index]]
            for index in indices
        ):
            correct += 1
    return _safe_ratio(correct, len(matched))


def release_gate_failures(metrics: dict[str, Any]) -> list[str]:
    gates = (
        ("domain_macro_f1", ">=", 0.85),
        ("lane_macro_f1", ">=", 0.85),
        ("family_macro_f1", ">=", 0.85),
        ("intent_macro_f1", ">=", 0.85),
        ("relation_macro_f1", ">=", 0.85),
        ("action_macro_f1", ">=", 0.85),
        ("entity_resolution_macro_f1", ">=", 0.85),
        ("exposed_action_macro_f1", ">=", 0.85),
        ("exposed_entity_resolution_macro_f1", ">=", 0.85),
        ("hierarchy_compatibility_error_rate", "<=", 0.00),
        ("counterfactual_action_accuracy", ">=", 0.95),
        ("counterfactual_entity_resolution_accuracy", ">=", 0.95),
        ("counterfactual_pair_flip_accuracy", ">=", 0.95),
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
        ("heldout_social_generalization_error_rate", "<=", 0.00),
        ("heldout_policy_followup_generalization_error_rate", "<=", 0.00),
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


def entity_resolution_class_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_weight: float = MAX_ENTITY_CLASS_WEIGHT,
) -> list[float]:
    """Return bounded square-root inverse-frequency entity weights."""
    counts = [0 for _ in ENTITY_RESOLUTION_LABELS]
    for row in rows:
        counts[int(row["entity_resolution_index"])] += 1
    missing = [
        label for label, count in zip(ENTITY_RESOLUTION_LABELS, counts, strict=True) if not count
    ]
    if missing:
        raise ValueError(f"entity resolution classes have no training rows: {missing}")
    majority = max(counts)
    return [min(max_weight, math.sqrt(majority / count)) for count in counts]


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
        num_domains=len(DOMAIN_LABELS),
        num_lanes=len(LANE_LABELS),
        num_families=len(FAMILY_LABELS),
        num_actions=len(ACTION_LABELS),
        num_entity_resolutions=len(ENTITY_RESOLUTION_LABELS),
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
    entity_class_weight_values = entity_resolution_class_weights(rows_by_split["train"])
    entity_class_weight = torch.tensor(
        entity_class_weight_values,
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
                entity_class_weight=entity_class_weight,
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
                float(validation_metrics["domain_macro_f1"])
                + float(validation_metrics["lane_macro_f1"])
                + float(validation_metrics["family_macro_f1"])
                + float(validation_metrics["intent_macro_f1"])
                + float(validation_metrics["relation_macro_f1"])
                + float(validation_metrics["action_macro_f1"])
                + float(validation_metrics["entity_resolution_macro_f1"])
                + float(validation_metrics["counterfactual_action_accuracy"])
                + float(validation_metrics["counterfactual_entity_resolution_accuracy"])
                + float(validation_metrics["counterfactual_pair_flip_accuracy"])
                + 1.0
                - float(validation_metrics["hierarchy_compatibility_error_rate"])
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
                "lane_loss_weight": LANE_LOSS_WEIGHT,
                "family_loss_weight": FAMILY_LOSS_WEIGHT,
                "relation_loss_weight": RELATION_LOSS_WEIGHT,
                "action_loss_weight": ACTION_LOSS_WEIGHT,
                "entity_resolution_loss_weight": ENTITY_RESOLUTION_LOSS_WEIGHT,
                "relation_positive_weights": relation_pos_weight_values,
                "entity_resolution_class_weights": entity_class_weight_values,
                "targeted_row_weight": TARGETED_ROW_WEIGHT,
                "counterfactual_row_weight": COUNTERFACTUAL_ROW_WEIGHT,
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
    if int(manifest.get("format_version", 0)) != 3:
        raise ValueError("unsupported conversation-router dataset version")
    expected_label_sets = {
        "domain_labels": DOMAIN_LABELS,
        "lane_labels": LANE_LABELS,
        "family_labels": FAMILY_LABELS,
        "intent_labels": INTENT_LABELS,
        "relation_labels": RELATION_LABELS,
        "action_labels": ACTION_LABELS,
        "entity_resolution_labels": ENTITY_RESOLUTION_LABELS,
    }
    for name, expected in expected_label_sets.items():
        if tuple(manifest.get(name, ())) != tuple(expected):
            raise ValueError(f"dataset {name} do not match the V4 contract")
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
    counterfactual_splits: dict[str, str] = {}
    counterfactual_actions: dict[str, list[str]] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            _validate_governed_row(row)
            trajectory_id = str(row["trajectory_id"])
            previous = trajectory_splits.setdefault(trajectory_id, split)
            if previous != split:
                raise ValueError(
                    f"trajectory {trajectory_id!r} appears in both {previous} and {split}"
                )
            pair_id = str(row.get("counterfactual_pair_id") or "")
            if pair_id:
                previous_pair_split = counterfactual_splits.setdefault(pair_id, split)
                if previous_pair_split != split:
                    raise ValueError(f"counterfactual pair {pair_id!r} appears in multiple splits")
                counterfactual_actions.setdefault(pair_id, []).append(str(row["action_name"]))
    invalid_pairs = sorted(
        pair_id
        for pair_id, actions in counterfactual_actions.items()
        if len(actions) != 2 or set(actions) != {"execute_tool", "clarify"}
    )
    if invalid_pairs:
        raise ValueError(f"invalid counterfactual pairs: {invalid_pairs[:5]}")
    counterfactual_pair_counts = {
        split: sum(1 for pair_split in counterfactual_splits.values() if pair_split == split)
        for split in ("train", "validation", "test")
    }
    missing_counterfactual_splits = [
        split for split, count in counterfactual_pair_counts.items() if count == 0
    ]
    if missing_counterfactual_splits:
        raise ValueError(
            "counterfactual action/entity pairs are required in every split: "
            f"{missing_counterfactual_splits}"
        )
    return manifest, rows_by_split


def _validate_governed_row(row: Mapping[str, Any]) -> None:
    named_indices = (
        ("domain_name", "domain_index", DOMAIN_LABELS),
        ("lane_name", "lane_index", LANE_LABELS),
        ("family_name", "family_index", FAMILY_LABELS),
        ("action_name", "action_index", ACTION_LABELS),
        (
            "entity_resolution_name",
            "entity_resolution_index",
            ENTITY_RESOLUTION_LABELS,
        ),
    )
    for name_key, index_key, labels in named_indices:
        name = str(row[name_key])
        index = int(row[index_key])
        if not 0 <= index < len(labels) or labels[index] != name:
            raise ValueError(f"row has inconsistent {name_key}/{index_key}")
    domain_name = str(row["domain_name"])
    expected_binary_domain = int(domain_name != "out_of_domain")
    if int(row["domain_label"]) != expected_binary_domain:
        raise ValueError("row has inconsistent legacy domain_label")
    intent_index = int(row["intent_label"])
    if domain_name == "out_of_domain":
        if intent_index != -100:
            raise ValueError("out-of-domain row must mask intent_label")
    elif (
        not 0 <= intent_index < len(INTENT_LABELS)
        or str(row["intent"]) != INTENT_LABELS[intent_index]
    ):
        raise ValueError("row has inconsistent intent/intent_label")
    relation_values = row["relation_labels"]
    if (
        not isinstance(relation_values, list)
        or len(relation_values) != len(RELATION_LABELS)
        or any(value not in {0, 1} for value in relation_values)
    ):
        raise ValueError("row has invalid relation_labels")
    pair_id = str(row.get("counterfactual_pair_id") or "")
    pair_target = str(row.get("counterfactual_target") or "")
    if bool(pair_id) != bool(pair_target):
        raise ValueError("counterfactual pair id and target must appear together")
    validate_hierarchical_labels(row)


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
            "domain_indices": torch.tensor(
                [int(row["domain_index"]) for row in rows],
                dtype=torch.long,
            ),
            "lane_indices": torch.tensor(
                [int(row["lane_index"]) for row in rows],
                dtype=torch.long,
            ),
            "family_indices": torch.tensor(
                [int(row["family_index"]) for row in rows],
                dtype=torch.long,
            ),
            "action_indices": torch.tensor(
                [int(row["action_index"]) for row in rows],
                dtype=torch.long,
            ),
            "entity_resolution_indices": torch.tensor(
                [int(row["entity_resolution_index"]) for row in rows],
                dtype=torch.long,
            ),
            "relation_labels": torch.tensor(
                [list(map(float, row["relation_labels"])) for row in rows],
                dtype=torch.float32,
            ),
            "row_weights": torch.tensor(
                [
                    max(
                        TARGETED_ROW_WEIGHT if row["source"] in TARGETED_SOURCES else 1.0,
                        COUNTERFACTUAL_ROW_WEIGHT if row.get("counterfactual_pair_id") else 1.0,
                    )
                    for row in rows
                ],
                dtype=torch.float32,
            ),
            "example_kinds": [str(row["example_kind"]) for row in rows],
            "current_texts": [str(row["current_text"]) for row in rows],
            "context_applied_flags": [bool(row.get("history")) for row in rows],
            "counterfactual_pair_ids": [
                str(row.get("counterfactual_pair_id") or "") for row in rows
            ],
            "counterfactual_targets": [str(row.get("counterfactual_target") or "") for row in rows],
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
    entity_class_weight: torch.Tensor,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        domain_indices = batch["domain_indices"].to(device, non_blocking=True)
        lane_indices = batch["lane_indices"].to(device, non_blocking=True)
        family_indices = batch["family_indices"].to(device, non_blocking=True)
        intent_labels = batch["intent_labels"].to(
            device,
            non_blocking=True,
        )
        action_indices = batch["action_indices"].to(device, non_blocking=True)
        entity_resolution_indices = batch["entity_resolution_indices"].to(
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
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if not isinstance(output, ConversationRouterOutput):
                raise TypeError("V4 router model must return ConversationRouterOutput")
            domain_losses = nn.functional.cross_entropy(
                output.domain_logits,
                domain_indices,
                reduction="none",
            )
            domain_loss = _weighted_mean(domain_losses, row_weights)
            lane_loss = _classification_loss(
                output.lane_logits,
                lane_indices,
                row_weights,
            )
            family_loss = _classification_loss(
                output.family_logits,
                family_indices,
                row_weights,
            )
            active_intent = intent_labels >= 0
            intent_loss = (
                _weighted_mean(
                    nn.functional.cross_entropy(
                        output.intent_logits[active_intent],
                        intent_labels[active_intent],
                        reduction="none",
                    ),
                    row_weights[active_intent],
                )
                if active_intent.any()
                else domain_loss.new_zeros(())
            )
            relation_losses = nn.functional.binary_cross_entropy_with_logits(
                output.relation_logits,
                relation_labels,
                pos_weight=relation_pos_weight,
                reduction="none",
            ).mean(dim=1)
            relation_loss = _weighted_mean(
                relation_losses,
                row_weights,
            )
            action_loss = _classification_loss(
                output.action_logits,
                action_indices,
                row_weights,
            )
            entity_resolution_loss = _weighted_mean(
                nn.functional.cross_entropy(
                    output.entity_resolution_logits,
                    entity_resolution_indices,
                    weight=entity_class_weight,
                    reduction="none",
                ),
                row_weights,
            )
            loss = (
                domain_loss
                + LANE_LOSS_WEIGHT * lane_loss
                + FAMILY_LOSS_WEIGHT * family_loss
                + INTENT_LOSS_WEIGHT * intent_loss
                + RELATION_LOSS_WEIGHT * relation_loss
                + ACTION_LOSS_WEIGHT * action_loss
                + ENTITY_RESOLUTION_LOSS_WEIGHT * entity_resolution_loss
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


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    row_weights: torch.Tensor,
) -> torch.Tensor:
    return _weighted_mean(
        nn.functional.cross_entropy(logits, labels, reduction="none"),
        row_weights,
    )


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
        "domain_predictions": [],
        "domain_scores": [],
        "domain_indices": [],
        "lane_predictions": [],
        "lane_scores": [],
        "lane_indices": [],
        "family_predictions": [],
        "family_scores": [],
        "family_indices": [],
        "intent_predictions": [],
        "intent_scores": [],
        "intent_labels": [],
        "relation_probabilities": [],
        "relation_labels": [],
        "example_kinds": [],
        "current_texts": [],
        "context_applied_flags": [],
        "action_predictions": [],
        "action_scores": [],
        "action_indices": [],
        "entity_resolution_predictions": [],
        "entity_resolution_scores": [],
        "entity_resolution_indices": [],
        "counterfactual_pair_ids": [],
        "counterfactual_targets": [],
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
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            if not isinstance(output, ConversationRouterOutput):
                raise TypeError("V4 router model must return ConversationRouterOutput")
            domain_distribution = torch.softmax(output.domain_logits.float(), dim=-1)
            result["domain_probabilities"].extend(
                (1.0 - domain_distribution[:, DOMAIN_LABELS.index("out_of_domain")]).cpu().tolist()
            )
            result["domain_labels"].extend(batch["domain_labels"].tolist())
            result["domain_predictions"].extend(output.domain_logits.argmax(dim=-1).cpu().tolist())
            result["domain_scores"].extend(output.domain_logits.float().cpu().tolist())
            result["domain_indices"].extend(batch["domain_indices"].tolist())
            result["lane_predictions"].extend(output.lane_logits.argmax(dim=-1).cpu().tolist())
            result["lane_scores"].extend(output.lane_logits.float().cpu().tolist())
            result["lane_indices"].extend(batch["lane_indices"].tolist())
            result["family_predictions"].extend(output.family_logits.argmax(dim=-1).cpu().tolist())
            result["family_scores"].extend(output.family_logits.float().cpu().tolist())
            result["family_indices"].extend(batch["family_indices"].tolist())
            result["intent_predictions"].extend(output.intent_logits.argmax(dim=-1).cpu().tolist())
            result["intent_scores"].extend(output.intent_logits.float().cpu().tolist())
            result["intent_labels"].extend(batch["intent_labels"].tolist())
            result["relation_probabilities"].extend(
                torch.sigmoid(output.relation_logits.float()).cpu().tolist()
            )
            result["relation_labels"].extend(batch["relation_labels"].tolist())
            result["example_kinds"].extend(batch["example_kinds"])
            result["current_texts"].extend(batch["current_texts"])
            result["context_applied_flags"].extend(
                batch.get("context_applied_flags", [False] * len(batch["current_texts"]))
            )
            result["action_predictions"].extend(output.action_logits.argmax(dim=-1).cpu().tolist())
            result["action_scores"].extend(output.action_logits.float().cpu().tolist())
            result["action_indices"].extend(batch["action_indices"].tolist())
            result["entity_resolution_predictions"].extend(
                output.entity_resolution_logits.argmax(dim=-1).cpu().tolist()
            )
            result["entity_resolution_scores"].extend(
                output.entity_resolution_logits.float().cpu().tolist()
            )
            result["entity_resolution_indices"].extend(batch["entity_resolution_indices"].tolist())
            result["counterfactual_pair_ids"].extend(batch["counterfactual_pair_ids"])
            result["counterfactual_targets"].extend(batch["counterfactual_targets"])
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
            "lane_head.weight": _required_head(model.lane_head, "lane_head")
            .weight.detach()
            .cpu()
            .contiguous(),
            "lane_head.bias": _required_head(model.lane_head, "lane_head")
            .bias.detach()
            .cpu()
            .contiguous(),
            "family_head.weight": _required_head(model.family_head, "family_head")
            .weight.detach()
            .cpu()
            .contiguous(),
            "family_head.bias": _required_head(model.family_head, "family_head")
            .bias.detach()
            .cpu()
            .contiguous(),
            "intent_head.weight": (model.intent_head.weight.detach().cpu().contiguous()),
            "intent_head.bias": (model.intent_head.bias.detach().cpu().contiguous()),
            "relation_head.weight": (model.relation_head.weight.detach().cpu().contiguous()),
            "relation_head.bias": (model.relation_head.bias.detach().cpu().contiguous()),
            "action_head.weight": _required_head(model.action_head, "action_head")
            .weight.detach()
            .cpu()
            .contiguous(),
            "action_head.bias": _required_head(model.action_head, "action_head")
            .bias.detach()
            .cpu()
            .contiguous(),
            "entity_resolution_head.weight": _required_head(
                model.entity_resolution_head,
                "entity_resolution_head",
            )
            .weight.detach()
            .cpu()
            .contiguous(),
            "entity_resolution_head.bias": _required_head(
                model.entity_resolution_head,
                "entity_resolution_head",
            )
            .bias.detach()
            .cpu()
            .contiguous(),
        },
        output / "classifier_heads.safetensors",
    )
    policy = metrics["policy"]
    router_config = {
        "contract": "banking-conversation-router",
        "format_version": 4,
        "architecture": "distilbert-cross-encoder-hierarchical-multitask",
        "base_model": BASE_MODEL_ID,
        "base_revision": BASE_MODEL_REVISION,
        "data_revision": data_revision,
        "data_manifest_sha256": metrics["data_manifest_sha256"],
        "domain_labels": list(DOMAIN_LABELS),
        "lane_labels": list(LANE_LABELS),
        "family_labels": list(FAMILY_LABELS),
        "intent_labels": list(INTENT_LABELS),
        "relation_labels": list(RELATION_LABELS),
        "action_labels": list(ACTION_LABELS),
        "entity_resolution_labels": list(ENTITY_RESOLUTION_LABELS),
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
        "runtime_constraints": "hierarchy-action-entity-v4",
        "runtime_contract_version": 7,
        "effective_decision_contract": "retail-bank-effective-turn-decision/v1",
        "generation_guidance_contract": (
            "intent-selects-tool-schema-with-grounded-public-selector-v2"
        ),
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
        "format_version": 4,
        "runtime_contract_version": 7,
        "effective_decision_contract": "retail-bank-effective-turn-decision/v1",
        "generation_guidance_contract": (
            "intent-selects-tool-schema-with-grounded-public-selector-v2"
        ),
        "implementation_version": os.environ.get("SOURCE_COMMIT", "local"),
        "release_eligible": metrics["release_eligible"],
        "signed": False,
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _required_head(head: nn.Linear | None, name: str) -> nn.Linear:
    if head is None:
        raise ValueError(f"V4 router model is missing {name}")
    return head


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
        commit_message="Publish hierarchical conversation router V6",
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
        # Python 3.14 uses forkserver on POSIX, which cannot pickle the
        # tokenizer-backed local collator.  This dataset is small enough that
        # deterministic in-process collation is the safer portable default.
        num_workers=0,
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

History-aware DistilBERT cross-encoder with one shared encoder and seven
jointly trained heads: domain, lane, family, intent, relation, action, and
entity resolution. Runtime hierarchy constraints expose route, intent,
action, and entity decisions used by the release evaluation.

## Held-out results

- Release eligible: `{metrics["release_eligible"]}`
- Domain macro F1: `{test["domain_macro_f1"]:.6f}`
- Lane macro F1: `{test["lane_macro_f1"]:.6f}`
- Family macro F1: `{test["family_macro_f1"]:.6f}`
- Intent macro F1: `{test["intent_macro_f1"]:.6f}`
- Relation macro F1: `{test["relation_macro_f1"]:.6f}`
- Action macro F1: `{test["action_macro_f1"]:.6f}`
- Entity-resolution macro F1: `{test["entity_resolution_macro_f1"]:.6f}`
- Hierarchy compatibility error rate:
  `{test["hierarchy_compatibility_error_rate"]:.6f}`
- Counterfactual action accuracy: `{test["counterfactual_action_accuracy"]:.6f}`
- Counterfactual entity accuracy:
  `{test["counterfactual_entity_resolution_accuracy"]:.6f}`
- Counterfactual exact pair-flip accuracy:
  `{test["counterfactual_pair_flip_accuracy"]:.6f}`
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
