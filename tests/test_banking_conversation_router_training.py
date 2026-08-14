from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import torch


def load_training_module() -> ModuleType:
    path = Path("scripts/retail_bank/train_conversation_router.py")
    spec = importlib.util.spec_from_file_location(
        "banking_conversation_router_training",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_route_policy_rescues_context_but_not_topic_shift() -> None:
    training = load_training_module()
    routes = training.route_predictions(
        domain_probabilities=[0.05, 0.05, 0.05, 0.90],
        relation_probabilities=[
            [0.95, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.95, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.95, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.05, 0.95],
        ],
        ood_banking_threshold=0.20,
        in_domain_threshold=0.50,
        relation_rescue_threshold=0.50,
    )

    assert routes == ["uncertain", "uncertain", "out_of_domain", "in_domain"]


def test_metrics_cover_intent_and_each_relation_slice() -> None:
    training = load_training_module()
    metrics = training.evaluate_predictions(
        domain_probabilities=[0.95, 0.90, 0.05, 0.10, 0.90],
        domain_labels=[1, 1, 0, 0, 1],
        intent_predictions=[0, 1, 0, 1, 0],
        intent_labels=[0, 1, -100, -100, 0],
        relation_probabilities=[
            [0.90, 0.05, 0.05, 0.05, 0.05],
            [0.90, 0.90, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.90, 0.05, 0.05],
            [0.05, 0.05, 0.90, 0.05, 0.05],
            [0.90, 0.05, 0.05, 0.05, 0.05],
        ],
        relation_labels=[
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        example_kinds=[
            "contextual_followup",
            "agent_repair",
            "external_topic_shift",
            "external_topic_shift",
            "heldout_screenshot_regression",
        ],
        current_texts=[
            "followup",
            "repair",
            "weather",
            "timer",
            "When was that created?",
        ],
        ood_banking_threshold=0.20,
        in_domain_threshold=0.50,
        relation_rescue_threshold=0.50,
        num_intents=2,
    )

    assert metrics["intent_macro_f1"] == 1.0
    assert metrics["relation_macro_f1"] == 1.0
    assert metrics["contextual_false_refusal_rate"] == 0.0
    assert metrics["repair_false_refusal_rate"] == 0.0
    assert metrics["topic_shift_ood_false_accept_rate"] == 0.0
    assert metrics["trajectory_resume_intent_error_rate"] == 0.0
    assert metrics["trajectory_resume_relation_error_rate"] == 0.0
    assert metrics["trajectory_state_route_error_rate"] == 0.0
    assert metrics["trajectory_state_intent_error_rate"] == 0.0
    assert metrics["trajectory_non_resume_false_positive_rate"] == 0.0
    assert metrics["heldout_regression_route_error_rate"] == 0.0
    assert metrics["heldout_regression_intent_error_rate"] == 0.0
    assert metrics["heldout_regression_relation_error_rate"] == 0.0
    assert metrics["heldout_regression_rows"] == 1
    assert metrics["heldout_regression_predictions"][0]["current_text"] == (
        "When was that created?"
    )
    assert training.release_gate_failures(metrics) == []


def test_release_gate_reports_use_case_regressions() -> None:
    training = load_training_module()
    failures = training.release_gate_failures(
        {
            "intent_macro_f1": 0.70,
            "relation_macro_f1": 0.70,
            "in_domain_false_refusal_rate": 0.06,
            "ood_false_accept_rate": 0.10,
            "contextual_false_refusal_rate": 0.06,
            "repair_false_refusal_rate": 0.06,
            "topic_shift_ood_false_accept_rate": 0.10,
            "trajectory_resume_intent_error_rate": 0.10,
            "trajectory_resume_relation_error_rate": 0.10,
            "trajectory_state_route_error_rate": 0.10,
            "trajectory_state_intent_error_rate": 0.10,
            "trajectory_non_resume_false_positive_rate": 0.10,
            "heldout_regression_route_error_rate": 0.10,
            "heldout_regression_intent_error_rate": 0.10,
            "heldout_regression_relation_error_rate": 0.10,
        }
    )

    assert len(failures) == 15


def test_state_negative_metrics_reject_prior_state_override() -> None:
    training = load_training_module()
    metrics = training.evaluate_predictions(
        domain_probabilities=[0.95, 0.95, 0.95],
        domain_labels=[1, 0, 1],
        intent_predictions=[0, 0, 1],
        intent_labels=[1, -100, 1],
        relation_probabilities=[
            [0.1, 0.1, 0.1, 0.1, 0.9],
            [0.1, 0.1, 0.1, 0.1, 0.9],
            [0.1, 0.1, 0.1, 0.1, 0.1],
        ],
        relation_labels=[
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        example_kinds=[
            "state_intent_switch",
            "state_ood_detour",
            "state_policy_followup",
        ],
        ood_banking_threshold=0.2,
        in_domain_threshold=0.5,
        relation_rescue_threshold=0.5,
        num_intents=2,
    )

    assert metrics["trajectory_state_route_error_rate"] == 1 / 3
    assert metrics["trajectory_state_intent_error_rate"] == 0.5
    assert metrics["trajectory_non_resume_false_positive_rate"] == 2 / 3


def test_resume_trajectory_metrics_require_exact_intent_and_relation() -> None:
    training = load_training_module()
    metrics = training.evaluate_predictions(
        domain_probabilities=[0.95],
        domain_labels=[1],
        intent_predictions=[1],
        intent_labels=[0],
        relation_probabilities=[[0.9, 0.1, 0.1, 0.1, 0.1]],
        relation_labels=[[1, 0, 0, 0, 1]],
        example_kinds=["resume_previous_service"],
        ood_banking_threshold=0.2,
        in_domain_threshold=0.5,
        relation_rescue_threshold=0.5,
        num_intents=2,
    )

    assert metrics["trajectory_resume_rows"] == 1
    assert metrics["trajectory_resume_intent_error_rate"] == 1.0
    assert metrics["trajectory_resume_relation_error_rate"] == 1.0


def test_relation_positive_weights_are_capped_for_rare_labels() -> None:
    training = load_training_module()
    rows = [
        {"relation_labels": [1, 0, 0, 0, 0]},
        {"relation_labels": [1, 0, 0, 0, 0]},
        {"relation_labels": [0, 1, 0, 0, 0]},
        {"relation_labels": [0, 0, 0, 0, 0]},
    ]

    assert training.relation_positive_weights(rows, max_weight=3.0) == [
        1.0,
        3.0,
        3.0,
        3.0,
        3.0,
    ]


def test_weighted_mean_prioritizes_targeted_rows() -> None:
    training = load_training_module()

    value = training._weighted_mean(
        torch.tensor([1.0, 3.0]),
        torch.tensor([1.0, 5.0]),
    )

    assert math.isclose(float(value), 8.0 / 3.0, rel_tol=1e-6)


def test_state_negative_rows_receive_targeted_training_weight() -> None:
    training = load_training_module()

    class TokenizerStub:
        def __call__(self, texts: list[str], **_kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
            }

    collate = training.make_collate(TokenizerStub(), max_length=256)
    common = {
        "text": "input",
        "domain_label": 1,
        "intent_label": 0,
        "relation_labels": [0, 0, 0, 0, 0],
        "example_kind": "state_intent_switch",
        "current_text": "switch",
    }
    batch = collate(
        [
            {**common, "source": "self-authored-router-v5-state-negatives"},
            {**common, "source": "unweighted-source"},
        ]
    )

    assert batch["row_weights"].tolist() == [training.TARGETED_ROW_WEIGHT, 1.0]


def test_relation_calibration_uses_lowest_exact_optimum_for_other_labels() -> None:
    training = load_training_module()
    probabilities = [
        [0.10, 0.20, 0.80, 0.10, 0.90],
        [0.01, 0.01, 0.01, 0.01, 0.01],
    ]
    labels = [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ]

    thresholds = training.calibrate_relation_thresholds(
        probabilities,
        labels,
    )

    assert thresholds == {
        "context_dependent": 0.05,
        "agent_repair": 0.05,
        "topic_shift": 0.05,
        "clarification_answer": 0.05,
        "resume_previous_service": 0.05,
    }


def test_relation_calibration_caps_agent_repair_threshold_for_recall() -> None:
    training = load_training_module()
    probabilities = [
        [0.1, 0.96, 0.1, 0.1, 0.1],
        [0.1, 0.90, 0.1, 0.1, 0.1],
    ]
    labels = [
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    thresholds = training.calibrate_relation_thresholds(probabilities, labels)

    assert thresholds["agent_repair"] == 0.85
