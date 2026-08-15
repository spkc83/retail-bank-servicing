from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pytest
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


def test_default_training_recipe_uses_verified_two_epoch_run() -> None:
    training = load_training_module()

    assert training.EPOCHS == 2


def test_training_loader_uses_portable_in_process_collation() -> None:
    training = load_training_module()
    loader = training._loader(
        [{}],
        batch_size=1,
        shuffle=False,
        collate=lambda rows: rows,
        device=torch.device("cpu"),
    )

    assert loader.num_workers == 0


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
    assert metrics["trajectory_runtime_transition_error_rate"] == 0.0
    assert metrics["heldout_social_generalization_error_rate"] == 0.0
    assert metrics["heldout_policy_followup_generalization_error_rate"] == 0.0
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
            "domain_macro_f1": 0.70,
            "lane_macro_f1": 0.70,
            "family_macro_f1": 0.70,
            "intent_macro_f1": 0.70,
            "relation_macro_f1": 0.70,
            "action_macro_f1": 0.70,
            "entity_resolution_macro_f1": 0.70,
            "exposed_action_macro_f1": 0.70,
            "exposed_entity_resolution_macro_f1": 0.70,
            "hierarchy_compatibility_error_rate": 0.10,
            "counterfactual_action_accuracy": 0.90,
            "counterfactual_entity_resolution_accuracy": 0.90,
            "counterfactual_pair_flip_accuracy": 0.90,
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
            "trajectory_runtime_transition_error_rate": 0.10,
            "heldout_social_generalization_error_rate": 0.10,
            "heldout_policy_followup_generalization_error_rate": 0.10,
            "heldout_regression_route_error_rate": 0.10,
            "heldout_regression_intent_error_rate": 0.10,
            "heldout_regression_relation_error_rate": 0.10,
        }
    )

    assert len(failures) == 29


def test_uncertain_in_scope_turn_counts_as_runtime_false_refusal() -> None:
    training = load_training_module()
    metrics = training.evaluate_predictions(
        domain_probabilities=[0.30],
        domain_labels=[1],
        intent_predictions=[training.INTENT_LABELS.index("replace_card")],
        intent_labels=[training.INTENT_LABELS.index("replace_card")],
        relation_probabilities=[[0.0] * len(training.RELATION_LABELS)],
        relation_labels=[[0] * len(training.RELATION_LABELS)],
        example_kinds=["servicing"],
        ood_banking_threshold=0.20,
        in_domain_threshold=0.50,
        relation_rescue_threshold=0.50,
        num_intents=len(training.INTENT_LABELS),
    )

    assert metrics["in_domain_false_refusal_rate"] == 1.0


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


def test_uncertain_route_fails_resume_and_switch_runtime_metrics() -> None:
    training = load_training_module()
    metrics = training.evaluate_predictions(
        domain_probabilities=[0.30, 0.30],
        domain_labels=[1, 1],
        intent_predictions=[0, 1],
        intent_labels=[0, 1],
        relation_probabilities=[
            [0.9, 0.1, 0.1, 0.1, 0.9],
            [0.1, 0.1, 0.9, 0.1, 0.1],
        ],
        relation_labels=[
            [1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
        ],
        example_kinds=["resume_previous_service", "state_intent_switch"],
        ood_banking_threshold=0.2,
        in_domain_threshold=0.5,
        relation_rescue_threshold=0.5,
        num_intents=2,
    )

    assert metrics["trajectory_resume_intent_error_rate"] == 1.0
    assert metrics["trajectory_state_route_error_rate"] == 1.0
    assert metrics["trajectory_state_intent_error_rate"] == 1.0
    assert metrics["trajectory_runtime_transition_error_rate"] == 1.0


def test_heldout_social_and_policy_gates_require_exact_runtime_intent() -> None:
    training = load_training_module()
    metrics = training.evaluate_predictions(
        domain_probabilities=[0.95, 0.95],
        domain_labels=[1, 1],
        intent_predictions=[0, 1],
        intent_labels=[1, 0],
        relation_probabilities=[
            [0.9, 0.1, 0.1, 0.1, 0.9],
            [0.1, 0.1, 0.1, 0.1, 0.9],
        ],
        relation_labels=[
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        example_kinds=[
            "heldout_social_generalization",
            "heldout_policy_followup_generalization",
        ],
        ood_banking_threshold=0.2,
        in_domain_threshold=0.5,
        relation_rescue_threshold=0.5,
        num_intents=2,
    )

    assert metrics["heldout_social_generalization_error_rate"] == 1.0
    assert metrics["heldout_policy_followup_generalization_error_rate"] == 1.0


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


def test_entity_resolution_weights_cover_and_upweight_rare_classes() -> None:
    training = load_training_module()
    rows = [
        {"entity_resolution_index": 0},
        {"entity_resolution_index": 0},
        {"entity_resolution_index": 0},
        {"entity_resolution_index": 0},
        {"entity_resolution_index": 1},
        {"entity_resolution_index": 2},
        {"entity_resolution_index": 3},
        {"entity_resolution_index": 4},
    ]

    weights = training.entity_resolution_class_weights(rows, max_weight=3.0)

    assert weights[0] == 1.0
    assert weights[1:] == [2.0, 2.0, 2.0, 2.0]
    with pytest.raises(ValueError, match="have no training rows"):
        training.entity_resolution_class_weights(rows[:-1])


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
        "domain_index": training.DOMAIN_LABELS.index("banking"),
        "intent_label": 0,
        "lane_index": training.LANE_LABELS.index("servicing"),
        "family_index": training.FAMILY_LABELS.index("accounts"),
        "action_index": training.ACTION_LABELS.index("execute_tool"),
        "entity_resolution_index": training.ENTITY_RESOLUTION_LABELS.index("not_required"),
        "relation_labels": [0, 0, 0, 0, 0],
        "example_kind": "state_intent_switch",
        "current_text": "switch",
    }
    batch = collate(
        [
            {**common, "source": "self-authored-router-v5-state-negatives"},
            {**common, "source": "self-authored-router-v5-resume-trajectory"},
            {**common, "source": "unweighted-source"},
        ]
    )

    assert batch["row_weights"].tolist() == [
        training.TARGETED_ROW_WEIGHT,
        training.TARGETED_ROW_WEIGHT,
        1.0,
    ]


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

    assert thresholds["agent_repair"] == 0.75


def test_relation_calibration_caps_clarification_answer_for_joint_decision_recall() -> None:
    training = load_training_module()
    probabilities = [
        [0.1, 0.1, 0.1, 0.96, 0.1],
        [0.1, 0.1, 0.1, 0.90, 0.1],
    ]
    labels = [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ]

    thresholds = training.calibrate_relation_thresholds(probabilities, labels)

    assert thresholds["clarification_answer"] == 0.80


def test_v4_counterfactual_metrics_use_runtime_exposed_decisions() -> None:
    training = load_training_module()

    def scores(index: int, width: int) -> list[float]:
        return [float(position == index) for position in range(width)]

    replace_card = training.INTENT_LABELS.index("replace_card")
    banking = training.DOMAIN_LABELS.index("banking")
    servicing = training.LANE_LABELS.index("servicing")
    cards = training.FAMILY_LABELS.index("cards")
    execute = training.ACTION_LABELS.index("execute_tool")
    clarify = training.ACTION_LABELS.index("clarify")
    resolved = training.ENTITY_RESOLUTION_LABELS.index("resolved")
    ambiguous = training.ENTITY_RESOLUTION_LABELS.index("ambiguous")

    metrics = training.evaluate_predictions(
        domain_probabilities=[0.99, 0.99],
        domain_labels=[1, 1],
        domain_predictions=[banking, banking],
        domain_indices=[banking, banking],
        lane_predictions=[servicing, servicing],
        lane_indices=[servicing, servicing],
        family_predictions=[cards, cards],
        family_indices=[cards, cards],
        intent_predictions=[replace_card, replace_card],
        intent_labels=[replace_card, replace_card],
        relation_probabilities=[[0.0] * 5, [0.0] * 5],
        relation_labels=[[0] * 5, [0] * 5],
        action_predictions=[execute, execute],
        action_indices=[execute, clarify],
        entity_resolution_predictions=[resolved, ambiguous],
        entity_resolution_indices=[resolved, ambiguous],
        domain_scores=[scores(banking, len(training.DOMAIN_LABELS))] * 2,
        lane_scores=[scores(servicing, len(training.LANE_LABELS))] * 2,
        family_scores=[scores(cards, len(training.FAMILY_LABELS))] * 2,
        intent_scores=[scores(replace_card, len(training.INTENT_LABELS))] * 2,
        action_scores=[scores(execute, len(training.ACTION_LABELS))] * 2,
        entity_resolution_scores=[
            scores(resolved, len(training.ENTITY_RESOLUTION_LABELS)),
            scores(ambiguous, len(training.ENTITY_RESOLUTION_LABELS)),
        ],
        counterfactual_pair_ids=["pair-1", "pair-1"],
        counterfactual_targets=["replace_card", "clarification"],
        example_kinds=["counterfactual_execute", "counterfactual_clarify"],
        ood_banking_threshold=0.2,
        in_domain_threshold=0.5,
        relation_rescue_threshold=0.5,
        num_intents=len(training.INTENT_LABELS),
    )

    # The runtime joint decoder selects the legal clarify+ambiguous tuple.
    assert metrics["counterfactual_action_accuracy"] == 1.0
    assert metrics["counterfactual_entity_resolution_accuracy"] == 1.0
    assert metrics["counterfactual_pair_flip_accuracy"] == 1.0
    assert metrics["hierarchy_compatibility_error_rate"] == 0.0
    assert metrics["raw_independent_head_incompatibility_rate"] == 0.5


def test_exposed_downstream_metrics_ignore_intentionally_suppressed_ood_values() -> None:
    training = load_training_module()
    banking = training.DOMAIN_LABELS.index("banking")
    ood = training.DOMAIN_LABELS.index("out_of_domain")
    servicing = training.LANE_LABELS.index("servicing")
    ood_lane = training.LANE_LABELS.index("out_of_domain")
    accounts = training.FAMILY_LABELS.index("accounts")
    external = training.FAMILY_LABELS.index("external")
    intent = training.INTENT_LABELS.index("view_accounts")
    execute = training.ACTION_LABELS.index("execute_tool")
    refuse = training.ACTION_LABELS.index("refuse_ood")
    not_required = training.ENTITY_RESOLUTION_LABELS.index("not_required")
    common = {
        "domain_probabilities": [0.99, 0.01],
        "domain_labels": [1, 0],
        "domain_predictions": [banking, ood],
        "domain_indices": [banking, ood],
        "lane_predictions": [servicing, ood_lane],
        "lane_indices": [servicing, ood_lane],
        "family_predictions": [accounts, external],
        "family_indices": [accounts, external],
        "intent_predictions": [intent, intent],
        "intent_labels": [intent, -100],
        "relation_probabilities": [[0.0] * 5, [0.0] * 5],
        "relation_labels": [[0] * 5, [0] * 5],
        "action_predictions": [execute, refuse],
        "action_indices": [execute, refuse],
        "entity_resolution_predictions": [not_required, not_required],
        "entity_resolution_indices": [not_required, not_required],
        "example_kinds": ["multi_turn", "clinc_external_ood"],
        "ood_banking_threshold": 0.2,
        "in_domain_threshold": 0.5,
        "relation_rescue_threshold": 0.5,
        "num_intents": len(training.INTENT_LABELS),
    }

    metrics = training.evaluate_predictions(**common)

    assert metrics["exposed_decision_rows"] == 2
    assert metrics["exposed_action_macro_f1"] == 1.0
    assert metrics["exposed_entity_resolution_macro_f1"] == 1.0
    refused = training.evaluate_predictions(
        **{**common, "domain_probabilities": [0.30, 0.01]}
    )
    assert refused["exposed_action_macro_f1"] == 0.5
    assert refused["exposed_entity_resolution_macro_f1"] == 2 / 3


def test_predict_returns_all_joint_decoder_head_scores() -> None:
    training = load_training_module()

    class ModelStub(torch.nn.Module):
        def forward(self, **_inputs: torch.Tensor) -> object:
            def logits(width: int) -> torch.Tensor:
                return torch.arange(width, dtype=torch.float32).unsqueeze(0)

            return training.ConversationRouterOutput(
                domain_logits=logits(len(training.DOMAIN_LABELS)),
                lane_logits=logits(len(training.LANE_LABELS)),
                family_logits=logits(len(training.FAMILY_LABELS)),
                intent_logits=logits(len(training.INTENT_LABELS)),
                relation_logits=logits(len(training.RELATION_LABELS)),
                action_logits=logits(len(training.ACTION_LABELS)),
                entity_resolution_logits=logits(
                    len(training.ENTITY_RESOLUTION_LABELS)
                ),
            )

    batch = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "domain_labels": torch.tensor([1]),
        "domain_indices": torch.tensor([1]),
        "lane_indices": torch.tensor([1]),
        "family_indices": torch.tensor([1]),
        "intent_labels": torch.tensor([0]),
        "relation_labels": torch.zeros((1, len(training.RELATION_LABELS))),
        "action_indices": torch.tensor([1]),
        "entity_resolution_indices": torch.tensor([0]),
        "example_kinds": ["test"],
        "current_texts": ["test"],
        "counterfactual_pair_ids": [""],
        "counterfactual_targets": [""],
    }

    predictions = training.predict(
        ModelStub(),
        [batch],
        device=torch.device("cpu"),
    )

    for key, labels in (
        ("domain_scores", training.DOMAIN_LABELS),
        ("lane_scores", training.LANE_LABELS),
        ("family_scores", training.FAMILY_LABELS),
        ("intent_scores", training.INTENT_LABELS),
        ("action_scores", training.ACTION_LABELS),
        ("entity_resolution_scores", training.ENTITY_RESOLUTION_LABELS),
    ):
        assert predictions[key] == [[float(index) for index in range(len(labels))]]


def test_release_gate_rejects_failed_counterfactual_and_hierarchy_metrics() -> None:
    training = load_training_module()
    passing = {
        "domain_macro_f1": 1.0,
        "lane_macro_f1": 1.0,
        "family_macro_f1": 1.0,
        "intent_macro_f1": 1.0,
        "relation_macro_f1": 1.0,
        "action_macro_f1": 1.0,
        "entity_resolution_macro_f1": 1.0,
        "exposed_action_macro_f1": 1.0,
        "exposed_entity_resolution_macro_f1": 1.0,
        "hierarchy_compatibility_error_rate": 0.0,
        "counterfactual_action_accuracy": 1.0,
        "counterfactual_entity_resolution_accuracy": 1.0,
        "counterfactual_pair_flip_accuracy": 1.0,
        "in_domain_false_refusal_rate": 0.0,
        "ood_false_accept_rate": 0.0,
        "contextual_false_refusal_rate": 0.0,
        "repair_false_refusal_rate": 0.0,
        "topic_shift_ood_false_accept_rate": 0.0,
        "trajectory_resume_intent_error_rate": 0.0,
        "trajectory_resume_relation_error_rate": 0.0,
        "trajectory_state_route_error_rate": 0.0,
        "trajectory_state_intent_error_rate": 0.0,
        "trajectory_non_resume_false_positive_rate": 0.0,
        "trajectory_runtime_transition_error_rate": 0.0,
        "heldout_social_generalization_error_rate": 0.0,
        "heldout_policy_followup_generalization_error_rate": 0.0,
        "heldout_regression_route_error_rate": 0.0,
        "heldout_regression_intent_error_rate": 0.0,
        "heldout_regression_relation_error_rate": 0.0,
    }

    assert training.release_gate_failures(passing) == []
    failures = training.release_gate_failures(
        {
            **passing,
            "hierarchy_compatibility_error_rate": 0.01,
            "counterfactual_action_accuracy": 0.94,
            "counterfactual_entity_resolution_accuracy": 0.94,
            "counterfactual_pair_flip_accuracy": 0.94,
        }
    )
    assert len(failures) == 4


def test_governed_loader_accepts_only_v3_canonical_label_contract(
    tmp_path: Path,
) -> None:
    training = load_training_module()

    def row(split: str, target: str) -> dict[str, object]:
        action = target
        resolution = "resolved" if action == "execute_tool" else "ambiguous"
        return {
            "text": "[CURRENT_USER]\nreplace it",
            "current_text": "replace it",
            "history": [],
            "domain_label": 1,
            "domain_name": "banking",
            "domain_index": training.DOMAIN_LABELS.index("banking"),
            "intent_label": training.INTENT_LABELS.index("replace_card"),
            "intent": "replace_card",
            "lane": "servicing",
            "lane_name": "servicing",
            "lane_index": training.LANE_LABELS.index("servicing"),
            "family_name": "cards",
            "family_index": training.FAMILY_LABELS.index("cards"),
            "action_name": action,
            "action_index": training.ACTION_LABELS.index(action),
            "entity_resolution_name": resolution,
            "entity_resolution_index": training.ENTITY_RESOLUTION_LABELS.index(resolution),
            "relation_labels": [1, 0, 0, 0, 0],
            "example_kind": f"counterfactual_{target}",
            "source": "test",
            "source_split": split,
            "group_id": f"{split}-group",
            "trajectory_id": f"{split}-trajectory-{target}",
            "prior_dialogue_state": {},
            "counterfactual_pair_id": f"{split}-pair",
            "counterfactual_target": (
                "replace_card" if target == "execute_tool" else "clarification"
            ),
            "counterfactual_phrase_family": f"{split}-phrase",
            "actionable_entity_count": 1 if target == "execute_tool" else 2,
        }

    split_entries = []
    for split in ("train", "validation", "test"):
        payload = "".join(
            json.dumps(row(split, target), sort_keys=True) + "\n"
            for target in ("execute_tool", "clarify")
        )
        path = tmp_path / f"{split}.jsonl"
        path.write_text(payload, encoding="utf-8")
        split_entries.append(
            {
                "name": split,
                "path": path.name,
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        )
    manifest = {
        "contract": "banking-conversation-router-data",
        "format_version": 3,
        "domain_labels": list(training.DOMAIN_LABELS),
        "lane_labels": list(training.LANE_LABELS),
        "family_labels": list(training.FAMILY_LABELS),
        "intent_labels": list(training.INTENT_LABELS),
        "relation_labels": list(training.RELATION_LABELS),
        "action_labels": list(training.ACTION_LABELS),
        "entity_resolution_labels": list(training.ENTITY_RESOLUTION_LABELS),
        "splits": split_entries,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    _manifest, splits = training.load_governed_data(tmp_path)
    assert all(len(rows) == 2 for rows in splits.values())

    manifest["format_version"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        training.load_governed_data(tmp_path)


def test_v4_artifact_saves_every_head_and_canonical_provenance(tmp_path: Path) -> None:
    training = load_training_module()

    class EncoderStub(torch.nn.Module):
        def save_pretrained(self, output: Path, **_kwargs: object) -> None:
            (output / "encoder.marker").write_text("encoder", encoding="utf-8")

    class TokenizerStub:
        def save_pretrained(self, output: Path) -> None:
            (output / "tokenizer.marker").write_text("tokenizer", encoding="utf-8")

    model = training.ConversationRouterModel(
        EncoderStub(),
        hidden_size=4,
        num_domains=len(training.DOMAIN_LABELS),
        num_lanes=len(training.LANE_LABELS),
        num_families=len(training.FAMILY_LABELS),
        num_intents=len(training.INTENT_LABELS),
        num_relations=len(training.RELATION_LABELS),
        num_actions=len(training.ACTION_LABELS),
        num_entity_resolutions=len(training.ENTITY_RESOLUTION_LABELS),
    )
    metrics = {
        "data_manifest_sha256": "a" * 64,
        "release_eligible": False,
        "policy": {
            "ood_banking_threshold": 0.2,
            "in_domain_threshold": 0.5,
            "relation_rescue_threshold": 0.5,
            "relation_thresholds": {label: 0.5 for label in training.RELATION_LABELS},
        },
        "test": {
            "domain_macro_f1": 0.0,
            "lane_macro_f1": 0.0,
            "family_macro_f1": 0.0,
            "intent_macro_f1": 0.0,
            "relation_macro_f1": 0.0,
            "action_macro_f1": 0.0,
            "entity_resolution_macro_f1": 0.0,
            "hierarchy_compatibility_error_rate": 0.0,
            "counterfactual_action_accuracy": 0.0,
            "counterfactual_entity_resolution_accuracy": 0.0,
            "counterfactual_pair_flip_accuracy": 0.0,
            "in_domain_false_refusal_rate": 0.0,
            "ood_false_accept_rate": 0.0,
            "contextual_false_refusal_rate": 0.0,
            "repair_false_refusal_rate": 0.0,
            "topic_shift_ood_false_accept_rate": 0.0,
        },
    }
    training.save_artifact(
        model=model,
        tokenizer=TokenizerStub(),
        output=tmp_path,
        data_manifest={
            "max_exchanges": 3,
            "report": {"split_counts": {"train": 2, "validation": 2, "test": 2}},
        },
        metrics=metrics,
        max_length=256,
        data_revision="b" * 40,
    )

    config = json.loads((tmp_path / "router_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    heads = training.load_file(tmp_path / "classifier_heads.safetensors")
    assert config["format_version"] == 4
    assert config["domain_labels"] == list(training.DOMAIN_LABELS)
    assert config["action_labels"] == list(training.ACTION_LABELS)
    assert config["entity_resolution_labels"] == list(training.ENTITY_RESOLUTION_LABELS)
    assert config["generation_guidance_contract"] == "intent-selects-tool-schema-no-arguments-v1"
    assert manifest["format_version"] == 4
    assert manifest["release_eligible"] is False
    assert {key.removesuffix(".weight") for key in heads if key.endswith(".weight")} == {
        "domain_head",
        "lane_head",
        "family_head",
        "intent_head",
        "relation_head",
        "action_head",
        "entity_resolution_head",
    }
