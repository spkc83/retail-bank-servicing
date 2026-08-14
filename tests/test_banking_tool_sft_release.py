from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_finalizer() -> ModuleType:
    path = Path("scripts/retail_bank/hf_job_finalize_tool_sft.py")
    spec = importlib.util.spec_from_file_location("hf_job_finalize_tool_sft", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINALIZER = load_finalizer()


def test_finalizer_accepts_versioned_step_and_artifact_subdirectories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "hf_job_finalize_tool_sft.py",
            "--source-commit",
            "a" * 40,
            "--training-job",
            "training-job",
            "--remerge-job",
            "merge-job",
            "--parity-job",
            "parity-job",
            "--selected-step",
            "750",
            "--merged-subdir",
            "merged",
            "--adapter-subdir",
            "adapter",
            "--release-dtype",
            "bfloat16",
        ],
    )

    args = FINALIZER.parse_args()

    assert args.selected_step == 750
    assert args.merged_subdir == "merged"
    assert args.adapter_subdir == "adapter"
    assert args.release_dtype == "bfloat16"


def parity_report(**overrides: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "all_logit_differences_finite": True,
        "all_greedy_generations_equal": True,
        "prompt_count": 8,
        "argmax_token_agreement": 1.0,
        "max_abs_logit_diff": 0.28125,
        "p999_abs_logit_diff": 0.064453125,
    }
    metrics.update(overrides)
    return {
        "contract": "banking-v3-bf16-merge-parity/v1",
        "dtype": "float16",
        "metrics": metrics,
    }


def test_fp16_merge_parity_accepts_behaviorally_identical_generation() -> None:
    metrics = FINALIZER.validate_parity(
        parity_report(),
        release_dtype="float16",
        minimum_argmax_agreement=0.999,
        maximum_logit_difference=0.3,
        maximum_p999_difference=0.07,
    )

    assert metrics["all_greedy_generations_equal"] is True
    assert metrics["argmax_token_agreement"] == 1.0


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"all_logit_differences_finite": False}, "non-finite"),
        ({"all_greedy_generations_equal": False}, "generations differ"),
        ({"prompt_count": 7}, "only 7 prompts"),
        ({"argmax_token_agreement": 0.98}, "argmax agreement"),
        ({"max_abs_logit_diff": 0.31}, "maximum logit difference"),
        ({"p999_abs_logit_diff": 0.08}, "p999 logit difference"),
    ],
)
def test_fp16_merge_parity_rejects_failed_release_gate(
    override: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        FINALIZER.validate_parity(
            parity_report(**override),
            release_dtype="float16",
            minimum_argmax_agreement=0.999,
            maximum_logit_difference=0.3,
            maximum_p999_difference=0.07,
        )


def test_finalizer_rejects_parity_from_different_release_dtype() -> None:
    report = parity_report()
    report["dtype"] = "bfloat16"

    with pytest.raises(RuntimeError, match="does not match release dtype"):
        FINALIZER.validate_parity(
            report,
            release_dtype="float16",
            minimum_argmax_agreement=0.999,
            maximum_logit_difference=0.3,
            maximum_p999_difference=0.07,
        )
