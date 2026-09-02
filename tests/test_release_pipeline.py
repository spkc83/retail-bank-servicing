from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    path = Path("scripts/retail_bank/run_release_pipeline.py")
    spec = importlib.util.spec_from_file_location("run_release_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pipeline = _load_module()


def _args(*values: str):
    return pipeline.parse_args(list(values))


def test_canonical_pipeline_keeps_two_distinct_granite_sft_stages() -> None:
    config = pipeline.load_config(Path("configs/retail-bank-release.toml"))
    stages = pipeline.build_stages(config, _args())

    assert tuple(stage.name for stage in stages) == pipeline.STAGE_ORDER
    base = stages[1]
    alignment = stages[2]
    assert base.environment["BASE_MODEL"] == "ibm-granite/granite-4.1-8b"
    assert alignment.environment["BASE_MODEL"] == "spkc83/retail-bank-agent-9b"
    assert alignment.environment["MAX_STEPS"] == "500"
    assert "Remediation SFT" in alignment.purpose
    router = stages[3]
    assert "--data-revision" in router.commands[0]
    assert "--dataset-revision" not in router.commands[0]
    deploy = stages[5]
    assert "scripts/retail_bank/deploy_zero_gpu_space.py" in deploy.commands[0]
    assert deploy.environment == {}


def test_execute_all_is_rejected_because_artifact_revisions_are_sequential() -> None:
    with pytest.raises(pipeline.ReleasePipelineError, match="one stage at a time"):
        pipeline.main(["--stage", "all", "--execute"])


def test_external_stages_require_explicit_publish_and_paid_guards() -> None:
    config = pipeline.load_config(Path("configs/retail-bank-release.toml"))
    alignment = pipeline.build_stages(config, _args())[2]

    with pytest.raises(pipeline.ReleasePipelineError, match="allow-paid"):
        pipeline.execute_stage(alignment, allow_paid=False, allow_publish=True)
    with pytest.raises(pipeline.ReleasePipelineError, match="allow-publish"):
        pipeline.execute_stage(alignment, allow_paid=True, allow_publish=False)


def test_revision_validation_rejects_branches_and_short_hashes() -> None:
    with pytest.raises(pipeline.ReleasePipelineError, match="exact 40-character"):
        pipeline.exact_revision("main", field="test revision")


def test_sft_dataset_revision_overrides_are_stage_specific() -> None:
    config = pipeline.load_config(Path("configs/retail-bank-release.toml"))
    base_revision = "a" * 40
    alignment_revision = "b" * 40
    stages = pipeline.build_stages(
        config,
        _args(
            "--base-dataset-revision",
            base_revision,
            "--alignment-dataset-revision",
            alignment_revision,
        ),
    )

    assert stages[1].commands[0][-1] == base_revision
    assert stages[2].commands[0][-1] == alignment_revision


def test_the_deploy_stage_states_every_runtime_identity() -> None:
    """The stage used to name only the model and the router, and let the rest default.

    Its --model-id came from the merged-weights lineage while the deploy
    script's own default supplied a v8 adapter and an empty subfolder, so
    running this stage would have silently rolled the live Space back two
    generations into a composition no release ever had.
    """
    config = pipeline.load_config(Path("configs/retail-bank-release.toml"))
    command = pipeline.build_stages(config, _args())[5].commands[0]
    flags = {
        command[index]: command[index + 1]
        for index, value in enumerate(command)
        if value.startswith("--") and index + 1 < len(command)
    }

    for required in (
        "--base-model-id",
        "--base-model-revision",
        "--adapter-id",
        "--adapter-revision",
        "--adapter-subfolder",
        "--model-dtype",
    ):
        assert required in flags, f"deploy must state {required} rather than inherit a default"

    # The adapter is the thing being deployed: model and adapter must agree.
    assert flags["--model-id"] == flags["--adapter-id"]
    assert flags["--model-revision"] == flags["--adapter-revision"]
    assert flags["--adapter-id"].endswith("-peft-v14-prompt-realized")
    assert flags["--adapter-subfolder"] == "adapter"
    assert flags["--base-model-id"] != flags["--adapter-id"]


def test_the_config_pins_the_router_the_space_actually_serves() -> None:
    """These pins were two releases stale and nothing recomputed them."""
    config = pipeline.load_config(Path("configs/retail-bank-release.toml"))
    router = config["router"]

    assert router["model_revision"] == "dd5ea26674a0f9808d42110a9ee51a9af6762a76"
    assert router["dataset_revision"] == "b33c27170e27cdb11783704ede14f7d25f70625e"
