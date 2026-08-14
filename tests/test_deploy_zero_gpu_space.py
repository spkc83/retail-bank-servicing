from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _load_module() -> ModuleType:
    path = Path("scripts/retail_bank/deploy_zero_gpu_space.py")
    spec = importlib.util.spec_from_file_location("deploy_zero_gpu_space", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


deploy_module = _load_module()
MODEL_REVISION = "1" * 40
ROUTER_REVISION = "2" * 40
SPACE_COMMIT = "3" * 40
BASE_MODEL_REVISION = "4" * 40
ADAPTER_REVISION = "5" * 40


class FakeApi:
    def __init__(self) -> None:
        self.upload: dict[str, Any] | None = None
        self.variables: dict[str, str] = {}
        self.waited: str | None = None

    def upload_folder(self, **kwargs: Any) -> Any:
        self.upload = kwargs
        return SimpleNamespace(oid=SPACE_COMMIT)

    def add_space_variable(
        self,
        repo_id: str,
        key: str,
        value: str,
        **_kwargs: Any,
    ) -> None:
        assert repo_id == "spkc83/test-space"
        self.variables[key] = value

    def wait_for_space(self, repo_id: str, **_kwargs: Any) -> Any:
        self.waited = repo_id
        return SimpleNamespace(stage="RUNNING")


def _args(tmp_path: Path, *extra: str):
    source_dir = tmp_path / "space"
    source_dir.mkdir()
    return deploy_module.parse_args(
        [
            "--space-id",
            "spkc83/test-space",
            "--source-dir",
            str(source_dir),
            "--model-id",
            "spkc83/model",
            "--model-revision",
            MODEL_REVISION,
            "--router-id",
            "spkc83/router",
            "--router-revision",
            ROUTER_REVISION,
            *extra,
        ]
    )


def test_plan_is_non_mutating_and_excludes_local_caches(tmp_path: Path) -> None:
    deployment_plan = deploy_module.plan(_args(tmp_path))

    assert deployment_plan["runtime_pins"]["RETAIL_BANK_MODEL_REVISION"] == MODEL_REVISION
    assert "tests/**" in deployment_plan["ignore_patterns"]
    assert "**/.*" in deployment_plan["ignore_patterns"]
    assert "LICENSE" in deployment_plan["allow_patterns"]


def test_deploy_persists_exact_runtime_pins_and_space_commit(tmp_path: Path) -> None:
    api = FakeApi()
    result = deploy_module.deploy(
        _args(tmp_path, "--execute", "--allow-publish"),
        api,
    )

    assert api.upload is not None
    assert "delete_patterns" not in api.upload
    assert api.variables == {
        "RETAIL_BANK_MODEL_ID": "spkc83/model",
        "RETAIL_BANK_MODEL_REVISION": MODEL_REVISION,
        "RETAIL_BANK_MODEL_DTYPE": "bf16",
        "RETAIL_BANK_BASE_MODEL_ID": "spkc83/retail-bank-servicing-agent-9b",
        "RETAIL_BANK_BASE_MODEL_REVISION": "1d56824995aa1adecfe20f62ca42fb1c0c443817",
        "RETAIL_BANK_ADAPTER_ID": "spkc83/retail-bank-servicing-agent-9b-peft",
        "RETAIL_BANK_ADAPTER_REVISION": "cc95e446af2b5e1d8d9df2751a8192613ad386e3",
        "RETAIL_BANK_ROUTER_ID": "spkc83/router",
        "RETAIL_BANK_ROUTER_REVISION": ROUTER_REVISION,
        "SPACE_COMMIT_SHA": SPACE_COMMIT,
    }
    assert api.waited == "spkc83/test-space"
    assert result["runtime_stage"] == "RUNNING"


def test_deploy_requires_both_explicit_guards(tmp_path: Path) -> None:
    with pytest.raises(deploy_module.DeployError, match="requires"):
        deploy_module.deploy(_args(tmp_path, "--execute"), FakeApi())


def test_peft_deploy_persists_exact_base_adapter_pins(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        "--base-model-id",
        "ibm-granite/base",
        "--base-model-revision",
        BASE_MODEL_REVISION,
        "--adapter-id",
        "spkc83/model",
        "--adapter-revision",
        ADAPTER_REVISION,
        "--model-dtype",
        "fp16",
    )

    pins = deploy_module.runtime_pins(args)

    assert pins["RETAIL_BANK_BASE_MODEL_ID"] == "ibm-granite/base"
    assert pins["RETAIL_BANK_BASE_MODEL_REVISION"] == BASE_MODEL_REVISION
    assert pins["RETAIL_BANK_ADAPTER_ID"] == "spkc83/model"
    assert pins["RETAIL_BANK_ADAPTER_REVISION"] == ADAPTER_REVISION
    assert pins["RETAIL_BANK_MODEL_DTYPE"] == "fp16"


def test_peft_deploy_rejects_partial_composition(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.base_model_revision = None

    with pytest.raises(deploy_module.DeployError, match="requires"):
        deploy_module.plan(args)


def test_merged_only_deploy_clears_adapter_variables(tmp_path: Path) -> None:
    pins = deploy_module.runtime_pins(_args(tmp_path, "--merged-model-only"))

    assert pins["RETAIL_BANK_BASE_MODEL_ID"] == ""
    assert pins["RETAIL_BANK_BASE_MODEL_REVISION"] == ""
    assert pins["RETAIL_BANK_ADAPTER_ID"] == ""
    assert pins["RETAIL_BANK_ADAPTER_REVISION"] == ""


def test_invalid_revision_is_rejected_before_upload(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.model_revision = "main"
    with pytest.raises(deploy_module.DeployError, match="exact 40-character"):
        deploy_module.plan(args)
