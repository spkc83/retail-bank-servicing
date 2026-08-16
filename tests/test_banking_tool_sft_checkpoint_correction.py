from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

WORKER_PATH = Path("scripts/retail_bank/cloud_continue_tool_sft.py")
JOB_PATH = Path("scripts/retail_bank/hf_job_correct_checkpoint_tool_sft.py")
LAUNCHER_PATH = Path("scripts/retail_bank/run_remote_checkpoint_correction.sh")


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WORKER = _load("cloud_continue_tool_sft_checkpoint_correction", WORKER_PATH)
JOB = _load("hf_job_correct_checkpoint_tool_sft", JOB_PATH)


def _job() -> SimpleNamespace:
    return SimpleNamespace(
        id=JOB.FAILED_JOB_ID,
        created_at=datetime(2026, 8, 16, 4, 14, 12, tzinfo=UTC),
        started_at=datetime(2026, 8, 16, 4, 14, 21, tzinfo=UTC),
        finished_at=datetime(2026, 8, 16, 4, 41, 57, tzinfo=UTC),
        command=JOB.expected_failed_job_command(),
        flavor="rtx-pro-6000",
        status=SimpleNamespace(stage="ERROR"),
        volumes=[
            SimpleNamespace(
                type="bucket",
                source="spkc83/jobs-artifacts",
                mount_path="/data",
                read_only=False,
            )
        ],
    )


def _checkpoint(root: Path, *, step: int = 964, dtype: torch.dtype = torch.bfloat16) -> Path:
    checkpoint = root / "trainer" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    save_file(
        {"adapter.layer": torch.ones((2, 2), dtype=dtype)},
        checkpoint / "adapter_model.safetensors",
    )
    (checkpoint / "adapter_config.json").write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
    (checkpoint / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    (checkpoint / "tokenizer_config.json").write_text(
        '{"model_max_length":2048}\n', encoding="utf-8"
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}) + "\n",
        encoding="utf-8",
    )
    checkpoint_time = datetime(2026, 8, 16, 4, 35, tzinfo=UTC).timestamp()
    for path in checkpoint.iterdir():
        os.utime(path, (checkpoint_time, checkpoint_time))
    return checkpoint


def _config(checkpoint: Path, provenance: Path, *, step: int = 964):
    config = WORKER.config_from_args(WORKER.parse_args([]))
    return replace(
        config,
        correction_checkpoint_dir=checkpoint,
        correction_checkpoint_step=step,
        correction_provenance=provenance,
        max_steps=64,
        checkpoint_every=16,
    )


def _provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "failed-v7"
    checkpoint = _checkpoint(root)
    monkeypatch.setattr(JOB, "FAILED_OUTPUT_ROOT", root)
    monkeypatch.setattr(WORKER, "V7_FAILED_OUTPUT_ROOT", root)
    payload = JOB.build_provenance(checkpoint, 964, _job())
    provenance = tmp_path / "checkpoint_correction_provenance.json"
    provenance.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return checkpoint, provenance


def test_checkpoint_correction_dry_run_declares_weights_only_semantics() -> None:
    config = WORKER.config_from_args(
        WORKER.parse_args(
            [
                "--correction-checkpoint-dir",
                str(WORKER.V7_DEFAULT_CORRECTION_CHECKPOINT),
                "--correction-checkpoint-step",
                "964",
                "--correction-provenance",
                "/tmp/provenance.json",
                "--max-steps",
                "64",
                "--checkpoint-every",
                "16",
            ]
        )
    )

    plan = WORKER.build_dry_run_plan(config)

    assert plan["training_parent"] == "exact failed V7 checkpoint weights"
    assert plan["checkpoint_correction"] == {
        "contract": "retail-bank-peft-v7-checkpoint-correction/v1",
        "path": str(WORKER.V7_DEFAULT_CORRECTION_CHECKPOINT),
        "expected_step": 964,
        "provenance": "/tmp/provenance.json",
        "initialization": "BF16 adapter weights and tokenizer only",
        "optimizer": "fresh",
        "scheduler": "fresh",
        "rng_state": "fresh seeded run",
        "exact_resume_claimed": False,
    }
    assert plan["training"]["max_steps"] == 64
    assert plan["training"]["checkpoint_every"] == 16


def test_exact_checkpoint_provenance_accepts_bf16_weights_and_fresh_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, provenance = _provenance(tmp_path, monkeypatch)

    identity = WORKER.validate_checkpoint_correction_source(_config(checkpoint, provenance))

    assert identity["source_step"] == 964
    assert identity["bf16_tensor_count"] == 1
    assert identity["optimizer_resumed"] is False
    assert identity["scheduler_resumed"] is False
    assert identity["rng_state_resumed"] is False
    assert set(identity["source_files"]) == set(WORKER.CORRECTION_SOURCE_FILES)


@pytest.mark.parametrize(
    "tampered_file",
    ["adapter_model.safetensors", "adapter_config.json", "tokenizer.json"],
)
def test_checkpoint_correction_rejects_file_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_file: str,
) -> None:
    checkpoint, provenance = _provenance(tmp_path, monkeypatch)
    with (checkpoint / tampered_file).open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(RuntimeError, match=f"file digest mismatch: {tampered_file}"):
        WORKER.validate_checkpoint_correction_source(_config(checkpoint, provenance))


def test_checkpoint_correction_rejects_job_command_and_time_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, provenance = _provenance(tmp_path, monkeypatch)
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["inspected_job"]["command"][-1] = "900"
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="inspected job provenance mismatch"):
        WORKER.validate_checkpoint_correction_source(_config(checkpoint, provenance))

    payload = JOB.build_provenance(checkpoint, 964, _job())
    payload["inspected_job"]["finished_at"] = "2026-08-16T04:00:00+00:00"
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="timestamps are not ordered"):
        WORKER.validate_checkpoint_correction_source(_config(checkpoint, provenance))


def test_checkpoint_correction_rejects_non_bf16_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "failed-v7"
    checkpoint = _checkpoint(root, dtype=torch.float32)
    monkeypatch.setattr(JOB, "FAILED_OUTPUT_ROOT", root)
    monkeypatch.setattr(WORKER, "V7_FAILED_OUTPUT_ROOT", root)
    payload = JOB.build_provenance(checkpoint, 964, _job())
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-BF16 tensors"):
        WORKER.validate_checkpoint_correction_source(_config(checkpoint, provenance))


def test_worker_never_claims_exact_optimizer_resume() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    training_body = source.split("def run_remote_continuation", 1)[1].split("def main", 1)[0]

    assert "trainer.train()" in training_body
    assert "resume_from_checkpoint=" not in training_body
    assert '"optimizer_resumed": False' in source
    assert '"scheduler_resumed": False' in source


def test_launcher_dry_run_prints_command_without_invoking_hf(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' 'V7_CHECKPOINT_CORRECTION_PROTOCOL = \""
        "retail-bank-peft-v7-checkpoint-correction/v1\"'\n"
        "printf '%s\\n' 'CORRECTION_PROTOCOL = \"retail-bank-peft-v7-checkpoint-correction/v1\"'\n",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    hf = fake_bin / "hf"
    hf.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    hf.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DRY_RUN": "1",
    }

    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH), "a" * 40],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stderr == ""
    assert completed.stdout.startswith("hf jobs uv run ")
    assert "--flavor rtx-pro-6000" in completed.stdout
    assert "checkpoint-964" in completed.stdout
    assert "--max-steps 64" in completed.stdout
    assert "6a813914c97db76cbdf32acc" in completed.stdout
