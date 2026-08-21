from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

JOB_PATH = Path("scripts/retail_bank/hf_job_tool_sft.py")


def _load_job() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hf_job_tool_sft", JOB_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_job_script_has_inline_dependencies_and_pinned_artifacts() -> None:
    source = JOB_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }

    assert "# /// script" in source
    assert '"trl==0.26.2"' in source
    assert '"trackio>=0.33,<0.34"' in source
    assert assignments["MODEL_REPO"] == "spkc83/retail-bank-servicing-agent-9b"
    assert assignments["DATASET_REPO"] == ("spkc83/retail-bank-servicing-alignment-sft")
    assert assignments["BASE_REVISION"] == ("1d56824995aa1adecfe20f62ca42fb1c0c443817")


def test_source_download_requires_a_commit_hash_before_network(
    tmp_path: Path,
) -> None:
    job = _load_job()

    with pytest.raises(ValueError, match="Git commit"):
        job.download_source("feat/tool-use-sft-v3", tmp_path)


def test_job_command_preserves_five_hour_internal_budget() -> None:
    source = JOB_PATH.read_text(encoding="utf-8")

    assert "default=14_400" in source
    assert 'default="/data/retail-bank-agent-9b"' in source
    assert "snapshot_download(" in source
    assert 'repo_type="dataset"' in source
    assert "dataset manifest is unavailable" in source
    assert '"--precision",' in source
    assert '"bf16-lora",' in source
    assert '"--push-to-hub",' in source
    assert 'parser.add_argument("--resume-from")' in source
    assert 'command.extend(["--resume-from", args.resume_from])' in source
    assert 'parser.add_argument("--learning-rate", type=float, default=1e-4)' in source
    assert "str(args.learning_rate)" in source
    assert "args.trackio_project" in source
    assert "args.trackio_run_name" in source


def test_remote_launcher_mounts_durable_job_bucket() -> None:
    launcher = Path("scripts/retail_bank/run_remote_training_job.sh").read_text(encoding="utf-8")

    assert "--volume hf://buckets/spkc83/jobs-artifacts:/data" in launcher
    assert (
        'output_prefix="${OUTPUT_PREFIX:-/data/retail-bank-agent-9b-'
        '${source_commit:0:8}}"' in launcher
    )
    assert "must be the exact 40-character lowercase Git commit" in launcher
    assert "/scripts/retail_bank/hf_job_tool_sft.py" in launcher
    assert "/scripts/banking_v2/hf_job_tool_sft.py" not in launcher
    assert 'if ! curl --fail --silent --head "$script_url"' in launcher
    assert 'script_url="$legacy_script_url"' not in launcher
    assert 'job_args+=(--resume-from "$resume_from")' in launcher
    assert '--max-steps "$max_steps"' in launcher
    assert '--learning-rate "$learning_rate"' in launcher
    assert '--trackio-project "$trackio_project"' in launcher


@pytest.mark.parametrize(
    ("launcher", "arguments", "bootstrap"),
    [
        (
            "run_remote_training_job.sh",
            ("a" * 40, "b" * 40),
            "hf_job_tool_sft.py",
        ),
        (
            "run_remote_continuation_job.sh",
            (
                "a" * 40,
                "b" * 40,
                "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2",
            ),
            "hf_job_continue_tool_sft.py",
        ),
        (
            "run_remote_continuation_export_recovery.sh",
            (
                "a" * 40,
                "b" * 40,
                "c" * 40,
                "d" * 40,
                "spkc83/job-123",
                "/data/retail-bank-agent-9b-continuation-test",
                "600",
            ),
            "hf_job_recover_continuation_export.py",
        ),
        (
            "run_remote_tool_eval_job.sh",
            ("a" * 40, "b" * 40, "c" * 40),
            "hf_job_tool_eval.py",
        ),
    ],
)
def test_remote_launchers_resolve_current_bootstraps(
    tmp_path: Path,
    launcher: str,
    arguments: tuple[str, ...],
    bootstrap: str,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_log = tmp_path / "curl.log"
    hf_log = tmp_path / "hf.log"
    curl = bin_dir / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
url="${@: -1}"
printf '%s\\n' "$url" >> "$CURL_LOG"
if [[ "$url" == *"hf_job_continue_tool_sft.py" || "$url" == *"cloud_continue_tool_sft.py" ]]; then
  printf '%s\\n' 'V6_CONTINUATION_PROTOCOL = "retail-bank-peft-v6-generation-contract/v1"'
fi
[[ "$url" == *"/scripts/retail_bank/"* ]]
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    hf = bin_dir / "hf"
    hf.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$HF_LOG"
""",
        encoding="utf-8",
    )
    hf.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CURL_LOG": str(curl_log),
        "HF_LOG": str(hf_log),
    }

    subprocess.run(
        ["bash", f"scripts/retail_bank/{launcher}", *arguments],
        check=True,
        env=env,
    )

    requested_urls = curl_log.read_text(encoding="utf-8").splitlines()
    submitted_args = hf_log.read_text(encoding="utf-8")
    assert requested_urls[0].endswith(f"/scripts/retail_bank/{bootstrap}")
    assert len(requested_urls) == (2 if launcher == "run_remote_continuation_job.sh" else 1)
    assert f"/scripts/retail_bank/{bootstrap}" in submitted_args


def test_continuation_launcher_rejects_source_without_v6_protocol(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    hf_log = tmp_path / "hf.log"
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' 'CANDIDATE4_PROTOCOL = true'\n",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    hf = bin_dir / "hf"
    hf.write_text(
        "#!/usr/bin/env bash\nprintf 'called\\n' > \"$HF_LOG\"\n",
        encoding="utf-8",
    )
    hf.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HF_LOG": str(hf_log),
    }

    completed = subprocess.run(
        [
            "bash",
            "scripts/retail_bank/run_remote_continuation_job.sh",
            "a" * 40,
            "b" * 40,
            "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "required V6 continuation protocol" in completed.stderr
    assert not hf_log.exists()


def test_continuation_publish_recovery_uses_cpu_and_publish_only_mode(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    hf_log = tmp_path / "hf.log"
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' 'V6_CONTINUATION_PROTOCOL = "
        '"retail-bank-peft-v6-generation-contract/v1"\'\n',
        encoding="utf-8",
    )
    curl.chmod(0o755)
    hf = bin_dir / "hf"
    hf.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$HF_LOG"\n', encoding="utf-8")
    hf.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HF_LOG": str(hf_log),
        "PUBLISH_ONLY": "1",
    }

    subprocess.run(
        [
            "bash",
            "scripts/retail_bank/run_remote_continuation_job.sh",
            "a" * 40,
            "b" * 40,
            "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2",
        ],
        check=True,
        env=env,
    )
    submitted = hf_log.read_text(encoding="utf-8")
    assert "cpu-basic" in submitted
    assert "30m" in submitted
    assert "--publish-only" in submitted


@pytest.mark.parametrize(
    ("environment", "expected_seconds"),
    [({"MAX_TRAIN_SECONDS": "1800"}, "1800"), ({}, "3600")],
)
@pytest.mark.parametrize(
    ("timeout_environment", "expected_timeout"),
    [({"JOB_TIMEOUT": "45m"}, "45m"), ({}, "5h")],
)
@pytest.mark.parametrize(
    ("min_steps_environment", "expected_min_steps"),
    [({"MIN_STEPS": "550"}, "550"), ({}, "0")],
)
def test_continuation_launcher_forwards_max_train_seconds(
    tmp_path: Path,
    environment: dict[str, str],
    expected_seconds: str,
    timeout_environment: dict[str, str],
    expected_timeout: str,
    min_steps_environment: dict[str, str],
    expected_min_steps: str,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    hf_log = tmp_path / "hf.log"
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' 'V6_CONTINUATION_PROTOCOL = "
        '"retail-bank-peft-v6-generation-contract/v1"\'\n',
        encoding="utf-8",
    )
    curl.chmod(0o755)
    hf = bin_dir / "hf"
    hf.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$HF_LOG"\n', encoding="utf-8")
    hf.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HF_LOG": str(hf_log),
    }
    env.pop("MAX_TRAIN_SECONDS", None)
    env.pop("JOB_TIMEOUT", None)
    env.pop("POSITIVE_MULTIPLIER", None)
    env.pop("AMBIGUITY_MULTIPLIER", None)
    env.pop("MIN_STEPS", None)
    env.update(environment)
    env.update(timeout_environment)
    env.update(min_steps_environment)
    if expected_seconds == "1800":
        env.update({"POSITIVE_MULTIPLIER": "3", "AMBIGUITY_MULTIPLIER": "6"})

    subprocess.run(
        [
            "bash",
            "scripts/retail_bank/run_remote_continuation_job.sh",
            "a" * 40,
            "b" * 40,
            "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2",
        ],
        check=True,
        env=env,
    )

    submitted = hf_log.read_text(encoding="utf-8").splitlines()
    assert "--max-train-seconds" in submitted
    assert submitted[submitted.index("--max-train-seconds") + 1] == expected_seconds
    assert submitted[submitted.index("--timeout") + 1] == expected_timeout
    expected_positive, expected_ambiguity = ("3", "6") if expected_seconds == "1800" else ("2", "4")
    assert submitted[submitted.index("--positive-multiplier") + 1] == expected_positive
    assert submitted[submitted.index("--ambiguity-multiplier") + 1] == expected_ambiguity
    assert submitted[submitted.index("--min-steps") + 1] == expected_min_steps


def test_continuation_launcher_rejects_non_numeric_max_train_seconds(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    hf_log = tmp_path / "hf.log"
    curl = bin_dir / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' 'V6_CONTINUATION_PROTOCOL = "
        '"retail-bank-peft-v6-generation-contract/v1"\'\n',
        encoding="utf-8",
    )
    curl.chmod(0o755)
    hf = bin_dir / "hf"
    hf.write_text('#!/usr/bin/env bash\nprintf \'called\\n\' > "$HF_LOG"\n', encoding="utf-8")
    hf.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HF_LOG": str(hf_log),
        "MAX_TRAIN_SECONDS": "30m",
    }

    completed = subprocess.run(
        [
            "bash",
            "scripts/retail_bank/run_remote_continuation_job.sh",
            "a" * 40,
            "b" * 40,
            "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "MAX_TRAIN_SECONDS" in completed.stderr
    assert not hf_log.exists()


def test_remote_training_launcher_forwards_v4_overrides(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    hf_log = tmp_path / "hf.log"
    curl = bin_dir / "curl"
    curl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    curl.chmod(0o755)
    hf = bin_dir / "hf"
    hf.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$HF_LOG"\n',
        encoding="utf-8",
    )
    hf.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HF_LOG": str(hf_log),
        "DATASET_REPO": "spkc83/retail-bank-servicing-alignment-sft",
        "BASE_MODEL": "spkc83/retail-bank-agent-9b",
        "BASE_REVISION": "c" * 40,
        "HF_HUB_DEST": "spkc83/retail-bank-servicing-agent-9b",
        "MAX_STEPS": "500",
        "LEARNING_RATE": "2e-5",
        "CHECKPOINT_EVERY": "100",
        "TRACKIO_PROJECT": "retail-bank-servicing-v4",
    }

    subprocess.run(
        [
            "bash",
            "scripts/retail_bank/run_remote_training_job.sh",
            "a" * 40,
            "b" * 40,
        ],
        check=True,
        env=env,
    )

    submitted_args = hf_log.read_text(encoding="utf-8")
    for expected in (
        "spkc83/retail-bank-servicing-alignment-sft",
        "spkc83/retail-bank-agent-9b",
        "spkc83/retail-bank-servicing-agent-9b",
        "500",
        "2e-5",
        "100",
        "retail-bank-servicing-v4",
    ):
        assert expected in submitted_args


def test_post_training_evaluation_detaches_closed_trackio_callback() -> None:
    worker_source = Path("scripts/retail_bank/cloud_train_tool_sft.py").read_text(encoding="utf-8")

    assert "trainer.remove_callback(TrackioCallback)" in worker_source
    assert worker_source.index("trainer.remove_callback(TrackioCallback)") < (
        worker_source.index("eval_metrics = trainer.evaluate()")
    )
