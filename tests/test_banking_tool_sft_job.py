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
    if launcher == "run_remote_training_job.sh":
        # The from-scratch lane refuses to run without an explicit, distinct destination.
        env["HF_HUB_DEST"] = "spkc83/retail-bank-servicing-agent-9b-peft-v9-scratch"
        # ...and, since the spend gate landed, without a priced authorisation
        # under the cost ceiling. The default 5h timeout is worth $13.75.
        env["JOB_TIMEOUT"] = "45m"
        env["CONFIRM_SPEND"] = "1"

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


def test_continuation_publish_recovery_targets_the_training_commit_output_dir(
    tmp_path: Path,
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
        "PUBLISH_ONLY": "1",
        "RECOVERY_SOURCE_COMMIT": "c" * 40,
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

    submitted = hf_log.read_text(encoding="utf-8").splitlines()
    assert "--publish-only" in submitted
    assert submitted[submitted.index("--recovery-source-commit") + 1] == "c" * 40
    output_dir = submitted[submitted.index("--output-dir") + 1]
    assert output_dir.startswith("/data/retail-bank-agent-9b-peft-v6-generation-contract-cccccccc-")
    assert output_dir.endswith("-d965816b-bbbbbbbb")


def test_continuation_launcher_rejects_recovery_commit_outside_publish_only(
    tmp_path: Path,
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
    hf.write_text("#!/usr/bin/env bash\nprintf 'called\\n' > \"$HF_LOG\"\n", encoding="utf-8")
    hf.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HF_LOG": str(hf_log),
        "RECOVERY_SOURCE_COMMIT": "c" * 40,
    }
    env.pop("PUBLISH_ONLY", None)

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
    assert "RECOVERY_SOURCE_COMMIT" in completed.stderr
    assert not hf_log.exists()


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
    hf.write_text("#!/usr/bin/env bash\nprintf 'called\\n' > \"$HF_LOG\"\n", encoding="utf-8")
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
        # Forwarding test, not a spend-gate test: authorise and lift the ceiling.
        "CONFIRM_SPEND": "1",
        "MAX_JOB_COST_USD": "1000",
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


def _training_launcher_harness(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    hf_log = tmp_path / "hf.log"
    curl = bin_dir / "curl"
    curl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    curl.chmod(0o755)
    hf = bin_dir / "hf"
    hf.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$HF_LOG"\n', encoding="utf-8")
    hf.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HF_LOG": str(hf_log),
    }
    for name in (
        "JOB_TIMEOUT",
        "SKIP_MERGE_ADAPTER",
        "HF_HUB_DEST",
        "BASE_MODEL",
        "POSITIVE_MULTIPLIER",
        "AMBIGUITY_MULTIPLIER",
        "POLICY_FAQ_MULTIPLIER",
        "TOOL_OUTCOME_MULTIPLIER",
    ):
        env.pop(name, None)
    # These tests are about argument forwarding, not about the spend gate, so
    # they authorise the run and lift the ceiling. The gate's own behaviour is
    # covered by the dedicated tests at the end of this module.
    env["CONFIRM_SPEND"] = "1"
    env["MAX_JOB_COST_USD"] = "1000"
    return env, hf_log


def _run_training_launcher(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "scripts/retail_bank/run_remote_training_job.sh",
            "a" * 40,
            "b" * 40,
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("environment", "expected_timeout"),
    [({"JOB_TIMEOUT": "80m"}, "80m"), ({}, "5h")],
)
def test_training_launcher_forwards_job_timeout(
    tmp_path: Path,
    environment: dict[str, str],
    expected_timeout: str,
) -> None:
    env, hf_log = _training_launcher_harness(tmp_path)
    env["HF_HUB_DEST"] = "spkc83/retail-bank-servicing-agent-9b-peft-v9-scratch"
    env.update(environment)

    completed = _run_training_launcher(env)

    assert completed.returncode == 0, completed.stderr
    submitted = hf_log.read_text(encoding="utf-8").splitlines()
    assert submitted[submitted.index("--timeout") + 1] == expected_timeout


def test_training_launcher_rejects_a_malformed_job_timeout(tmp_path: Path) -> None:
    env, hf_log = _training_launcher_harness(tmp_path)
    env["HF_HUB_DEST"] = "spkc83/retail-bank-servicing-agent-9b-peft-v9-scratch"
    env["JOB_TIMEOUT"] = "80 minutes"

    completed = _run_training_launcher(env)

    assert completed.returncode == 2
    assert "JOB_TIMEOUT" in completed.stderr
    assert not hf_log.exists()


@pytest.mark.parametrize(
    ("environment", "expected_flag"),
    [({"SKIP_MERGE_ADAPTER": "1"}, True), ({}, False)],
)
def test_training_launcher_forwards_skip_merge_adapter(
    tmp_path: Path,
    environment: dict[str, str],
    expected_flag: bool,
) -> None:
    env, hf_log = _training_launcher_harness(tmp_path)
    env["HF_HUB_DEST"] = "spkc83/retail-bank-servicing-agent-9b-peft-v9-scratch"
    env.update(environment)

    completed = _run_training_launcher(env)

    assert completed.returncode == 0, completed.stderr
    submitted = hf_log.read_text(encoding="utf-8").splitlines()
    assert ("--skip-merge-adapter" in submitted) is expected_flag


@pytest.mark.parametrize(
    "environment",
    [
        pytest.param({}, id="missing"),
        pytest.param(
            {"HF_HUB_DEST": "spkc83/retail-bank-servicing-agent-9b"},
            id="equal-to-default-base",
        ),
        pytest.param(
            {
                "HF_HUB_DEST": "spkc83/some-other-base",
                "BASE_MODEL": "spkc83/some-other-base",
            },
            id="equal-to-overridden-base",
        ),
    ],
)
def test_training_launcher_requires_a_distinct_explicit_hub_destination(
    tmp_path: Path,
    environment: dict[str, str],
) -> None:
    env, hf_log = _training_launcher_harness(tmp_path)
    env.update(environment)

    completed = _run_training_launcher(env)

    assert completed.returncode == 2
    assert "HF_HUB_DEST" in completed.stderr
    assert not hf_log.exists()


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (
            {
                "POSITIVE_MULTIPLIER": "3",
                "AMBIGUITY_MULTIPLIER": "6",
                "POLICY_FAQ_MULTIPLIER": "4",
                "TOOL_OUTCOME_MULTIPLIER": "6",
            },
            ("3", "6", "4", "6"),
        ),
        ({}, ("1", "1", "1", "1")),
    ],
)
def test_training_launcher_forwards_mix_multipliers(
    tmp_path: Path,
    environment: dict[str, str],
    expected: tuple[str, str, str, str],
) -> None:
    env, hf_log = _training_launcher_harness(tmp_path)
    env["HF_HUB_DEST"] = "spkc83/retail-bank-servicing-agent-9b-peft-v9-scratch"
    env.update(environment)

    completed = _run_training_launcher(env)

    assert completed.returncode == 0, completed.stderr
    submitted = hf_log.read_text(encoding="utf-8").splitlines()
    assert (
        submitted[submitted.index("--positive-multiplier") + 1],
        submitted[submitted.index("--ambiguity-multiplier") + 1],
        submitted[submitted.index("--policy-faq-multiplier") + 1],
        submitted[submitted.index("--tool-outcome-multiplier") + 1],
    ) == expected


def test_training_launcher_rejects_a_non_numeric_multiplier(tmp_path: Path) -> None:
    env, hf_log = _training_launcher_harness(tmp_path)
    env["HF_HUB_DEST"] = "spkc83/retail-bank-servicing-agent-9b-peft-v9-scratch"
    env["AMBIGUITY_MULTIPLIER"] = "six"

    completed = _run_training_launcher(env)

    assert completed.returncode == 2
    assert "MULTIPLIER" in completed.stderr
    assert not hf_log.exists()


def test_bootstrap_forwards_guard_flags_to_the_worker(tmp_path: Path) -> None:
    job = _load_job()
    args = job.parse_args_from(
        [
            "--source-commit",
            "a" * 40,
            "--dataset-revision",
            "b" * 40,
            "--hub-dest",
            "spkc83/retail-bank-servicing-agent-9b-peft-v9-scratch",
            "--skip-merge-adapter",
            "--positive-multiplier",
            "3",
            "--ambiguity-multiplier",
            "6",
            "--policy-faq-multiplier",
            "4",
            "--tool-outcome-multiplier",
            "6",
        ]
    )

    command = job.build_worker_command(
        args,
        source_root=tmp_path,
        manifest=tmp_path / "manifest.json",
    )

    assert "--skip-merge-adapter" in command
    assert command[command.index("--positive-multiplier") + 1] == "3"
    assert command[command.index("--ambiguity-multiplier") + 1] == "6"
    assert command[command.index("--policy-faq-multiplier") + 1] == "4"
    assert command[command.index("--tool-outcome-multiplier") + 1] == "6"
    assert command[command.index("--hub-dest") + 1] == (
        "spkc83/retail-bank-servicing-agent-9b-peft-v9-scratch"
    )


def test_bootstrap_defaults_keep_merging_and_an_unweighted_mix(tmp_path: Path) -> None:
    job = _load_job()
    args = job.parse_args_from(
        [
            "--source-commit",
            "a" * 40,
            "--dataset-revision",
            "b" * 40,
            "--hub-dest",
            "spkc83/retail-bank-servicing-agent-9b-peft-v9-scratch",
        ]
    )

    command = job.build_worker_command(
        args,
        source_root=tmp_path,
        manifest=tmp_path / "manifest.json",
    )

    assert "--skip-merge-adapter" not in command
    for flag in (
        "--positive-multiplier",
        "--ambiguity-multiplier",
        "--policy-faq-multiplier",
        "--tool-outcome-multiplier",
    ):
        assert command[command.index(flag) + 1] == "1"


def test_bootstrap_refuses_publishing_over_the_training_base() -> None:
    job = _load_job()
    args = job.parse_args_from(
        [
            "--source-commit",
            "a" * 40,
            "--dataset-revision",
            "b" * 40,
            "--hub-dest",
            "spkc83/retail-bank-servicing-agent-9b",
        ]
    )

    with pytest.raises(ValueError, match="must differ from the training base model"):
        job.validate_arguments(args)


def test_bootstrap_rejects_out_of_range_multipliers() -> None:
    job = _load_job()
    args = job.parse_args_from(
        [
            "--source-commit",
            "a" * 40,
            "--dataset-revision",
            "b" * 40,
            "--hub-dest",
            "spkc83/retail-bank-servicing-agent-9b-peft-v9-scratch",
            "--ambiguity-multiplier",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="multiplier"):
        job.validate_arguments(args)


LAUNCHER = Path("scripts/retail_bank/run_remote_training_job.sh")


def _run_launcher(tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    """Invoke the launcher with DRY_RUN so no test can ever submit a job.

    `curl` is stubbed because the bootstrap-URL check runs before the dry-run
    exit and a placeholder commit legitimately 404s. `hf` is stubbed too, as a
    belt-and-braces guarantee that a regression in DRY_RUN handling cannot
    reach the real submission -- this suite launched a billable job once.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("curl", "hf"):
        stub = bin_dir / name
        stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
    base = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DRY_RUN": "1",
        "HF_HUB_DEST": "spkc83/launcher-gate-test",
        **env,
    }
    return subprocess.run(
        ["bash", str(LAUNCHER), "0" * 40, "1" * 40],
        env=base,
        capture_output=True,
        text=True,
    )


def test_launcher_refuses_a_timeout_whose_worst_case_exceeds_the_ceiling(tmp_path: Path) -> None:
    """A mistyped timeout was a four-figure mistake nothing on this side caught."""
    result = _run_launcher(tmp_path, JOB_TIMEOUT="999h", CONFIRM_SPEND="1")

    assert result.returncode == 2
    assert "2747.25" in result.stderr
    assert "exceeds MAX_JOB_COST_USD" in result.stderr


def test_launcher_refuses_without_explicit_spend_confirmation(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path, JOB_TIMEOUT="45m")

    assert result.returncode == 2
    assert "CONFIRM_SPEND=1" in result.stderr
    assert "2.06" in result.stderr, "the price must be shown even when refusing"


def test_launcher_prices_the_run_before_submitting_anything(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path, JOB_TIMEOUT="45m", CONFIRM_SPEND="1")

    assert result.returncode == 0
    assert "Worst case if it runs to the timeout: $2.06" in result.stderr
    assert "not submitting" in result.stderr


def test_the_five_hour_default_cannot_launch_silently(tmp_path: Path) -> None:
    """The default timeout is worth $13.75; it must require a deliberate act."""
    result = _run_launcher(tmp_path, CONFIRM_SPEND="1")

    assert result.returncode == 2
    assert "13.75" in result.stderr
