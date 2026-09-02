#!/usr/bin/env python
"""Canonical data, two-stage Granite SFT, router, evaluation, and deploy entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("configs/retail-bank-release.toml")
STAGE_ORDER = (
    "prepare-data",
    "base-sft",
    "alignment-sft",
    "router",
    "evaluate",
    "deploy",
)
EXTERNAL_STAGES = frozenset({"base-sft", "alignment-sft", "router", "evaluate", "deploy"})
PAID_STAGES = frozenset({"base-sft", "alignment-sft", "evaluate"})


class ReleasePipelineError(ValueError):
    """Raised when a release stage is ambiguous or insufficiently guarded."""


@dataclass(frozen=True)
class Stage:
    name: str
    purpose: str
    commands: tuple[tuple[str, ...], ...]
    environment: Mapping[str, str]
    paid: bool = False
    publishes: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "commands": [shlex.join(command) for command in self.commands],
            "environment": dict(sorted(self.environment.items())),
            "paid": self.paid,
            "publishes": self.publishes,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=(*STAGE_ORDER, "all"), default="all")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute one selected stage. Without this flag the command prints a plan.",
    )
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="Required with --execute for a paid Hugging Face Jobs stage.",
    )
    parser.add_argument(
        "--allow-publish",
        action="store_true",
        help="Required with --execute for any Hub/Space publishing stage.",
    )
    parser.add_argument("--source-commit")
    parser.add_argument("--base-dataset-revision")
    parser.add_argument("--alignment-dataset-revision")
    parser.add_argument("--base-revision")
    parser.add_argument("--model-revision")
    parser.add_argument("--router-data-revision")
    parser.add_argument("--router-revision")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReleasePipelineError(f"release config is unavailable: {path}")
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    for section in ("source", "granite_base", "granite_alignment", "router", "space"):
        if not isinstance(config.get(section), dict):
            raise ReleasePipelineError(f"release config is missing [{section}]")
    return config


def exact_revision(value: Any, *, field: str) -> str:
    revision = str(value or "")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ReleasePipelineError(f"{field} must be an exact 40-character lowercase revision")
    return revision


def build_stages(config: Mapping[str, Any], args: argparse.Namespace) -> tuple[Stage, ...]:
    source = _section(config, "source")
    base = _section(config, "granite_base")
    alignment = _section(config, "granite_alignment")
    router = _section(config, "router")
    peft = _section(config, "granite_peft")
    source_commit = exact_revision(
        args.source_commit or source["release_commit"],
        field="source commit",
    )

    base_dataset_revision = exact_revision(
        args.base_dataset_revision or base["dataset_revision"],
        field="base SFT dataset revision",
    )
    base_weights_revision = exact_revision(
        args.base_revision or base["weights_revision"],
        field="base-stage weights revision",
    )
    alignment_dataset_revision = exact_revision(
        args.alignment_dataset_revision or alignment["training_dataset_revision"],
        field="alignment dataset revision",
    )
    aligned_weights_revision = exact_revision(
        args.model_revision or alignment["weights_revision"],
        field="aligned model revision",
    )
    router_data_revision = exact_revision(
        args.router_data_revision or router["dataset_revision"],
        field="router dataset revision",
    )
    router_revision = exact_revision(
        args.router_revision or router["model_revision"],
        field="router model revision",
    )

    prepare = Stage(
        name="prepare-data",
        purpose="Generate and lock base tool SFT, v4 remediation SFT, and router data.",
        commands=(
            (
                sys.executable,
                "scripts/retail_bank/prepare_tool_sft_data.py",
                "--output-dir",
                "data/banking-v3-tool-sft",
                "--pilot-count",
                "9000",
                "--split-seed",
                "711",
            ),
            (sys.executable, "scripts/retail_bank/prepare_servicing_alignment_data.py"),
            (sys.executable, "scripts/retail_bank/prepare_conversation_router_data.py"),
        ),
        environment={"PYTHONPATH": "src"},
    )
    base_sft = Stage(
        name="base-sft",
        purpose="Initial tool-use/domain SFT from the pinned IBM Granite foundation checkpoint.",
        commands=(
            (
                "scripts/retail_bank/run_remote_training_job.sh",
                source_commit,
                base_dataset_revision,
            ),
        ),
        environment={
            "BASE_MODEL": str(base["model_repo"]),
            "BASE_REVISION": exact_revision(base["model_revision"], field="IBM base revision"),
            "DATASET_REPO": str(base["dataset_repo"]),
            "HF_HUB_DEST": str(base["output_repo"]),
            "MAX_STEPS": "3000",
            "LEARNING_RATE": "1e-4",
            "PROJECT_LABEL": "retail-bank-base-tool-sft",
        },
        paid=True,
        publishes=True,
    )
    alignment_sft = Stage(
        name="alignment-sft",
        purpose=(
            "Remediation SFT for observed multi-turn, tool-use, agent-repair, "
            "and topic-shift failures."
        ),
        commands=(
            (
                "scripts/retail_bank/run_remote_training_job.sh",
                source_commit,
                alignment_dataset_revision,
            ),
        ),
        environment={
            "BASE_MODEL": str(base["output_repo"]),
            "BASE_REVISION": base_weights_revision,
            "DATASET_REPO": str(alignment["dataset_repo"]),
            "HF_HUB_DEST": str(alignment["output_repo"]),
            "MAX_STEPS": "500",
            "LEARNING_RATE": "2e-5",
            "CHECKPOINT_EVERY": "100",
            "PROJECT_LABEL": "retail-bank-servicing-v4",
        },
        paid=True,
        publishes=True,
    )
    router_stage = Stage(
        name="router",
        purpose="Train and publish the history-aware domain/capability/relation cross-encoder.",
        commands=(
            (
                sys.executable,
                "scripts/retail_bank/train_conversation_router.py",
                "--data-revision",
                router_data_revision,
                "--publish",
                "--destination-id",
                str(router["model_repo"]),
            ),
        ),
        environment={"PYTHONPATH": "src", "SOURCE_COMMIT": source_commit},
        publishes=True,
    )
    evaluate = Stage(
        name="evaluate",
        purpose="Generate and enforce the exact frozen tool/conversation release gates.",
        commands=(
            (
                "scripts/retail_bank/run_remote_tool_eval_job.sh",
                source_commit,
                aligned_weights_revision,
                exact_revision(
                    alignment["scoring_dataset_revision"],
                    field="scoring dataset revision",
                ),
            ),
        ),
        environment={
            "MODEL_REPO": str(alignment["output_repo"]),
            "DATASET_REPO": str(alignment["dataset_repo"]),
        },
        paid=True,
        publishes=True,
    )
    # Every identity is stated. The stage used to name only the model and the
    # router and let the rest default, which paired a --model-id from the
    # merged-weights lineage with the deploy script's hard-coded v8 adapter and
    # an empty subfolder -- a silent two-generation rollback of the live demo,
    # and a composition that never existed in any release.
    adapter_revision = exact_revision(
        str(peft["adapter_revision"]), field="deployed adapter revision"
    )
    deploy = Stage(
        name="deploy",
        purpose="Deploy the gated model and router source to the public ZeroGPU Space.",
        commands=(
            (
                sys.executable,
                "scripts/retail_bank/deploy_zero_gpu_space.py",
                "--space-id",
                str(_section(config, "space")["repo"]),
                "--model-id",
                str(peft["adapter_repo"]),
                "--model-revision",
                adapter_revision,
                "--base-model-id",
                str(peft["base_model_repo"]),
                "--base-model-revision",
                exact_revision(
                    str(peft["base_model_revision"]), field="deployed base model revision"
                ),
                "--adapter-id",
                str(peft["adapter_repo"]),
                "--adapter-revision",
                adapter_revision,
                "--adapter-subfolder",
                str(peft["adapter_subfolder"]),
                "--model-dtype",
                str(peft["model_dtype"]),
                "--router-id",
                str(router["model_repo"]),
                "--router-revision",
                router_revision,
                "--best-of-n",
                str(peft["best_of_n"]),
                "--execute",
                "--allow-publish",
            ),
        ),
        environment={},
        publishes=True,
    )
    return (prepare, base_sft, alignment_sft, router_stage, evaluate, deploy)


def select_stages(stages: Sequence[Stage], selected: str) -> tuple[Stage, ...]:
    if selected == "all":
        return tuple(stages)
    return tuple(stage for stage in stages if stage.name == selected)


def execute_stage(stage: Stage, *, allow_paid: bool, allow_publish: bool) -> None:
    if stage.paid and not allow_paid:
        raise ReleasePipelineError(f"{stage.name} requires --allow-paid")
    if stage.publishes and not allow_publish:
        raise ReleasePipelineError(f"{stage.name} requires --allow-publish")
    environment = {**os.environ, **stage.environment}
    for command in stage.commands:
        subprocess.run(command, check=True, env=environment)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ReleasePipelineError(f"release config is missing [{name}]")
    return section


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    selected = select_stages(build_stages(config, args), args.stage)
    if args.execute:
        if args.stage == "all":
            raise ReleasePipelineError(
                "execute one stage at a time so newly published revisions can be pinned downstream"
            )
        execute_stage(
            selected[0],
            allow_paid=args.allow_paid,
            allow_publish=args.allow_publish,
        )
    print(
        json.dumps(
            {
                "contract": "retail-bank-release-pipeline/v1",
                "mode": "executed" if args.execute else "plan",
                "stage_order": list(STAGE_ORDER),
                "selected": [stage.as_dict() for stage in selected],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleasePipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
