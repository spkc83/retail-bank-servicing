from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")

CANONICAL_DOCS = (
    "docs/README.md",
    "docs/01-system-overview.md",
    "docs/02-data-generation.md",
    "docs/03-model-and-peft.md",
    "docs/04-training-and-recovery.md",
    "docs/05-hierarchical-router.md",
    "docs/06-evaluation.md",
    "docs/07-inference-and-poc.md",
    "docs/08-end-to-end-runbook.md",
    "docs/09-conversation-router-v4.md",
    "docs/10-servicing-alignment-v4.md",
    "docs/reference/file-map.md",
    "docs/reference/artifacts.md",
)

OBSOLETE_FILES = (
    "src/hello_slm/banking_moe.py",
    "src/hello_slm/banking_chat_runtime.py",
    "src/hello_slm/banking_hf_generator.py",
    "src/hello_slm/banking_policy.py",
    "src/hello_slm/banking_shiny_app.py",
    "scripts/retail_bank/cloud_train_banking_moe.py",
    "scripts/retail_bank/train_banking_moe.py",
    "configs/banking-v2-moe-9b.toml",
    "configs/banking-v2-dense-adapter.toml",
    "model_cards/retail-bank-servicing-moe-9b.md",
)

OBSOLETE_FILE_TREES = ("specs", ".github/workflows", "scripts/banking_v2")

FORBIDDEN_HISTORY = (
    "custom Qwen2-MoE",
    "dense-to-MoE",
    "retail-bank-servicing-moe-9b",
    "banking_shiny_app",
    "arithmetic-30m",
)


def test_canonical_junior_documentation_exists() -> None:
    missing = [path for path in CANONICAL_DOCS if not (ROOT / path).is_file()]
    assert missing == []


def test_obsolete_implementation_and_specs_are_absent() -> None:
    present = [path for path in OBSOLETE_FILES if (ROOT / path).exists()]
    for tree in OBSOLETE_FILE_TREES:
        root = ROOT / tree
        if root.exists():
            present.extend(
                str(path.relative_to(ROOT))
                for path in root.rglob("*")
                if path.is_file()
            )
    assert present == []


def test_relative_markdown_links_resolve() -> None:
    failures: list[str] = []
    markdown_paths = [
        ROOT / "README.md",
        *(ROOT / path for path in CANONICAL_DOCS),
        *sorted((ROOT / "data_cards").glob("*.md")),
        *sorted((ROOT / "model_cards").glob("*.md")),
        ROOT / "poc/retail-bank-customer-service-poc/README.md",
    ]
    for markdown in markdown_paths:
        text = markdown.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if (
                not target
                or target.startswith(("#", "http://", "https://", "mailto:"))
            ):
                continue
            relative = target.split("#", maxsplit=1)[0]
            if not relative:
                continue
            resolved = (markdown.parent / relative).resolve()
            if not resolved.exists():
                failures.append(
                    f"{markdown.relative_to(ROOT)} -> {target}"
                )
    assert failures == []


def test_canonical_docs_do_not_reference_retired_architectures() -> None:
    markdown_paths = [
        ROOT / "README.md",
        *(ROOT / path for path in CANONICAL_DOCS),
        *sorted((ROOT / "data_cards").glob("*.md")),
        *sorted((ROOT / "model_cards").glob("*.md")),
        ROOT / "poc/retail-bank-customer-service-poc/README.md",
    ]
    failures: list[str] = []
    for path in markdown_paths:
        text = path.read_text(encoding="utf-8")
        for phrase in FORBIDDEN_HISTORY:
            if phrase.casefold() in text.casefold():
                failures.append(f"{path.relative_to(ROOT)}: {phrase}")
    assert failures == []
