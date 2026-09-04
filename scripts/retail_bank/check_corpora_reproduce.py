#!/usr/bin/env python
"""Assert the documented regeneration commands reproduce the committed corpora.

The corpora are generated, not collected, so the commands in
``docs/02-data-generation.md`` are the only record of how. Nothing checked that
they still worked: the alignment command had lost its prompt-realization flags
and no longer reproduced the committed files at all, and the prompt teacher it
credits (``claude-opus-4-8``) is stamped into 256 rows' provenance, so passing
the wrong one silently rewrites authorship and moves every digest.

This reads the commands **out of the documentation** rather than keeping a copy
of them, so the thing under test is the instruction a person would actually
follow -- every ``prepare_*`` command the page documents, not a fixed list that
would itself drift. Each is run into a scratch directory and every ``.jsonl``
is compared by SHA-256 against the committed file.

usage: check_corpora_reproduce.py [--docs docs/02-data-generation.md]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Corpora that are deliberately NOT expected to reproduce from HEAD, each with
#: the reason. This is not a place to silence an inconvenient failure: an entry
#: means the committed files are a frozen release artifact pinned to a deployed
#: model, so HEAD moving away from them is the intended state rather than a
#: regression. Anything not listed here must reproduce exactly.
FROZEN_RELEASE_ARTIFACTS = {
    "banking-conversation-router-v8-first-turn-mutation": (
        "Frozen at 0ebbd73 (2026-08-20) and pinned to the deployed router "
        "spkc83/retail-bank-conversation-router@dd5ea266. It derives from "
        "data/banking-servicing-alignment-v5, which has moved repeatedly since "
        "-- the coreference phrase families on 2026-08-21 and the prompt "
        "realization passes on 2026-08-31 -- so a rebuild no longer matches its "
        "release lock. The corpus HEAD builds is "
        "data/banking-conversation-router-v9-surface-form, documented in docs/02 "
        "and checked here like any other; docs/08 section 4 records why the two "
        "differ and why the deployed router is still the v8 artifact."
    ),
}
#: The fenced block in the docs that holds both regeneration commands.
COMMAND_PATTERN = re.compile(
    r"PYTHONPATH=src uv run python (scripts/retail_bank/prepare_\w+\.py[^`]*?)(?=\n\n|\n```)",
    re.DOTALL,
)


def documented_commands(docs: Path) -> list[list[str]]:
    text = docs.read_text(encoding="utf-8")
    commands = []
    for match in COMMAND_PATTERN.finditer(text):
        commands.append(shlex.split(match.group(1).replace("\\\n", " ")))
    if not commands:
        raise SystemExit(f"no documented regeneration commands found in {docs}")
    return commands


def redirect_output(command: list[str], destination: Path) -> tuple[list[str], Path]:
    """Point the command at a scratch directory, returning the committed one."""
    try:
        index = command.index("--output-dir")
    except ValueError as error:
        raise SystemExit(f"documented command has no --output-dir: {command}") from error
    committed = REPO_ROOT / command[index + 1]
    rewritten = list(command)
    rewritten[index + 1] = str(destination)
    return rewritten, committed


def digests(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.jsonl"))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=REPO_ROOT / "docs/02-data-generation.md")
    args = parser.parse_args()

    failures: list[str] = []
    for command in documented_commands(args.docs):
        with tempfile.TemporaryDirectory(prefix="corpus-check-") as scratch:
            rewritten, committed = redirect_output(command, Path(scratch))
            print(f"--- regenerating {committed.name} from the documented command", flush=True)
            result = subprocess.run(
                [sys.executable, *rewritten],
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONPATH": "src"},
                capture_output=True,
                text=True,
            )
            frozen = FROZEN_RELEASE_ARTIFACTS.get(committed.name)
            if frozen is not None:
                verb = "still does not" if result.returncode != 0 else "now does"
                print(f"    frozen release artifact; {verb} rebuild from HEAD", flush=True)
                print(f"    reason: {frozen}", flush=True)
                continue
            if result.returncode != 0:
                failures.append(
                    f"{committed.name}: the documented command failed\n{result.stderr[-2000:]}"
                )
                continue

            regenerated, existing = digests(Path(scratch)), digests(committed)
            if not existing:
                failures.append(f"{committed}: no committed splits to compare against")
                continue
            for name, digest in sorted(existing.items()):
                if name not in regenerated:
                    failures.append(f"{committed.name}/{name}: not produced by the command")
                elif regenerated[name] != digest:
                    failures.append(
                        f"{committed.name}/{name}: committed {digest[:16]}, "
                        f"regenerated {regenerated[name][:16]}"
                    )
            print(f"    {len(existing)} splits compared", flush=True)

    if failures:
        print("\nThe documented commands no longer reproduce the committed corpora:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nEvery committed split reproduces from its documented command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
