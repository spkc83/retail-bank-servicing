"""Every committed digest is recomputed from the bytes it claims to describe.

The manifests and locks under ``data/`` are the repository's provenance: they
are what a reader checks a corpus against, what the release pipeline pins, and
what `check_corpora_reproduce.py` compares a rebuild to. Nothing recomputed
them. A digest that describes no file, or describes a file that has since
moved, is indistinguishable from a correct one until someone tries to use it.

These tests are deliberately blunt. They do not rebuild anything and they do
not care how a corpus was produced; they read the recorded digest, hash the
bytes on disk, and compare. That is cheap enough to run on every commit and it
is the check that would have caught a manifest going stale under an edit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

DATA_ROOT = Path("data")

#: Manifest keys holding ``{path, sha256}`` split entries. The value is a list
#: in most corpora and a name-keyed dict in the counterfactual eval one, so both
#: shapes are read rather than one being quietly skipped.
SPLIT_LIST_KEYS = ("splits", "tool_sft", "behavioral_gates", "evaluation_fixtures")

#: The one lock whose digests match nothing on disk, kept visible rather than
#: silenced. Two independent records agree against it: the files themselves, and
#: ``data/banking-counterfactual-eval-v1/manifest.json``, whose ``source_inputs``
#: pins the v4 train split at the digest the file actually has. The lock is the
#: stale artifact, not the corpus, and rewriting it would erase the only
#: evidence of that. ``test_the_stale_v4_lock_stays_visible`` asserts the
#: discrepancy so it cannot quietly change or quietly resolve.
STALE_LOCKS = {"banking-servicing-alignment-v4.lock.json"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_paths() -> list[Path]:
    return sorted(DATA_ROOT.glob("*/manifest.json"))


def _lock_paths() -> list[Path]:
    return sorted((DATA_ROOT / "sources").glob("*.lock.json"))


def _split_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key in SPLIT_LIST_KEYS:
        value = manifest.get(key)
        candidates: list[Any]
        if isinstance(value, list):
            candidates = value
        elif isinstance(value, dict):
            candidates = list(value.values())
        else:
            continue
        entries.extend(
            entry
            for entry in candidates
            if isinstance(entry, dict) and "sha256" in entry and "path" in entry
        )
    return entries


@pytest.mark.parametrize("manifest_path", _manifest_paths(), ids=lambda p: p.parent.name)
def test_manifest_digests_match_the_committed_bytes(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _split_entries(manifest)

    assert entries, f"{manifest_path} records no split digests to check"

    mismatches = []
    for entry in entries:
        target = manifest_path.parent / str(entry["path"])
        if not target.is_file():
            mismatches.append(f"{entry['path']}: recorded but missing from disk")
            continue
        actual = _sha256(target)
        if actual != entry["sha256"]:
            mismatches.append(f"{entry['path']}: recorded {entry['sha256']}, on disk {actual}")

    assert not mismatches, f"{manifest_path}\n  " + "\n  ".join(mismatches)


@pytest.mark.parametrize("manifest_path", _manifest_paths(), ids=lambda p: p.parent.name)
def test_manifest_record_counts_match_the_committed_lines(manifest_path: Path) -> None:
    """A digest proves the bytes; the count proves someone read them."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mismatches = []
    for entry in _split_entries(manifest):
        recorded = entry.get("record_count")
        target = manifest_path.parent / str(entry["path"])
        if recorded is None or not target.is_file():
            continue
        actual = sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip())
        if actual != recorded:
            mismatches.append(f"{entry['path']}: recorded {recorded} rows, on disk {actual}")

    assert not mismatches, f"{manifest_path}\n  " + "\n  ".join(mismatches)


@pytest.mark.parametrize("lock_path", _lock_paths(), ids=lambda p: p.name)
def test_lock_digests_match_the_corpus_they_pin(lock_path: Path) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    recorded = lock.get("prepared_split_sha256")
    if not isinstance(recorded, dict) or not recorded:
        pytest.skip(f"{lock_path.name} pins no prepared splits")

    corpus = DATA_ROOT / lock_path.name.removesuffix(".lock.json")
    if not corpus.is_dir():
        pytest.skip(f"{corpus.name} is not in the working tree; the lock outlives its corpus")
    if lock_path.name in STALE_LOCKS:
        pytest.skip(
            f"{lock_path.name} is a known stale lock; see test_the_stale_v4_lock_stays_visible"
        )

    mismatches = []
    for split, digest in sorted(recorded.items()):
        target = corpus / f"{split}.jsonl"
        if not target.is_file():
            mismatches.append(f"{split}: pinned but {target} is missing")
            continue
        actual = _sha256(target)
        if actual != digest:
            mismatches.append(f"{split}: pinned {digest}, on disk {actual}")

    assert not mismatches, f"{lock_path}\n  " + "\n  ".join(mismatches)


def test_the_stale_v4_lock_stays_visible() -> None:
    """Pin the one provenance defect rather than exempting it out of sight.

    If someone regenerates the v4 corpus or rewrites the lock, this test fails
    and the change has to be argued for. That is the point: a lock is a claim
    about what a release was built from, and quietly making the claim true
    again destroys the evidence that it once was not.
    """

    lock = json.loads(
        (DATA_ROOT / "sources" / "banking-servicing-alignment-v4.lock.json").read_text(
            encoding="utf-8"
        )
    )
    corpus = DATA_ROOT / "banking-servicing-alignment-v4"
    counterfactual = json.loads(
        (DATA_ROOT / "banking-counterfactual-eval-v1" / "manifest.json").read_text(encoding="utf-8")
    )

    pinned = lock["prepared_split_sha256"]
    on_disk = {split: _sha256(corpus / f"{split}.jsonl") for split in pinned}

    assert all(pinned[split] != on_disk[split] for split in pinned), (
        "the v4 lock now agrees with the corpus; if that was deliberate, delete this test"
    )

    # The independent record that decides which artifact is wrong.
    declared = counterfactual["source_inputs"]["data/banking-servicing-alignment-v4/train.jsonl"]
    assert declared["sha256"] == on_disk["train"], (
        "the counterfactual eval manifest no longer corroborates the corpus on disk, "
        "so the conclusion that the lock is the stale artifact needs revisiting"
    )


def test_every_corpus_directory_carries_a_manifest() -> None:
    """A corpus with no manifest has no provenance at all, which is worse."""

    without = sorted(
        directory.name
        for directory in DATA_ROOT.iterdir()
        if directory.is_dir()
        and directory.name != "sources"
        and any(directory.glob("*.jsonl"))
        and not (directory / "manifest.json").is_file()
    )

    assert without == [], f"corpora with jsonl but no manifest.json: {without}"
