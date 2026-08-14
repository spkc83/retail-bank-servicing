#!/usr/bin/env python
"""Prepare governed V5 conversation-router data from SFT records and CLINC OOD."""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hello_slm.banking_conversation_router_data import (
    CLINC_EXTERNAL_OOD_LABELS,
    INTENT_LABELS,
    RELATION_LABELS,
    ROUTER_SPLITS,
    build_conversation_router_splits,
    rows_jsonl_bytes,
    rows_sha256,
)
from hello_slm.config import file_sha256

CLINC_URL = "https://archive.ics.uci.edu/static/public/570/clinc150.zip"
CLINC_ZIP_SHA256 = "0d8ecc3e1edd7b25cabde0177544ce536ddf773844bc80ef1a75f36e7f030ea2"
CLINC_MEMBER = "clinc150_uci/data_oos_plus.json"
CLINC_MEMBER_SHA256 = "bfcca9ae515623541dc1983c94c4ed7cae9d26b42ae47d74b972e51bb6f7a21f"
DEFAULT_SFT_DIR = Path("data/banking-servicing-alignment-v5")
DEFAULT_OUTPUT_DIR = Path("data/banking-conversation-router-v5")
DEFAULT_RELEASE_LOCK = Path("data/sources/banking-conversation-router-v5.lock.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-dir", type=Path, default=DEFAULT_SFT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7404)
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=DEFAULT_RELEASE_LOCK,
        help="Tracked source and prepared-split lock written after validation.",
    )
    parser.add_argument(
        "--expected-release-lock",
        type=Path,
        default=DEFAULT_RELEASE_LOCK,
        help="Optional V5 lock whose prepared split digests should match when present.",
    )
    parser.add_argument(
        "--skip-release-digest-check",
        action="store_true",
        help="Allow new V5 splits before a release lock exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_release_lock = (
        read_json(args.expected_release_lock) if args.expected_release_lock.is_file() else None
    )
    sft_manifest, sft_records = load_sft_records(args.sft_dir)
    clinc_bytes = download_clinc_member()
    clinc_payload = json.loads(clinc_bytes)
    splits, report = build_conversation_router_splits(
        sft_records,
        clinc_payload,
        seed=args.seed,
    )
    if report["pii_matches"] != 0:
        raise ValueError(
            f"conversation router data contains {report['pii_matches']} PII-like matches"
        )
    if report["leakage"]["group_split_leak_count"] != 0:
        raise ValueError("conversation router data has group leakage across splits")
    if report["leakage"]["trajectory_split_leak_count"] != 0:
        raise ValueError("conversation router data has trajectory leakage across splits")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_entries = []
    for split in ROUTER_SPLITS:
        rows = splits[split]
        path = args.output_dir / f"{split}.jsonl"
        path.write_bytes(rows_jsonl_bytes(rows))
        split_entries.append(
            {
                "name": split,
                "path": path.name,
                "rows": len(rows),
                "sha256": rows_sha256(rows),
                "bytes": path.stat().st_size,
                "allowed_use": [
                    "conversation-router-training"
                    if split == "train"
                    else "conversation-router-evaluation"
                ],
            }
        )

    created_at = release_created_at(expected_release_lock)
    manifest = {
        "contract": "banking-conversation-router-data",
        "format_version": 2,
        "created_at": created_at,
        "seed": args.seed,
        "max_exchanges": 3,
        "schema": {
            "text": "cross-encoder input rendered by render_router_input",
            "domain_label": "1=in-domain banking/conversation, 0=external OOD",
            "intent_label": "index into intent_labels or -100",
            "lane": "deterministically derived from intent_label",
            "relation_labels": "multi-hot vector in relation_labels order",
        },
        "intent_labels": INTENT_LABELS,
        "relation_labels": RELATION_LABELS,
        "sources": {
            "sft": {
                "path": str(args.sft_dir),
                "manifest_sha256": file_sha256(args.sft_dir / "manifest.json"),
                "split_sha256": {
                    entry["name"]: entry["sha256"]
                    for entry in sft_manifest.get("tool_sft", [])
                    if isinstance(entry, dict)
                },
                "allowed_use": ["conversation-router-domain-intent-training"],
            },
            "UCI/clinc150": {
                "url": CLINC_URL,
                "archive_sha256": CLINC_ZIP_SHA256,
                "member": CLINC_MEMBER,
                "member_sha256": CLINC_MEMBER_SHA256,
                "used_labels": sorted(CLINC_EXTERNAL_OOD_LABELS),
                "allowed_use": ["conversation-router-external-ood-training"],
            },
        },
        "splits": split_entries,
        "report": report,
        "review_status": "automated-policy-pass",
        "signed": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_data_card(args.output_dir / "README.md", manifest)

    if not args.skip_release_digest_check and expected_release_lock is not None:
        verify_release_split_digests(split_entries, expected_release_lock)
    if should_write_source_lock(
        source_lock=args.source_lock,
        expected_release_lock=args.expected_release_lock,
        expected_release_lock_contents=expected_release_lock,
    ):
        args.source_lock.parent.mkdir(parents=True, exist_ok=True)
        write_source_lock(args.source_lock, manifest, clinc_bytes)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def load_sft_records(sft_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_path = sft_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("contract") != "banking-tool-sft-manifest":
        raise ValueError("unexpected SFT manifest contract")
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest.get("tool_sft", []):
        if not isinstance(entry, dict):
            continue
        split = str(entry["name"])
        if split not in ROUTER_SPLITS:
            continue
        path = sft_dir / str(entry["path"])
        if file_sha256(path) != str(entry["sha256"]):
            raise ValueError(f"SFT {split} digest mismatch")
        rows_by_split[split] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    missing = [split for split in ROUTER_SPLITS if split not in rows_by_split]
    if missing:
        raise ValueError(f"SFT manifest is missing splits: {missing}")
    return manifest, rows_by_split


def download_clinc_member() -> bytes:
    payload = download(CLINC_URL)
    if _bytes_sha256(payload) != CLINC_ZIP_SHA256:
        raise ValueError("CLINC150 archive digest mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = archive.read(CLINC_MEMBER)
    if _bytes_sha256(member) != CLINC_MEMBER_SHA256:
        raise ValueError("CLINC150 member digest mismatch")
    return member


def verify_release_split_digests(
    split_entries: list[dict[str, Any]],
    release_lock: dict[str, Any],
) -> None:
    expected = release_lock.get("prepared_split_sha256")
    if not isinstance(expected, dict):
        raise ValueError("release lock is missing prepared_split_sha256")
    actual = {str(entry["name"]): str(entry["sha256"]) for entry in split_entries}
    for split in ROUTER_SPLITS:
        if actual.get(split) != expected.get(split):
            raise ValueError(
                f"{split} split digest drift: expected {expected.get(split)}, "
                f"got {actual.get(split)}"
            )


def should_write_source_lock(
    *,
    source_lock: Path,
    expected_release_lock: Path,
    expected_release_lock_contents: dict[str, Any] | None,
) -> bool:
    return (
        expected_release_lock_contents is None
        or source_lock.resolve() != expected_release_lock.resolve()
    )


def release_created_at(release_lock: dict[str, Any] | None) -> str:
    if release_lock is not None:
        created_at = release_lock.get("created_at")
        if not isinstance(created_at, str) or not created_at.strip():
            raise ValueError("release lock is missing a valid created_at")
        return created_at
    return datetime.now(UTC).isoformat()


def write_data_card(path: Path, manifest: dict[str, Any]) -> None:
    counts = manifest["report"]["split_counts"]
    path.write_text(
        "\n".join(
            [
                "# Retail Bank Conversation Router V5 Data",
                "",
                "Governed cross-encoder data for a history-aware OOD, "
                "fine-intent, and relation classifier.",
                "",
                "Rows include only prior visible user/assistant messages "
                "and the current user message.",
                "They exclude current-turn tool plans, tool results, "
                "expected outputs, and final assistant responses.",
                "",
                f"- Train rows: {counts['train']}",
                f"- Validation rows: {counts['validation']}",
                f"- Test rows: {counts['test']}",
                f"- Intent labels: {', '.join(manifest['intent_labels'])}",
                f"- Relation labels: {', '.join(manifest['relation_labels'])}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_source_lock(path: Path, manifest: dict[str, Any], clinc_bytes: bytes) -> None:
    path.write_text(
        json.dumps(
            {
                "contract": "banking-conversation-router-v5-source-lock",
                "format_version": 2,
                "created_at": manifest["created_at"],
                "sources": manifest["sources"],
                "clinc_member_sha256": _bytes_sha256(clinc_bytes),
                "prepared_split_sha256": {
                    entry["name"]: entry["sha256"] for entry in manifest["splits"]
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "retail-bank-servicing/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bytes_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
