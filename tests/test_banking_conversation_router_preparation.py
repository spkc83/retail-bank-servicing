from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def load_preparation_module() -> ModuleType:
    path = Path("scripts/retail_bank/prepare_conversation_router_data.py")
    spec = importlib.util.spec_from_file_location(
        "banking_conversation_router_preparation",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_sft_records_verifies_manifest_split_digests(tmp_path: Path) -> None:
    preparation = load_preparation_module()
    row = {"record_id": "a", "messages": [{"role": "user", "content": "hi"}]}
    payload = json.dumps(row, sort_keys=True) + "\n"
    split_path = tmp_path / "train.jsonl"
    split_path.write_text(payload, encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text(payload, encoding="utf-8")
    (tmp_path / "test.jsonl").write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    shadow_row = {
        "record_id": "shadow-a",
        "messages": [{"role": "user", "content": "that one"}],
        "metadata": {"trainable": False},
    }
    shadow_payload = json.dumps(shadow_row, sort_keys=True) + "\n"
    (tmp_path / "shadow.jsonl").write_text(shadow_payload, encoding="utf-8")
    shadow_digest = hashlib.sha256(shadow_payload.encode("utf-8")).hexdigest()
    manifest = {
        "contract": "banking-tool-sft-manifest",
        "tool_sft": [
            {"name": "train", "path": "train.jsonl", "sha256": digest},
            {"name": "validation", "path": "validation.jsonl", "sha256": digest},
            {"name": "test", "path": "test.jsonl", "sha256": digest},
        ],
        "behavioral_gates": [
            {
                "name": "coreference-shadow",
                "path": "shadow.jsonl",
                "sha256": shadow_digest,
                "record_count": 1,
                "trainable": False,
                "allowed_use": ["post-selection-evaluation-once"],
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded_manifest, records = preparation.load_sft_records(tmp_path)

    assert loaded_manifest == manifest
    assert records["train"] == [row]
    assert records["test"] == [row, shadow_row]
    split_path.write_text(payload + payload, encoding="utf-8")
    with pytest.raises(ValueError, match="SFT train digest mismatch"):
        preparation.load_sft_records(tmp_path)


def test_release_digest_check_is_optional_until_v4_lock_exists() -> None:
    preparation = load_preparation_module()
    entries = [
        {"name": "train", "sha256": "a" * 64},
        {"name": "validation", "sha256": "b" * 64},
        {"name": "test", "sha256": "c" * 64},
    ]

    preparation.verify_release_split_digests(
        entries,
        {
            "prepared_split_sha256": {
                "train": "a" * 64,
                "validation": "b" * 64,
                "test": "c" * 64,
            }
        },
    )
    with pytest.raises(ValueError, match="validation split digest drift"):
        preparation.verify_release_split_digests(
            entries,
            {
                "prepared_split_sha256": {
                    "train": "a" * 64,
                    "validation": "x" * 64,
                    "test": "c" * 64,
                }
            },
        )


def test_existing_release_lock_is_not_rewritten_by_reproducibility_run(
    tmp_path: Path,
) -> None:
    preparation = load_preparation_module()
    lock_path = tmp_path / "release.lock.json"
    lock_path.write_text('{"created_at":"stable"}\n', encoding="utf-8")

    assert not preparation.should_write_source_lock(
        source_lock=lock_path,
        expected_release_lock=lock_path,
        expected_release_lock_contents=preparation.read_json(lock_path),
    )
    assert preparation.should_write_source_lock(
        source_lock=tmp_path / "candidate.lock.json",
        expected_release_lock=lock_path,
        expected_release_lock_contents=preparation.read_json(lock_path),
    )


def test_reproducibility_run_reuses_release_timestamp() -> None:
    preparation = load_preparation_module()

    assert (
        preparation.release_created_at({"created_at": "2026-07-31T07:00:09+00:00"})
        == "2026-07-31T07:00:09+00:00"
    )
    with pytest.raises(ValueError, match="valid created_at"):
        preparation.release_created_at({})


def test_v3_manifest_schema_documents_hierarchy_and_legacy_domain_label() -> None:
    preparation = load_preparation_module()

    manifest = preparation.build_manifest(
        sft_dir=Path("data/sft"),
        sft_manifest={"tool_sft": []},
        split_entries=[],
        report={"split_counts": {"train": 0, "validation": 0, "test": 0}},
        seed=7404,
        created_at="2026-08-15T00:00:00+00:00",
    )

    assert manifest["format_version"] == 3
    assert manifest["schema"]["domain_label"].startswith("legacy binary")
    assert manifest["schema"]["domain_name"] == "name in domain_labels"
    assert manifest["schema"]["action_name"] == "name in action_labels"
    assert manifest["schema"]["entity_resolution_name"] == ("name in entity_resolution_labels")
    assert manifest["domain_labels"] == ["out_of_domain", "banking", "social"]
    assert manifest["action_labels"] == [
        "refuse_ood",
        "execute_tool",
        "clarify",
        "retrieve_policy",
        "converse",
    ]
