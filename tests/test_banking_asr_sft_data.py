from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from hello_slm.banking_asr_sft_data import AsrSftDataError, prepare_asr_sft_data


def _write_base_dataset(root: Path) -> Path:
    root.mkdir(parents=True)
    entries = []
    for split in ("train", "validation", "test"):
        row = {
            "schema_version": "banking-tool-sft/v1",
            "record_id": f"source-{split}",
            "messages": [
                {"role": "system", "content": "system", "loss": False},
                {"role": "user", "content": f"Show my {split} accounts", "loss": False},
                {
                    "role": "assistant",
                    "content": None,
                    "loss": True,
                    "tool_calls": [
                        {
                            "id": f"call-{split}",
                            "index": 0,
                            "type": "function",
                            "function": {"name": "list_accounts", "arguments": {}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{split}",
                    "name": "list_accounts",
                    "content": {"ok": True, "result": {"accounts": []}},
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": "You do not have any accounts in this synthetic profile.",
                    "loss": True,
                },
            ],
            "expected": {
                "path": "tool",
                "ordered_calls": [f"call-{split}"],
                "tool_calls": [{"name": "list_accounts", "arguments": {}}],
            },
            "split_keys": {"source": split},
            "provenance": {"source": "self-authored-synthetic"},
            "validation": {"accepted": True},
            "metadata": {
                "split": split,
                "split_group": f"customer-session-{split}",
                "trainable": True,
            },
        }
        path = root / f"{split}.jsonl"
        path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        payload = path.read_bytes()
        entries.append(
            {
                "name": split,
                "path": path.name,
                "record_count": 1,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    manifest = {
        "contract": "banking-tool-sft-manifest",
        "schema_version": "banking-tool-sft/v1",
        "tool_sft": entries,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _asr_row(source_record_id: str, transcript: str) -> dict[str, Any]:
    return {
        "record_id": f"asr-{source_record_id}",
        "source_record_id": source_record_id,
        "utterance_id": f"utt-{source_record_id}",
        "recording_id": f"recording-{source_record_id}",
        "speaker": "customer",
        "transcript": transcript,
        "start_ms": 1250,
        "end_ms": 3680,
        "language": "en-US",
        "confidence": 0.87,
        "alternatives": [{"text": transcript + " please", "confidence": 0.11}],
        "asr_model_id": "example/asr-small",
        "asr_model_revision": "0123456789abcdef",
        "audio_sha256": "a" * 64,
        "review": {
            "semantic_match": True,
            "pii_reviewed": True,
            "consent_for_training": True,
            "reviewer": "synthetic-data-review",
            "license": "MIT",
        },
    }


def test_asr_overlay_preserves_targets_and_inherits_source_split(tmp_path: Path) -> None:
    manifest = _write_base_dataset(tmp_path / "base")
    input_path = tmp_path / "asr.jsonl"
    rows = [
        _asr_row("source-train", "show me my train uh accounts"),
        _asr_row("source-validation", "show my validation accounts"),
        _asr_row("source-test", "show my test accounts please"),
    ]
    input_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    result = prepare_asr_sft_data(
        asr_input=input_path,
        base_manifest=manifest,
        output_dir=output,
    )

    assert result["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    generated = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    source = json.loads((manifest.parent / "train.jsonl").read_text(encoding="utf-8"))
    assert [m for m in generated["messages"] if m["role"] == "user"][-1][
        "content"
    ] == "show me my train uh accounts"
    assert generated["messages"][-3:-1] == source["messages"][-3:-1]
    assert generated["messages"][-1]["content"] == source["messages"][-1]["content"]
    assert generated["messages"][-1]["loss"] is False
    assert generated["messages"][-3]["loss"] is True
    assert generated["metadata"]["split"] == "train"
    assert generated["metadata"]["split_group"] == "customer-session-train"
    assert generated["provenance"]["source_record_id"] == "source-train"
    assert generated["provenance"]["audio_sha256"] == "a" * 64
    assert generated["metadata"]["target_policy"] == "tool_selection_host_rendered_read"
    assert generated["metadata"]["asr"]["speaker"] == "customer"
    assert generated["metadata"]["asr"]["start_ms"] == 1250
    assert generated["metadata"]["asr"]["end_ms"] == 3680
    output_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert output_manifest["contract"] == "banking-asr-sft-manifest/v1"
    assert output_manifest["tool_sft"][0]["sha256"] == hashlib.sha256(
        (output / "train.jsonl").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"transcript": "my ssn is 123-45-6789"}, "PII"),
        ({"review": {"semantic_match": False}}, "semantic_match"),
        ({"confidence": 1.1}, "confidence"),
        ({"audio_sha256": "not-a-digest"}, "audio_sha256"),
        ({"speaker": "agent"}, "speaker"),
        ({"start_ms": 4000, "end_ms": 3000}, "timestamps"),
    ],
)
def test_asr_overlay_fails_closed_on_unapproved_or_invalid_rows(
    tmp_path: Path,
    change: dict[str, Any],
    message: str,
) -> None:
    manifest = _write_base_dataset(tmp_path / "base")
    row = _asr_row("source-train", "show my accounts")
    if "review" in change:
        row["review"] = {**row["review"], **change["review"]}
    else:
        row.update(change)
    input_path = tmp_path / "asr.json"
    input_path.write_text(json.dumps([row]), encoding="utf-8")

    with pytest.raises(AsrSftDataError, match=message):
        prepare_asr_sft_data(
            asr_input=input_path,
            base_manifest=manifest,
            output_dir=tmp_path / "output",
        )


def test_asr_overlay_requires_evaluation_split_coverage(tmp_path: Path) -> None:
    manifest = _write_base_dataset(tmp_path / "base")
    input_path = tmp_path / "asr.jsonl"
    input_path.write_text(
        json.dumps(_asr_row("source-train", "show my accounts")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AsrSftDataError, match="empty splits: validation, test"):
        prepare_asr_sft_data(
            asr_input=input_path,
            base_manifest=manifest,
            output_dir=tmp_path / "output",
        )


def test_asr_overlay_rejects_recording_family_cross_split_leakage(tmp_path: Path) -> None:
    manifest = _write_base_dataset(tmp_path / "base")
    rows = [
        _asr_row("source-train", "show my train accounts"),
        _asr_row("source-validation", "show my validation accounts"),
        _asr_row("source-test", "show my test accounts"),
    ]
    rows[1]["recording_id"] = rows[0]["recording_id"]
    input_path = tmp_path / "asr.jsonl"
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(AsrSftDataError, match="recording_id crosses"):
        prepare_asr_sft_data(
            asr_input=input_path,
            base_manifest=manifest,
            output_dir=tmp_path / "output",
        )


def test_asr_overlay_rejects_unrecognized_base_manifest_contract(tmp_path: Path) -> None:
    manifest_path = _write_base_dataset(tmp_path / "base")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract"] = "untrusted-manifest"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    input_path = tmp_path / "asr.jsonl"
    rows = [
        _asr_row("source-train", "show my train accounts"),
        _asr_row("source-validation", "show my validation accounts"),
        _asr_row("source-test", "show my test accounts"),
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(AsrSftDataError, match="unsupported contract"):
        prepare_asr_sft_data(
            asr_input=input_path,
            base_manifest=manifest_path,
            output_dir=tmp_path / "output",
        )
