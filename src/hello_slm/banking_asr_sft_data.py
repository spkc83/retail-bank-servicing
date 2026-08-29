from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from hello_slm.config import canonical_json_bytes, file_sha256

ASR_SFT_CONTRACT = "banking-asr-tool-sft/v1"
ASR_SFT_MANIFEST_CONTRACT = "banking-asr-sft-manifest/v1"
GENERATOR_VERSION = "banking-asr-overlay/v1"
SPLITS = ("train", "validation", "test")
PII_PATTERNS = (
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]?){12,}\b"),
)
SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
READ_TOOLS = {
    "list_accounts",
    "list_cards",
    "list_service_cases",
    "list_transactions",
    "list_transfers",
}


class AsrSftDataError(ValueError):
    """Raised when an ASR overlay cannot safely become supervised training data."""


def prepare_asr_sft_data(
    *,
    asr_input: Path,
    base_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Overlay reviewed ASR transcripts on immutable, validated semantic SFT targets."""

    source_records = _load_source_records(base_manifest)
    input_rows = _read_rows(asr_input)
    generated = _build_overlays(input_rows, source_records)
    split_rows = {
        split: [row for row in generated if row["metadata"]["split"] == split]
        for split in SPLITS
    }
    empty_splits = [split for split, rows in split_rows.items() if not rows]
    if empty_splits:
        raise AsrSftDataError(
            "ASR SFT requires train, validation, and test coverage; empty splits: "
            + ", ".join(empty_splits)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [_write_split(output_dir, split, split_rows[split]) for split in SPLITS]
    report = {
        "contract": "banking-asr-sft-preparation-report/v1",
        "generator_version": GENERATOR_VERSION,
        "source_manifest": str(base_manifest.resolve()),
        "source_manifest_sha256": file_sha256(base_manifest),
        "input_sha256": file_sha256(asr_input),
        "input_records": len(input_rows),
        "accepted_records": len(generated),
        "split_counts": {split: len(split_rows[split]) for split in SPLITS},
        "language_counts": dict(Counter(row["metadata"]["asr"]["language"] for row in generated)),
        "asr_model_counts": dict(
            Counter(row["metadata"]["asr"]["model_id"] for row in generated)
        ),
        "rejected_records": 0,
    }
    manifest = {
        "format_version": 1,
        "name": "retail-bank-reviewed-asr-tool-sft",
        "contract": ASR_SFT_MANIFEST_CONTRACT,
        "schema_version": ASR_SFT_CONTRACT,
        "generator_version": GENERATOR_VERSION,
        "training_allowed": True,
        "tool_sft": entries,
        "source_roles": {
            "reviewed-asr-overlay": {
                "role": "speech-robustness-tool-use-sft",
                "trainable": True,
                "source_manifest_sha256": report["source_manifest_sha256"],
            }
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "preparation-report.json", report)
    return report


def _load_source_records(manifest_path: Path) -> dict[str, tuple[str, dict[str, Any]]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AsrSftDataError(f"cannot read base manifest: {manifest_path}") from error
    entries = manifest.get("tool_sft")
    if manifest.get("contract") != "banking-tool-sft-manifest":
        raise AsrSftDataError("base manifest has an unsupported contract")
    if manifest.get("schema_version") != "banking-tool-sft/v1":
        raise AsrSftDataError("base manifest has an unsupported schema_version")
    if not isinstance(entries, list):
        raise AsrSftDataError("base manifest must contain tool_sft entries")
    records: dict[str, tuple[str, dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("name") not in SPLITS:
            continue
        split = str(entry["name"])
        declared = Path(str(entry.get("path", "")))
        path = declared if declared.is_absolute() else manifest_path.parent / declared
        if not path.is_file():
            raise AsrSftDataError(f"base split does not exist: {path}")
        declared_sha = entry.get("sha256")
        if not isinstance(declared_sha, str) or not SHA256_RE.fullmatch(declared_sha):
            raise AsrSftDataError(f"base split is missing a valid digest: {split}")
        if file_sha256(path) != declared_sha.removeprefix("sha256:"):
            raise AsrSftDataError(f"base split digest mismatch: {split}")
        split_records = _read_rows(path)
        record_count = entry.get("record_count")
        if not isinstance(record_count, int) or isinstance(record_count, bool):
            raise AsrSftDataError(f"base split is missing record_count: {split}")
        if record_count != len(split_records):
            raise AsrSftDataError(f"base split record_count mismatch: {split}")
        for row in split_records:
            record_id = _required_text(row, "record_id")
            if record_id in records:
                raise AsrSftDataError(f"duplicate source record_id: {record_id}")
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get("trainable") is False:
                raise AsrSftDataError(f"source record is not trainable: {record_id}")
            validation = row.get("validation")
            if not isinstance(validation, Mapping) or validation.get("accepted") is not True:
                raise AsrSftDataError(f"source record is not validated: {record_id}")
            if metadata.get("split") not in (None, split):
                raise AsrSftDataError(f"source split mismatch: {record_id}")
            split_group = metadata.get("split_group")
            if not isinstance(split_group, str) or not split_group.strip():
                raise AsrSftDataError(f"source record has no split_group: {record_id}")
            records[record_id] = (split, row)
    if not records:
        raise AsrSftDataError("base manifest does not contain any trainable records")
    return records


def _build_overlays(
    rows: Iterable[dict[str, Any]],
    sources: Mapping[str, tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    seen_utterance_ids: set[str] = set()
    seen_transcripts: set[str] = set()
    recording_groups: dict[str, tuple[str, str]] = {}
    for row in rows:
        record_id = _required_text(row, "record_id")
        source_record_id = _required_text(row, "source_record_id")
        utterance_id = _required_text(row, "utterance_id")
        transcript = _required_text(row, "transcript")
        if record_id in seen_record_ids:
            raise AsrSftDataError(f"duplicate record_id: {record_id}")
        if utterance_id in seen_utterance_ids:
            raise AsrSftDataError(f"duplicate utterance_id: {utterance_id}")
        transcript_key = NORMALIZE_RE.sub(" ", transcript.casefold()).strip()
        if transcript_key in seen_transcripts:
            raise AsrSftDataError(f"duplicate normalized transcript: {record_id}")
        seen_record_ids.add(record_id)
        seen_utterance_ids.add(utterance_id)
        seen_transcripts.add(transcript_key)
        if any(pattern.search(transcript) for pattern in PII_PATTERNS):
            raise AsrSftDataError(f"{record_id} transcript contains PII-like text")
        source_entry = sources.get(source_record_id)
        if source_entry is None:
            raise AsrSftDataError(f"unknown source_record_id: {source_record_id}")
        split, source = source_entry
        asr = _validate_asr_metadata(row, record_id)
        review = _validate_review(row, record_id)
        source_metadata = source.get("metadata")
        if not isinstance(source_metadata, Mapping):
            raise AsrSftDataError(f"source {source_record_id} has invalid metadata")
        split_group = str(source_metadata["split_group"])
        recording_id = str(asr["recording_id"])
        prior_group = recording_groups.setdefault(recording_id, (split, split_group))
        if prior_group != (split, split_group):
            raise AsrSftDataError(
                f"recording_id crosses source split or split_group: {recording_id}"
            )
        overlay = deepcopy(source)
        messages = overlay.get("messages")
        if not isinstance(messages, list):
            raise AsrSftDataError(f"source {source_record_id} has no messages")
        user_messages = [message for message in messages if message.get("role") == "user"]
        if not user_messages:
            raise AsrSftDataError(f"source {source_record_id} has no user message")
        user_messages[-1]["content"] = transcript
        overlay["schema_version"] = ASR_SFT_CONTRACT
        overlay["record_id"] = record_id
        overlay["provenance"] = {
            "source": "reviewed-asr-overlay",
            "source_record_id": source_record_id,
            "source_semantic_sha256": _semantic_digest(source),
            "utterance_id": utterance_id,
            "audio_sha256": asr["audio_sha256"],
            "license": review["license"],
            "reviewer": review["reviewer"],
            "generator_version": GENERATOR_VERSION,
        }
        overlay["metadata"] = {
            **(dict(source_metadata) if isinstance(source_metadata, Mapping) else {}),
            "split": split,
            "trainable": True,
            "record_type": "asr_tool_use_sft",
            "asr": asr,
        }
        source_validation = source.get("validation")
        overlay["validation"] = {
            **(dict(source_validation) if isinstance(source_validation, Mapping) else {}),
            "accepted": True,
            "semantic_match_reviewed": True,
            "pii_reviewed": True,
            "consent_for_training": True,
            "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        }
        if _semantic_digest(overlay) != _semantic_digest(source):
            raise AsrSftDataError(f"{record_id} changed immutable source semantics")
        if _read_only_target(overlay):
            final_assistant = _last_assistant_message(overlay)
            final_assistant["loss"] = False
            overlay["metadata"]["target_policy"] = "tool_selection_host_rendered_read"
        else:
            overlay["metadata"]["target_policy"] = "model_authored_answer"
        generated.append(overlay)
    return generated


def _validate_asr_metadata(row: Mapping[str, Any], record_id: str) -> dict[str, Any]:
    confidence = row.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise AsrSftDataError(f"{record_id} confidence must be numeric")
    if not 0 <= float(confidence) <= 1:
        raise AsrSftDataError(f"{record_id} confidence must be between 0 and 1")
    audio_sha256 = _required_text(row, "audio_sha256")
    if not SHA256_RE.fullmatch(audio_sha256):
        raise AsrSftDataError(f"{record_id} has invalid audio_sha256")
    alternatives = row.get("alternatives", [])
    if not isinstance(alternatives, list):
        raise AsrSftDataError(f"{record_id} alternatives must be a list")
    normalized_alternatives = []
    for alternative in alternatives:
        if not isinstance(alternative, Mapping):
            raise AsrSftDataError(f"{record_id} has an invalid ASR alternative")
        text = _required_text(alternative, "text")
        alt_confidence = alternative.get("confidence")
        if isinstance(alt_confidence, bool) or not isinstance(alt_confidence, int | float):
            raise AsrSftDataError(f"{record_id} alternative confidence must be numeric")
        if not 0 <= float(alt_confidence) <= 1:
            raise AsrSftDataError(f"{record_id} alternative confidence is out of range")
        if any(pattern.search(text) for pattern in PII_PATTERNS):
            raise AsrSftDataError(f"{record_id} ASR alternative contains PII-like text")
        normalized_alternatives.append({"text": text, "confidence": float(alt_confidence)})
    return {
        "model_id": _required_text(row, "asr_model_id"),
        "model_revision": _required_text(row, "asr_model_revision"),
        "recording_id": _required_text(row, "recording_id"),
        "speaker": _validated_speaker(row, record_id),
        "language": _required_text(row, "language"),
        "confidence": float(confidence),
        **_validated_timestamps(row, record_id),
        "audio_sha256": audio_sha256.removeprefix("sha256:"),
        "alternatives": normalized_alternatives,
    }


def _validate_review(row: Mapping[str, Any], record_id: str) -> dict[str, str]:
    review = row.get("review")
    if not isinstance(review, Mapping):
        raise AsrSftDataError(f"{record_id} review is required")
    for field in ("semantic_match", "pii_reviewed", "consent_for_training"):
        if review.get(field) is not True:
            raise AsrSftDataError(f"{record_id} review.{field} must be true")
    return {
        "reviewer": _required_text(review, "reviewer"),
        "license": _required_text(review, "license"),
    }


def _validated_speaker(row: Mapping[str, Any], record_id: str) -> str:
    speaker = _required_text(row, "speaker").casefold()
    if speaker != "customer":
        raise AsrSftDataError(
            f"{record_id} speaker must be customer; agent transcripts are not user targets"
        )
    return speaker


def _validated_timestamps(row: Mapping[str, Any], record_id: str) -> dict[str, int]:
    start_ms = row.get("start_ms")
    end_ms = row.get("end_ms")
    if (
        not isinstance(start_ms, int)
        or isinstance(start_ms, bool)
        or not isinstance(end_ms, int)
        or isinstance(end_ms, bool)
        or start_ms < 0
        or end_ms <= start_ms
    ):
        raise AsrSftDataError(
            f"{record_id} timestamps must satisfy 0 <= start_ms < end_ms"
        )
    return {"start_ms": start_ms, "end_ms": end_ms}


def _semantic_digest(source: Mapping[str, Any]) -> str:
    messages = deepcopy(source.get("messages", []))
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                message["content"] = "<ASR_USER_UTTERANCE>"
                break
    payload = {
        "messages": messages,
        "expected": source.get("expected"),
        "split_keys": source.get("split_keys"),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _read_only_target(record: Mapping[str, Any]) -> bool:
    expected = record.get("expected")
    tool_calls = expected.get("tool_calls") if isinstance(expected, Mapping) else None
    return bool(tool_calls) and isinstance(tool_calls, list) and all(
        isinstance(call, Mapping) and call.get("name") in READ_TOOLS for call in tool_calls
    )


def _last_assistant_message(record: Mapping[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if (
                isinstance(message, dict)
                and message.get("role") == "assistant"
                and isinstance(message.get("content"), str)
            ):
                return message
    raise AsrSftDataError("record has no final assistant message")


def _write_split(output_dir: Path, split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = output_dir / f"{split}.jsonl"
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    return {
        "name": split,
        "path": path.name,
        "record_count": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "included": True,
        "allowed_use": [
            "granite-asr-robustness-sft" if split == "train" else "granite-asr-evaluation"
        ],
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AsrSftDataError(f"cannot read data file: {path}") from error
    try:
        if path.suffix.casefold() == ".jsonl":
            values = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            value = json.loads(text)
            values = value if isinstance(value, list) else [value]
    except json.JSONDecodeError as error:
        raise AsrSftDataError(f"invalid JSON in {path}") from error
    if any(not isinstance(value, dict) for value in values):
        raise AsrSftDataError(f"{path} must contain JSON objects")
    return values


def _required_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise AsrSftDataError(f"{key} must be non-empty text")
    return item.strip()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert reviewed ASR output into Granite banking tool-use SFT overlays."
    )
    parser.add_argument("--asr-input", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = prepare_asr_sft_data(
        asr_input=args.asr_input,
        base_manifest=args.base_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
