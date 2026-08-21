#!/usr/bin/env python
"""Hard-rule checker for banking teacher realization batches.

Every rule (a-n) mirrors a validator the rewritten rows will meet later: the
teacher realizer contract, ``validate_records``, the runtime response policy of
the POC, or the voice specification in the retrain plan (Appendix A). A batch is
acceptable only when every rule holds for every row, so the checker prints one
``record_id: rule: detail`` line per violation and exits 2; a clean batch prints
a JSON summary and exits 0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hello_slm.banking_servicing_alignment_data import (  # noqa: E402
    FINAL_CLOSERS,
    FINAL_OPENERS,
    SCREENSHOT_HELDOUT_CURRENTS,
)
from hello_slm.banking_tool_sft_data import (  # noqa: E402
    FAQ_REQUIRED_MARKERS,
    FINAL_RESPONSE_FORBIDDEN,
    POC_PRESET_KEYS,
    POLICY_CHUNKS,
    REALIZER_FINAL_CLOSERS,
    REALIZER_FINAL_PREFIXES,
    load_canonical_policy_corpus,
    normalized_user_text,
)


def _load_script_module(path: Path, name: str) -> ModuleType:
    """Load a non-packaged script or POC module by path."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POC_DIR = REPO_ROOT / "poc" / "retail-bank-customer-service-poc"
RESPONSE_POLICY = _load_script_module(
    POC_DIR / "response_policy.py", "_check_teacher_batch_response_policy"
)
REALIZER = _load_script_module(
    REPO_ROOT / "scripts" / "retail_bank" / "realize_tool_sft_teacher.py",
    "_check_teacher_batch_realizer",
)

SPLIT_LEADS = ("For this request,", "In this session,")
FILLER_PHRASES: tuple[str, ...] = tuple(
    sorted(
        {
            phrase
            for phrase in (
                *REALIZER_FINAL_PREFIXES,
                *REALIZER_FINAL_CLOSERS,
                *FINAL_OPENERS,
                *FINAL_CLOSERS,
                *RESPONSE_POLICY.REALIZER_FILLER_PREFIXES,
                *RESPONSE_POLICY.REALIZER_FILLER_CLOSERS,
                *SPLIT_LEADS,
            )
            if phrase.strip()
        },
        key=len,
        reverse=True,
    )
)
CONDITIONAL_LITERALS: tuple[str, ...] = (
    "could not",
    "was not",
    "unchanged",
    "last four digits",
    "retail banking",
    "account numbers",
    "customer ids",
    "which",
    "card",
    "available",
    "current",
    "sorry",
    "replacement",
    "dispute",
    "froze",
    "frozen",
    "cancelled",
    "canceled",
)
ENVELOPE_LITERAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("card", "last4"),
    ("transfer", "recipient"),
    ("transaction", "description"),
)
EXTRA_BANNED_TERMS: tuple[str, ...] = ("session", "as an ai")
# Mirrors the inline regexes of ``realize_tool_sft_teacher._assistant_requests_private_data``.
# The realizer cannot tell "do not share a password" from "share your password", and seven
# released policy finals carry that required claim, so rule a waives an error the unchanged
# final already produces and only fails a rewrite that widens the credential signature.
CREDENTIAL_VERB_RE = re.compile(
    r"\b(?:provide|send|enter|share|tell me|confirm)\b", re.IGNORECASE
)
CREDENTIAL_SECRET_RE = re.compile(
    r"\b(?:password|pin|account number|customer id|private id)\b", re.IGNORECASE
)
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
ALPHA_TOKEN_RE = re.compile(r"[a-z]+")
ALLOWED_NON_ASCII = frozenset("’—–")
CLARIFY_GATE_WORDS = 45
MAX_SENTENCES = 4
MAX_PROSE_WORDS = 70
MAX_TRIGRAM_USES = 4
POLICY_PATH = "retrieval_grounded_policy"
REALIZER_RESPONSE_KEYS = ("record_id", "immutable_hash", "user_content", "final_response")
CLARIFY_PATHS = frozenset({"clarification"})
POLICY_CHUNKS_BY_ID: dict[str, Mapping[str, Any]] = {
    str(chunk["chunk_id"]): chunk for chunk in load_canonical_policy_corpus()["chunks"]
}
HELDOUT_CURRENT_KEYS = frozenset(normalized_user_text(text) for text in SCREENSHOT_HELDOUT_CURRENTS)


class CheckerInputError(ValueError):
    """Raised when the checker cannot read its own inputs."""


@dataclass(frozen=True)
class Violation:
    record_id: str
    rule: str
    detail: str

    def render(self) -> str:
        return f"{self.record_id}: {self.rule}: {self.detail}"


@dataclass(frozen=True)
class RecordContext:
    record_id: str
    split: str
    path: str
    mode: str
    scenario_family: str
    template_id: str
    grounding_facts: tuple[str, ...]
    forbidden_facts: tuple[str, ...]
    policy_citations: tuple[str, ...]
    envelope_literals: tuple[str, ...]
    has_tool_message: bool
    original_final: str
    original_user: str
    coreference_pair_id: str

    @property
    def is_policy(self) -> bool:
        return self.path == POLICY_PATH or "[Policy:" in self.original_final

    @property
    def is_clarify(self) -> bool:
        return self.mode == "clarify" or self.path in CLARIFY_PATHS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise CheckerInputError(f"missing JSONL file: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise CheckerInputError(f"{path}:{line_number} row must be a JSON object")
            rows.append(row)
    return rows


def _last_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _final_text(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def _tool_envelope(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    content = message.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return None
    return content if isinstance(content, Mapping) else None


def _envelope_literals(messages: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    literals: list[str] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        envelope = _tool_envelope(message)
        if envelope is None or envelope.get("ok") is not True:
            continue
        result = envelope.get("result")
        if not isinstance(result, Mapping):
            continue
        for entity, field in ENVELOPE_LITERAL_FIELDS:
            payload = result.get(entity)
            if not isinstance(payload, Mapping):
                continue
            value = payload.get(field)
            if isinstance(value, str) and value.strip() and value not in literals:
                literals.append(value)
    return tuple(literals)


def build_record_context(record: Mapping[str, Any]) -> RecordContext:
    messages = [m for m in record.get("messages", []) if isinstance(m, Mapping)]
    expected = record.get("expected", {})
    expected = expected if isinstance(expected, Mapping) else {}
    contract = expected.get("generation_contract", {})
    contract = contract if isinstance(contract, Mapping) else {}
    split_keys = record.get("split_keys", {})
    split_keys = split_keys if isinstance(split_keys, Mapping) else {}
    metadata = record.get("metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return RecordContext(
        record_id=str(record.get("record_id", "")),
        split=str(metadata.get("split", "")),
        path=str(expected.get("path", "")),
        mode=str(contract.get("mode", "")),
        scenario_family=str(split_keys.get("scenario_family", "")),
        template_id=str(split_keys.get("template_id", "")),
        grounding_facts=tuple(str(fact) for fact in expected.get("grounding_facts") or ()),
        forbidden_facts=tuple(str(fact) for fact in expected.get("forbidden_facts") or ()),
        policy_citations=tuple(str(cid) for cid in expected.get("policy_citations") or ()),
        envelope_literals=_envelope_literals(messages),
        has_tool_message=any(message.get("role") == "tool" for message in messages),
        original_final=_final_text(messages),
        original_user=_last_user_text(messages),
        coreference_pair_id=str(metadata.get("coreference_pair_id", "")),
    )


def table_block(text: str) -> str | None:
    """Return the contiguous span from the first to the last markdown table line."""

    lines = text.splitlines()
    indexes = [index for index, line in enumerate(lines) if line.lstrip().startswith("|")]
    if not indexes:
        return None
    return "\n".join(lines[indexes[0] : indexes[-1] + 1])


def prose_of(text: str) -> str:
    """Return the text with markdown table lines removed."""

    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("|")
    ).strip()


def leading_prose_of(text: str) -> str:
    """Return the prose before the first table line (the read-path deliverable)."""

    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        if line.lstrip().startswith("|"):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def count_sentences(text: str) -> int:
    return len(SENTENCE_END_RE.findall(text))


def opening_trigram(text: str) -> tuple[str, ...]:
    return tuple(ALPHA_TOKEN_RE.findall(text.lower())[:3])


def _grounding_literal(fact: str) -> str:
    return fact.split("=", 1)[1] if "=" in fact else fact


def _policy_chunks(context: RecordContext, final: str) -> list[Mapping[str, Any]]:
    citations = list(context.policy_citations)
    if not citations:
        citations = [
            citation.strip()
            for citation in RESPONSE_POLICY.POLICY_CITATION.findall(context.original_final or final)
        ]
    chunks = [POLICY_CHUNKS_BY_ID[cid] for cid in citations if cid in POLICY_CHUNKS_BY_ID]
    if not chunks and context.template_id in POLICY_CHUNKS:
        chunks = [POLICY_CHUNKS[context.template_id]]
    return chunks


def _credential_signature(text: str) -> frozenset[str]:
    return frozenset(
        match.group(0).lower()
        for pattern in (CREDENTIAL_VERB_RE, CREDENTIAL_SECRET_RE)
        for match in pattern.finditer(text)
    ) | frozenset(match.lower() for match in REALIZER.PRIVATE_VALUE_RE.findall(text))


def _realizer_error(request: Mapping[str, Any], response: Mapping[str, Any]) -> str | None:
    try:
        REALIZER.validate_response_row(dict(request), dict(response))
    except REALIZER.TeacherRealizationError as error:
        return str(error)
    return None


def check_rule_a(
    record_id: str, request: Mapping[str, Any], response: Mapping[str, Any]
) -> tuple[list[Violation], str | None]:
    """Run the realizer contract, waiving a defect the released final already has."""

    error = _realizer_error(request, response)
    if error is None:
        return [], None
    identity = {key: request.get(key) for key in REALIZER_RESPONSE_KEYS}
    baseline = _realizer_error(request, identity)
    if baseline is not None and _credential_signature(
        str(response.get("final_response", ""))
    ) <= _credential_signature(str(request.get("final_response", ""))):
        return [], baseline
    return [Violation(record_id, "a", error)], None


def check_rule_b(record_id: str, normalized_final: str) -> list[Violation]:
    words = len(normalized_final.split())
    if words < 7:
        return [Violation(record_id, "b", f"final has {words} normalized words, needs 7")]
    return []


def check_rule_c(record_id: str, final: str, normalized_final: str) -> list[Violation]:
    violations = [
        Violation(record_id, "c", f"final contains banned term {term!r}")
        for term in (*FINAL_RESPONSE_FORBIDDEN, *EXTRA_BANNED_TERMS)
        if normalized_user_text(term) in normalized_final
    ]
    violations.extend(
        Violation(record_id, "c", f"final contains internal language ({label})")
        for label, pattern in RESPONSE_POLICY.INTERNAL_LANGUAGE_PATTERNS.items()
        if pattern.search(final)
    )
    return violations


def check_rule_d(record_id: str, final: str, normalized_final: str) -> list[Violation]:
    violations: list[Violation] = []
    for phrase in FILLER_PHRASES:
        normalized_phrase = normalized_user_text(phrase)
        if not normalized_phrase or normalized_phrase not in normalized_final:
            continue
        if normalized_final.startswith(normalized_phrase):
            where = "opens with"
        elif normalized_final.endswith(normalized_phrase):
            where = "ends with"
        else:
            where = "reuses"
        violations.append(Violation(record_id, "d", f"final {where} realizer filler {phrase!r}"))
    return violations


def check_rule_e(record_id: str, context: RecordContext, final: str) -> list[Violation]:
    original = context.original_final
    original_lower = original.lower()
    final_lower = final.lower()
    required: list[str] = [
        literal for literal in CONDITIONAL_LITERALS if literal in original_lower
    ]
    required.extend(
        _grounding_literal(fact)
        for fact in context.grounding_facts
        if _grounding_literal(fact).lower() in original_lower
    )
    required.extend(match.group(0) for match in RESPONSE_POLICY.POLICY_CITATION.finditer(original))
    required.extend(ISO_DATE_RE.findall(original))
    required.extend(context.envelope_literals)
    seen: set[str] = set()
    violations: list[Violation] = []
    for literal in required:
        key = literal.lower()
        if not literal.strip() or key in seen:
            continue
        seen.add(key)
        if key not in final_lower:
            violations.append(
                Violation(record_id, "e", f"rewrite dropped load-bearing literal {literal!r}")
            )
    return violations


def check_rule_f(record_id: str, context: RecordContext, final: str) -> list[Violation]:
    block = table_block(context.original_final)
    if block is None:
        return []
    index = final.find(block)
    if index < 0:
        return [Violation(record_id, "f", "original markdown table block is missing or altered")]
    trailing = final[index + len(block) :]
    if trailing.strip():
        return [
            Violation(record_id, "f", f"text follows the table: {trailing.strip()[:60]!r}")
        ]
    return []


def check_rule_g(record_id: str, context: RecordContext, final: str) -> list[Violation]:
    if not context.is_clarify:
        return []
    original_lower = context.original_final.lower()
    gate_text = " ".join(final.split()[:CLARIFY_GATE_WORDS]).lower()
    return [
        Violation(
            record_id,
            "g",
            f"clarify final places {word!r} after the first {CLARIFY_GATE_WORDS} words",
        )
        for word in ("which", "card")
        if word in original_lower and word not in gate_text
    ]


def check_rule_h(
    record_id: str,
    context: RecordContext,
    final: str,
    *,
    min_sentences: int,
    exempt_families: frozenset[str],
) -> tuple[list[Violation], int]:
    has_table = table_block(final) is not None
    prose = leading_prose_of(final) if has_table else prose_of(final)
    sentences = count_sentences(prose)
    words = len(prose.split())
    violations: list[Violation] = []
    if has_table:
        if sentences < 1:
            violations.append(
                Violation(record_id, "h", "table-bearing final has no lead-in sentence")
            )
    elif sentences < min_sentences and context.scenario_family not in exempt_families:
        violations.append(
            Violation(record_id, "h", f"final has {sentences} sentences, needs {min_sentences}")
        )
    if sentences > MAX_SENTENCES:
        violations.append(
            Violation(record_id, "h", f"final has {sentences} sentences, at most {MAX_SENTENCES}")
        )
    if words > MAX_PROSE_WORDS:
        violations.append(
            Violation(record_id, "h", f"final prose has {words} words, at most {MAX_PROSE_WORDS}")
        )
    exclamations = final.count("!")
    if exclamations > 1:
        violations.append(
            Violation(record_id, "h", f"final uses {exclamations} exclamation marks, at most 1")
        )
    stray = sorted(
        {
            character
            for character in final
            if not (character.isascii() and (character.isprintable() or character == "\n"))
            and character not in ALLOWED_NON_ASCII
        }
    )
    if stray:
        violations.append(Violation(record_id, "h", f"final uses unsupported characters: {stray}"))
    return violations, sentences


def check_rule_i(
    record_id: str, request: Mapping[str, Any], response: Mapping[str, Any]
) -> list[Violation]:
    if str(response.get("user_content", "")) != str(request.get("user_content", "")):
        return [Violation(record_id, "i", "--finals-only forbids editing user_content")]
    return []


def check_rule_k(
    record_id: str, context: RecordContext, final: str, record: Mapping[str, Any]
) -> list[Violation]:
    conversation = [
        message
        for message in record.get("messages", [])
        if isinstance(message, Mapping) and message.get("role") == "tool"
    ]
    result = RESPONSE_POLICY.validate_no_unsupported_action_claims(final, (), conversation)
    if result.valid:
        return []
    return [Violation(record_id, "k", error) for error in result.errors]


def check_rule_l(
    record_id: str, context: RecordContext, final: str, normalized_final: str
) -> list[Violation]:
    if not context.is_policy:
        return []
    violations: list[Violation] = []
    for claim in context.forbidden_facts:
        if normalized_user_text(claim) in normalized_final:
            violations.append(Violation(record_id, "l", f"final states forbidden claim {claim!r}"))
    required = [*context.grounding_facts, *FAQ_REQUIRED_MARKERS.get(context.template_id, ())]
    for claim in required:
        if normalized_user_text(claim) not in normalized_final:
            violations.append(Violation(record_id, "l", f"final omits required claim {claim!r}"))
    chunks = _policy_chunks(context, final)
    if not chunks:
        violations.append(Violation(record_id, "l", "no policy chunk is available to check"))
        return violations
    result = RESPONSE_POLICY.validate_policy_answer(final, chunks)
    violations.extend(
        Violation(record_id, "l", error)
        for error in result.errors
        if not error.startswith("answer contains internal term")
    )
    return violations


def check_rule_m(
    record_id: str, context: RecordContext, response: Mapping[str, Any]
) -> list[Violation]:
    user_content = str(response.get("user_content", ""))
    normalized_user = normalized_user_text(user_content)
    violations: list[Violation] = []
    if normalized_user in POC_PRESET_KEYS:
        violations.append(Violation(record_id, "m", "user turn normalizes to a POC preset"))
    if normalized_user in HELDOUT_CURRENT_KEYS:
        violations.append(Violation(record_id, "m", "user turn normalizes to a held-out current"))
    missing = sorted(
        digits
        for digits in set(REALIZER.DIGIT_RE.findall(context.original_user))
        if digits not in user_content
    )
    if missing:
        violations.append(Violation(record_id, "m", f"user turn dropped digits: {missing}"))
    return violations


def _shares_governed_pair(first: RecordContext | None, second: RecordContext | None) -> bool:
    """Simplified mirror of ``banking_tool_sft_data._is_governed_counterfactual_group``.

    That validator's full shape (paired targets, matching card counts, tool-call
    argument checks, history-form signatures) is too involved to replicate here, so
    this only mirrors its practical effect: records that share a non-empty governed
    counterfactual pair id are allowed to share normalized user text.
    """
    if first is None or second is None:
        return False
    return bool(first.coreference_pair_id) and (
        first.coreference_pair_id == second.coreference_pair_id
    )


@dataclass(frozen=True)
class CheckerConfig:
    requests: tuple[Path, ...]
    responses: tuple[Path, ...]
    records_dirs: tuple[Path, ...]
    finals_only: bool
    min_sentences: int
    exempt_families: frozenset[str]


def load_records(records_dirs: Iterable[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for directory in records_dirs:
        if not directory.is_dir():
            raise CheckerInputError(f"missing records directory: {directory}")
        for path in sorted(directory.glob("*.jsonl")):
            for record in read_jsonl(path):
                record_id = str(record.get("record_id", ""))
                if record_id:
                    records[record_id] = record
    return records


def check(config: CheckerConfig) -> tuple[list[Violation], dict[str, Any]]:
    records = load_records(config.records_dirs)
    contexts = {record_id: build_record_context(record) for record_id, record in records.items()}
    requests: dict[str, dict[str, Any]] = {}
    for path in config.requests:
        for row in read_jsonl(path):
            requests[str(row.get("record_id", ""))] = row

    violations: list[Violation] = []
    files_summary: list[dict[str, Any]] = []
    sentence_counts: Counter[int] = Counter()
    families: set[str] = set()
    seen_finals: dict[str, str] = {}
    seen_users: dict[str, str] = {}
    trigram_uses: Counter[tuple[str, tuple[str, ...]]] = Counter()
    trigram_records: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    responded: set[str] = set()
    waivers: list[dict[str, str]] = []

    for path in config.responses:
        rows = read_jsonl(path)
        files_summary.append({"path": str(path), "rows": len(rows)})
        for row in rows:
            record_id = str(row.get("record_id", ""))
            if not record_id:
                violations.append(Violation("<unknown>", "input", f"{path} row has no record_id"))
                continue
            if record_id in responded:
                violations.append(Violation(record_id, "input", "duplicate response row"))
                continue
            responded.add(record_id)
            request = requests.get(record_id)
            context = contexts.get(record_id)
            record = records.get(record_id)
            if request is None:
                violations.append(Violation(record_id, "input", "no matching request row"))
                continue
            if context is None or record is None:
                violations.append(Violation(record_id, "input", "no matching record"))
                continue
            families.add(context.scenario_family)
            final = str(row.get("final_response", ""))
            normalized_final = normalized_user_text(final)

            rule_a_violations, waived = check_rule_a(record_id, request, row)
            violations.extend(rule_a_violations)
            if waived is not None:
                waivers.append({"record_id": record_id, "rule": "a", "reason": waived})
            violations.extend(check_rule_b(record_id, normalized_final))
            violations.extend(check_rule_c(record_id, final, normalized_final))
            violations.extend(check_rule_d(record_id, final, normalized_final))
            violations.extend(check_rule_e(record_id, context, final))
            violations.extend(check_rule_f(record_id, context, final))
            violations.extend(check_rule_g(record_id, context, final))
            row_violations, sentences = check_rule_h(
                record_id,
                context,
                final,
                min_sentences=config.min_sentences,
                exempt_families=config.exempt_families,
            )
            violations.extend(row_violations)
            sentence_counts[sentences] += 1
            if config.finals_only:
                violations.extend(check_rule_i(record_id, request, row))
            else:
                violations.extend(check_rule_m(record_id, context, row))
            violations.extend(check_rule_k(record_id, context, final, record))
            violations.extend(check_rule_l(record_id, context, final, normalized_final))

            if normalized_final in seen_finals:
                violations.append(
                    Violation(
                        record_id,
                        "j",
                        f"final duplicates the rewrite of {seen_finals[normalized_final]}",
                    )
                )
            else:
                seen_finals[normalized_final] = record_id
            normalized_user = normalized_user_text(str(row.get("user_content", "")))
            if normalized_user in seen_users:
                other_id = seen_users[normalized_user]
                if not _shares_governed_pair(context, contexts.get(other_id)):
                    violations.append(
                        Violation(record_id, "n", f"user text duplicates {other_id}")
                    )
            else:
                seen_users[normalized_user] = record_id
            prose = leading_prose_of(final) if table_block(final) else prose_of(final)
            trigram = opening_trigram(prose)
            if trigram:
                key = (context.scenario_family, trigram)
                trigram_uses[key] += 1
                trigram_records.setdefault(key, []).append(record_id)

    for record_id, context in contexts.items():
        if record_id in responded:
            continue
        normalized_untouched = normalized_user_text(context.original_final)
        owner = seen_finals.get(normalized_untouched)
        if owner is not None:
            violations.append(
                Violation(owner, "j", f"final duplicates untouched record {record_id}")
            )
        normalized_untouched_user = normalized_user_text(context.original_user)
        user_owner = seen_users.get(normalized_untouched_user)
        if user_owner is not None and not _shares_governed_pair(
            contexts.get(user_owner), context
        ):
            violations.append(
                Violation(user_owner, "n", f"user text duplicates {record_id}")
            )

    for (family, trigram), count in sorted(trigram_uses.items()):
        if count <= MAX_TRIGRAM_USES:
            continue
        for record_id in trigram_records[(family, trigram)]:
            violations.append(
                Violation(
                    record_id,
                    "j",
                    f"opening trigram {' '.join(trigram)!r} used {count} times in {family}",
                )
            )

    summary = {
        "checked": len(responded),
        "files": files_summary,
        "records_dirs": [str(path) for path in config.records_dirs],
        "finals_only": config.finals_only,
        "min_sentences": config.min_sentences,
        "exempt_families": sorted(config.exempt_families),
        "sentence_histogram": {
            str(count): sentence_counts[count] for count in sorted(sentence_counts)
        },
        "families": sorted(families),
        "waivers": waivers,
    }
    return violations, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a teacher realization batch against the hard authoring rules."
    )
    parser.add_argument("--requests", type=Path, action="append", required=True)
    parser.add_argument("--responses", type=Path, action="append", required=True)
    parser.add_argument("--records-dir", type=Path, action="append", required=True)
    parser.add_argument("--finals-only", action="store_true")
    parser.add_argument("--min-sentences", type=int, default=2)
    parser.add_argument("--exempt-family", action="append", default=[])
    args = parser.parse_args(argv)

    config = CheckerConfig(
        requests=tuple(args.requests),
        responses=tuple(args.responses),
        records_dirs=tuple(args.records_dir),
        finals_only=bool(args.finals_only),
        min_sentences=int(args.min_sentences),
        exempt_families=frozenset(str(family) for family in args.exempt_family),
    )
    try:
        violations, summary = check(config)
    except (CheckerInputError, json.JSONDecodeError) as error:
        print(f"checker input error: {error}")
        return 2
    if violations:
        for violation in sorted(
            violations, key=lambda item: (item.record_id, item.rule, item.detail)
        ):
            print(violation.render())
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
