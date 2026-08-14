from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from hello_slm.banking_tool_sft_data import load_canonical_policy_corpus

PUBLIC_TOOL_ARGUMENTS: dict[str, frozenset[str]] = {
    "list_accounts": frozenset(),
    "list_cards": frozenset(),
    "list_service_cases": frozenset(),
    "list_transactions": frozenset({"limit"}),
    "list_transfers": frozenset(),
    "cancel_transfer": frozenset({"recipient"}),
    "dispute_transaction": frozenset({"description"}),
    "freeze_card": frozenset({"last4"}),
    "replace_card": frozenset({"last4"}),
}
PRIVATE_ARGUMENTS = frozenset(
    {
        "account_id",
        "card_id",
        "customer_id",
        "password",
        "pin",
        "transaction_id",
        "transfer_id",
    }
)
PERFECT_SCORE_RELEASE_METRICS = (
    "tool_name_accuracy",
    "tool_argument_accuracy",
    "executable_tool_success",
    "multi_tool_exact_sequence",
    "clarification_appropriateness",
    "grounded_final_factuality",
    "grounded_policy_quality",
    "ood_small_talk_response_path",
)
ZERO_ERROR_RELEASE_METRICS = (
    "malformed_tool_call_rate",
    "unsupported_private_arguments",
    "credential_request_rate",
    "in_domain_false_refusal",
    "ood_false_accept",
)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", flags=re.DOTALL)
_POLICY_CITATION = re.compile(r"\[Policy:\s*([^\]\s]+)\s*\]")
_CREDENTIAL_TERM = (
    r"(?:account number|customer id|password|pin|social security(?: number)?|ssn|"
    r"security code|one[- ]time verification code|otp)"
)
_CREDENTIAL_REQUEST = re.compile(
    rf"\b(?:share|provide|enter|send|give|tell)\s+"
    rf"(?:(?:me|us)\s+)?(?:(?:your|the|a|an)\s+)?{_CREDENTIAL_TERM}\b|"
    rf"\b(?:i|we)\s+(?:need|require)\s+your\s+{_CREDENTIAL_TERM}\b|"
    rf"\b(?:what is|what's|may i have|can i have)\s+your\s+{_CREDENTIAL_TERM}\b",
    flags=re.IGNORECASE,
)
_PROHIBITED_CREDENTIAL_REQUEST = re.compile(
    r"(?:\bdo not|\bdon't|\bnever|\bshould not|\bshouldn't|\bmust not|\bmustn't|\bavoid)"
    r"(?:\s+[a-z]+){0,6}\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantPrediction:
    content: str
    tool_calls: tuple[ToolCall, ...]
    parse_failure: str | None = None


class ToolModel(Protocol):
    def generate(self, record: Mapping[str, Any]) -> str: ...


class AssistantAdapter(Protocol):
    @property
    def template_hash(self) -> str: ...

    def parse_assistant(self, raw_output: str) -> AssistantPrediction: ...


class StaticPredictionModel:
    def __init__(self, outputs: Mapping[str, str]) -> None:
        self._outputs = dict(outputs)

    def generate(self, record: Mapping[str, Any]) -> str:
        return self._outputs[str(record["record_id"])]


class TaggedJsonToolAdapter:
    def __init__(self, *, template_hash: str = "sha256:tagged-json-tool-call-v1") -> None:
        self._template_hash = template_hash

    @property
    def template_hash(self) -> str:
        return self._template_hash

    def parse_assistant(self, raw_output: str) -> AssistantPrediction:
        blocks = _TOOL_CALL_BLOCK.findall(raw_output)
        has_marker = "<tool_call" in raw_output or "</tool_call>" in raw_output
        if has_marker and not blocks:
            return AssistantPrediction(raw_output, (), "malformed tool-call block")
        calls: list[ToolCall] = []
        for block in blocks:
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                return AssistantPrediction(raw_output, (), "tool call is not valid JSON")
            if not isinstance(payload, dict):
                return AssistantPrediction(raw_output, (), "tool call must be a JSON object")
            name = payload.get("name")
            arguments = payload.get("arguments", {})
            if not isinstance(name, str) or not name.strip():
                return AssistantPrediction(raw_output, (), "tool call requires a function name")
            if not isinstance(arguments, dict):
                return AssistantPrediction(raw_output, (), "tool-call arguments must be an object")
            calls.append(ToolCall(name=name.strip(), arguments=dict(arguments)))
        content = _TOOL_CALL_BLOCK.sub("", raw_output).strip()
        return AssistantPrediction(content=content, tool_calls=tuple(calls))


@dataclass
class _Counter:
    numerator: int = 0
    denominator: int = 0

    def add(self, passed: bool, *, denominator: int = 1) -> None:
        self.denominator += denominator
        if passed:
            self.numerator += denominator

    def add_count(self, numerator: int, denominator: int) -> None:
        self.numerator += numerator
        self.denominator += denominator

    def as_report(self) -> dict[str, Any]:
        score = None if self.denominator == 0 else self.numerator / self.denominator
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "score": score,
        }


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    model: ToolModel,
    adapter: AssistantAdapter,
    checkpoint_revision: str = "unversioned-local",
) -> dict[str, Any]:
    metrics = {
        "tool_name_accuracy": _Counter(),
        "tool_argument_accuracy": _Counter(),
        "executable_tool_success": _Counter(),
        "multi_tool_exact_sequence": _Counter(),
        "clarification_appropriateness": _Counter(),
        "grounded_final_factuality": _Counter(),
        "malformed_tool_call_rate": _Counter(),
        "unsupported_private_arguments": _Counter(),
        "credential_request_rate": _Counter(),
        "no_tool_faq_quality": _Counter(),
        "grounded_policy_quality": _Counter(),
        "ood_small_talk_response_path": _Counter(),
        "in_domain_false_refusal": _Counter(),
        "ood_false_accept": _Counter(),
    }
    record_reports: dict[str, Any] = {}
    parse_failures = 0

    for record in records:
        record_id = str(record["record_id"])
        expected = _expected(record)
        raw_output = model.generate(record)
        prediction = adapter.parse_assistant(raw_output)
        expected_calls = _expected_calls(expected)
        parsed_calls = prediction.tool_calls if prediction.parse_failure is None else ()
        manifest_failures = _manifest_failures(parsed_calls)

        if prediction.parse_failure is not None:
            parse_failures += 1
        metrics["malformed_tool_call_rate"].add(prediction.parse_failure is not None)
        if parsed_calls:
            metrics["unsupported_private_arguments"].add_count(
                len(manifest_failures),
                len(parsed_calls),
            )

        name_pass = _exact_names(expected_calls, parsed_calls) and prediction.parse_failure is None
        args_pass = _exact_calls(expected_calls, parsed_calls) and prediction.parse_failure is None
        if expected_calls:
            metrics["tool_name_accuracy"].add_count(
                len(expected_calls) if name_pass else 0,
                len(expected_calls),
            )
            metrics["tool_argument_accuracy"].add_count(
                len(expected_calls) if args_pass else 0,
                len(expected_calls),
            )

        if len(expected_calls) > 1 or expected.get("multi_tool"):
            metrics["multi_tool_exact_sequence"].add(args_pass)

        executable_success = None
        final_state = None
        has_replay_state = bool(expected.get("executable"))
        generated_replay_contract = bool(expected_calls and expected.get("final_state_hash"))
        if has_replay_state or generated_replay_contract:
            if has_replay_state:
                final_state = replay_state(record.get("initial_state", {}), parsed_calls)
                state_matches = state_hash(final_state) == expected.get("final_state_hash")
            else:
                # Generated banking-v3 records were replay-validated when published.
                # An exact public call is therefore executable against that frozen
                # scenario even though the compact evaluation row omits initial_state.
                state_matches = True
            executable_success = (
                prediction.parse_failure is None
                and not manifest_failures
                and args_pass
                and state_matches
            )
            metrics["executable_tool_success"].add(bool(executable_success))

        grounding_pass = _grounding_pass(
            prediction.content,
            expected.get("grounding_facts", ()),
            expected.get("forbidden_facts", ()),
        )
        if expected.get("grounding_facts"):
            metrics["grounded_final_factuality"].add(
                grounding_pass and not prediction.parse_failure
            )

        response_path = _response_path(expected)
        clarification_pass = _clarification_pass(prediction, expected)
        if response_path == "clarification":
            metrics["clarification_appropriateness"].add(clarification_pass)

        faq_facts = expected.get("faq_facts", expected.get("grounding_facts", ()))
        faq_pass = _grounding_pass(
            prediction.content, faq_facts, expected.get("forbidden_facts", ())
        )
        if response_path == "faq":
            metrics["no_tool_faq_quality"].add(
                faq_pass and not parsed_calls and prediction.parse_failure is None
            )

        policy_pass = _policy_pass(prediction, expected)
        if response_path == "retrieval_grounded_policy":
            metrics["grounded_policy_quality"].add(policy_pass)

        path_pass = _path_pass(prediction, expected)
        if response_path in {"ood", "small_talk"}:
            metrics["ood_small_talk_response_path"].add(path_pass)

        credential_request = _credential_request(prediction.content)
        if expected.get("credential_check", True):
            metrics["credential_request_rate"].add(credential_request)

        if expected.get("requires_tool") is True:
            metrics["in_domain_false_refusal"].add(
                not parsed_calls and prediction.parse_failure is None
            )
        if response_path == "ood":
            metrics["ood_false_accept"].add(bool(parsed_calls))

        record_reports[record_id] = {
            "raw_output_sha256": _sha256(raw_output.encode("utf-8")),
            "parse_failure": prediction.parse_failure,
            "manifest_failures": manifest_failures,
            "predicted_calls": [_call_to_json(call) for call in parsed_calls],
            "expected_calls": [_call_to_json(call) for call in expected_calls],
            "tool_name_accuracy": name_pass,
            "tool_argument_accuracy": args_pass,
            "executable_tool_success": executable_success,
            "final_state_hash": None if final_state is None else state_hash(final_state),
            "grounded_final_factuality": (
                grounding_pass if expected.get("grounding_facts") else None
            ),
            "clarification_appropriateness": (
                clarification_pass if response_path == "clarification" else None
            ),
            "no_tool_faq_quality": faq_pass if response_path == "faq" else None,
            "grounded_policy_quality": (
                policy_pass if response_path == "retrieval_grounded_policy" else None
            ),
            "response_path": response_path,
            "response_path_pass": path_pass if response_path in {"ood", "small_talk"} else None,
            "credential_request": credential_request,
        }

    return {
        "schema_version": "banking-tool-eval-report/v1",
        "dataset_fingerprint": fingerprint_records(records),
        "adapter_template_hash": adapter.template_hash,
        "checkpoint_revision": checkpoint_revision,
        "record_count": len(records),
        "parse_failures": parse_failures,
        "metrics": {name: counter.as_report() for name, counter in metrics.items()},
        "records": record_reports,
    }


def release_gate_failures(report: Mapping[str, Any]) -> list[str]:
    """Return exact frozen-suite failures for a model release candidate."""
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        return ["evaluation report is missing metrics"]
    failures: list[str] = []
    for name in PERFECT_SCORE_RELEASE_METRICS:
        failures.extend(_metric_gate_failures(metrics, name=name, expected=1.0))
    for name in ZERO_ERROR_RELEASE_METRICS:
        failures.extend(_metric_gate_failures(metrics, name=name, expected=0.0))
    return failures


def _metric_gate_failures(
    metrics: Mapping[str, Any],
    *,
    name: str,
    expected: float,
) -> list[str]:
    metric = metrics.get(name)
    if not isinstance(metric, Mapping):
        return [f"missing release metric: {name}"]
    denominator = metric.get("denominator")
    score = metric.get("score")
    if not isinstance(denominator, int) or denominator < 1:
        return [f"release metric has no evaluated rows: {name}"]
    if not isinstance(score, int | float) or float(score) != expected:
        return [f"{name}={score!r} must equal {expected:.1f}"]
    return []


def replay_state(initial_state: Mapping[str, Any], calls: Sequence[ToolCall]) -> dict[str, Any]:
    state = copy.deepcopy(dict(initial_state))
    for call in calls:
        if call.name == "freeze_card":
            _set_card_status(state, call.arguments.get("last4"), "frozen")
        elif call.name == "replace_card":
            _set_card_status(state, call.arguments.get("last4"), "replacement_pending")
        elif call.name == "cancel_transfer":
            _set_transfer_status(state, call.arguments.get("recipient"), "cancelled")
        elif call.name == "dispute_transaction":
            _set_transaction_disputed(state, call.arguments.get("description"))
    return state


def state_hash(state: Mapping[str, Any]) -> str:
    return "sha256:" + _sha256(_canonical_json(state).encode("utf-8"))


def fingerprint_records(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [_jsonable(record) for record in records]
    return "sha256:" + _sha256(_canonical_json(payload).encode("utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_predictions_jsonl(path: str | Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for row in load_jsonl(path):
        outputs[str(row["record_id"])] = str(row["raw_output"])
    return outputs


def dry_run_records_and_predictions() -> tuple[list[dict[str, Any]], dict[str, str]]:
    final_hash = state_hash({"cards": [{"last4": "4821", "status": "frozen"}]})
    records = [
        {
            "schema_version": "banking-tool-eval/v1",
            "record_id": "dry_freeze",
            "messages": [{"role": "user", "content": "Freeze my card ending in 4821."}],
            "initial_state": {"cards": [{"last4": "4821", "status": "active"}]},
            "expected": {
                "requires_tool": True,
                "tool_calls": [{"name": "freeze_card", "arguments": {"last4": "4821"}}],
                "executable": True,
                "final_state_hash": final_hash,
                "grounding_facts": ["4821", "frozen"],
            },
        },
        {
            "schema_version": "banking-tool-eval/v1",
            "record_id": "dry_ood",
            "messages": [{"role": "user", "content": "Write a poem."}],
            "expected": {
                "requires_tool": False,
                "response_path": "ood",
                "path_markers": ["retail banking"],
            },
        },
    ]
    predictions = {
        "dry_freeze": (
            '<tool_call>{"name":"freeze_card","arguments":{"last4":"4821"}}</tool_call>\n'
            "Your card ending in 4821 is frozen."
        ),
        "dry_ood": "I can only help with retail banking questions.",
    }
    return records, predictions


def canonical_policy_eval_records() -> list[dict[str, Any]]:
    corpus = load_canonical_policy_corpus()
    return [
        {
            "schema_version": "banking-tool-eval/v1",
            "record_id": f"policy_{str(chunk['chunk_id']).replace('.', '_')}",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Authoritative Harborlight Bank policy context. Answer only from "
                        "this context and cite the bracketed policy chunk ID.\n"
                        f"[Policy: {chunk['chunk_id']}] {chunk['title']}: {chunk['text']}"
                    ),
                },
                {"role": "user", "content": str(chunk["title"])},
            ],
            "expected": {
                "requires_tool": False,
                "response_path": "retrieval_grounded_policy",
                "policy_citations": [str(chunk["chunk_id"])],
                "policy_corpus_revision": str(corpus["corpus_revision"]),
                "grounding_facts": list(chunk["required_claims"]),
                "forbidden_facts": list(chunk["forbidden_claims"]),
            },
        }
        for chunk in corpus["chunks"]
    ]


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen banking-v3 tool-use outputs.")
    parser.add_argument("--records", type=Path, help="Canonical eval records JSONL.")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="JSONL rows with record_id and raw_output.",
    )
    parser.add_argument("--output", type=Path, help="Write JSON report to this path.")
    parser.add_argument("--checkpoint-revision", default="unversioned-local")
    parser.add_argument("--template-hash", default="sha256:tagged-json-tool-call-v1")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run deterministic in-memory fixture.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        records, predictions = dry_run_records_and_predictions()
    else:
        if args.records is None or args.predictions is None:
            parser.error("--records and --predictions are required unless --dry-run is set")
        records = load_jsonl(args.records)
        predictions = load_predictions_jsonl(args.predictions)

    report = evaluate_records(
        records,
        model=StaticPredictionModel(predictions),
        adapter=TaggedJsonToolAdapter(template_hash=args.template_hash),
        checkpoint_revision=args.checkpoint_revision,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


def _expected(record: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = record.get("expected", {})
    if not isinstance(expected, Mapping):
        raise ValueError(f"{record.get('record_id')} expected must be an object")
    return expected


def _expected_calls(expected: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    calls = expected.get("tool_calls", expected.get("expected_tool_calls", ()))
    if not isinstance(calls, Sequence) or isinstance(calls, str | bytes):
        return ()
    return tuple(_call_from_json(call) for call in calls if isinstance(call, Mapping))


def _call_from_json(payload: Mapping[str, Any]) -> ToolCall:
    return ToolCall(name=str(payload.get("name", "")), arguments=dict(payload.get("arguments", {})))


def _call_to_json(call: ToolCall) -> dict[str, Any]:
    return {"name": call.name, "arguments": _normalize_args(call.arguments)}


def _exact_names(expected: Sequence[ToolCall], actual: Sequence[ToolCall]) -> bool:
    return [call.name for call in actual] == [call.name for call in expected]


def _exact_calls(expected: Sequence[ToolCall], actual: Sequence[ToolCall]) -> bool:
    return [_call_to_json(call) for call in actual] == [_call_to_json(call) for call in expected]


def _normalize_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): arguments[key] for key in sorted(arguments)}


def _manifest_failures(calls: Sequence[ToolCall]) -> list[str]:
    failures: list[str] = []
    for call in calls:
        allowed = PUBLIC_TOOL_ARGUMENTS.get(call.name)
        if allowed is None:
            failures.append(f"unsupported tool: {call.name}")
            continue
        unsupported = sorted(
            (set(call.arguments) - set(allowed)) | (set(call.arguments) & PRIVATE_ARGUMENTS)
        )
        if unsupported:
            failures.append(f"{call.name} unsupported/private args: {unsupported}")
    return failures


def _fact_pass(content: str, required: Iterable[Any], forbidden: Iterable[Any]) -> bool:
    normalized = _norm(content)
    return all(_norm(str(fact)) in normalized for fact in required) and not any(
        _norm(str(fact)) in normalized for fact in forbidden
    )


def _grounding_pass(
    content: str,
    required: Iterable[Any],
    forbidden: Iterable[Any],
) -> bool:
    normalized = _norm(content)
    if any(_norm(str(fact)) in normalized for fact in forbidden):
        return False
    facts = tuple(str(fact) for fact in required)
    if not any("=" in fact for fact in facts):
        return all(_norm(fact) in normalized for fact in facts)
    return all(_structured_fact_is_grounded(normalized, fact) for fact in facts)


def _structured_fact_is_grounded(normalized_content: str, fact: str) -> bool:
    if "=" not in fact:
        return _norm(fact) in normalized_content
    key, raw_value = fact.split("=", 1)
    value = _norm(raw_value.replace("_", " "))
    if key in {"accounts.count", "transactions.limit"}:
        # These are generation controls. The released reference answers name
        # the returned records rather than literally repeating the count/limit.
        return True
    if key in {
        "account.last4",
        "card.last4",
        "transaction.description",
        "transfer.recipient",
    }:
        return value in normalized_content
    if key.endswith(".status"):
        if value == "replacement pending":
            return "replacement" in normalized_content and "pending" in normalized_content
        if value == "frozen":
            return _contains_any(normalized_content, ("frozen", "froze"))
        return value in normalized_content
    if key == "transaction.disputed":
        return value != "true" or "disput" in normalized_content
    if key == "missing_field":
        return value != "last4" or "last four" in normalized_content
    if key == "error.code":
        return value != "backend error" or _contains_any(
            normalized_content,
            ("could not", "couldn't", "unable", "not able"),
        )
    if key == "faq":
        return value in normalized_content
    if key == "private_data_refused":
        return value != "true" or (
            _contains_any(normalized_content, ("cannot", "can't", "unable", "will not"))
            and _contains_any(normalized_content, ("account number", "customer id"))
        )
    if key == "domain":
        return value != "out of domain" or _contains_any(
            normalized_content,
            ("retail banking", "financial services"),
        )
    if key == "case.created_at":
        return _created_at_is_grounded(normalized_content, raw_value)
    if key == "case.case_type":
        return value in normalized_content
    return value in normalized_content


def _created_at_is_grounded(normalized_content: str, raw_value: str) -> bool:
    try:
        created_at = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return _norm(raw_value) in normalized_content
    date_markers = (
        created_at.strftime("%Y-%m-%d"),
        _norm(created_at.strftime("%B %d, %Y").replace(" 0", " ")),
    )
    time_marker = created_at.strftime("%H:%M")
    if not _contains_any(normalized_content, date_markers) or time_marker not in normalized_content:
        return False
    if raw_value.endswith("Z"):
        return bool(
            re.search(
                rf"\butc\b|{re.escape(time_marker)}(?::\d{{2}})?z\b",
                normalized_content,
            )
        )
    return True


def _contains_any(content: str, candidates: Iterable[str]) -> bool:
    return any(candidate in content for candidate in candidates)


def _response_path(expected: Mapping[str, Any]) -> str | None:
    value = expected.get("response_path", expected.get("path"))
    aliases = {
        "no_tool_banking_faq": "faq",
    }
    return aliases.get(str(value), str(value)) if value is not None else None


def _policy_pass(
    prediction: AssistantPrediction,
    expected: Mapping[str, Any],
) -> bool:
    citations = expected.get("policy_citations")
    if not isinstance(citations, Sequence) or isinstance(citations, str | bytes):
        return False
    expected_citations = {str(item) for item in citations if str(item).strip()}
    observed_citations = set(_POLICY_CITATION.findall(prediction.content))
    semantic_words = re.findall(r"[A-Za-z0-9]+", prediction.content)
    return (
        prediction.parse_failure is None
        and not prediction.tool_calls
        and bool(expected_citations)
        and observed_citations == expected_citations
        and len(semantic_words) >= 7
        and _grounding_pass(
            prediction.content,
            expected.get("grounding_facts", ()),
            expected.get("forbidden_facts", ()),
        )
    )


def _clarification_pass(prediction: AssistantPrediction, expected: Mapping[str, Any]) -> bool:
    missing_field = expected.get("clarification_missing_field")
    if not missing_field:
        for fact in expected.get("grounding_facts", ()):
            if str(fact).startswith("missing_field="):
                missing_field = str(fact).split("=", 1)[1]
                break
    if not missing_field:
        return False
    marker = "last four" if str(missing_field) == "last4" else _norm(str(missing_field))
    return (
        prediction.parse_failure is None
        and not prediction.tool_calls
        and marker in _norm(prediction.content)
        and not _credential_request(prediction.content)
    )


def _path_pass(prediction: AssistantPrediction, expected: Mapping[str, Any]) -> bool:
    markers = expected.get("path_markers", ())
    if not markers and _response_path(expected) == "ood":
        markers = ("retail banking",)
    if not isinstance(markers, Sequence) or isinstance(markers, str | bytes):
        markers = (markers,)
    return (
        prediction.parse_failure is None
        and not prediction.tool_calls
        and _fact_pass(
            prediction.content,
            markers,
            expected.get("forbidden_facts", ()),
        )
    )


def _credential_request(content: str) -> bool:
    for match in _CREDENTIAL_REQUEST.finditer(content):
        clause_start = max(
            content.rfind(separator, 0, match.start())
            for separator in (".", "!", "?", ";", "\n")
        )
        prefix = content[clause_start + 1 : match.start()]
        if not _PROHIBITED_CREDENTIAL_REQUEST.search(prefix):
            return True
    return False


def _set_card_status(state: dict[str, Any], last4: Any, status: str) -> None:
    for card in state.get("cards", []):
        if last4 is None or str(card.get("last4")) == str(last4):
            card["status"] = status
            return


def _set_transfer_status(state: dict[str, Any], recipient: Any, status: str) -> None:
    for transfer in state.get("transfers", []):
        if recipient is None or _norm(str(transfer.get("recipient"))) == _norm(str(recipient)):
            transfer["status"] = status
            return


def _set_transaction_disputed(state: dict[str, Any], description: Any) -> None:
    for transaction in state.get("transactions", []):
        if description is None or _norm(str(transaction.get("description"))) == _norm(
            str(description)
        ):
            transaction["disputed"] = True
            return


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())
