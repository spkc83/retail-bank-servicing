#!/usr/bin/env python
"""Run the nine authenticated screenshot-regression turns against a Gradio Space."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from hello_slm.banking_servicing_alignment_data import build_screenshot_regression_fixture

DEFAULT_USERNAME = "alex.demo"
PASSWORD_ENV = "RETAIL_BANK_DEMO_PASSWORD"
_GENERIC_ERRORS = (
    "internal server error",
    "traceback (most recent call last)",
    "something went wrong",
    "classifier_error",
    "model failure",
)


class SmokeError(ValueError):
    """Raised when the remote Space violates the smoke contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", required=True)
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=os.environ.get(PASSWORD_ENV))
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def plan(args: argparse.Namespace) -> dict[str, Any]:
    fixtures = build_screenshot_regression_fixture()
    return {
        "contract": "retail-bank-zero-gpu-smoke/v1",
        "space_id": str(args.space_id),
        "username": str(args.username),
        "case_count": len(fixtures),
        "case_ids": [str(case["record_id"]) for case in fixtures],
    }


def run(
    args: argparse.Namespace,
    client_factory: Callable[..., Any],
) -> dict[str, Any]:
    smoke_plan = plan(args)
    if not args.execute:
        return {**smoke_plan, "mode": "plan"}
    if not isinstance(args.password, str) or not args.password:
        raise SmokeError(f"--password or {PASSWORD_ENV} is required with --execute")

    fixtures = build_screenshot_regression_fixture()
    results = []
    for case in fixtures:
        client = client_factory(
            str(args.space_id),
            auth=(str(args.username), args.password),
            verbose=False,
        )
        output = client.predict(
            str(case["current"]),
            list(case["history"]),
            api_name="/chat",
        )
        results.append(validate_case(case, output))
    return {
        **smoke_plan,
        "mode": "executed",
        "passed": len(results),
        "cases": results,
    }


def validate_case(case: Mapping[str, Any], output: Any) -> dict[str, Any]:
    if not isinstance(output, list | tuple) or len(output) != 10:
        raise SmokeError(f"{case['record_id']}: /chat returned an unexpected output shape")
    chat, diagnostics = output[1], output[5]
    if not isinstance(diagnostics, str):
        raise SmokeError(f"{case['record_id']}: diagnostics were not text")
    lowered = diagnostics.casefold()
    if "\n- error: `" in lowered or any(marker in lowered for marker in _GENERIC_ERRORS):
        raise SmokeError(f"{case['record_id']}: diagnostics expose a generic hidden error")

    _assert_router_tuple(str(case["record_id"]), diagnostics)
    passes = _granite_passes(str(case["record_id"]), diagnostics)
    expected = case["expected"]
    tool_name = expected["tool_name"]
    if tool_name:
        _require(diagnostics, f'Exposed tools: `["{tool_name}"]`', case)
        _require(diagnostics, f"{tool_name}: success", case)
        if passes < 1:
            raise SmokeError(f"{case['record_id']}: Granite did not record a model event")
    else:
        _require(diagnostics, "Exposed tools: `[]`", case)
        _require(diagnostics, "Tool result: `none`", case)

    response = _assistant_response(chat)
    for required in expected["response_properties"]["must_include"]:
        if str(required).casefold() not in response.casefold():
            raise SmokeError(f"{case['record_id']}: response omitted {required!r}")
    for forbidden in expected["response_properties"]["must_not_include"]:
        if str(forbidden).casefold() in response.casefold():
            raise SmokeError(f"{case['record_id']}: response included {forbidden!r}")
    return {
        "record_id": str(case["record_id"]),
        "granite_passes": passes,
        "tool": tool_name,
    }


def _assert_router_tuple(record_id: str, diagnostics: str) -> None:
    for label in ("domain", "lane", "family", "intent", "action", "entity resolution"):
        match = re.search(rf"- V6 {label}: `([^`]+)`", diagnostics, flags=re.IGNORECASE)
        if match is None or match.group(1).startswith("not available"):
            raise SmokeError(f"{record_id}: diagnostics omitted router {label}")


def _granite_passes(record_id: str, diagnostics: str) -> int:
    match = re.search(r"- Granite passes: `(\d+)`", diagnostics)
    if match is None:
        raise SmokeError(f"{record_id}: diagnostics omitted the Granite event count")
    return int(match.group(1))


def _assistant_response(chat: Any) -> str:
    if not isinstance(chat, list):
        raise SmokeError("/chat did not return message history")
    for message in reversed(chat):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str):
                return content
    raise SmokeError("/chat did not return an assistant response")


def _require(diagnostics: str, expected: str, case: Mapping[str, Any]) -> None:
    if expected not in diagnostics:
        raise SmokeError(f"{case['record_id']}: diagnostics omitted {expected!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute:
        try:
            from gradio_client import Client
        except ModuleNotFoundError as error:
            raise SmokeError("gradio_client is required for remote smoke execution") from error
        result = run(args, Client)
    else:
        result = run(args, lambda *_args, **_kwargs: None)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeError as error:
        print(json.dumps({"error": str(error)}, sort_keys=True))
        raise SystemExit(2) from error
