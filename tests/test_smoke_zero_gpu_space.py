from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from hello_slm.banking_servicing_alignment_data import build_screenshot_regression_fixture


def _load_module() -> ModuleType:
    path = Path("scripts/retail_bank/smoke_zero_gpu_space.py")
    spec = importlib.util.spec_from_file_location("smoke_zero_gpu_space", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _diagnostics(case: dict[str, Any]) -> str:
    expected = case["expected"]
    tool = expected["tool_name"]
    tool_result = f"{tool}: success" if tool else "none"
    exposed = f'["{tool}"]' if tool else "[]"
    passes = 1 if tool else 0
    return (
        "### Diagnostic summary\n\n"
        f"- Outcome: `{'base tool' if tool else 'OOD stock response'}`\n"
        "- Route hierarchy: `banking → servicing → service → view service cases → "
        "execute tool → resolved`\n"
        f"- Granite passes: `{passes}`\n"
        f"- Tool result: `{tool_result}`\n"
        "- Effective grounding/source: `application response`\n\n"
        "### Full technical details\n\n"
        "- V6 domain: `banking`\n"
        "- V6 lane: `servicing`\n"
        "- V6 family: `service`\n"
        "- V6 intent: `view_service_cases`\n"
        "- V6 action: `execute_tool`\n"
        "- V6 entity resolution: `resolved`\n"
        f"- Exposed tools: `{exposed}`\n"
        f"**Tool results**\n- `{tool or 'none'}`: success"
    )


def test_remote_smoke_uses_static_auth_and_all_nine_isolated_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixtures = build_screenshot_regression_fixture()
    clients: list[Any] = []

    class FakeClient:
        def __init__(self, case: dict[str, Any]) -> None:
            self.case = case
            self.calls: list[tuple[Any, ...]] = []

        def predict(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
            self.calls.append((*args, kwargs))
            response = " ".join(self.case["expected"]["response_properties"]["must_include"])
            chat = [*self.case["history"], {"role": "assistant", "content": response}]
            return "", chat, "snapshot", "activity", _diagnostics(self.case)

    def factory(src: str, *, auth: tuple[str, str], verbose: bool) -> FakeClient:
        assert src == "spkc83/test-space"
        assert auth == ("alex.demo", "existing-static-password")
        assert verbose is False
        client = FakeClient(fixtures[len(clients)])
        clients.append(client)
        return client

    args = module.parse_args(
        [
            "--space-id",
            "spkc83/test-space",
            "--password",
            "existing-static-password",
            "--execute",
        ]
    )
    result = module.run(args, factory)

    assert result["case_count"] == 9
    assert result["passed"] == 9
    assert len(clients) == 9
    assert all(client.calls[0][-1] == {"api_name": "/chat"} for client in clients)
    assert [client.calls[0][0] for client in clients] == [case["current"] for case in fixtures]
    assert [client.calls[0][1] for client in clients] == [case["history"] for case in fixtures]


def test_remote_smoke_plan_is_non_networked_and_does_not_expose_password() -> None:
    module = _load_module()
    args = module.parse_args(
        ["--space-id", "spkc83/test-space", "--password", "existing-static-password"]
    )

    result = module.run(args, lambda *_args, **_kwargs: pytest.fail("must not connect"))

    assert result["mode"] == "plan"
    assert result["case_count"] == 9
    assert "password" not in str(result).casefold()


def test_remote_smoke_rejects_generic_hidden_error_diagnostics() -> None:
    module = _load_module()
    case = build_screenshot_regression_fixture()[0]
    diagnostics = f"{_diagnostics(case)}\n- Error: `internal server error`"
    response = "created"

    with pytest.raises(module.SmokeError, match="hidden error"):
        module.validate_case(
            case,
            ("", [{"role": "assistant", "content": response}], "", "", diagnostics),
        )
