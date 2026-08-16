"""Shared concise diagnostics for the hosted and local POC interfaces."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from typing import Any

_SECRET_PATTERN = re.compile(
    r"\b(api[_ -]?key|token|password|secret)\s*([=:])\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    flags=re.IGNORECASE,
)
_MAX_ERROR_CHARS = 300


def sanitized_error_text(error: object) -> str:
    """Return a bounded, single-line, Markdown-safe error message."""
    collapsed = " ".join(str(error).split())
    redacted = _SECRET_PATTERN.sub(r"\1\2<redacted>", collapsed)
    bounded = redacted[:_MAX_ERROR_CHARS]
    if len(redacted) > _MAX_ERROR_CHARS:
        bounded = f"{bounded.rstrip()}…"
    return html.escape(bounded.replace("`", "'"), quote=False)


def error_model_passes(error: object) -> tuple[Any, ...]:
    """Retain model traces carried by an otherwise generic exception."""
    traces = getattr(error, "model_passes", ())
    return tuple(traces) if isinstance(traces, list | tuple) else ()


def diagnostic_summary(
    *,
    route: dict[str, Any],
    calls: Sequence[Any],
    results: Sequence[dict[str, Any]],
    response_path: str,
    model_passes: Sequence[Any],
    policy_sources: Sequence[str] = (),
    error: object | None = None,
) -> str:
    """Render the same compact, customer-safe execution summary for both UIs."""
    hierarchy = " → ".join(_route_hierarchy(route))
    tool_results = _tool_result_summary(calls, results)
    grounding, grounding_source = _effective_grounding(
        route,
        calls,
        model_passes,
        policy_sources,
    )
    grounding_source_line = (
        f"\n- Grounding source: `{grounding_source}`" if grounding_source else ""
    )
    error_line = (
        f"\n- Error: `{sanitized_error_text(error)}`" if error is not None and str(error) else ""
    )
    return (
        "### Diagnostic summary\n\n"
        f"- Outcome: `{_display(response_path)}`\n"
        f"- Route hierarchy: `{hierarchy}`\n"
        f"- Granite passes: `{len(model_passes)}`\n"
        f"- Tool result: `{tool_results}`\n"
        f"- Effective grounding/source: `{grounding}`"
        f"{grounding_source_line}"
        f"{error_line}"
    )


def _route_hierarchy(route: dict[str, Any]) -> tuple[str, ...]:
    compatibility = "not available"
    return tuple(
        _display(value)
        for value in (
            route.get("domain", compatibility),
            route.get("lane", compatibility),
            route.get("family", route.get("capability", compatibility)),
            route.get("fine_intent", route.get("intent", compatibility)),
            route.get("action", compatibility),
            route.get("entity_resolution", compatibility),
        )
    )


def _tool_result_summary(
    calls: Sequence[Any],
    results: Sequence[dict[str, Any]],
) -> str:
    if not calls:
        return "none"
    summaries: list[str] = []
    for index, call in enumerate(calls):
        result = results[index] if index < len(results) else {}
        status = "success" if result.get("ok") else "error"
        error_payload = result.get("error")
        code = error_payload.get("code") if isinstance(error_payload, dict) else None
        suffix = f" ({code})" if isinstance(code, str) and code else ""
        summaries.append(f"{getattr(call, 'name', 'unknown')}: {status}{suffix}")
    return "; ".join(summaries)


def _effective_grounding(
    route: dict[str, Any],
    calls: Sequence[Any],
    model_passes: Sequence[Any],
    policy_sources: Sequence[str],
) -> tuple[str, str | None]:
    if policy_sources:
        return f"policy: {', '.join(str(source) for source in policy_sources)}", None
    source = route.get("entity_grounding_source")
    if isinstance(source, str) and source not in {
        "",
        "not_applicable",
        "not available",
        "not available (V3 compatibility)",
    }:
        constraints = route.get("argument_constraints")
        if isinstance(constraints, dict) and constraints:
            grounded = ", ".join(
                f"{_display(key)}={_display(value)}"
                for key, value in sorted(constraints.items(), key=lambda item: str(item[0]))
            )
        else:
            grounded = _display(route.get("entity_resolution", "unresolved"))
        return f"{grounded} via {source}", source
    if calls:
        names = ", ".join(str(getattr(call, "name", "unknown")) for call in calls)
        return f"tool result: {names}", None
    if model_passes:
        return "Granite generation", None
    return "application response", None


def _display(value: object) -> str:
    return str(value if value is not None else "not available").replace("_", " ").strip()
