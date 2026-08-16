from __future__ import annotations

from types import SimpleNamespace

from diagnostics import diagnostic_summary, sanitized_error_text


def test_shared_summary_is_concise_and_covers_effective_execution() -> None:
    route = {
        "route": "in_domain",
        "domain": "banking",
        "lane": "servicing",
        "family": "cards",
        "intent": "freeze_card",
        "action": "execute_tool",
        "entity_resolution": "resolved",
        "relation_probabilities": {"context_dependent": 0.91},
    }
    calls = (SimpleNamespace(name="freeze_card"),)
    results = ({"ok": False, "error": {"code": "CARD_ALREADY_FROZEN"}},)
    passes = (SimpleNamespace(label="base", raw_output="secret raw output"),)

    rendered = diagnostic_summary(
        route=route,
        calls=calls,
        results=results,
        response_path="base_tool",
        model_passes=passes,
        policy_sources=(),
    )

    assert "Outcome: `base tool`" in rendered
    assert "banking → servicing → cards → freeze card → execute tool → resolved" in rendered
    assert "Granite passes: `1`" in rendered
    assert "freeze_card: error (CARD_ALREADY_FROZEN)" in rendered
    assert "Effective grounding/source: `tool result: freeze_card`" in rendered
    assert "SHA-256" not in rendered
    assert "secret raw output" not in rendered
    assert "relation_probabilities" not in rendered


def test_shared_summary_reports_policy_as_effective_source() -> None:
    rendered = diagnostic_summary(
        route={"route": "in_domain", "lane": "policy", "intent": "policy_knowledge"},
        calls=(),
        results=(),
        response_path="policy_grounded",
        model_passes=(SimpleNamespace(label="policy_grounded"),),
        policy_sources=("disputes.timeline.us.v1",),
    )

    assert "Effective grounding/source: `policy: disputes.timeline.us.v1`" in rendered


def test_shared_summary_prefers_effective_entity_grounding() -> None:
    rendered = diagnostic_summary(
        route={
            "route": "in_domain",
            "lane": "servicing",
            "intent": "freeze_card",
            "entity_resolution": "resolved",
            "argument_constraints": {"last4": "4821"},
            "entity_grounding_source": "live_candidate",
        },
        calls=(SimpleNamespace(name="freeze_card"),),
        results=({"ok": True},),
        response_path="base_tool",
        model_passes=(SimpleNamespace(label="base"),),
    )

    assert "Effective grounding/source: `last4=4821 via live_candidate`" in rendered
    assert "Grounding source: `live_candidate`" in rendered


def test_shared_summary_reports_ambiguous_effective_grounding_without_raw_json() -> None:
    rendered = diagnostic_summary(
        route={
            "route": "in_domain",
            "lane": "servicing",
            "intent": "freeze_card",
            "entity_resolution": "ambiguous",
            "argument_constraints": {},
            "entity_grounding_source": "live_candidate",
        },
        calls=(),
        results=(),
        response_path="clarification",
        model_passes=(),
    )

    assert "Effective grounding/source: `ambiguous via live_candidate`" in rendered
    assert "argument_constraints" not in rendered


def test_error_text_is_safe_for_inline_diagnostics() -> None:
    rendered = sanitized_error_text("bad `payload`\napi_key=super-secret-value password: hunter2")

    assert rendered == "bad 'payload' api_key=&lt;redacted&gt; password:&lt;redacted&gt;"
    assert "super-secret-value" not in rendered
    assert "hunter2" not in rendered
