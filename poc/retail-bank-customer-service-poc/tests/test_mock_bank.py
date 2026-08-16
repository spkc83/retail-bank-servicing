from __future__ import annotations

import json
from pathlib import Path

import pytest

from mock_bank import SessionBankRegistry

ROOT = Path(__file__).parents[1]


def registry() -> SessionBankRegistry:
    return SessionBankRegistry.from_json(ROOT / "synthetic_bank.json", max_sessions=4)


def test_seed_is_explicitly_synthetic_and_contains_two_login_scoped_customers() -> None:
    payload = json.loads((ROOT / "synthetic_bank.json").read_text(encoding="utf-8"))

    assert payload["contract"] == "synthetic-retail-bank-v1"
    assert "fictional" in payload["notice"].lower()
    assert {customer["login"] for customer in payload["customers"]} == {
        "alex.demo",
        "maya.demo",
    }
    assert "password" not in json.dumps(payload).lower()


def test_sessions_and_authenticated_customers_are_isolated() -> None:
    bank = registry()

    bank.execute("alex.demo", "alex-session", "freeze_card", {"last4": "4821"})

    assert bank.snapshot("alex.demo", "alex-session")["cards"][0]["status"] == "frozen"
    assert bank.snapshot("alex.demo", "second-alex-session")["cards"][0]["status"] == "active"
    assert bank.snapshot("maya.demo", "maya-session")["cards"][0]["last4"] == "7319"


def test_file_backed_state_is_shared_across_worker_registries(tmp_path: Path) -> None:
    first = SessionBankRegistry.from_json(
        ROOT / "synthetic_bank.json",
        database_dir=tmp_path,
    )
    second = SessionBankRegistry.from_json(
        ROOT / "synthetic_bank.json",
        database_dir=tmp_path,
    )

    first.execute("alex.demo", "shared-session", "freeze_card", {"last4": "4821"})

    assert second.snapshot("alex.demo", "shared-session")["cards"][0]["status"] == "frozen"
    assert second.snapshot("alex.demo", "other-session")["cards"][0]["status"] == "active"

    second.reset("alex.demo", "shared-session")

    assert first.snapshot("alex.demo", "shared-session")["cards"][0]["status"] == "active"


def test_supported_mock_actions_mutate_only_session_database() -> None:
    bank = registry()
    user = "alex.demo"
    session = "action-session"

    disputed = bank.execute(
        user,
        session,
        "dispute_transaction",
        {"description": "North Harbor Market"},
    )
    cancelled = bank.execute(
        user,
        session,
        "cancel_transfer",
        {"recipient": "River Consulting"},
    )
    replaced = bank.execute(user, session, "replace_card", {"last4": "4821"})

    snapshot = bank.snapshot(user, session)
    assert disputed["transaction"]["disputed"] is True
    assert cancelled["transfer"]["status"] == "cancelled"
    assert replaced["card"]["status"] == "replacement_pending"
    assert len(snapshot["service_cases"]) == 3


def test_read_bundle_returns_ordered_consistent_results_without_mutation() -> None:
    bank = registry()

    result = bank.execute_read_bundle(
        "alex.demo",
        "read-session",
        (
            ("list_transfers", {}),
            ("list_transactions", {"limit": 3}),
        ),
    )

    assert tuple(result) == ("list_transfers", "list_transactions")
    assert len(result["list_transfers"]["transfers"]) == 2
    assert len(result["list_transactions"]["transactions"]) == 3
    assert bank.snapshot("alex.demo", "read-session")["transfers"][0]["status"] == "pending"


def test_read_bundle_rejects_write_tools_before_execution() -> None:
    bank = registry()

    with pytest.raises(ValueError, match="read bundle"):
        bank.execute_read_bundle(
            "alex.demo",
            "read-session",
            (("list_accounts", {}), ("freeze_card", {})),
        )

    assert bank.snapshot("alex.demo", "read-session")["cards"][0]["status"] == "active"


def test_tool_scope_rejects_unknown_users_sessions_and_cross_customer_arguments() -> None:
    bank = registry()

    with pytest.raises(ValueError, match="unknown authenticated user"):
        bank.snapshot("unknown.demo", "session")
    with pytest.raises(ValueError, match="session hash"):
        bank.snapshot("alex.demo", "")
    with pytest.raises(ValueError, match="unsupported arguments"):
        bank.execute(
            "alex.demo",
            "session",
            "list_accounts",
            {"customer_id": "cust_maya"},
        )


def test_customer_facing_write_selectors_match_synthetic_records() -> None:
    bank = registry()

    cancelled = bank.execute(
        "alex.demo",
        "friendly-selectors",
        "cancel_transfer",
        {"recipient": "river consulting"},
    )
    disputed = bank.execute(
        "alex.demo",
        "friendly-selectors",
        "dispute_transaction",
        {"description": "north harbor market"},
    )

    assert cancelled["transfer"]["recipient"] == "River Consulting"
    assert cancelled["transfer"]["status"] == "cancelled"
    assert disputed["transaction"]["description"] == "North Harbor Market"
    assert disputed["transaction"]["disputed"] is True


def test_customer_facing_write_selector_reports_no_match() -> None:
    bank = registry()

    with pytest.raises(ValueError, match="matching"):
        bank.execute(
            "alex.demo",
            "friendly-selectors",
            "cancel_transfer",
            {"recipient": "Nobody"},
        )


@pytest.mark.parametrize(
    "tool_name",
    ["freeze_card", "replace_card", "dispute_transaction", "cancel_transfer"],
)
def test_write_tools_reject_selector_free_execution(tool_name: str) -> None:
    bank = registry()

    with pytest.raises(ValueError, match="requires public selector"):
        bank.execute("alex.demo", "no-default", tool_name, {})


def test_write_eligibility_is_revalidated_inside_each_transaction() -> None:
    bank = registry()
    user = "alex.demo"
    session = "eligibility"

    bank.execute(user, session, "freeze_card", {"last4": "4821"})
    with pytest.raises(ValueError, match="eligible"):
        bank.execute(user, session, "freeze_card", {"last4": "4821"})

    bank.execute(user, session, "replace_card", {"last4": "4821"})
    with pytest.raises(ValueError, match="eligible"):
        bank.execute(user, session, "replace_card", {"last4": "4821"})

    bank.execute(
        user,
        session,
        "dispute_transaction",
        {"description": "North Harbor Market"},
    )
    with pytest.raises(ValueError, match="eligible"):
        bank.execute(
            user,
            session,
            "dispute_transaction",
            {"description": "North Harbor Market"},
        )

    bank.execute(
        user,
        session,
        "cancel_transfer",
        {"recipient": "River Consulting"},
    )
    with pytest.raises(ValueError, match="eligible"):
        bank.execute(
            user,
            session,
            "cancel_transfer",
            {"recipient": "River Consulting"},
        )
