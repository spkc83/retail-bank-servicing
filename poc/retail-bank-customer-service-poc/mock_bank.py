from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

READ_TOOLS = {
    "list_accounts",
    "list_cards",
    "list_service_cases",
    "list_transactions",
    "list_transfers",
}
WRITE_TOOLS = {
    "cancel_transfer",
    "dispute_transaction",
    "freeze_card",
    "replace_card",
}
SUPPORTED_TOOLS = READ_TOOLS | WRITE_TOOLS
TOOL_ARGUMENTS = {
    "list_accounts": set(),
    "list_cards": set(),
    "list_service_cases": set(),
    "list_transactions": {"limit"},
    "list_transfers": set(),
    "cancel_transfer": {"recipient", "transfer_id"},
    "dispute_transaction": {"description", "transaction_id"},
    "freeze_card": {"last4"},
    "replace_card": {"last4"},
}
T = TypeVar("T")


@dataclass
class SessionEntry:
    connection: sqlite3.Connection
    last_access: float
    lock: threading.RLock = field(default_factory=threading.RLock)


class SessionBankRegistry:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        ttl_seconds: float = 7200,
        max_sessions: int = 32,
        database_dir: str | Path | None = None,
    ) -> None:
        if payload.get("contract") != "synthetic-retail-bank-v1":
            raise ValueError("unexpected synthetic bank contract")
        if ttl_seconds <= 0 or max_sessions < 1:
            raise ValueError("session limits must be positive")
        customers = payload.get("customers")
        if not isinstance(customers, list):
            raise ValueError("synthetic bank customers must be a list")
        self._customers = {
            str(customer["login"]): customer
            for customer in customers
            if isinstance(customer, dict) and isinstance(customer.get("login"), str)
        }
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._database_dir = Path(database_dir) if database_dir is not None else None
        if self._database_dir is not None:
            self._database_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._sessions: dict[tuple[str, str], SessionEntry] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        ttl_seconds: float = 7200,
        max_sessions: int = 32,
        database_dir: str | Path | None = None,
    ) -> SessionBankRegistry:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            payload,
            ttl_seconds=ttl_seconds,
            max_sessions=max_sessions,
            database_dir=database_dir,
        )

    def snapshot(self, username: str, session_hash: str) -> dict[str, Any]:
        entry = self._entry(username, session_hash)
        with entry.lock:
            return _snapshot(entry.connection)

    def reset(self, username: str, session_hash: str) -> dict[str, Any]:
        key = self._validated_key(username, session_hash)
        if self._database_dir is not None:
            entry = self._entry(username, session_hash)
            with entry.lock:
                _reset_seed(entry.connection, self._customers[username])
                return _snapshot(entry.connection)
        with self._lock:
            if key in self._sessions:
                self._sessions.pop(key).connection.close()
        return self.snapshot(username, session_hash)

    def execute(
        self,
        username: str,
        session_hash: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        result, _ = self.execute_atomic(
            username,
            session_hash,
            tool_name,
            arguments,
            finalize=lambda _result: None,
        )
        return result

    def execute_atomic(
        self,
        username: str,
        session_hash: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        finalize: Callable[[dict[str, Any]], T],
    ) -> tuple[dict[str, Any], T]:
        """Commit only after the caller validates its final response."""

        if tool_name not in SUPPORTED_TOOLS:
            raise ValueError(f"unsupported tool: {tool_name}")
        allowed_arguments = TOOL_ARGUMENTS[tool_name]
        extras = set(arguments) - allowed_arguments
        if extras:
            raise ValueError(f"unsupported arguments for {tool_name}: {sorted(extras)}")
        entry = self._entry(username, session_hash)
        with entry.lock:
            entry.connection.execute("BEGIN IMMEDIATE")
            try:
                result = _execute(entry.connection, tool_name, arguments)
                finalized = finalize(result)
            except Exception:
                entry.connection.rollback()
                raise
            entry.connection.commit()
            return result, finalized

    def execute_read_bundle(
        self,
        username: str,
        session_hash: str,
        calls: tuple[tuple[str, dict[str, Any]], ...],
    ) -> dict[str, dict[str, Any]]:
        """Execute an ordered read-only workflow against one consistent snapshot."""

        if not calls:
            raise ValueError("read bundle must contain at least one tool")
        for tool_name, arguments in calls:
            if tool_name not in READ_TOOLS:
                raise ValueError(f"read bundle cannot execute {tool_name}")
            extras = set(arguments) - TOOL_ARGUMENTS[tool_name]
            if extras:
                raise ValueError(f"unsupported arguments for {tool_name}: {sorted(extras)}")
        entry = self._entry(username, session_hash)
        with entry.lock:
            entry.connection.execute("BEGIN")
            try:
                result = {
                    tool_name: _execute(entry.connection, tool_name, arguments)
                    for tool_name, arguments in calls
                }
            except Exception:
                entry.connection.rollback()
                raise
            entry.connection.commit()
            return result

    def _validated_key(self, username: str, session_hash: str) -> tuple[str, str]:
        if username not in self._customers:
            raise ValueError("unknown authenticated user")
        if not isinstance(session_hash, str) or not session_hash.strip():
            raise ValueError("session hash must be a non-empty string")
        return username, session_hash

    def _entry(self, username: str, session_hash: str) -> SessionEntry:
        key = self._validated_key(username, session_hash)
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            entry = self._sessions.get(key)
            if entry is None:
                while len(self._sessions) >= self._max_sessions:
                    oldest_key = min(
                        self._sessions,
                        key=lambda candidate: self._sessions[candidate].last_access,
                    )
                    self._sessions.pop(oldest_key).connection.close()
                database = (
                    ":memory:"
                    if self._database_dir is None
                    else str(self._database_path(username, session_hash))
                )
                connection = sqlite3.connect(
                    database,
                    check_same_thread=False,
                    timeout=30,
                )
                connection.row_factory = sqlite3.Row
                _initialize(connection)
                customer_count = connection.execute("SELECT COUNT(*) FROM customer").fetchone()[0]
                if customer_count == 0:
                    _seed_customer(connection, self._customers[username])
                entry = SessionEntry(connection=connection, last_access=now)
                self._sessions[key] = entry
            entry.last_access = now
            return entry

    def _database_path(self, username: str, session_hash: str) -> Path:
        if self._database_dir is None:
            raise RuntimeError("database directory is not configured")
        digest = hashlib.sha256(f"{username}\0{session_hash}".encode()).hexdigest()
        return self._database_dir / f"{digest}.sqlite3"

    def _purge(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._sessions.items()
            if now - entry.last_access > self._ttl_seconds
        ]
        for key in expired:
            self._sessions.pop(key).connection.close()


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS customer (
            customer_id TEXT PRIMARY KEY,
            login TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            segment TEXT NOT NULL,
            city TEXT NOT NULL,
            member_since TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS account (
            account_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customer(customer_id),
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            last4 TEXT NOT NULL,
            currency TEXT NOT NULL,
            available_balance_cents INTEGER NOT NULL,
            current_balance_cents INTEGER NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS card (
            card_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customer(customer_id),
            account_id TEXT NOT NULL REFERENCES account(account_id),
            name TEXT NOT NULL,
            last4 TEXT NOT NULL,
            status TEXT NOT NULL,
            wallet_status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bank_transaction (
            transaction_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL REFERENCES account(account_id),
            posted_at TEXT NOT NULL,
            description TEXT NOT NULL,
            amount_cents INTEGER NOT NULL,
            currency TEXT NOT NULL,
            status TEXT NOT NULL,
            category TEXT NOT NULL,
            disputed INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bank_transfer (
            transfer_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customer(customer_id),
            from_account_id TEXT NOT NULL REFERENCES account(account_id),
            recipient TEXT NOT NULL,
            amount_cents INTEGER NOT NULL,
            currency TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            reference TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS service_case (
            case_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customer(customer_id),
            case_type TEXT NOT NULL,
            subject TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )


def _seed_customer(connection: sqlite3.Connection, customer: dict[str, Any]) -> None:
    customer_id = str(customer["customer_id"])
    connection.execute(
        "INSERT INTO customer VALUES (?, ?, ?, ?, ?, ?)",
        (
            customer_id,
            customer["login"],
            customer["display_name"],
            customer["segment"],
            customer["city"],
            customer["member_since"],
        ),
    )
    for account in customer["accounts"]:
        connection.execute(
            "INSERT INTO account VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                account["account_id"],
                customer_id,
                account["name"],
                account["type"],
                account["last4"],
                account["currency"],
                account["available_balance_cents"],
                account["current_balance_cents"],
                account["status"],
            ),
        )
    for card in customer["cards"]:
        connection.execute(
            "INSERT INTO card VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                card["card_id"],
                customer_id,
                card["account_id"],
                card["name"],
                card["last4"],
                card["status"],
                card["wallet_status"],
            ),
        )
    for transaction in customer["transactions"]:
        connection.execute(
            "INSERT INTO bank_transaction VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                transaction["transaction_id"],
                transaction["account_id"],
                transaction["posted_at"],
                transaction["description"],
                transaction["amount_cents"],
                transaction["currency"],
                transaction["status"],
                transaction["category"],
                int(transaction["disputed"]),
            ),
        )
    for transfer in customer["transfers"]:
        connection.execute(
            "INSERT INTO bank_transfer VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                transfer["transfer_id"],
                customer_id,
                transfer["from_account_id"],
                transfer["recipient"],
                transfer["amount_cents"],
                transfer["currency"],
                transfer["created_at"],
                transfer["status"],
                transfer["reference"],
            ),
        )
    for case in customer["service_cases"]:
        connection.execute(
            "INSERT INTO service_case VALUES (?, ?, ?, ?, ?, ?)",
            (
                case["case_id"],
                customer_id,
                case["case_type"],
                case["subject"],
                case["status"],
                case["created_at"],
            ),
        )
    connection.commit()


def _reset_seed(connection: sqlite3.Connection, customer: dict[str, Any]) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        for table in (
            "service_case",
            "bank_transfer",
            "bank_transaction",
            "card",
            "account",
            "customer",
        ):
            connection.execute(f"DELETE FROM {table}")
        _seed_customer(connection, customer)
    except Exception:
        connection.rollback()
        raise


def _snapshot(connection: sqlite3.Connection) -> dict[str, Any]:
    customer = _one(connection, "SELECT * FROM customer")
    return {
        "customer": customer,
        "accounts": _many(connection, "SELECT * FROM account ORDER BY type, name"),
        "cards": _many(connection, "SELECT * FROM card ORDER BY name"),
        "transactions": _many(
            connection,
            "SELECT * FROM bank_transaction ORDER BY posted_at DESC LIMIT 10",
        ),
        "transfers": _many(
            connection,
            "SELECT * FROM bank_transfer ORDER BY created_at DESC",
        ),
        "service_cases": _many(
            connection,
            "SELECT * FROM service_case ORDER BY created_at DESC",
        ),
    }


def _execute(
    connection: sqlite3.Connection,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if tool_name == "list_accounts":
        return {"accounts": _snapshot(connection)["accounts"]}
    if tool_name == "list_cards":
        return {"cards": _snapshot(connection)["cards"]}
    if tool_name == "list_service_cases":
        return {"service_cases": _snapshot(connection)["service_cases"]}
    if tool_name == "list_transactions":
        limit = int(arguments.get("limit", 5))
        return {
            "transactions": _many(
                connection,
                "SELECT * FROM bank_transaction ORDER BY posted_at DESC LIMIT ?",
                (limit,),
            )
        }
    if tool_name == "list_transfers":
        return {"transfers": _snapshot(connection)["transfers"]}
    if tool_name in {"freeze_card", "replace_card"}:
        last4 = _required_public_selector(tool_name, arguments, "last4")
        card = _selected_card(connection, tool_name, last4)
        status = "frozen" if tool_name == "freeze_card" else "replacement_pending"
        connection.execute(
            "UPDATE card SET status = ? WHERE card_id = ?",
            (status, card["card_id"]),
        )
        case_type = "stolen_card" if tool_name == "freeze_card" else "replacement_card"
        _create_case(
            connection,
            case_type,
            f"{case_type.replace('_', ' ').title()} for card ending {card['last4']}",
        )
        card["status"] = status
        return {"card": card, "simulated": True}
    if tool_name == "dispute_transaction":
        if arguments.get("transaction_id") is None:
            _required_public_selector(tool_name, arguments, "description")
        transaction = _selected_transaction(
            connection,
            arguments.get("transaction_id"),
            arguments.get("description"),
        )
        connection.execute(
            "UPDATE bank_transaction SET disputed = 1 WHERE transaction_id = ?",
            (transaction["transaction_id"],),
        )
        _create_case(
            connection,
            "transaction_dispute",
            f"Dispute {transaction['description']} ({transaction['transaction_id']})",
        )
        transaction["disputed"] = True
        return {"transaction": transaction, "simulated": True}
    if tool_name == "cancel_transfer":
        if arguments.get("transfer_id") is None:
            _required_public_selector(tool_name, arguments, "recipient")
        transfer = _selected_transfer(
            connection,
            arguments.get("transfer_id"),
            arguments.get("recipient"),
        )
        connection.execute(
            "UPDATE bank_transfer SET status = 'cancelled' WHERE transfer_id = ?",
            (transfer["transfer_id"],),
        )
        transfer["status"] = "cancelled"
        return {"transfer": transfer, "simulated": True}
    raise ValueError(f"unsupported tool: {tool_name}")


def _required_public_selector(
    tool_name: str,
    arguments: dict[str, Any],
    selector: str,
) -> str:
    value = arguments.get(selector)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool_name} requires public selector {selector}")
    return value.strip()


def _selected_card(
    connection: sqlite3.Connection,
    tool_name: str,
    last4: str,
) -> dict[str, Any]:
    eligible_statuses = ("active",) if tool_name == "freeze_card" else ("active", "frozen")
    placeholders = ", ".join("?" for _status in eligible_statuses)
    return _one_eligible(
        connection,
        f"SELECT * FROM card WHERE last4 = ? AND status IN ({placeholders})",
        (last4, *eligible_statuses),
        "card",
    )


def _selected_transaction(
    connection: sqlite3.Connection,
    transaction_id: Any,
    description: Any = None,
) -> dict[str, Any]:
    if transaction_id is not None and description is not None:
        raise ValueError("select a transaction by ID or description, not both")
    if description is not None:
        return _one_casefold_match(
            connection,
            """
            SELECT * FROM bank_transaction
            WHERE amount_cents < 0 AND status = 'posted' AND disputed = 0
              AND lower(description) = lower(?)
            """,
            str(description),
            "transaction",
        )
    return _one_eligible(
        connection,
        """
        SELECT * FROM bank_transaction
        WHERE transaction_id = ? AND amount_cents < 0
          AND status = 'posted' AND disputed = 0
        """,
        (str(transaction_id),),
        "transaction",
    )


def _selected_transfer(
    connection: sqlite3.Connection,
    transfer_id: Any,
    recipient: Any = None,
) -> dict[str, Any]:
    if transfer_id is not None and recipient is not None:
        raise ValueError("select a transfer by ID or recipient, not both")
    if recipient is not None:
        return _one_casefold_match(
            connection,
            """
            SELECT * FROM bank_transfer
            WHERE status = 'pending' AND lower(recipient) = lower(?)
            """,
            str(recipient),
            "transfer",
        )
    return _one_eligible(
        connection,
        "SELECT * FROM bank_transfer WHERE transfer_id = ? AND status = 'pending'",
        (str(transfer_id),),
        "transfer",
    )


def _one_casefold_match(
    connection: sqlite3.Connection,
    query: str,
    value: str,
    record_type: str,
) -> dict[str, Any]:
    rows = connection.execute(query, (value,)).fetchall()
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one matching synthetic eligible {record_type}; found {len(rows)}"
        )
    return _row(rows[0])


def _one_eligible(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[Any, ...],
    record_type: str,
) -> dict[str, Any]:
    rows = connection.execute(query, parameters).fetchall()
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one matching synthetic eligible {record_type}; found {len(rows)}"
        )
    return _row(rows[0])


def _create_case(connection: sqlite3.Connection, case_type: str, subject: str) -> None:
    customer_id = str(_one(connection, "SELECT customer_id FROM customer")["customer_id"])
    connection.execute(
        "INSERT INTO service_case VALUES (?, ?, ?, ?, 'open', datetime('now'))",
        (
            f"case_{uuid.uuid4().hex[:10]}",
            customer_id,
            case_type,
            subject,
        ),
    )


def _one(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[Any, ...] = (),
) -> dict[str, Any]:
    row = connection.execute(query, parameters).fetchone()
    if row is None:
        raise ValueError("no matching synthetic banking record")
    return _row(row)


def _many(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    return [_row(row) for row in connection.execute(query, parameters).fetchall()]


def _row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    if "disputed" in result:
        result["disputed"] = bool(result["disputed"])
    return result
