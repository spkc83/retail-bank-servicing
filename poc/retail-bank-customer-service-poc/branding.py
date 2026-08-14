"""Shared customer-facing brand and presentation helpers for the POC apps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

BANK_NAME = "Harborlight Bank"
ASSISTANT_NAME = "Harbor"
PROTOTYPE_NOTICE = (
    "Prototype experience · Fictional customer data · No real banking systems connected"
)

GRADIO_CSS = """
:root {
  --harbor-navy: #082f49;
  --harbor-teal: #0f766e;
  --harbor-mist: #f0fdfa;
  --harbor-line: #d8e3e8;
  --harbor-ink: #163042;
}
.gradio-container {
  max-width: 1220px !important;
  color: var(--harbor-ink);
  background: #f7fafb;
}
.harbor-header {
  border-radius: 18px;
  padding: 22px 26px;
  color: #ffffff;
  background: linear-gradient(125deg, var(--harbor-navy), #0b4f64);
  box-shadow: 0 12px 32px rgba(8, 47, 73, 0.14);
}
.harbor-header h1 { margin: 0; color: #ffffff; font-size: 1.8rem; }
.harbor-header p { margin: 6px 0 0; color: #dff7f4; }
.prototype-notice {
  margin: 10px 2px 18px;
  color: #526977;
  font-size: 0.82rem;
}
.profile-card {
  border: 1px solid var(--harbor-line);
  border-radius: 16px;
  padding: 16px;
  background: #ffffff;
  box-shadow: 0 8px 24px rgba(8, 47, 73, 0.07);
}
.profile-card h3 { color: var(--harbor-navy); }
.status-ok { color: var(--harbor-teal); font-weight: 700; }
.primary { background: var(--harbor-teal) !important; }
"""

STREAMLIT_CSS = """
<style>
:root {
  --harbor-navy: #082f49;
  --harbor-teal: #0f766e;
  --harbor-mist: #f0fdfa;
}
[data-testid="stAppViewContainer"] { background: #f7fafb; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #d8e3e8; }
h1, h2, h3 { color: var(--harbor-navy); }
.stButton > button[kind="primary"] {
  background: var(--harbor-teal);
  border-color: var(--harbor-teal);
}
.harbor-kicker { color: var(--harbor-teal); font-weight: 700; letter-spacing: .02em; }
.prototype-notice { color: #607684; font-size: .82rem; margin: -.35rem 0 1.25rem; }
</style>
"""

_ACCOUNT_TYPE_LABELS = {
    "checking": "Checking account",
    "savings": "Savings account",
    "money_market": "Money market account",
    "credit": "Credit account",
}


def account_type_label(value: Any) -> str:
    """Return a friendly product label while preserving unknown source values."""
    normalized = str(value or "").strip().lower()
    if not normalized:
        return "Bank account"
    return _ACCOUNT_TYPE_LABELS.get(normalized, f"{normalized.replace('_', ' ').title()} account")


def response_provenance(response_path: str, model_passes: Sequence[Any]) -> str:
    """Describe only the response path and model passes recorded by orchestration."""
    path = str(response_path).replace("_", " ").strip()
    labels = [str(item.label) for item in model_passes]
    pass_detail = f" ({', '.join(labels)})" if labels else ""
    return f"Response path: {path} · Recorded model passes: {len(labels)}{pass_detail}"
