from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from branding import STREAMLIT_CSS
from streamlit_app import render_snapshot, resolve_local_router_artifact

APP_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_is_available_for_the_local_app() -> None:
    assert importlib.util.find_spec("streamlit") is not None


def test_local_streamlit_prefers_the_canonical_hierarchical_router() -> None:
    artifact = resolve_local_router_artifact()

    assert artifact is not None
    assert artifact.name == "banking-conversation-router-v6-hierarchical"
    assert (artifact / "classifier_heads.safetensors").is_file()
    assert (artifact / "router_config.json").is_file()


def test_local_streamlit_does_not_fall_back_to_obsolete_local_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("LOCAL_ROUTER_ARTIFACT_DIR", raising=False)
    monkeypatch.setattr(
        "streamlit_app.DEFAULT_LOCAL_ROUTER_ARTIFACT",
        tmp_path / "missing-published-final",
    )
    obsolete = tmp_path / "banking-conversation-router-v5"
    obsolete.mkdir()
    (obsolete / "manifest.json").write_text("{}", encoding="utf-8")

    assert resolve_local_router_artifact() is None


def test_streamlit_app_renders_local_login_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LOCAL_POC_SKIP_MODEL_LOAD", "1")
    monkeypatch.setenv("LOCAL_POC_SKIP_ROUTER_LOAD", "1")
    monkeypatch.delenv("DEMO_AUTH_JSON", raising=False)

    app = AppTest.from_file(str(APP_ROOT / "streamlit_app.py"), default_timeout=15)
    app.run()

    assert not app.exception
    assert any("Harborlight Bank" in title.value for title in app.title)
    assert {item.label for item in app.text_input} >= {"Username", "Password"}
    page_text = "\n".join(markdown.value for markdown in app.markdown)
    assert "alex.demo" in page_text
    assert "maya.demo" in page_text
    assert "Meet Harbor" in page_text
    assert "Prototype experience" in page_text


def test_local_launcher_does_not_use_the_gradio_skip_flags() -> None:
    launcher = (
        Path(__file__).resolve().parents[3] / "scripts" / "retail_bank" / "run_local_streamlit.py"
    ).read_text(encoding="utf-8")

    assert "LOCAL_POC_SKIP_MODEL_LOAD" in launcher
    assert "LOCAL_POC_SKIP_ROUTER_LOAD" in launcher
    assert '"POC_SKIP_MODEL_LOAD"' not in launcher
    assert '"POC_SKIP_ROUTER_LOAD"' not in launcher
    assert "pytorch-cu126" in launcher
    assert "streamlit_app.py" in launcher


def test_experiment_diagnostics_use_an_on_demand_sidebar_popover() -> None:
    source = (APP_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    assert 'st.sidebar.popover("Experiment diagnostics"' in source
    assert "router_manifest_sha256=" in source
    assert "router_data_sha256=" in source
    assert "router_artifact=" in source
    assert 'st.expander("Model, router, tool-call, and raw-output diagnostics"' not in source


def test_streamlit_tables_are_scoped_to_assistant_chat_and_scroll_horizontally() -> None:
    assistant_scope = (
        '[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])'
    )

    assert f"{assistant_scope} table" in STREAMLIT_CSS
    assert "overflow-x: auto" in STREAMLIT_CSS
    assert "white-space: nowrap" in STREAMLIT_CSS
    assert "overflow-wrap: normal" in STREAMLIT_CSS


def test_streamlit_presentation_uses_shared_brand_and_warm_copy() -> None:
    source = (APP_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    assert "from branding import" in source
    assert "Local Granite 9B Retail Bank POC" not in source
    assert "Ask the signed-in synthetic bank agent" not in source
    assert 'st.chat_input("How can Harbor help today?")' in source
    assert "#082f49" not in source


def test_streamlit_snapshot_uses_customer_friendly_product_labels() -> None:
    rendered = render_snapshot(
        {
            "accounts": [
                {
                    "name": "Everyday Checking",
                    "type": "checking",
                    "last4": "1042",
                    "available_balance_cents": 324567,
                    "currency": "USD",
                }
            ],
            "cards": [],
            "transfers": [],
            "service_cases": [],
        }
    )

    assert "Your products" in rendered
    assert "Checking account" in rendered
    assert "Synthetic state" not in rendered


def test_execute_turn_reruns_so_sidebar_snapshot_is_fresh() -> None:
    source = Path("poc/retail-bank-customer-service-poc/streamlit_app.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def _execute_turn", 1)[1].split("\ndef ", 1)[0]

    stores = body.index('st.session_state["diagnostics"] = result.diagnostics')
    rerun = body.index("st.rerun()")
    assert stores < rerun, "turn results must be stored before the rerun"
