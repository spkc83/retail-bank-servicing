from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

from auth import load_demo_auth
from branding import (
    ASSISTANT_NAME,
    BANK_NAME,
    PROTOTYPE_NOTICE,
    STREAMLIT_CSS,
    account_type_label,
    response_provenance,
)
from local_app_service import (
    LocalBankingController,
    visible_conversation,
)
from local_gpu_runtime import MODEL_ID, MODEL_REVISION, LocalGraniteRuntime
from responses import render_policy_citations
from router import ROUTER_REVISION, LearnedBankingRouter
from state import BANK

LOCAL_AUTH_DEFAULT_JSON = json.dumps(
    {
        "alex.demo": "alex-local-demo",
        "maya.demo": "maya-local-demo",
    }
)
DEFAULT_LOCAL_ROUTER_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "banking-conversation-router-v6-hierarchical"
)

PRESETS = (
    "Hello, how are you?",
    "Show my account balances.",
    "Show my five most recent transactions.",
    "What is the status of my debit card?",
    "My card was stolen. Freeze it.",
    "Please replace my debit card.",
    "What happened with the money I sent recently?",
    "Cancel the pending transfer to River Consulting.",
    "When was my mailing address changed?",
    "How does savings interest work?",
    "What is the weather tomorrow?",
)


@st.cache_resource(show_spinner=False)
def get_local_runtime() -> LocalGraniteRuntime:
    return LocalGraniteRuntime.load()


@st.cache_resource(show_spinner=False)
def get_local_router() -> LearnedBankingRouter | None:
    if os.environ.get("LOCAL_POC_SKIP_ROUTER_LOAD") == "1":
        return None
    local_artifact = resolve_local_router_artifact()
    if local_artifact is not None:
        return LearnedBankingRouter.from_artifact_dir(local_artifact)
    return LearnedBankingRouter.from_hub()


@st.cache_resource(show_spinner=False)
def get_local_controller() -> LocalBankingController:
    return LocalBankingController(
        bank=BANK,
        runtime=get_local_runtime(),
        router=get_local_router(),
    )


def local_auth_credentials() -> list[tuple[str, str]]:
    return load_demo_auth(os.environ.get("DEMO_AUTH_JSON", LOCAL_AUTH_DEFAULT_JSON))


def resolve_local_router_artifact() -> Path | None:
    configured = os.environ.get("LOCAL_ROUTER_ARTIFACT_DIR")
    candidate = Path(configured).expanduser() if configured else DEFAULT_LOCAL_ROUTER_ARTIFACT
    return candidate if (candidate / "manifest.json").is_file() else None


def main() -> None:
    st.set_page_config(
        page_title=f"{ASSISTANT_NAME} | {BANK_NAME}",
        page_icon="🏦",
        layout="wide",
    )
    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="harbor-kicker">HARBORLIGHT BANK</div>', unsafe_allow_html=True)
    st.title(BANK_NAME)
    st.markdown(f"### Meet {ASSISTANT_NAME}")
    st.markdown(
        f'<div class="prototype-notice">{PROTOTYPE_NOTICE}</div>',
        unsafe_allow_html=True,
    )
    if not _authenticated():
        _render_login()
        return

    username = str(st.session_state["username"])
    session_hash = str(st.session_state["session_hash"])
    with st.spinner("Loading the pinned Granite 9B model and conversation router…"):
        controller = get_local_controller()
    _initialize_chat_state()
    _render_sidebar(controller, username, session_hash)
    st.caption("Ask a question in your own words, or start with one of the ideas below.")
    for item in visible_conversation(
        st.session_state["conversation"], controller.policy_chunk_lookup
    ):
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            _render_policy_references(item.get("policy_references", []))

    left, right = st.columns([5, 1])
    with left:
        selected = st.selectbox("Preset test case", PRESETS)
    with right:
        st.write("")
        st.write("")
        run_preset = st.button("Run preset", use_container_width=True)
    entered = st.chat_input("How can Harbor help today?")
    prompt = selected if run_preset else entered
    if prompt:
        _execute_turn(controller, username, session_hash, str(prompt))


def _authenticated() -> bool:
    return isinstance(st.session_state.get("username"), str) and isinstance(
        st.session_state.get("session_hash"), str
    )


def _render_login() -> None:
    credentials = local_auth_credentials()
    st.info(f"Welcome. Sign in to explore {BANK_NAME} with fictional customer data.")
    st.markdown(
        "Use either local account:\n\n"
        + "\n".join(f"- `{username}` / `{password}`" for username, password in credentials)
    )
    with st.form("local-login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        accepted = dict(credentials).get(username) == password
        if not accepted:
            st.error("Invalid local demo credentials.")
            return
        st.session_state["username"] = username
        st.session_state["session_hash"] = uuid.uuid4().hex
        st.session_state["conversation"] = []
        st.session_state["diagnostics"] = (
            "### Local experiment diagnostics\n\nNo model turn has run yet."
        )
        st.rerun()


def _initialize_chat_state() -> None:
    st.session_state.setdefault("conversation", [])
    st.session_state.setdefault(
        "diagnostics",
        "### Local experiment diagnostics\n\nNo model turn has run yet.",
    )


def _render_sidebar(
    controller: LocalBankingController,
    username: str,
    session_hash: str,
) -> None:
    snapshot = controller.snapshot(username, session_hash)
    customer = snapshot["customer"]
    metadata = controller.runtime_metadata()
    router = get_local_router()
    if router is not None:
        metadata.update(router.artifact_metadata())
    with st.sidebar:
        st.caption(BANK_NAME)
        st.header(str(customer["display_name"]))
        st.write(f"Authenticated as `{username}`")
        st.write(f"{customer['segment']} · {customer['city']}")
    _render_experiment_diagnostics(metadata)
    with st.sidebar:
        st.markdown(render_snapshot(snapshot))
        if st.button("Start over", use_container_width=True):
            controller.reset(username, session_hash)
            st.session_state["conversation"] = []
            st.session_state["diagnostics"] = (
                "### Local experiment diagnostics\n\nSession reset; no new model turn yet."
            )
            st.rerun()
        if st.button("Sign out", use_container_width=True):
            for key in ("username", "session_hash", "conversation", "diagnostics"):
                st.session_state.pop(key, None)
            st.rerun()


def _render_experiment_diagnostics(metadata: dict[str, str]) -> None:
    with st.sidebar.popover("Experiment diagnostics", use_container_width=True):
        st.caption(
            "Developer-only runtime provenance, routing decisions, tool calls, "
            "and raw model output."
        )
        st.markdown("#### Runtime proof")
        st.code(
            "\n".join(
                (
                    f"model={metadata.get('model_id', MODEL_ID)}",
                    f"revision={metadata.get('model_revision', MODEL_REVISION)}",
                    f"router_source={metadata.get('router_source', 'hub')}",
                    f"router_repo={metadata.get('router_repo_id', 'local')}",
                    f"router_revision={metadata.get('router_revision', ROUTER_REVISION)}",
                    f"router_artifact={metadata.get('router_artifact_path', 'hub snapshot')}",
                    "router_manifest_sha256="
                    f"{metadata.get('router_manifest_sha256', 'hub-verified')}",
                    "router_data_sha256="
                    f"{metadata.get('router_data_manifest_sha256', 'unavailable')}",
                    f"device={metadata.get('runtime_device', 'unavailable')}",
                    f"gpu={metadata.get('cuda_device_name', 'unavailable')}",
                    f"quantization={metadata.get('weight_quantization', 'unavailable')}",
                    f"allocated_gib={metadata.get('cuda_memory_allocated_gib', 'unavailable')}",
                )
            )
        )
        st.divider()
        st.markdown(st.session_state["diagnostics"])


def _execute_turn(
    controller: LocalBankingController,
    username: str,
    session_hash: str,
    prompt: str,
) -> None:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Granite is generating and may use the available banking actions…"):
        result = controller.run_turn(
            username=username,
            session_hash=session_hash,
            message=prompt,
            conversation=st.session_state["conversation"],
        )
    st.session_state["conversation"] = result.conversation
    st.session_state["diagnostics"] = result.diagnostics
    display_text, references = render_policy_citations(
        result.response, controller.policy_chunk_lookup
    )
    with st.chat_message("assistant"):
        st.markdown(display_text)
        _render_policy_references(references)
    st.caption(response_provenance(result.response_path, result.model_passes))
    # Rerun so the sidebar snapshot and diagnostics reflect this turn's tool effects.
    st.rerun()


_MARKDOWN_ESCAPE_PATTERN = re.compile(r"([\\`*_{}\[\]()#+\-.!<>|~])")


def _escape_markdown(text: str) -> str:
    """Escape Markdown metacharacters so corpus text can't alter the rendered layout."""
    return _MARKDOWN_ESCAPE_PATTERN.sub(r"\\\1", text)


def _render_policy_references(references: list[dict[str, str]]) -> None:
    if not references:
        return
    with st.expander("Policy references"):
        for reference in references:
            st.markdown(_policy_reference_heading(reference))
            st.caption(_escape_markdown(reference.get("excerpt") or "No excerpt available."))


def _policy_reference_heading(reference: dict[str, str]) -> str:
    marker = _escape_markdown(str(reference["marker"]))
    title = _escape_markdown(str(reference["title"]))
    effective_from = _escape_markdown(str(reference["effective_from"]))
    return f"**[{marker}] {title}** — effective {effective_from}"


def render_snapshot(snapshot: dict[str, Any]) -> str:
    accounts = "\n".join(
        (
            f"- **{item['name']} ····{item['last4']}**  "
            f"_{account_type_label(item.get('type'))}_ — "
            f"{_money(item['available_balance_cents'], item['currency'])} available"
        )
        for item in snapshot["accounts"]
    )
    cards = "\n".join(
        f"- **{item['name']} ····{item['last4']}** — `{item['status']}`"
        for item in snapshot["cards"]
    )
    transfers = "\n".join(
        f"- {item['recipient']} — {_money(item['amount_cents'], item['currency'])} "
        f"`{item['status']}`"
        for item in snapshot["transfers"][:3]
    )
    cases = "\n".join(
        f"- {item['subject']} — `{item['status']}`" for item in snapshot["service_cases"][:4]
    )
    return (
        f"### Your products\n\n#### Bank accounts\n{accounts or '- None'}\n\n"
        f"#### Debit cards\n{cards or '- None'}\n\n"
        f"#### Recent transfers\n{transfers or '- None'}\n\n"
        f"#### Support requests\n{cases or '- None'}"
    )


def _money(cents: Any, currency: Any) -> str:
    return f"{str(currency)} {int(cents) / 100:,.2f}"


if __name__ == "__main__":
    main()
