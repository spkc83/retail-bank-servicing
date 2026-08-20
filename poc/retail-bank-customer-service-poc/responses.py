from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from response_policy import POLICY_CITATION

OOD_RESPONSE = (
    "I’m here to help with Harborlight Bank accounts, cards, transactions, transfers, "
    "service requests, and banking policies. What banking question can I help with?"
)
MODEL_FAILURE_RESPONSE = (
    "I’m sorry, I couldn’t complete that request right now. Please try again in a moment."
)
POLICY_NOT_FOUND_RESPONSE = (
    "I’m sorry, I couldn’t find an approved current policy that answers that question. "
    "Please ask about another banking topic or contact a Harborlight Bank specialist."
)

POLICY_REFERENCES_LABEL = "Policy references"
_EXCERPT_LIMIT = 400
_SENTENCE_BOUNDARY = re.compile(r"[.!?](?=\s|$)")


def policy_chunk_lookup(chunks: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """Build a chunk_id -> metadata dict from PolicyChunk/PolicyMatch objects or dicts."""
    lookup: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        payload = chunk.as_dict() if hasattr(chunk, "as_dict") else dict(chunk)
        chunk_id = payload.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            lookup[chunk_id] = payload
    return lookup


def render_policy_citations(
    text: str,
    chunk_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Replace literal ``[Policy: id]`` tags with compact numbered display markers.

    Returns ``(display_text, references)``. References are ordered by first
    appearance in `text`. A cited id absent from `chunk_lookup` still gets a
    reference entry (falling back to the bare id), so a citation is never
    silently dropped or made to crash rendering.
    """
    chunk_lookup = chunk_lookup or {}
    order: list[str] = []
    markers: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        chunk_id = match.group(1).strip()
        if chunk_id not in markers:
            markers[chunk_id] = str(len(order) + 1)
            order.append(chunk_id)
        return f"[{markers[chunk_id]}]"

    display_text = POLICY_CITATION.sub(_replace, text)
    references = [
        _reference_entry(chunk_id, markers[chunk_id], chunk_lookup.get(chunk_id))
        for chunk_id in order
    ]
    return display_text, references


def _reference_entry(
    chunk_id: str,
    marker: str,
    chunk: Mapping[str, Any] | None,
) -> dict[str, str]:
    if chunk is None:
        return {
            "marker": marker,
            "chunk_id": chunk_id,
            "title": chunk_id,
            "effective_from": "unknown",
            "excerpt": "",
        }
    excerpt = _excerpt(str(chunk.get("text", "")))
    return {
        "marker": marker,
        "chunk_id": chunk_id,
        "title": str(chunk.get("title") or chunk_id),
        "effective_from": str(chunk.get("effective_from") or "unknown"),
        "excerpt": excerpt,
    }


def _excerpt(source_text: str, limit: int = _EXCERPT_LIMIT) -> str:
    """Return an excerpt of `source_text`, preferring a clean sentence boundary.

    Text within `limit` is returned whole. Longer text is cut at the last
    sentence end at or before `limit` so a trailing disclaimer sentence is
    never chopped mid-word; only when no sentence boundary is found within
    the window does this fall back to a hard cut with an ellipsis.
    """
    if len(source_text) <= limit:
        return source_text
    window = source_text[:limit]
    boundaries = [match.end() for match in _SENTENCE_BOUNDARY.finditer(window)]
    if boundaries:
        return window[: boundaries[-1]]
    return window.rstrip() + "…"


def policy_references_html(references: Sequence[Mapping[str, str]]) -> str:
    """Render a Gradio-safe ``<details>`` reference block; empty when there is nothing to show."""
    if not references:
        return ""
    items = "".join(
        "<li>"
        f"<strong>[{html.escape(str(ref['marker']))}] {html.escape(str(ref['title']))}</strong>"
        f" — effective {html.escape(str(ref['effective_from']))}"
        f"<br>{html.escape(str(ref['excerpt']))}"
        "</li>"
        for ref in references
    )
    return f"<details><summary>{POLICY_REFERENCES_LABEL}</summary><ul>{items}</ul></details>"


def render_gradio_assistant_message(
    text: str,
    chunk_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Return chat-message-ready markdown/HTML: numbered citations plus a references block."""
    display_text, references = render_policy_citations(text, chunk_lookup)
    block = policy_references_html(references)
    return f"{display_text}\n\n{block}" if block else display_text
