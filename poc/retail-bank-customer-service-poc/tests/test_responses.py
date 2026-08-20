from __future__ import annotations

from responses import (
    policy_chunk_lookup,
    policy_references_html,
    render_gradio_assistant_message,
    render_policy_citations,
)


def chunk(chunk_id: str, *, title: str, effective_from: str, text: str) -> dict[str, str]:
    return {
        "chunk_id": chunk_id,
        "title": title,
        "effective_from": effective_from,
        "text": text,
    }


def test_render_policy_citations_replaces_tags_with_numbered_markers_in_order() -> None:
    lookup = policy_chunk_lookup(
        [
            chunk(
                "disputes.timeline.us.v1",
                title="Transaction dispute review",
                effective_from="2026-01-01",
                text="A submitted dispute is reviewed and status updates are provided.",
            )
        ]
    )

    display_text, references = render_policy_citations(
        "Disputes are reviewed after submission. [Policy: disputes.timeline.us.v1]",
        lookup,
    )

    assert display_text == "Disputes are reviewed after submission. [1]"
    assert references == [
        {
            "marker": "1",
            "chunk_id": "disputes.timeline.us.v1",
            "title": "Transaction dispute review",
            "effective_from": "2026-01-01",
            "excerpt": "A submitted dispute is reviewed and status updates are provided.",
        }
    ]


def test_render_policy_citations_reuses_marker_for_repeated_tag() -> None:
    lookup = policy_chunk_lookup(
        [chunk("a.v1", title="A", effective_from="2026-01-01", text="Text A")]
    )

    display_text, references = render_policy_citations(
        "[Policy: a.v1] and again [Policy: a.v1]",
        lookup,
    )

    assert display_text == "[1] and again [1]"
    assert len(references) == 1


def test_render_policy_citations_numbers_multiple_distinct_tags_in_first_seen_order() -> None:
    lookup = policy_chunk_lookup(
        [
            chunk("b.v1", title="B", effective_from="2026-02-01", text="Text B"),
            chunk("a.v1", title="A", effective_from="2026-01-01", text="Text A"),
        ]
    )

    display_text, references = render_policy_citations(
        "[Policy: b.v1] then [Policy: a.v1]",
        lookup,
    )

    assert display_text == "[1] then [2]"
    assert [reference["chunk_id"] for reference in references] == ["b.v1", "a.v1"]


def test_render_policy_citations_excerpt_renders_a_disclaimer_final_chunk_in_full() -> None:
    # Real corpus chunks run 326-389 chars and often end on a qualifying/disclaimer
    # sentence (e.g. mortgage's "does not guarantee approval..."); a naive 200-char
    # cut used to drop that sentence even though the whole chunk fits comfortably.
    mortgage_text = (
        "Customers may begin a mortgage application online or with a mortgage "
        "specialist. Eligibility and approval depend on verified identity, income, "
        "employment, assets, debts, credit history, the property, and applicable "
        "underwriting. Starting an application does not guarantee approval or a "
        "particular rate. A specialist can explain required documents and available "
        "loan options."
    )
    assert len(mortgage_text) == 374
    lookup = policy_chunk_lookup(
        [
            chunk(
                "mortgage.opening.us.v1", title="M", effective_from="2026-01-01", text=mortgage_text
            )
        ]
    )

    _display, references = render_policy_citations("[Policy: mortgage.opening.us.v1]", lookup)

    excerpt = references[0]["excerpt"]
    assert excerpt == mortgage_text
    assert "does not guarantee approval or a particular rate" in excerpt
    assert not excerpt.endswith("…")


def test_render_policy_citations_excerpt_truncates_at_the_last_sentence_boundary() -> None:
    long_text = "First sentence ends cleanly here. " + "y" * 500
    lookup = policy_chunk_lookup(
        [chunk("a.v1", title="A", effective_from="2026-01-01", text=long_text)]
    )

    _display, references = render_policy_citations("[Policy: a.v1]", lookup)

    excerpt = references[0]["excerpt"]
    assert excerpt == "First sentence ends cleanly here."
    assert not excerpt.endswith("…")


def test_render_policy_citations_excerpt_falls_back_to_a_hard_cut_when_no_sentence_boundary() -> (
    None
):
    long_text = "x" * 450  # no sentence-ending punctuation anywhere in the window

    lookup = policy_chunk_lookup(
        [chunk("a.v1", title="A", effective_from="2026-01-01", text=long_text)]
    )

    _display, references = render_policy_citations("[Policy: a.v1]", lookup)

    excerpt = references[0]["excerpt"]
    assert excerpt.endswith("…")
    assert len(excerpt) == 401


def test_render_policy_citations_falls_back_to_chunk_id_when_metadata_missing() -> None:
    display_text, references = render_policy_citations(
        "See the policy. [Policy: unknown.chunk.v1]",
        {},
    )

    assert display_text == "See the policy. [1]"
    assert references == [
        {
            "marker": "1",
            "chunk_id": "unknown.chunk.v1",
            "title": "unknown.chunk.v1",
            "effective_from": "unknown",
            "excerpt": "",
        }
    ]


def test_render_policy_citations_leaves_uncited_text_unchanged() -> None:
    text = "This answer has no policy citation at all."

    display_text, references = render_policy_citations(text, {})

    assert display_text == text
    assert references == []


def test_policy_references_html_is_empty_for_no_references() -> None:
    assert policy_references_html([]) == ""


def test_policy_references_html_escapes_untrusted_content() -> None:
    html_block = policy_references_html(
        [
            {
                "marker": "1",
                "chunk_id": "a.v1",
                "title": "<script>alert(1)</script>",
                "effective_from": "2026-01-01",
                "excerpt": "safe & sound",
            }
        ]
    )

    assert "<details><summary>Policy references</summary>" in html_block
    assert "&lt;script&gt;" in html_block
    assert "<script>" not in html_block
    assert "safe &amp; sound" in html_block


def test_render_gradio_assistant_message_appends_details_block_when_cited() -> None:
    lookup = policy_chunk_lookup(
        [chunk("a.v1", title="A policy", effective_from="2026-01-01", text="Some policy text.")]
    )

    message = render_gradio_assistant_message("Answer. [Policy: a.v1]", lookup)

    assert message.startswith("Answer. [1]")
    assert "<details><summary>Policy references</summary>" in message
    assert "A policy" in message


def test_render_gradio_assistant_message_is_unchanged_when_no_citation() -> None:
    text = "Your balance is USD 100.00."

    assert render_gradio_assistant_message(text, {}) == text


def test_policy_chunk_lookup_accepts_objects_with_as_dict_and_plain_dicts() -> None:
    class Chunk:
        def as_dict(self) -> dict[str, str]:
            return {"chunk_id": "obj.v1", "title": "Object chunk"}

    lookup = policy_chunk_lookup([Chunk(), {"chunk_id": "dict.v1", "title": "Dict chunk"}])

    assert set(lookup) == {"obj.v1", "dict.v1"}
    assert lookup["obj.v1"]["title"] == "Object chunk"
    assert lookup["dict.v1"]["title"] == "Dict chunk"


def test_policy_chunk_lookup_skips_entries_without_a_usable_chunk_id() -> None:
    lookup = policy_chunk_lookup([{"title": "No id here"}, {"chunk_id": "", "title": "Blank id"}])

    assert lookup == {}
