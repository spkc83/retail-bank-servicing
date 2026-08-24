# Data Pipeline Remediation Implementation Plan

> **Status as of 2026-08-24 — historical.** This plan is the origin of the runtime-guard
> and corpus work that followed; it is kept for its findings, not as an open work list.
> Where its findings stand now:
>
> - **F1 corpus diversity** — addressed. Teacher realization was applied; the released
>   alignment corpus carries teacher metadata (a publish that silently stripped it was
>   caught by diffing the regenerated tree against git and republished).
> - **F2 entity-state starvation** — partially addressed by the ambiguity and ineligible
>   curricula added since; not separately re-measured.
> - **F3 fabricated action claims** — addressed, in two layers and over several rounds.
>   `validate_no_unsupported_action_claims` now rejects completed-action claims,
>   retrieved-account-data claims, credential claims (absolutely), and evidence-free
>   assertions of account state; the converse and clarify turn guidance forbid the same
>   things at the prompt. Verified on live v10 output: fabrications 2/8 → 0/8.
> - **F4 router action-head instability** — addressed by the first-turn mutation router
>   data (the shipped v8 router artifact).
> - **F5 write on a question turn** — addressed by the retrospective-status curriculum.
> - **F6 stale sidebar** — addressed; `test_execute_turn_reruns_so_sidebar_snapshot_is_fresh`
>   guards it.
> - **F7 seed-data gap** — addressed.
> - **F8 OOD misroute** — still unconfirmed, still out of scope.
>
> See `2026-08-21-conversational-voice-retrain.md` for everything after this plan,
> including the router regression that is currently open.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every defect discovered in the 2026-08-17 demo review at its actual root — corpus answer-diversity collapse, entity-state starvation, seed-data gaps, and the runtime paths that let a model fabricate actions — then retrain and redeploy once.

**Architecture:** Phase 0 ships deterministic runtime guards that need no retraining. Phase 1 fixes the data generators using the pipeline's own (never-used) teacher-realization mechanism and post-split mutation pattern, keeping every frozen artifact byte-identical. Phase 2 regenerates, retrains once, and redeploys.

**Tech Stack:** Python 3.11/3.12, pytest, Streamlit POC, HF Jobs (`rtx-pro-6000`), Granite 9B + LoRA, DistilBERT router.

**Spec:** The Findings section below is the spec; there is no separate document.

## Findings (spec)

Evidence gathered in this session (screenshots in `/home/pavan/Pictures/slm-screenshots/`, live Playwright repro, code reading):

- **F1 — Corpus diversity collapse.** `data/banking-v5-tool-sft` holds 1200 rows but only **202 distinct answer bodies**; uniqueness comes from rotating 16 `REALIZER_FINAL_PREFIXES` × 8 `REALIZER_FINAL_CLOSERS` (`src/hello_slm/banking_tool_sft_data.py:229-258`, applied at `:2468`). Stripping filler creates 743 duplicate finals, so the global-dedup validator (`:818`) is satisfied by scaffolding, not content. The model therefore learned the filler as the answer format (it leaks verbatim into the demo). The pipeline's teacher-realization flow (`export_teacher_realization_requests` / `import_teacher_realizations`, CLI flags `--export-teacher-requests` / `--teacher-responses`) was built for exactly this and **was never applied** (0 released records carry teacher metadata).
- **F2 — Entity-state starvation.** Across both training corpora, `generation_contract.entity_state` counts are: `ineligible` **1**, `missing` 74 — while the runtime grounding (`poc/.../entity_grounding.py:101,116,124,152`) produces `ineligible` in three branches. The fabricated-freeze repro happened on a `clarify`/`ineligible` turn.
- **F3 — Fabricated action claims.** Live repro: "My card was stolen. Freeze it." → response "I found your active card and froze it…" with diagnostics `action: converse`, `Exposed tools: []`, `Tool result: none`. No runtime guard forbids past-tense action claims on zero-tool turns.
- **F4 — Router action-head instability on mutation imperatives.** Same input produced `list_cards` table in one session and `converse` in another; a freeze imperative must never route to `converse`.
- **F5 — Write executed on a question turn.** In the user's session, "was the card frozen ?" (a status question) triggered the pending freeze mutation and opened a support case.
- **F6 — Stale sidebar.** `streamlit_app.py` renders the sidebar (`:112`) before `_execute_turn` (`:128`) and never reruns, so account/card state and diagnostics lag one turn behind.
- **F7 — Seed-data gap.** `synthetic_bank.json` seeds 4 transactions for alex.demo and 3 for maya.demo; the preset asks for "five most recent". The screenshot-regression case checks only route/tool/constraints + `must_include: ["transaction"]`, so adding rows is safe.
- **F8 — (unconfirmed) OOD misroute.** One hidden turn scored `ood_probability 0.997`. Not reproducible without knowing the input; out of scope, noted for later.

## Global Constraints

- The frozen 215-record test split must stay byte-identical: `test_candidate5_preserves_all_215_test_behavior_fields_byte_equivalent` pins sha `4ac64ad9…186e` and MUST NOT be updated.
- The coreference shadow (`sha 55c9df4b…34a9` in the manifest), granite-v7 shadow, and screenshot heldout guards must pass unchanged; never edit a `*-shadow*` or frozen-test artifact.
- Run tests as `rtk proxy python -m pytest …` (the rtk hook misreports plain pytest output in this repo).
- Full local gate before any commit: `rtk proxy python -m pytest tests/test_banking_*.py poc/retail-bank-customer-service-poc/tests -q`, `rtk proxy python -m ruff check src scripts tests poc/retail-bank-customer-service-poc`, `rtk proxy python -m mypy src scripts tests` (baseline: 8 pre-existing errors in 5 files — zero NEW errors allowed).
- Commit style: single-line imperative subject (`fix: …` / `feat: …` / `test: …`), matching `git log`.
- No paid GPU launch without explicit user approval; remaining credit ≈ $1.31, one 550-step run ≈ $0.57–0.80.
- The base dataset regenerates byte-for-byte from `prepare()` (verified this session) — regeneration is safe, but always diff against `git` before committing data.

## File Structure

- `poc/retail-bank-customer-service-poc/streamlit_app.py` — sidebar rerun (F6)
- `poc/retail-bank-customer-service-poc/response_policy.py` — no-fabricated-action validator (F3)
- `poc/retail-bank-customer-service-poc/model_service.py` — wire validator into `_ensure_customer_facing` (F3)
- `poc/retail-bank-customer-service-poc/router.py` — mutation-imperative constraint (F4)
- `poc/retail-bank-customer-service-poc/dialogue_state.py` — interrogative resume gate (F5)
- `poc/retail-bank-customer-service-poc/synthetic_bank.json` — transaction seeds (F7)
- `src/hello_slm/banking_tool_sft_data.py` — teacher-responses integration already exists; no change expected (F1)
- `data/sources/banking-v5-tool-sft-teacher-realizations.jsonl` — NEW tracked teacher artifact (F1)
- `src/hello_slm/banking_servicing_alignment_data.py` — ineligible/missing clarify curriculum (F2)
- Tests: `poc/.../tests/test_streamlit_app.py`, `test_response_policy.py`, `test_model_service.py`, `test_router.py`, `test_dialogue_state.py`, `tests/test_banking_tool_sft_data.py`, `tests/test_banking_servicing_alignment_data.py`

---

## Phase 0 — Runtime guards (deployable without retraining)

### Task 1: Sidebar freshness (F6)

**Files:**
- Modify: `poc/retail-bank-customer-service-poc/streamlit_app.py:233-252` (`_execute_turn`)
- Test: `poc/retail-bank-customer-service-poc/tests/test_streamlit_app.py`

**Interfaces:**
- Consumes: `st.session_state["conversation"|"diagnostics"]`, `controller.run_turn`
- Produces: `_execute_turn` ends with `st.rerun()` after state is stored

- [ ] **Step 1: Write the failing test** (this repo tests Streamlit behavior by asserting on source bodies — follow that idiom; see `test_local_streamlit_prefers_the_canonical_hierarchical_router`)

```python
def test_execute_turn_reruns_so_sidebar_snapshot_is_fresh() -> None:
    source = Path("poc/retail-bank-customer-service-poc/streamlit_app.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def _execute_turn", 1)[1].split("\ndef ", 1)[0]

    stores = body.index('st.session_state["diagnostics"] = result.diagnostics')
    rerun = body.index("st.rerun()")
    assert stores < rerun, "turn results must be stored before the rerun"
```

Match the existing import style at the top of `test_streamlit_app.py` (it already imports `Path`).

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk proxy python -m pytest poc/retail-bank-customer-service-poc/tests/test_streamlit_app.py -q -k rerun`
Expected: FAIL with `ValueError: substring not found` (no `st.rerun()` in body)

- [ ] **Step 3: Implement** — at the end of `_execute_turn` (after the `st.caption(...)` line), append:

```python
    # Rerun so the sidebar snapshot and diagnostics reflect this turn's tool effects.
    st.rerun()
```

Also delete the two now-dead display lines `with st.chat_message("assistant"): st.markdown(result.response)` and `st.caption(response_provenance(...))`? **No — keep them.** On rerun the conversation loop at `:114-116` re-renders the full history including this turn, but keeping the immediate render costs nothing and covers any rerun interruption. Only append `st.rerun()`.

- [ ] **Step 4: Run the full POC suite**

Run: `rtk proxy python -m pytest poc/retail-bank-customer-service-poc/tests -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add poc/retail-bank-customer-service-poc/streamlit_app.py poc/retail-bank-customer-service-poc/tests/test_streamlit_app.py
git commit -m "fix: rerun after each turn so the sidebar reflects tool effects"
```

### Task 2: Block fabricated action claims on zero-tool turns (F3)

**Files:**
- Modify: `poc/retail-bank-customer-service-poc/response_policy.py` (add validator after `strip_realizer_filler`)
- Modify: `poc/retail-bank-customer-service-poc/model_service.py:612-641` (`_ensure_customer_facing`)
- Test: `poc/retail-bank-customer-service-poc/tests/test_response_policy.py`, `tests/test_model_service.py`

**Interfaces:**
- Consumes: `_ensure_customer_facing(..., authoritative_evidence=results)` — already receives tool results at both call sites (`model_service.py:486` with results, `:581` without).
- Produces: `validate_no_unsupported_action_claims(answer: str, results: Sequence[Mapping[str, Any]]) -> GroundingValidation` in `response_policy`.

Rule: when a turn executed **no tools at all**, the answer must not assert a completed banking mutation. This is exactly the observed failure (converse turn, zero tools, "…and froze it"). Turns with any tool evidence are left to the existing grounding validator.

- [ ] **Step 1: Write the failing tests** in `test_response_policy.py`:

```python
def test_zero_tool_answers_must_not_claim_completed_actions() -> None:
    fabricated = validate_no_unsupported_action_claims(
        "I found your active card and froze it to stop unauthorized use.", ()
    )
    frozen_state = validate_no_unsupported_action_claims(
        "Your card ending in 4821 is now frozen.", ()
    )

    assert not fabricated.valid
    assert not frozen_state.valid
    assert any("without tool evidence" in error for error in fabricated.errors)


def test_zero_tool_answers_may_ask_and_describe_without_action_claims() -> None:
    question = validate_no_unsupported_action_claims(
        "Would you like me to freeze the card ending in 4821?", ()
    )
    neutral = validate_no_unsupported_action_claims(
        "Hi, I’m Harbor. How can I help with your banking today?", ()
    )

    assert question.valid
    assert neutral.valid


def test_action_claims_are_allowed_when_any_tool_evidence_exists() -> None:
    evidence = ({"ok": True, "result": {"card": {"last4": "4821", "status": "frozen"}}},)

    validation = validate_no_unsupported_action_claims(
        "Your Everyday Visa Debit ending in 4821 is now frozen.", evidence
    )

    assert validation.valid
```

And in `test_model_service.py` (uses the same `RecordingModel`/`bank()`/`v4_router_guidance` helpers as neighboring tests):

```python
def test_converse_turn_claiming_an_action_is_repaired_or_rejected() -> None:
    model = RecordingModel(
        [
            "I found your active card and froze it to stop unauthorized use.",
            "I can’t complete that from this conversation yet — would you like me to freeze the card ending in 4821?",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="My card was stolen. Freeze it.",
        conversation=[],
        router_result=v4_router_guidance(
            action="converse",
            fine_intent="freeze_card",
            entity_resolution="not_required",
        ),
    )

    assert "froze it" not in result.response
    assert len(model.calls) == 2  # draft + customer-experience repair
```

Note: check `v4_router_guidance`'s signature in the test file first; if `action="converse"` with `entity_resolution="not_required"` routes differently, use the router_result shape the `direct_answer` path tests already use — the requirement is a zero-tool turn.

- [ ] **Step 2: Run to verify both fail** (`ImportError` then assertion)

Run: `rtk proxy python -m pytest poc/retail-bank-customer-service-poc/tests -q -k "unsupported_action or claiming_an_action"`

- [ ] **Step 3: Implement the validator** in `response_policy.py`:

```python
COMPLETED_ACTION_CLAIMS = re.compile(
    r"\b(?:froze|has been frozen|is (?:now )?frozen|replaced|replacement is pending|"
    r"cancelled|canceled|disputed|has been closed|is now closed)\b",
    re.IGNORECASE,
)


def validate_no_unsupported_action_claims(
    answer: str,
    results: Sequence[Mapping[str, Any]],
) -> GroundingValidation:
    """Zero-tool turns must never assert a completed banking action or state change."""

    if results:
        return GroundingValidation(True, ())
    if not isinstance(answer, str):
        return GroundingValidation(False, ("final answer is not text",))
    match = COMPLETED_ACTION_CLAIMS.search(answer)
    if match is None:
        return GroundingValidation(True, ())
    return GroundingValidation(
        False,
        (f"answer claims a completed action ({match.group(0)!r}) without tool evidence",),
    )
```

- [ ] **Step 4: Wire it into `_ensure_customer_facing`** (`model_service.py:612`), merging errors with the existing customer-facing validation so the existing repair pass corrects both:

```python
        draft = strip_realizer_filler(draft) or draft
        validation = validate_customer_facing_answer(draft)
        action_validation = validate_no_unsupported_action_claims(
            draft, tuple(authoritative_evidence)
        )
        errors = (*validation.errors, *action_validation.errors)
        if validation.valid and action_validation.valid:
            return draft, response_path
        repair_messages = build_customer_experience_repair_messages(
            user_message=user_message,
            draft=draft,
            errors=errors,
            authoritative_evidence=authoritative_evidence,
        )
```

The rest of the method is unchanged except the final re-check: after the repair pass, run BOTH `validate_customer_facing_answer(repaired)` and `validate_no_unsupported_action_claims(repaired, tuple(authoritative_evidence))` and raise the existing `AgentProtocolError("customer-experience repair failed validation: …")` if either is invalid. Import `validate_no_unsupported_action_claims` in the `from response_policy import (...)` block at the top of `model_service.py`.

- [ ] **Step 5: Run the tests, then the full local gate; fix regressions** (some existing zero-tool tests may legitimately claim actions in canned RecordingModel outputs — if any fail, inspect: if the test models a zero-tool turn asserting an action, the test itself encodes the bug and its canned output should be updated to a question form).

- [ ] **Step 6: Commit**

```bash
git add poc/retail-bank-customer-service-poc/response_policy.py poc/retail-bank-customer-service-poc/model_service.py poc/retail-bank-customer-service-poc/tests
git commit -m "fix: reject completed-action claims on zero-tool turns"
```

### Task 3: Router constraint — mutation imperative never converses (F4)

**Files:**
- Modify: `poc/retail-bank-customer-service-poc/router.py` (constraint function at `:1000-1034`)
- Test: `poc/retail-bank-customer-service-poc/tests/test_router.py`

**Interfaces:**
- Produces: `MUTATION_FINE_INTENTS = frozenset({"freeze_card", "replace_card", "dispute_transaction", "cancel_transfer"})` module constant in `router.py`; constraint diagnostics tag `constraint:mutation-intent-cannot-converse`.

- [ ] **Step 1: Write the failing test** (follow the shape of the neighboring constraint tests in `test_router.py` — find one exercising `constraint:clarify-requires-unresolved-entity` and mirror its setup):

```python
def test_mutation_intent_with_converse_action_downgrades_to_clarify() -> None:
    route, action, diagnostics = router._apply_decision_constraints(  # use the actual fn name at router.py:1000
        domain="banking",
        lane="servicing",
        family="cards",
        intent="freeze_card",
        route="in_domain",
        action="converse",
        entity_resolution="resolved",
        action_labels=("execute_tool", "clarify", "converse"),
    )

    assert action == "clarify"
    assert "constraint:mutation-intent-cannot-converse" in diagnostics
```

Read `router.py:990-1000` first for the exact function name/signature and adjust the call — the assertion block is the contract.

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Implement** inside the constraint function, after the existing `action == "clarify"` check (`router.py:1027-1029`):

```python
    if action == "converse" and intent in MUTATION_FINE_INTENTS:
        if "clarify" in action_labels:
            action = "clarify"
            diagnostics.append("constraint:mutation-intent-cannot-converse")
        else:
            route = "uncertain"
            diagnostics.append("constraint:mutation-intent-cannot-converse")
```

With the module constant near the top of `router.py`:

```python
MUTATION_FINE_INTENTS = frozenset(
    {"freeze_card", "replace_card", "dispute_transaction", "cancel_transfer"}
)
```

Check the actual fine-intent label spellings against `SERVICING_TOOLS` / the taxonomy in `router.py` before committing — the set must use the router's own label strings.

- [ ] **Step 4: Run `test_router.py` + full POC suite**

- [ ] **Step 5: Commit**

```bash
git add poc/retail-bank-customer-service-poc/router.py poc/retail-bank-customer-service-poc/tests/test_router.py
git commit -m "fix: never converse on a mutation-intent turn"
```

### Task 4: Interrogative turns must not auto-resume a pending mutation (F5)

**Files:**
- Read first: `poc/retail-bank-customer-service-poc/dialogue_state.py` (entire file — only fragments were reviewed)
- Modify: `poc/retail-bank-customer-service-poc/dialogue_state.py` (resume path around `:131-166`)
- Test: `poc/retail-bank-customer-service-poc/tests/test_dialogue_state.py`

**Interfaces:**
- Produces: a module predicate `is_interrogative(message: str) -> bool` and resume gating: a pending servicing action is resumed only by a non-interrogative turn.

- [ ] **Step 1: Read `dialogue_state.py` fully and locate the exact branch** where `state.pending_servicing` causes the next turn to resume the pending action (around `:131-166`, `begin_turn`). Record the branch condition verbatim in the task notes before editing.

- [ ] **Step 2: Write the failing tests** (mirror the setup style already in `test_dialogue_state.py` for pending-servicing transitions):

```python
def test_status_question_does_not_resume_pending_mutation() -> None:
    # arrange a state with pending_servicing = freeze_card (copy the fixture
    # used by the existing resume test in this file)
    transition = begin_turn_with_pending("was the card frozen ?")

    assert transition.resumed_pending is False


def test_affirmative_continuation_still_resumes_pending_mutation() -> None:
    transition = begin_turn_with_pending("yes please go ahead")

    assert transition.resumed_pending is True


def test_is_interrogative_detects_questions() -> None:
    assert is_interrogative("was the card frozen ?")
    assert is_interrogative("Did that go through")
    assert not is_interrogative("yes do it")
    assert not is_interrogative("freeze the card please")
```

`begin_turn_with_pending` is a small local helper you write in the test file wrapping the existing fixture; `transition.resumed_pending` must be whatever field the existing resume test asserts — align names with what Step 1 found (the plan's names are the contract, adapt to the file's actual vocabulary and note the mapping in the commit message if it differs).

- [ ] **Step 3: Implement** `is_interrogative` in `dialogue_state.py`:

```python
_INTERROGATIVE_OPENERS = (
    "was", "is", "are", "were", "did", "does", "do", "has", "have", "can",
    "could", "will", "would", "what", "when", "where", "which", "who", "why", "how",
)


def is_interrogative(message: str) -> bool:
    text = message.strip().lower()
    if text.endswith("?"):
        return True
    first = text.split(maxsplit=1)[0] if text else ""
    return first in _INTERROGATIVE_OPENERS
```

and guard the resume branch: `if pending is not None and not is_interrogative(message): <existing resume>` — an interrogative turn leaves `pending_servicing` intact (so a later "yes, go ahead" still resumes) but routes the current turn normally.

- [ ] **Step 4: Run `test_dialogue_state.py` + full POC suite** — the orphan-resume tests from commit `3d063b4` must still pass; if one fails, the failing case defines the boundary: only gate resumes for messages where `is_interrogative` is true, never restructure the resume itself.

- [ ] **Step 5: Commit**

```bash
git add poc/retail-bank-customer-service-poc/dialogue_state.py poc/retail-bank-customer-service-poc/tests/test_dialogue_state.py
git commit -m "fix: keep pending servicing paused on status questions"
```

---

## Phase 1 — Data-generation fixes

### Task 5: Teacher-realize train/validation finals (F1)

**Files:**
- Create: `data/sources/banking-v5-tool-sft-teacher-realizations.jsonl` (tracked artifact)
- Modify: `data/banking-v5-tool-sft/{train,validation}.jsonl` + `manifest.json` + `preparation-report.json` (regenerated)
- Test: `tests/test_banking_tool_sft_data.py`

**Interfaces:**
- Consumes: `python scripts/retail_bank/prepare_tool_sft_data.py --export-teacher-requests <path>` and `--teacher-responses <path> --teacher-model <id> --teacher-prompt-hash <sha>` (already implemented, `banking_tool_sft_data.py:351-365`).
- Produces: train/val finals rewritten into genuinely distinct natural sentences; frozen test split untouched (teacher file contains **no** test-split record_ids).

- [ ] **Step 1: Export the request file**

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_tool_sft_data.py \
  --output-dir /tmp/claude-1000/-stora-work-home-pavan-learn-hello-SLM/28ae6d3c-74e6-458b-bbba-1f0dbc4176e4/scratchpad/base-export \
  --export-teacher-requests /tmp/claude-1000/-stora-work-home-pavan-learn-hello-SLM/28ae6d3c-74e6-458b-bbba-1f0dbc4176e4/scratchpad/teacher-requests.jsonl
```

Then filter to train/validation rows only (join `record_id` against the split assignment in the export dir), producing the working request list (~1020 rows).

- [ ] **Step 2: Author the realization file** (the agentic worker writes it, batched ~50 rows at a time). For each row emit `{"record_id", "immutable_hash", "user_content", "final_response"}` where `user_content` is copied **unchanged** and `final_response` is rewritten. Authoring rules — these are the whole point, follow them exactly:
  - Preserve every digit, amount, last4, name, status word, and table row from the original final; the import validator rejects fact drift, but do not rely on it to catch omissions.
  - Vary *sentence structure*, not just word choice: lead with the outcome ("Your card ending in 4821 is frozen."), or the object ("The transfer to River Consulting has been cancelled…"), or a short confirmation ("Done — …"). No two rewrites of the same answer body may share their first four words.
  - Never use any phrase from `REALIZER_FINAL_PREFIXES` or `REALIZER_FINAL_CLOSERS`, and never invent facts, dates, or hedges.
  - Keep markdown tables intact wherever the original final contains one; rewrite only the surrounding prose.
- [ ] **Step 3: Import + regenerate into a scratch dir; verify invariants**

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_tool_sft_data.py \
  --output-dir <scratch>/base-teacher \
  --teacher-responses data/sources/banking-v5-tool-sft-teacher-realizations.jsonl \
  --teacher-model "claude-opus-5" \
  --teacher-prompt-hash "$(sha256sum <requests-file> | cut -d' ' -f1)"
sha256sum data/banking-v5-tool-sft/test.jsonl <scratch>/base-teacher/test.jsonl   # MUST match
```

Expected: import validation passes (immutable hashes match, wording-only edits), `test.jsonl` byte-identical.

- [ ] **Step 4: Add the failing corpus-quality tests** to `tests/test_banking_tool_sft_data.py` (run against the regenerated output — write them against `generate_records()` with the teacher file threaded through `prepare`'s in-memory API if available, else against the written files):

```python
def test_training_finals_carry_no_realizer_scaffolding_after_teacher_pass() -> None:
    rows = [json.loads(line) for line in Path("data/banking-v5-tool-sft/train.jsonl").open()]
    prefixes = tuple(filter(None, banking_tool_sft_data.REALIZER_FINAL_PREFIXES))
    closers = tuple(filter(None, banking_tool_sft_data.REALIZER_FINAL_CLOSERS))

    offenders = [
        row["record_id"]
        for row in rows
        for final in [_final_text(row)]
        if final.startswith(prefixes) or final.endswith(closers)
    ]

    assert offenders == []


def test_training_finals_are_diverse_without_scaffolding() -> None:
    rows = [json.loads(line) for line in Path("data/banking-v5-tool-sft/train.jsonl").open()]
    finals = [_final_text(row) for row in rows]

    assert len(set(finals)) == len(finals)
```

(`_final_text` = last assistant message without tool_calls; copy the helper from this session's reverted attempt.)

- [ ] **Step 5: Replace the tracked dataset** with the teacher-realized regeneration, run the FULL local gate (all suites — the frozen-sha tests are the real check), and diff:

```bash
git diff --stat data/banking-v5-tool-sft   # test.jsonl MUST NOT appear
```

- [ ] **Step 6: Commit**

```bash
git add data/sources/banking-v5-tool-sft-teacher-realizations.jsonl data/banking-v5-tool-sft tests/test_banking_tool_sft_data.py
git commit -m "feat: teacher-realize training finals for genuine diversity"
```

### Task 6: Ineligible and missing clarify curriculum (F2)

**Files:**
- Modify: `src/hello_slm/banking_servicing_alignment_data.py` (new `_deictic_ineligible_curriculum`, called from the same place `_deictic_replace_curriculum` is)
- Modify: `tests/test_banking_servicing_alignment_data.py` (pinned counts)
- Modify: `data/banking-servicing-alignment-v5/{train,manifest,preparation-report}` (regenerated)

**Interfaces:**
- Produces: train-only records with `generation_contract = {"mode": "clarify", "entity_state": "ineligible", "tool_names": [], "argument_constraints": {}}`; scenario_family `deictic_ineligible_clarification`. The frozen validation/shadow gates are NOT extended.

- [ ] **Step 1: Write the failing coverage test:**

```python
def test_train_split_covers_ineligible_and_missing_clarifications() -> None:
    splits, _report = build_servicing_alignment_splits()
    states = Counter(
        row["expected"]["generation_contract"]["entity_state"]
        for row in splits["train"]
        if row["expected"].get("generation_contract", {}).get("mode") == "clarify"
    )

    assert states["ineligible"] >= 64
    assert states["missing"] >= 64
```

- [ ] **Step 2: Run to verify it fails** (today: ineligible ≈ 0–1 in train).

- [ ] **Step 3: Implement `_deictic_ineligible_curriculum(split)`** modeled directly on `_deictic_replace_curriculum` (`banking_servicing_alignment_data.py:1030+`), train split only. Six train-only phrase families with unused products (verify with `grep -c` before choosing) — e.g.:

```python
        {
            "phrase_family": "frozen-replace-bridge",
            "prompt": "replace that card",           # history shows the card FROZEN
            "product": "Cobble",
        },
        {"phrase_family": "pending-freeze-bridge", "prompt": "freeze that card", "product": "Drift"},
        {"phrase_family": "closed-card-copy-bridge", "prompt": "send a new copy of that card", "product": "Ember"},
        {"phrase_family": "pending-again-bridge", "prompt": "order another replacement for it", "product": "Foss"},
        {"phrase_family": "frozen-swap-bridge", "prompt": "swap that one out", "product": "Gale"},
        {"phrase_family": "closed-freeze-bridge", "prompt": "freeze the card you listed", "product": "Hollow"},
```

Record shape per family × 4 prompt_forms × 2 history forms (8 realizations): history = assistant listing exactly one card whose status makes the request ineligible (`frozen`, `replacement_pending`, or `closed`); final = one concise clarify sentence naming the blocker and asking for an eligible choice, e.g. `"Your {card_name} ending in {card_last4} already has a replacement pending, so I can’t order another. Which active card should I work with instead?"` — vary final phrasing per family (write six distinct finals; the alignment dedup requires it). Contract: attach via the same `metadata`/`expected` update pattern the replace curriculum uses, with `"coreference_target": "clarification"`, `actionable_card_count: 0`, entity_state `ineligible`. Add a small `missing`-state set the same way (user asks to freeze "my card" with NO card list in history → clarify asking which card; entity_state `missing`) to reach the ≥64 floor.

- [ ] **Step 4: Update pinned counts** in `tests/test_banking_servicing_alignment_data.py` — `split_counts.train`, `train_families` Counter (new `deictic_ineligible_clarification` entry), following the exact procedure used three times this session (search for the previous values, bump by the added record count).

- [ ] **Step 5: Leak pre-check, then full alignment suite**

```bash
python3 - <<'EOF'
import sys; sys.path.insert(0,'src')
from hello_slm import banking_servicing_alignment_data as m
splits = {"train": m._train_records(), "validation": m._validation_records(), "test": m._test_records()}
assert not m._heldout_long_ngram_leaks_in_train(splits)
assert not m._heldout_exact_currents_in_train(splits)
print("clean", len(splits["train"]))
EOF
rtk proxy python -m pytest tests/test_banking_servicing_alignment_data.py -q
```

- [ ] **Step 6: Regenerate `data/banking-servicing-alignment-v5`** (`PYTHONPATH=src python scripts/retail_bank/prepare_servicing_alignment_data.py`), verify only `train.jsonl`/`manifest.json`/`preparation-report.json` changed and the shadow sha is still `55c9df4b…`, run the FULL gate, commit:

```bash
git add src/hello_slm/banking_servicing_alignment_data.py tests/test_banking_servicing_alignment_data.py data/banking-servicing-alignment-v5
git commit -m "feat: cover ineligible and missing clarify states in training"
```

### Task 7: Seed enough transactions for the demo presets (F7)

**Files:**
- Modify: `poc/retail-bank-customer-service-poc/synthetic_bank.json`
- Test: `tests/test_banking_tool_sft_data.py` (POC preset coherence) / `poc/.../tests/test_mock_bank.py` if present — check `ls poc/.../tests/` first

**Interfaces:**
- Produces: ≥6 transactions for alex.demo, ≥5 for maya.demo; new rows strictly OLDER than the existing oldest (`2026-07-22T17:25:00Z`) so every existing "recent"-ordered expectation is unchanged.

- [ ] **Step 1: Write the failing test** (place it beside the existing synthetic-bank/preset tests; find them with `grep -rn "synthetic_bank" tests/ poc/*/tests/ | grep -i test`):

```python
def test_every_customer_can_answer_the_five_most_recent_preset() -> None:
    payload = json.loads(
        Path("poc/retail-bank-customer-service-poc/synthetic_bank.json").read_text()
    )

    for customer in payload["customers"]:
        assert len(customer["transactions"]) >= 5
```

- [ ] **Step 2: Run to verify it fails** (alex has 4, maya has 3).

- [ ] **Step 3: Add transactions** — for each customer append rows with ids continuing the existing sequence (`txn_alex_005`…), `posted_at` values in `2026-07-18` – `2026-07-21` (older than all existing rows), plausible descriptions/categories consistent with existing style, `"disputed": false`, `"status": "posted"`. Copy the exact field set of an existing row.

- [ ] **Step 4: Run the FULL gate** — the screenshot case asserts only `must_include: ["transaction"]`, and frozen splits don't embed synthetic_bank rows (verified this session), so everything must stay green. If a test fails on transaction data, STOP and report — that means a frozen artifact embeds seed rows and the addition needs user sign-off.

- [ ] **Step 5: Commit**

```bash
git add poc/retail-bank-customer-service-poc/synthetic_bank.json tests/
git commit -m "fix: seed enough transactions for the recent-activity presets"
```

---

## Phase 2 — Regenerate, retrain, release (requires user approval before Step 2)

### Task 8: Publish the dataset revision and retrain

**Files:**
- No code; operates the release pipeline.

- [ ] **Step 1: Publish the regenerated dataset**

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_servicing_alignment_data.py --push-to-hub
python3 -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('spkc83/retail-bank-servicing-alignment-sft').sha)"
```

Verify the pushed manifest sha-matches local (same check used four times this session), push the code branch to `github-model`.

- [ ] **Step 2: ASK THE USER before launching** (billed ~$0.57–0.80 of ≈$1.31), then:

```bash
bash scripts/retail_bank/run_remote_continuation_job.sh \
  <HEAD-full-sha> <dataset-revision-sha> \
  d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2 \
  spkc83/retail-bank-servicing-agent-9b-peft-v7-grounded-generation 550
```

The worker publishes automatically only after two consecutive perfect dev gates plus a 32/32 shadow pass. If the gate fails: **stop and report** — do not iterate without the user (standing instruction from this session).

- [ ] **Step 3: Record the new adapter revision** and the gate trajectory from the durable bucket (`hf buckets sync .../behavioral-evaluations`).

### Task 9: Redeploy both surfaces and verify

- [ ] **Step 1: Deploy the Space** with the new adapter revision (same `deploy_zero_gpu_space.py` invocation as this session, `--adapter-revision`/`--model-revision` updated; router pin `c0d71b433fd1eef510fce36f6308eb36e423e329` unchanged; plan first, then `--execute --allow-publish`).
- [ ] **Step 2: Launch the local instance** (same env-pinned `uv run scripts/retail_bank/run_local_streamlit.py` as this session) and re-run the three-turn Playwright probe (`scratchpad/poc_e2e.py`) plus the two regression sequences from the review: the stolen-card freeze flow and "Show my five most recent transactions" (must now return 5 rows and a natural, filler-free lead-in).
- [ ] **Step 3: Ask the user to run the authenticated Space smoke** (`! RETAIL_BANK_DEMO_PASSWORD=… PYTHONPATH=src python scripts/retail_bank/smoke_zero_gpu_space.py --space-id spkc83/retail-bank-servicing-poc --execute`).
- [ ] **Step 4: Update the release table** in `docs/08-end-to-end-runbook.md` (+ README pins) and commit as `docs: record …`, following commit `1fcbd9a`'s shape.

---

## Explicitly out of scope

- **F8 (suspected OOD misroute):** unconfirmed; revisit only if it reproduces with a known input.
- **Router retrain (F4 root cause):** the Task 3 constraint neutralizes the failure deterministically. Retraining the DistilBERT router on expanded mutation-imperative coverage is a follow-up (local GPU, free) once the Granite release lands.
- **Frozen release-eligibility eval** (215+13+9 via `run_remote_tool_eval_job.sh`, ~$0.3–0.7): required for formal release eligibility; needs a credit top-up after Task 8.
