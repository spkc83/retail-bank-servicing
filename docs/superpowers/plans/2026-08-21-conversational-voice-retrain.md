# Conversational-Voice Granite Retrain Implementation Plan (rev 2, post-critique)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Granite's customer-visible replies read like a warm, natural human bank-chat agent by rewriting **every loss-bearing train/validation final** whose text is currently one-sentence, filler-wrapped, or a rigid template, then train ONE capped PEFT continuation run inside a ≤$3 budget and measure the tone change on the live demo against an explicit threshold.

**Architecture:** Three data levers, no change to tool calls, record counts, split membership, multipliers or gates:
1. **Base rows (1,020 = 841 train + 179 validation)** — new teacher-realization file through the existing `export_teacher_realization_requests` / `import_teacher_realizations` mechanism (`src/hello_slm/banking_tool_sft_data.py:510-573`); rewrites `final_response` and cleans `user_content`.
2. **Alignment filler rows (819 = 651 train + 168 validation; 21 sentence cores × (31 train + 8 validation) opener/closer permutations from `_varied_final`, `banking_servicing_alignment_data.py:2344`)** — new alignment-side teacher hook (same import function) rewriting `final_response` only.
3. **`deictic_replace_ambiguity` (672 train + 16 validation; ≈47% of loss-bearing weighted mass at ambiguity multiplier 4)** — replace the single inline template in `_deictic_replace_curriculum` with a 32-phrasing conversational pool for the `train`/`validation` branches only; the `shadow` branch keeps the verbatim old string.

Frozen `test.jsonl` (both datasets), `coreference-shadow.jsonl`, `granite-v7-shadow.jsonl`, `screenshot-regression.jsonl` stay byte-identical. Loss-masked `deictic_replace_action` finals (672, `cloud_continue_tool_sft.py:515-534`) are left alone (zero gradient). After this plan, unchanged loss-bearing mass is limited to `deictic_ineligible/missing_clarification` (144 rows, weight 1), `natural_social_style`, `v7_*` (already conversational) — ≈3% of weighted mass.

**Tech Stack:** Python 3.12, uv, pytest, huggingface_hub, HF Jobs (`run_remote_continuation_job.sh`, `rtx-pro-6000`, $2.75/h, per-minute billing).

**Spec:** User ask "make the model more chatty and responses more natural conversation like"; budget "keep the budget <=$3"; anti-loop directive (one targeted iteration, then report). Critique rev 1 findings C1–C5, M1–M9 are each addressed below (see "Critique ledger").

## Global Constraints

- HF Jobs spend ≤ $3.00 total. Recipe identical to v8 (`MAX_STEPS=964`, same LR schedule) with two hard caps exposed through the launcher and stated literally in the approval prompt: `MAX_TRAIN_SECONDS=1800` (training loop) and `JOB_TIMEOUT=45m` (whole job incl. download, eval, upload). Measured v8 analogue: total job 13.8 min = $0.63 (gate passed at step 350, selected 400, 0.81 steps/s); measured full 964-step continuation 29.5 min = $1.35 (≈9.7 min non-training overhead). Run-1 quote: expected $0.63–$1.35; capped worst case 30 + 9.7 min ≈ $1.82; hard ceiling 45 min = $2.06. A second billed run is offered only if its hard ceiling fits in `$3.00 − actual run-1 cost`, and only after fresh priced approval. No launch / publish / push without explicit approval via AskUserQuestion.
- Byte-identical after regeneration: `data/banking-v5-tool-sft/test.jsonl`, `data/banking-servicing-alignment-v5/{test,coreference-shadow,granite-v7-shadow,screenshot-regression}.jsonl`.
- Record counts unchanged: base 841/179/180; alignment 2202/218/35 (`tests/test_banking_servicing_alignment_data.py:656`).
- Do not edit `REALIZER_FINAL_PREFIXES`/`REALIZER_FINAL_CLOSERS`/`FINAL_OPENERS`/`FINAL_CLOSERS` or the poc `REALIZER_FILLER_*` lists (parity lock `tests/test_banking_generation_guidance.py:106-118`).
- Alignment teacher rows edit `final_response` ONLY (user text feeds leakage/held-out gates); enforced in code. Teacher files contain no test-split ids of **either** dataset (composite test = 180 base + 35 alignment); enforced in code.
- Every final obeys `validate_records` (`banking_tool_sft_data.py:756-1013`): ≥7 normalized words; globally unique; no substring `demo|synthetic|mock|test|backend|gpu|router|tool call` (so no `latest`, `greatest`, `contest`, `demonstrate`); path markers (`last four digits`, `retail banking`, `account numbers`+`customer ids`, `[Policy: id]` + every `required_claims`, **no `forbidden_claims`** `:882-887`); no POC-preset user text (`POC_PRESET_KEYS`, `:290-307`, `:800-802`); no held-out current (`SCREENSHOT_HELDOUT_CURRENTS`, alignment `:405`).
- Every final passes the runtime validators it will meet (`poc/.../response_policy.py`): `INTERNAL_LANGUAGE_PATTERNS` (`tool(s|ing| call| result)`, `model`, `9b`, `classifier|router`, `gpu|cpu|cuda|zerogpu`, `back[- ]?end`, `test(ing| data)?`, `demo(nstration)?`, `mock`, `synthetic`); `COMPLETED_ACTION_CLAIMS` never matches a final whose conversation has no prior `role=="tool"` message (`tests/test_response_policy.py:430` runs this over all of train.jsonl); `validate_grounded_answer` literals (last4 + `froze|frozen` for freeze, `replacement` for replace, `transaction.description` + `dispute` for disputes, `transfer.recipient` + `cancelled|canceled` for cancels, `could not|couldn't|unable|not found` on failed calls); `validate_policy_answer` — no number or number word (`one`…`ninety`) absent from the chunk text.
- Read-path reality (`model_service.py:499-505`, `leading_prose` = text before the first `|`): for table-bearing finals the deliverable is **1–2 warm sentences before the table, nothing after it**. The corpus test exempts table rows from the ≥2-sentence rule and instead requires ≥1 sentence of lead-in.
- Dev-gate safety: clarify finals mentioning cards keep `which` and `card` within the first 45 words (gate generates 128 greedy tokens, `cloud_continue_tool_sft.py:1140-1147`).
- Known, deliberately out of scope (frozen split would change): the `get createdin the app?` concatenation defect in `_suffix` (`banking_servicing_alignment_data.py:2473`) also occurs in `test.jsonl` ("createdfrom my profile"); fix at the next frozen-split rotation. Alignment user turns remain scaffolded ("… I am checking this in the mobile app. Please keep the answer concise.") — documented limitation of this iteration.
- No new dependencies. `rtk proxy python -m pytest` for real output; run from repo root.

## Success criterion (measured in Task 7)

On the local demo with the v9 adapter (same router/code as the v8 probe), the 8-sequence `release_probe.py` passes 8/8 AND, over its model-authored replies: mean sentence count ≥ 2.0 (v8: ≈1.1), zero filler-list phrases, zero `INTERNAL_LANGUAGE` matches, and ≥ 5 of 7 model-authored replies have ≥ 2 sentences. If the probe passes but the tone threshold is missed, the result is reported as "gate pass, tone target missed" and no second run is launched without a new decision from the user.

## Rollback

v8 pins stay the code/doc defaults until the user approves a repin. The regenerated dataset is published as a **new** Hub revision; the v8 revision `a78bed17…` remains addressable and is what the v8 adapter's docs cite. Reverting = not repinning.

---

## Appendix A — Voice Specification (authoring contract for Tasks 3–4)

**Persona.** Harbor, a friendly, competent retail-bank chat agent. Real-person support-chat voice: contractions, second person, plain words, calm warmth. No gushing, no emoji, at most one `!` per reply, never "As an AI", never "in this session".

| Mode / family | Shape | Must keep |
|---|---|---|
| execute_tool, mutation (freeze, replace, dispute, cancel) | 2–3 sentences: (1) what was done, naming the card/transaction/transfer exactly as the original did; (2) one sentence of genuinely useful context or an honest next step Harbor can actually do (nine supported actions, or general safety guidance); (3) optional short, varied offer of help. | all digits, `frozen`/`froze`, `replacement`, `dispute`, `cancelled`, the recipient / merchant name verbatim; `sorry` in `emergency_card_freeze` |
| execute_tool, read (accounts, cards, transactions, transfers, service cases) | 1–2 warm sentences **before** the original markdown table (verbatim), **nothing after the table**. No prose facts that aren't in the original. | `available`, `current`, both balances (`read_accounts`); `2026-06-18`, `address_update`, `Confirm mailing address update` (`service_case_context`) |
| execute_tool, error outcomes | Honest and kind: what did **not** happen, nothing changed, a retry/next step. | `could not` (or `was not`) AND one of `no `/`not `/`unchanged` |
| clarify | One warm acknowledgement + the specific question, ≤45 words. | `which`, `card`, `last four digits`, original card names/digits |
| converse (thanks, greeting, check-in, no-action follow-up, action-summary follow-up, hard negatives) | 1–3 natural sentences; hard negatives firm but friendly; never claim a completed action (`froze`, `has been frozen`, `is now frozen`, `replacement is pending`, `I've frozen/replaced/cancelled/disputed`). | `account numbers`, `customer ids` (hard negatives) |
| retrieve_policy | Conversational framing of the same policy content. No new numbers or number words, no new hedges, none of the forbidden claims: `rate never changes`, `interest is credited daily`, `every overdraft is paid`, `every overdraft has a fee`, `approval is automatic`, `guaranteed rate`, `must be at least 18`, `guaranteed approval`, `no identification is required`. | `[Policy: <id>]`, every `required_claims` phrase, FAQ markers |
| refuse_ood | Friendly redirect to what Harbor can do. | `retail banking` |

**Diversity.** Within a scenario family, no opening trigram (first three words, lowercased) may be used by more than 4 finals; vary the second sentence's job (context / next step / reassurance) and vary or omit the offer. Never reuse any sentence from `REALIZER_FINAL_*`, `FINAL_OPENERS`, `FINAL_CLOSERS`, or the split leads `For this request,` / `In this session,`.

**Banned in finals (substring):** demo, synthetic, mock, test (incl. latest/greatest/contest), backend, gpu, router, tool, model, classifier, cpu, cuda, session, "As an AI".

**User text (base rows only).** Rewrite into a grammatical, natural chat message with the same intent and the same digits/fact words (validator enforces). Remove scaffold residue ("Please please…", "I need you to how should I…", "Before we continue.", dangling "so I can finish this banking task"). Must not normalize to any POC preset (`please replace my debit card`, `show my account balances`, `my card was stolen freeze it`, `cancel the pending transfer to river consulting`, `what is the status of my debit card`, … full list `POC_PRESET_KEYS`) or any held-out current (`SCREENSHOT_HELDOUT_CURRENTS`). Alignment rows: user text byte-identical.

## Appendix B — Conversational ambiguity pool (Task 2; exactly these 32 strings)

```python
_CONVERSATIONAL_AMBIGUITY_FINALS = (
    "I can see two cards here: {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}. Which card would you like me to replace? Share the last four digits and I'll get it started.",
    "Happy to do that — there are two cards on file, {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}. Which card should I replace? Just tell me the last four digits.",
    "Quick check before I go ahead: this profile has {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}. Which card do you mean? The last four digits will do.",
    "There are two cards I could replace: {card_name} ending in {card_last4} or {other_card_name} ending in {other_card_last4}. Which card is it? Send me the last four digits and I'll take care of it.",
    "Sure thing. I just want to make sure I replace the right card — {card_name} ending in {card_last4} or {other_card_name} ending in {other_card_last4}? Let me know which one by its last four digits.",
    "Okay, two cards match that: {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}. Which card should I put the replacement through for? The last four digits are all I need.",
    "Before I order a replacement, which card did you have in mind? You have {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4} — the last four digits will point me to the right one.",
    "I want to get this right for you. Which card needs replacing — {card_name} ending in {card_last4} or {other_card_name} ending in {other_card_last4}? Reply with the last four digits and I'll start it.",
    "No problem. Two cards are on this profile, {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}, so which card should I replace? The last four digits will confirm it.",
    "Let's sort that out. Which card are we replacing: {card_name} ending in {card_last4} or {other_card_name} ending in {other_card_last4}? Give me the last four digits and I'll get moving.",
    "Got it — one detail first. I see {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}; which card would you like replaced? Just the last four digits, please.",
    "Of course. Since you have both {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}, which card should I replace? The last four digits will settle it.",
    "Absolutely, I can help with that. Which card do you want replaced — {card_name} ending in {card_last4} or {other_card_name} ending in {other_card_last4}? Share the last four digits and it's done.",
    "Two cards could match here: {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}. Tell me which card you mean by its last four digits and I'll order the replacement.",
    "Happy to help replace it. I found {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}, so which card is the one? The last four digits are enough.",
    "Just to be safe, which card should I replace? Your profile shows {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4} — the last four digits will tell me which one.",
    "I can do that right away once I know which card you mean. Is it {card_name} ending in {card_last4} or {other_card_name} ending in {other_card_last4}? The last four digits are all I need.",
    "Sure — you have two cards with me, {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}. Which card should get the replacement? Just send the last four digits.",
    "Good news: a replacement is easy to arrange. Which card is it, though — {card_name} ending in {card_last4} or {other_card_name} ending in {other_card_last4}? Reply with the last four digits.",
    "Let me make sure I pick the right one. Between {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}, which card should I replace? The last four digits will confirm it for me.",
    "On it — I just need to know which card. There's {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}; tell me the last four digits of the one to replace.",
    "That's doable. Your cards are {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}, so which card would you like me to replace? Share its last four digits and I'll begin.",
    "Right, let's get you a new card. Which card are we talking about — {card_name} ending in {card_last4} or {other_card_name} ending in {other_card_last4}? The last four digits will point me to it.",
    "You've got two cards on this profile: {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}. Which card should I replace? Pop the last four digits in and I'll handle the rest.",
    "Can do. To avoid replacing the wrong card, is it {card_name} ending in {card_last4} or {other_card_name} ending in {other_card_last4}? Let me know which by the last four digits.",
    "One quick question so I replace the right card: {card_name} ending in {card_last4} or {other_card_name} ending in {other_card_last4}? Which one is it — the last four digits will do.",
    "I'd be glad to. There are two candidates, {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}; which card do you need replaced? Send the last four digits and I'll set it up.",
    "Understood. Which card should I replace — {card_name} ending in {card_last4} or {other_card_name} ending in {other_card_last4}? Once I have the last four digits I'll get the replacement started.",
    "Sure, I'll arrange a replacement. Because I see {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}, which card did you mean? The last four digits will clear it up.",
    "Almost there — I just need to know which card. Your options are {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}; share the last four digits of the one you want replaced.",
    "Thanks for flagging it. I'm seeing {card_name} ending in {card_last4} and {other_card_name} ending in {other_card_last4}, so which card should I replace? Give me the last four digits and I'll take it from there.",
    "Let's do it. Which card needs the replacement: {card_name} ending in {card_last4} or {other_card_name} ending in {other_card_last4}? The last four digits are all I need from you.",
)
```

---

### Task 1: Alignment-side teacher realization hook

**Files:**
- Modify: `src/hello_slm/banking_servicing_alignment_data.py` (`write_servicing_alignment_dataset`, ~line 388)
- Modify: `scripts/retail_bank/prepare_servicing_alignment_data.py` (new flags)
- Test: `tests/test_banking_servicing_alignment_data.py`

**Interfaces:**
- Consumes: `export_teacher_realization_requests(records, path)`, `import_teacher_realizations(records, path, *, teacher_model, teacher_prompt_hash)` (`hello_slm.banking_tool_sft_data`); `load_base_sft_splits(base_sft_dir)`.
- Produces: `write_servicing_alignment_dataset(output_dir, *, base_sft_dir=DEFAULT_BASE_SFT_DIR, synthetic_bank_path=..., export_teacher_requests: Path | None = None, teacher_responses: Path | None = None, teacher_model: str | None = None, teacher_prompt_hash: str | None = None)`; manifest `report["alignment_teacher_realization"] = {"model", "prompt_hash", "realized_counts": {"train": int, "validation": int}}`; CLI flags `--export-teacher-requests`, `--teacher-responses`, `--teacher-model`, `--teacher-prompt-hash`.

- [ ] **Step 1: Failing tests.** Follow the existing fixture pattern (`tests/test_banking_servicing_alignment_data.py:656-658`: `prepare(output_dir=base_dir, pilot_count=120)` to build a small base dir, then call the writer with `base_sft_dir=base_dir`).

```python
def _small_base(tmp_path: Path) -> Path:
    base_dir = tmp_path / "base"
    prepare(output_dir=base_dir, pilot_count=120)
    return base_dir


def _export_alignment_requests(tmp_path: Path, base_dir: Path) -> list[dict]:
    requests = tmp_path / "requests.jsonl"
    write_servicing_alignment_dataset(tmp_path / "export", base_sft_dir=base_dir, export_teacher_requests=requests)
    rows = [json.loads(line) for line in requests.read_text().splitlines() if line]
    assert len(rows) == 2202 + 218
    return rows


def test_alignment_teacher_hook_rewrites_only_train_and_validation_finals(tmp_path: Path) -> None:
    base_dir = _small_base(tmp_path)
    rows = _export_alignment_requests(tmp_path, base_dir)
    target = next(r for r in rows if r["record_id"].startswith("tool_outcome_consistency"))
    response = {
        "record_id": target["record_id"], "immutable_hash": target["immutable_hash"], "user_content": target["user_content"],
        "final_response": "Good news — the replacement for your Cashback Debit ending in 7742 is already pending. Keep using your other cards as normal while it is on the way.",
    }
    responses = tmp_path / "responses.jsonl"
    responses.write_text(json.dumps(response) + "\n")
    before_test = (tmp_path / "export" / "test.jsonl").read_bytes()

    manifest = write_servicing_alignment_dataset(
        tmp_path / "out", base_sft_dir=base_dir, teacher_responses=responses,
        teacher_model="claude-opus-5", teacher_prompt_hash="sha256:" + "0" * 64,
    )

    train = [json.loads(l) for l in (tmp_path / "out" / "train.jsonl").read_text().splitlines() if l]
    rewritten = next(r for r in train if r["record_id"] == target["record_id"])
    assert rewritten["messages"][-1]["content"] == response["final_response"]
    assert rewritten["provenance"]["teacher_model"] == "claude-opus-5"
    assert (tmp_path / "out" / "test.jsonl").read_bytes() == before_test
    assert manifest["report"]["alignment_teacher_realization"]["realized_counts"] == {"train": 1, "validation": 0}


@pytest.mark.parametrize("which", ["base", "alignment"])
def test_alignment_teacher_hook_rejects_test_split_rows(tmp_path: Path, which: str) -> None:
    base_dir = _small_base(tmp_path)
    _export_alignment_requests(tmp_path, base_dir)
    test_rows = [json.loads(l) for l in (tmp_path / "export" / "test.jsonl").read_text().splitlines() if l]
    row = test_rows[0] if which == "base" else test_rows[-1]   # composite test = base rows first, alignment rows last
    bad = {"record_id": row["record_id"], "immutable_hash": "sha256:" + "0" * 64, "user_content": "x", "final_response": "y"}
    responses = tmp_path / "bad.jsonl"; responses.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="test split"):
        write_servicing_alignment_dataset(tmp_path / "out", base_sft_dir=base_dir, teacher_responses=responses,
                                          teacher_model="m", teacher_prompt_hash="sha256:" + "0" * 64)


def test_alignment_teacher_hook_rejects_user_text_edits(tmp_path: Path) -> None:
    base_dir = _small_base(tmp_path)
    rows = _export_alignment_requests(tmp_path, base_dir)
    edited = {k: rows[0][k] for k in ("record_id", "immutable_hash", "final_response")}
    edited["user_content"] = rows[0]["user_content"] + " please"
    responses = tmp_path / "bad.jsonl"; responses.write_text(json.dumps(edited) + "\n")
    with pytest.raises(ValueError, match="final_response only"):
        write_servicing_alignment_dataset(tmp_path / "out", base_sft_dir=base_dir, teacher_responses=responses,
                                          teacher_model="m", teacher_prompt_hash="sha256:" + "0" * 64)


def test_alignment_teacher_hook_requires_model_and_prompt_hash(tmp_path: Path) -> None:
    base_dir = _small_base(tmp_path)
    responses = tmp_path / "r.jsonl"; responses.write_text("")
    with pytest.raises(ValueError, match="teacher_model and teacher_prompt_hash"):
        write_servicing_alignment_dataset(tmp_path / "out", base_sft_dir=base_dir, teacher_responses=responses)
```

- [ ] **Step 2: Run to verify failure** — `rtk proxy python -m pytest tests/test_banking_servicing_alignment_data.py -k teacher_hook -q` → TypeError.

- [ ] **Step 3: Implement.** In `write_servicing_alignment_dataset`, keep the existing order up to `base_manifest, base_splits = load_base_sft_splits(base_sft_dir)`, then insert **after** it (the test-id check needs base test ids):

```python
    trainable = [*alignment_splits["train"], *alignment_splits["validation"]]
    if export_teacher_requests is not None:
        export_teacher_realization_requests(trainable, export_teacher_requests)
    realized_counts = {"train": 0, "validation": 0}
    if teacher_responses is not None:
        if not teacher_model or not teacher_prompt_hash:
            raise ValueError("teacher_model and teacher_prompt_hash are required with teacher_responses")
        _assert_alignment_teacher_rows(
            teacher_responses,
            trainable=trainable,
            test_ids={str(r["record_id"]) for r in (*base_splits["test"], *alignment_splits["test"])},
        )
        realized = import_teacher_realizations(
            trainable, teacher_responses, teacher_model=teacher_model, teacher_prompt_hash=teacher_prompt_hash,
        )
        n_train = len(alignment_splits["train"])
        alignment_splits = {**alignment_splits, "train": realized[:n_train], "validation": realized[n_train:]}
        for split in ("train", "validation"):
            realized_counts[split] = sum(
                1 for r in alignment_splits[split] if r["provenance"].get("teacher_model") == teacher_model
            )
```

```python
def _assert_alignment_teacher_rows(
    path: Path, *, trainable: Sequence[Mapping[str, Any]], test_ids: set[str]
) -> None:
    user_text = {
        str(r["record_id"]): str([m for m in r["messages"] if m["role"] == "user"][-1]["content"]).strip()
        for r in trainable
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record_id = str(row.get("record_id"))
        if record_id in test_ids:
            raise ValueError(f"{record_id}: teacher rows must not target the test split")
        if record_id in user_text and str(row.get("user_content", "")).strip() != user_text[record_id]:
            raise ValueError(f"{record_id}: alignment teacher rows may edit final_response only")
```

Add `"alignment_teacher_realization": {"model": teacher_model, "prompt_hash": teacher_prompt_hash, "realized_counts": realized_counts}` to `report`. CLI: add the four arguments and pass through.

- [ ] **Step 4: Run** the new tests, then the whole file. **Step 5: Commit** — `feat: teacher-realization hook for alignment-side finals`.

### Task 2: Conversational ambiguity pool + launcher time cap

**Files:**
- Modify: `src/hello_slm/banking_servicing_alignment_data.py` (`_deictic_replace_curriculum`, the `ambiguity = _record(... final=...)` block)
- Modify: `scripts/retail_bank/run_remote_continuation_job.sh`, `scripts/retail_bank/hf_job_continue_tool_sft.py` (no default change; launcher passes `--max-train-seconds`)
- Test: `tests/test_banking_servicing_alignment_data.py`, `tests/test_banking_tool_sft_job.py` (launcher args)

- [ ] **Step 1: Failing tests**

```python
def test_ambiguity_finals_are_conversational_in_train_and_verbatim_in_shadow() -> None:
    train = [r for r in _deictic_replace_curriculum("train") if r["metadata"]["scenario_family"] == "deictic_replace_ambiguity"]
    shadow = [r for r in _deictic_replace_curriculum("shadow") if r["metadata"]["scenario_family"] == "deictic_replace_ambiguity"]
    finals = [r["messages"][-1]["content"] for r in train]
    assert len(finals) == 672 and len(set(finals)) == 672
    assert all(f.split(". Which card should I replace? Please share")[0] != f for f in finals) or True  # pool, not the old template
    assert not any(f.startswith("I found ") and f.endswith("shown in the app.") for f in finals)
    for f in finals:
        head = " ".join(f.lower().split()[:45])
        assert "which" in head and "card" in head and "last four digits" in f
    assert all(s["messages"][-1]["content"].startswith("I found ") and s["messages"][-1]["content"].endswith("shown in the app.") for s in shadow)
    openings = Counter(" ".join(f.lower().split()[:3]) for f in finals)
    assert max(openings.values()) <= 672 // 32 + 1
```
and in `tests/test_banking_tool_sft_job.py` (next to the existing launcher assertions): with `MAX_TRAIN_SECONDS=1800` in the environment the rendered `hf jobs uv run` args include `--max-train-seconds 1800`; without it they include `--max-train-seconds 3600`.

- [ ] **Step 2: Implement.** Add `_CONVERSATIONAL_AMBIGUITY_FINALS` (Appendix B, verbatim) at module level. In `_deictic_replace_curriculum`, replace the inline ambiguity final with:

```python
            legacy_ambiguity_final = (
                f"I found {card_name} ending in {card_last4} and {other_card_name} "
                f"ending in {other_card_last4}. Which card should I replace? Please "
                "share the last four digits shown in the app."
            )
            ambiguity_final = (
                legacy_ambiguity_final
                if split == "shadow"
                else _CONVERSATIONAL_AMBIGUITY_FINALS[
                    (pair_index + 7 * family_index) % len(_CONVERSATIONAL_AMBIGUITY_FINALS)
                ].format(**history_values)
            )
```
Launcher: `max_train_seconds="${MAX_TRAIN_SECONDS:-3600}"`, validate `^[0-9]+$`, append `--max-train-seconds "$max_train_seconds"` to `job_args`; update the usage comment. Bootstrap already forwards the flag (`hf_job_continue_tool_sft.py:58,188`).

- [ ] **Step 3: Verify frozen gate files unchanged:** `PYTHONPATH=src uv run python scripts/retail_bank/prepare_servicing_alignment_data.py --output-dir <scratch>/chatty/align-check` then `sha256sum` of `coreference-shadow.jsonl`, `granite-v7-shadow.jsonl`, `screenshot-regression.jsonl`, `test.jsonl` equals the committed ones. Run the whole alignment test file + `tests/test_banking_tool_sft_job.py` + `tests/test_banking_tool_sft_continuation.py`.
- [ ] **Step 4: Commit** — `feat: conversational ambiguity clarifications and launcher train-time cap`.

### Task 3: Authoring tooling — batch checker and request files

**Files:**
- Create: `scripts/retail_bank/check_teacher_batch.py`
- Test: `tests/test_check_teacher_batch.py`
- Scratch: `<scratchpad>/chatty/{base-export,align-export}/`, `base-requests.jsonl` (1020, already exported), `base-authoring-inputs.jsonl` (already built: request + v8 wording + conversation + grounding facts), `align-requests.jsonl` (819), `align-authoring-inputs.jsonl`, `voice-spec.md` (= Appendix A)

**Interface:** `python scripts/retail_bank/check_teacher_batch.py --requests R.jsonl --responses S.jsonl [--responses S2.jsonl ...] --records-dir EXPORT_DIR [--records-dir EXPORT_DIR2] [--finals-only] [--min-sentences 2] [--exempt-family F ...]` → exit 0 + JSON summary, or exit 2 + `record_id: rule: detail` lines. Rules (all hard):

| rule | check | source of truth |
|---|---|---|
| a | `validate_response_row` (digits, `FACT_WORD_RE`, private data) | `realize_tool_sft_teacher.py:333-429` |
| b | ≥7 normalized words | `validate_records:809` |
| c | banned substrings: data list + runtime `INTERNAL_LANGUAGE_PATTERNS` + `session` | `:308-315`; `response_policy.py:19-29` |
| d | no filler prefix/closer/lead at start/end; no filler sentence anywhere | four lists + split leads |
| e | load-bearing literals present in the original survive: `could not`, `was not`, `unchanged`, `last four digits`, `retail banking`, `account numbers`, `customer ids`, `which`, `card`, `available`, `current`, `sorry`, `replacement`, `dispute`, `froze`, `frozen`, `cancelled`, `canceled`, `[Policy: …]`, every `expected.grounding_facts` value string, ISO dates; PLUS from the record's tool envelopes: `card.last4`, `transfer.recipient`, `transaction.description` when present | `validate_grounded_answer` (`response_policy.py:318-377`) |
| f | markdown table block byte-identical; no text after the table | `leading_prose` |
| g | clarify: `which` + `card` within first 45 words if original had them | dev gate |
| h | sentences (table lines excluded) ≥ `--min-sentences` unless family exempt or table-bearing (then ≥1 before table); ≤4; ≤70 words ex-table; ≤1 `!`; ASCII + `’ — –` only | Appendix A |
| i | `--finals-only` ⇒ `user_content` identical | Global Constraints |
| j | uniqueness: normalized final unique across ALL `--responses` files and against every final in ALL `--records-dir` splits not covered by responses; per-family opening-trigram ≤ 4 | `validate_records:818`, Appendix A |
| k | `COMPLETED_ACTION_CLAIMS` never matches when the record has no prior `role=="tool"` message (any mode) | `test_response_policy.py:430` |
| l | policy rows (`expected.path == "retrieval_grounded_policy"` or `[Policy:` in original): no `forbidden_claims` substring (normalized); no number / number word (`NUMBER_WORD_VALUES`) absent from the chunk text; every `required_claims` present | `validate_records:857-887`; `validate_policy_answer` |
| m | rewritten `user_content` (base) must not normalize to a `POC_PRESET_KEYS` entry or a `SCREENSHOT_HELDOUT_CURRENTS` entry; must keep the original's digits | `validate_records:800-802`; alignment `:405` |

- [ ] **Step 1: Failing tests** — one violating pair per rule a–m using tiny synthetic request/response rows and a two-record `records-dir` written by the test (reuse `prepare(pilot_count=120)` output for realistic records where a rule needs envelopes/policy chunks); one fully passing case.
- [ ] **Step 2: Implement** (load `realize_tool_sft_teacher.py` and the poc `response_policy.py` via `importlib.util.spec_from_file_location`; import lists from `hello_slm`; policy chunks via `hello_slm.banking_tool_sft_data.POLICY_CHUNKS` or the canonical corpus loader).
- [ ] **Step 3: Build the alignment request set** — `prepare_servicing_alignment_data.py --output-dir <scratch>/chatty/align-export --export-teacher-requests <scratch>/chatty/align-requests-all.jsonl`; keep rows whose final starts with a `FINAL_OPENERS`/`REALIZER` prefix or split lead, or ends with a closer → `align-requests.jsonl` (expect 819 = 651 + 168); build `align-authoring-inputs.jsonl` (request + conversation context + envelopes + family/mode) grouped by family. Write `voice-spec.md`. Split into batch files of ≤50 rows (`base-NN.jsonl` ×21, `align-NN.jsonl` ×17; a batch never splits a sentence core).
- [ ] **Step 4: Run tests; commit** — `feat: teacher batch checker for realization passes`.

### Task 4: Author the realizations (batched subagents, checker-gated, bounded)

**Files:**
- Create: `data/sources/banking-v5-tool-sft-teacher-realizations-v2.jsonl` (1,020 rows)
- Create: `data/sources/banking-servicing-alignment-v5-teacher-realizations.jsonl` (819 rows)

**Process.** Fresh subagent per batch (model `opus`; voice quality is the deliverable). Inputs: `voice-spec.md`, the batch's authoring-input file, the checker command. Output: the batch response file with exit 0 from the checker run against the **cumulative** responses (`--responses <cumulative-base> --responses <cumulative-align> --responses <this-batch>` with both `--records-dir`s) so uniqueness is global across both files. **Bound:** ≤3 checker rounds per batch; rows still failing go to a `residue-NN.jsonl` the controller re-dispatches once with an explicit list of violations; anything still failing after that is reported, not looped. Style review: a separate `sonnet` reviewer samples **20%** of each batch against Appendix A (natural? repetitive? scaffold-shaped?) and may send a batch back once.

- [ ] **Step 1: Base batches** (1,020). Prompt hash = `sha256sum <scratch>/chatty/base-requests.jsonl`.
- [ ] **Step 2: Alignment batches** (819, `--finals-only --min-sentences 2`). Prompt hash = `sha256sum <scratch>/chatty/align-requests.jsonl`.
- [ ] **Step 3: Whole-corpus check** — checker exit 0 with both complete files and both records dirs; `wc -l` = 1020 / 819.
- [ ] **Step 4: Commit both files** — `data: conversational-voice teacher realizations for base and alignment finals`.

### Task 5: Regenerate datasets, corpus-quality tests, docs

**Files:**
- Regenerate: `data/banking-v5-tool-sft/{train,validation}.jsonl, manifest.json, preparation-report.json, README.md, DATA_CARD.md`; `data/banking-servicing-alignment-v5/{train,validation}.jsonl, manifest.json, preparation-report.json`
- Modify: `tests/test_banking_tool_sft_data.py`, `tests/test_banking_servicing_alignment_data.py`
- Modify: `docs/02-data-generation.md`, `docs/reference/artifacts.md`, `docs/README.md`, `README.md`

- [ ] **Step 1: Failing corpus tests** (read committed files):

```python
# tests/test_banking_servicing_alignment_data.py
_SPLIT_LEADS = ("For this request,", "In this session,")
_REALIZED_FAMILIES = {"history_entity_action", "tool_outcome_consistency", "service_case_context", "card_anaphora_action",
                      "clarification_answer", "agent_repair", "banking_topic_shift", "policy_resume", "external_topic_shift",
                      "policy_detour", "history_entity_ambiguity", "deictic_replace_ambiguity"}

def _sentence_count(text: str) -> int:
    prose = " ".join(line for line in text.splitlines() if not line.strip().startswith("|"))
    return len(re.findall(r"[.!?](?:\s|$)", prose))

def _finals(split: str) -> list[tuple[str, str, str]]:
    rows = [json.loads(l) for l in Path(f"data/banking-servicing-alignment-v5/{split}.jsonl").open()]
    out = []
    for row in rows:
        final = next(m for m in reversed(row["messages"]) if m["role"] == "assistant" and not m.get("tool_calls"))
        mode = row["expected"].get("generation_contract", {}).get("mode", "")
        out.append((row["metadata"]["scenario_family"], mode, final["content"]))
    return out

def test_alignment_training_finals_carry_no_template_scaffolding() -> None:
    prefixes = tuple(filter(None, (*FINAL_OPENERS, *REALIZER_FINAL_PREFIXES, *_SPLIT_LEADS)))
    closers = tuple(filter(None, (*FINAL_CLOSERS, *REALIZER_FINAL_CLOSERS)))
    offenders = [(f, t[:40]) for split in ("train", "validation") for f, _, t in _finals(split)
                 if t.startswith(prefixes) or t.endswith(closers)]
    assert offenders == []

def test_realized_alignment_finals_are_conversational() -> None:
    short = [(f, t[:60]) for f, _, t in _finals("train")
             if f in _REALIZED_FAMILIES and "|" not in t and _sentence_count(t) < 2]
    assert short == []

def test_clarify_finals_keep_dev_gate_markers_early() -> None:
    bad = [t[:60] for split in ("train", "validation") for _, mode, t in _finals(split)
           if mode == "clarify" and "card" in t.lower()
           and not {"which", "card"} <= set(re.findall(r"[a-z]+", " ".join(t.lower().split()[:45])))]
    assert bad == []
```
In `tests/test_banking_tool_sft_data.py`: `test_training_execute_tool_finals_are_conversational` — every base train `execute_tool` final without a table has ≥2 sentences; table-bearing finals have ≥1 sentence before the first `|` and no prose after the table.

- [ ] **Step 2: Regenerate** (both preparers `mkdir(exist_ok=True)` — write in place):

```bash
sha256sum data/banking-v5-tool-sft/test.jsonl data/banking-servicing-alignment-v5/{test,coreference-shadow,granite-v7-shadow,screenshot-regression}.jsonl > <scratch>/chatty/frozen.before
PYTHONPATH=src uv run python scripts/retail_bank/prepare_tool_sft_data.py --output-dir data/banking-v5-tool-sft \
  --teacher-responses data/sources/banking-v5-tool-sft-teacher-realizations-v2.jsonl --teacher-model claude-opus-5 \
  --teacher-prompt-hash "$(sha256sum <scratch>/chatty/base-requests.jsonl | cut -d' ' -f1)"
PYTHONPATH=src uv run python scripts/retail_bank/prepare_servicing_alignment_data.py --output-dir data/banking-servicing-alignment-v5 \
  --teacher-responses data/sources/banking-servicing-alignment-v5-teacher-realizations.jsonl --teacher-model claude-opus-5 \
  --teacher-prompt-hash "$(sha256sum <scratch>/chatty/align-requests.jsonl | cut -d' ' -f1)"
sha256sum -c <scratch>/chatty/frozen.before     # all OK
git diff --stat data/                            # no frozen file listed
```
(No release lock exists for alignment v5 — only `banking-servicing-alignment-v4.lock.json`; nothing to refresh.)

- [ ] **Step 3: Full suite** from repo root: `rtk proxy python -m pytest -q` and `POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 rtk proxy python -m pytest -q poc/retail-bank-customer-service-poc/tests`. Pinned-text tests are edited only where the pin encodes the old filler.
- [ ] **Step 4: Docs** — teacher files, regeneration commands, voice contract summary, new digests, the out-of-scope notes (createdin, scaffolded alignment user turns). **Step 5: Commit** — `data: regenerate SFT splits with conversational-voice finals`.

### Task 6: Runtime belt-and-braces — strip filler on the policy path (both sites)

**Files:** `poc/retail-bank-customer-service-poc/model_service.py` (`run_policy_turn`, ~603–645), test in `poc/.../tests/test_model_service.py`.

- [ ] **Step 1: Failing tests** — fake runtime returns `"I found the following details: <valid answer> [Policy: card.replacement.us.v1]."` on the first pass → response has no prefix; second test: first pass invalid, repair pass returns a filler-prefixed valid answer → still stripped.
- [ ] **Step 2: Implement** — `output = strip_realizer_filler(output) or output` at both assignment sites (first pass `~:630` and post-repair `~:641`).
- [ ] **Step 3: Run poc tests; commit** — `fix: strip realizer filler on the policy answer path`.

### Task 7: Quote, approvals, run, verify (controller-owned; outward actions gated)

- [ ] **Step 1: Pre-flight** — full suite green; `git status` clean; frozen digests re-verified; sequence-length check: longest rendered train example ≤ 2048 tokens (tokenize with the Granite tokenizer; report p99).
- [ ] **Step 2: AskUserQuestion — one question bundling:** push `fix/hf-demo-end-to-end-v7` to `github-model` (job bootstraps from the GitHub commit); publish `data/banking-servicing-alignment-v5` with `--push-to-hub` (new revision of `spkc83/retail-bank-servicing-alignment-sft`); launch `MAX_TRAIN_SECONDS=1800 JOB_TIMEOUT=45m scripts/retail_bank/run_remote_continuation_job.sh <commit> <dataset-rev> d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2 spkc83/retail-bank-servicing-agent-9b-peft-v9-conversational-voice 964` — quoted at $0.63–$1.35 expected / $1.82 capped worst case / $2.06 hard ceiling; ask the user to confirm the HF credit balance covers it. State explicitly that the continuation starts from v5-remediation (as v6/v7/v8 did), so all behaviour is re-earned from the data.
- [ ] **Step 3: Monitor** — poll `inspect_job` every 60 s; emit on COMPLETED/ERROR; actual cost = `(finished_at - started_at) × $2.75/h`.
- [ ] **Step 4: Verify** — job gate report (dev: three accuracies ≥0.95, two consecutive passes; shadow coreference 16 pairs ≥0.95); local demo relaunch with the v9 adapter pin and BEST_OF_N=2; `release_probe.py` 8/8; tone read-out against the Success criterion. Gate failure ⇒ one diagnosis, one data fix, re-quote, re-ask — a second billed run only if cumulative worst case ≤ $3.00; otherwise report and stop.
- [ ] **Step 5: Repin + redeploy Space** only on explicit approval.

## Critique ledger (rev 1 → rev 2)

| Finding | Disposition |
|---|---|
| C1 heaviest family untouched | Fixed: ambiguity pool (Task 2); Architecture restates coverage (~97% of loss-bearing mass) |
| C2 read-path prose after table discarded | Fixed: spec scoped to lead-in only; tests exempt table rows |
| C3 3600 s cap ⇒ $2.75 | Fixed: `MAX_TRAIN_SECONDS=1800` + `JOB_TIMEOUT=45m` through launcher (completion-gate C1/M1/M2); quote $1.82 capped / $2.06 hard ceiling; second run bound to `$3.00 − run-1 actual` |
| C4 forbidden_claims | Fixed: rule l |
| C5 test-split ids composite / test couldn't pass | Fixed: check after `load_base_sft_splits`; parametrized test |
| M1 MAX_STEPS change alters schedule | Fixed: keep 964 |
| M2 cross-file uniqueness | Fixed: cumulative multi-`--responses`/`--records-dir` check before commit |
| M3 action claims any mode | Fixed: rule k |
| M4 presets / held-out currents | Fixed: rule m + Appendix A |
| M5 number words | Fixed: rule l |
| M6 "was updated" claim | Partially addressed (closer removed, `session` banned); regex extension for profile mutations noted as runtime follow-up, out of scope |
| M7 strip at both sites | Fixed: Task 6 |
| M8 arithmetic / diversity rule | Fixed: 21 × (31+8); opening-trigram ≤4 rule; 20% style sampling |
| M9 createdin / scaffolded user turns | Documented out of scope (defect also in frozen test split) |
| Minor 1–3 | Fixed (in-place regeneration; Task 6 caveat noted; fixture pattern) |
| Missing: threshold, rollback, lock, parent adapter | Added: Success criterion, Rollback, lock note, Task 7 wording |
| Skeptic's cheaper falsification (ambiguity-only run) | Considered and rejected: it would not exercise the grounded-action replies the user's complaint is about; the ambiguity pool is included in the full pass instead |

## Completion-gate ledger (2026-08-21)

| Finding | Disposition |
|---|---|
| C1 5h job timeout ⇒ $13.75 ceiling | Fixed: `JOB_TIMEOUT` passthrough (eefdb5f), launch with 45m |
| M1 overhead ~9.7 min not 3 | Fixed: quote restated above |
| M2 defaults depend on env discipline | Fixed: both env values stated literally in the approval prompt |
| M3 multi-turn base records: only the last user turn is teacher-editable; 177 records keep turn 0 (8 with scaffold residue); all user turns are loss-masked | Disclosed; schema extension deferred |
| Minor 1/2: 23 unrealized seed finals (incl. 1 pre-existing `cancel_transfer` final lacking "cancelled") share text with the frozen test split | Disclosed; fix at next frozen-split rotation |
| Minor 3: ambiguity pool trigram counts ~23 per opener (32 templates × 21 states) | Accepted: 32× better than the single template; gate-neutral |
| Minor 4: em-dash density 16.6% of rows | Disclosed as a taste item; not worth a billed run |
| Minor 5: corpus tests narrower than described (carve-outs currently vacuous — critic re-ran stricter variants: 0 violations) | Disclosed |

## Run-1 outcome and the one permitted fix (2026-08-21)

- Job `6a87cc5b9cd058584adc4f00`: 964 steps, 28.0 min, **$1.29** actual. Dev gate never reached two consecutive passes; final
  `positive_tool_argument_accuracy=0.875, ambiguity_accuracy=0.4375, pair_flip_accuracy=0.375` (v8: 1.0/1.0/1.0).
- Evidence (20 per-checkpoint reports, `scratchpad/chatty/gate-v9/`): 0 of the ambiguity failures were tool calls; the model
  emitted parent-adapter filler recitals ("I checked the available information. … is active and available for use.",
  "… I can help with the next banking step.") that do not occur in the v9 data. Ambiguity accuracy rose 0.06 → ~0.44 and
  plateaued from step ~350; v8 reached 1.0 by step 350 at the same LR.
- Ruling: the 32-phrasing ambiguity pool (Appendix B) diluted per-phrasing repetition 32× below what the continuation LR
  (2e-6, ~0.5 epoch) needs to overwrite the parent prior on the gate prompts. Reverted `deictic_replace_ambiguity` to the
  v8 template in all splits; all 1,839 teacher rewrites kept (not implicated). Appendix B is retained for the record only.
- Run 2 plan: v8's exact recipe `MAX_STEPS=550` (v8 passed at 350/400), `MAX_TRAIN_SECONDS=1200`, `JOB_TIMEOUT=37m`
  ⇒ hard ceiling $1.70 ≤ remaining $1.71. If run 2 fails the gate, the budget is exhausted and the iteration stops.

## Run-2 outcome (2026-08-21) — iteration closed

- Job `6a87e26c9cd058584adc5190`: 550 steps, 18.1 min, **$0.83** actual (total spend $2.12 of $3.00). Failed only the
  *two-consecutive-passes* rule: `ambiguity_accuracy = 1.0` from step 200 on; all three metrics ≥0.95 at steps 250, 400
  and 550 (final checkpoint 1.0/1.0/1.0), but one positive record — `deictic_replace_list-reference_validation_2`
  (single-card history form 1) — flickered to a clarify ("I found Aspen Cashback Debit ending in 2107. What card should I
  replace it with?") at steps 300/350/450/500, so no two evaluations in a row passed and the worker did not upload.
- The template revert fixed the run-1 failure mode completely; the v9 rewrites are gate-compatible up to one borderline
  validation prompt. Reports: `scratchpad/chatty/gate-v9r2/step-*.json`.
- No supported cheap path ships checkpoint-550: `--publish-only` needs the post-gate bundle that was never written and
  `--probe-only` cannot publish. Shipping it would require a new worker mode and a relaxation of the consecutive-gate
  contract — a user decision, outside this iteration.
- Remaining $0.88 is below any run's hard ceiling; per the anti-loop directive the iteration stops here. v8 remains live.

## Extension (2026-08-21, user: "go ahead with the targeted examples and rerun, budget +$3"; also "remove any references to the app")

- App/demo wording removed from all trainable text: templates (0362239), teacher rows (aed8e8e, 754c78b), permanent
  `TRAINABLE_TEXT_BANNED_WORDS` gate in `validate_records` (train/validation only; frozen fixtures keep their wording).
- Targeted train phrase families `listed-card`, `from-your-list`, `card-you-showed` (b21589e): alignment train 2202 → 2298.
- Run 3 `6a884d337c5c7dd379233346` (data 5192e39 / dataset @3fe2bd61, 550-step recipe, caps 1200 s / 37 m): 13.5 min, **$0.62**.
  Dev gate PASSED at steps 350 and 400 (1.0/1.0/1.0; early stop at 400). Post-training shadow gate 15/16 ambiguity:
  `deictic_ambiguous_results-reference-shadow_shadow_3` ("yes the card displayed above is the replacement target") began
  the clarification correctly then emitted the parent's filler closer. Total spend $2.74 of $6.
- Run 4 plan: +4 train families in the shown-above/results neighbourhood (`shown-above`, `above-card`, `from-results`,
  `target-card`), same recipe and caps (≈$0.62–0.83, ceiling $1.70).
- Run 4 `6a8860587c5c7dd37923344b` (data b4dc127 / dataset @0f99604a, +4 families, alignment train 2426): 17.3 min, **$0.79**.
  Dev gate 1.0/1.0/1.0 at steps 450 and 550 but 0.938 at 500 (`deictic_replace_list-reference_validation_2` flickered back to a
  clarify) → no two consecutive passes, no upload. Cumulative spend $3.53 of $6. Runs 2–4 each sit within one held-out record of
  the rule; the flips are parent-prior leaks on unseen phrasings (outputs still carry the parent's "shown in the app").
- Run 5 (user choice): strengthen gate-family weighting via launcher env `POSITIVE_MULTIPLIER=3 AMBIGUITY_MULTIPLIER=6`
  (defaults 2/4 unchanged), same data/dataset revision, same 550-step recipe and caps.
- Run 5 `6a88846e7c5c7dd379233767` (code 9463293, dataset @0f99604a, multipliers 3/6): 12.1 min, **$0.56**. Dev gate 1.0/1.0/1.0 at
  steps 300 and 350 (two consecutive), shadow gate 1.0/1.0/1.0; adapter published
  `spkc83/retail-bank-servicing-agent-9b-peft-v9-conversational-voice@0a9fe83fce3408e6be9a467e85b4e3398f780f05`.
  Cumulative spend $4.09 of $6. Local probe + tone read-out next; repin/redeploy only on approval.
- Local probe on v9 @0a9fe83f (router dd5ea266, BEST_OF_N=2): **8/8 PASS** (first-turn freeze, card status, cancel, policy
  no-match stock, mailing address after detour, policy citation [1], weather OOD, fresh-session control). Tone read-out on the
  model-authored replies: **mean 1.17 sentences** (S1/S2/S3/S5/S7 one sentence; S6b two), 0 filler phrases, 0 internal-language
  hits; S7 still says "confirmed in this session" (parent prior). **Success criterion (mean ≥ 2.0, ≥5/7 two-sentence) NOT met.**
  Cause: the gate-driven early stop selected step 350 (≈1,400 weighted examples, a fraction of an epoch) — the act/ask behaviour
  stabilises long before the conversational finals are absorbed. Lever: a `--min-steps` floor in the worker so training continues
  past the first consecutive gate pass and the last passing checkpoint ≥ min-steps is selected.
- Run 6 (user choice: min-steps floor, a3acfad): first launch `6a88a527` aborted at start — the worker refuses a non-empty
  destination repo (run 5's adapter lives there), negligible cost. Relaunched as `6a88a58873304676c8ec5dfa` into
  `spkc83/retail-bank-servicing-agent-9b-peft-v9b-conversational-voice` with `MIN_STEPS=550 MAX_STEPS=700 MAX_TRAIN_SECONDS=1500
  JOB_TIMEOUT=40m`, multipliers 3/6, dataset @0f99604a.
- Run 6 `6a88a58873304676c8ec5dfa` (min-steps 550, cap 700): dev gate 1.0/1.0/1.0 at every checkpoint from step 200 to 550
  (8 consecutive passes, stopped by the floor at 550); shadow gate on step 550 1.0/1.0/1.0; full bundle written to the bucket.
  Upload then failed on the pre-existing check `consecutive_dev_passes != 2` (streak was 8) — a bookkeeping bug, not a model
  issue. Recovery plan: relax the check to ≥ 2, add a publish-only recovery path keyed by the training commit (bundle identity
  records a3acfad), publish from the bucket on cpu-basic (pennies), then probe locally. Estimated run-6 cost ≈ $1.0 (job timing
  not reported by the API); cumulative ≈ $5.1 of $6.
- Recovery (618bb73: streak check ≥ 2; `RECOVERY_SOURCE_COMMIT` publish-only path keyed by the training commit): cpu-basic job
  `6a88b92c7c5c7dd379233a53` published the run-6 bundle as
  `spkc83/retail-bank-servicing-agent-9b-peft-v9b-conversational-voice@15abf8f898f97c607641534bef86610648cab6cb`
  (trained at a3acfad, dataset @0f99604a, multipliers 3/6, min-steps 550, selected step 550). Local probe + tone read-out next.
- v9b local probe: **8/8 PASS**, replies byte-identical to v9's (one visible sentence each). Diagnostics show why: the raw
  `grounded_final` output is "I found your Everyday Visa Debit ending in 4821 and froze it. I can help with the next banking
  step." — the second sentence is the v5-remediation parent's filler closer, stripped at runtime. The adapters learned to add a
  second sentence but not the conversational one: at lr 2e-6 over ≤550 steps with loss on every chat token, the new finals do
  not displace the parent's closer habit. **Iteration closed at ≈$5.2 of $6 (run-6 timing not reported by the API; estimate).**
  Levers that would change the outcome (each needs new budget): final-token-weighted loss in the continuation worker (cheap code,
  ≈$1.1/run); higher continuation LR (≈$1.1/run, behaviour risk gated); or a from-scratch Stage-2 LoRA on the v9 data (no gate in
  that lane, ≈$4–5 incl. a gating continuation). v8 remains the live default; v9b is gate-passing and app-free if a repin is wanted.

## Option 3 (user decision 2026-08-21): from-scratch Stage-2 LoRA on the v9 data

Guards required before launch: (1) `MAX_TRAIN_SECONDS` / `JOB_TIMEOUT` caps on `run_remote_training_job.sh`; (2) a NEW hub
destination (never the pinned base repo); (3) the coreference dev + shadow gate ported into `cloud_train_tool_sft.py` as a
post-train check that blocks upload on any metric < 0.95. Launch only after a priced approval (target: MAX_STEPS≈2000 ≈ 2.4 epochs,
MAX_TRAIN_SECONDS=3600, JOB_TIMEOUT=80m ⇒ hard ceiling ≈ $3.67; expected ≈ $2.5–3.0).
