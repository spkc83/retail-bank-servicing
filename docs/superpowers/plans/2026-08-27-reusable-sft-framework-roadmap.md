# Reusable SFT Framework Roadmap

**Goal:** evolve hello-SLM + sft-dataforge from a single-project pipeline into a reusable
framework that lets anyone (a) build governed synthetic SFT data — including from production
ASR transcripts with zero PII leakage, (b) fine-tune and gate an SLM on it, and (c) run that
SLM inside an omni-channel banking customer-servicing harness.

**Status:** roadmap (phased), revised after an adversarial review on 2026-08-27 (the review
confirmed the strategy but rejected the first draft on five factual errors and four missing
workstreams; every finding below marked "review:" traces to it). Each later phase gets its
own writing-plans implementation plan before execution.

---

## What exists today (the extraction sources)

| Concern | Where it lives now | Reusability today |
| --- | --- | --- |
| Governed data construction | `sft-dataforge` (guards, curricula, teacher, emit, checks; v9 mechanisms pushed; v11 port in flight, un-reviewed until committed) | Library, generic, tested — the seed of the framework |
| Domain data generation | `src/hello_slm/banking_*_data.py` (~500KB of banking-specific curricula) | Banking-specific; the *patterns* are proven, the code is not generic |
| ASR route | `src/hello_slm/banking_asr_sft_data.py` + `docs/15-asr-to-sft-pipeline.md` + `examples/asr/` | **A live, shipped transcript-overlay route**: externally supplied transcripts are applied verbatim onto synthetic records, guarded today only by operator-attested review booleans (`semantic_match`, `pii_reviewed`, `consent_for_training`) and four US-centric PII regexes. It contains NO disfluency/homophone synthesis (review: first draft had this backwards). Phase 3 is therefore a *retrofit hardening a live route*, not greenfield. |
| Training + gates | `scripts/retail_bank/` (~14k lines, 38 files, six near-duplicate recovery variants). Gate mechanics: `cloud_train_tool_sft.py` runs a single end-of-training gate at `COREFERENCE_GATE_MINIMUM = 0.95`; the 2-consecutive-passes rule (`ConsecutiveGateTracker`, `required_passes=2`) lives in the continuation lane `cloud_continue_tool_sft.py`; the "1.0" figures in run reports are measured outcomes, not thresholds (review: first draft mis-cited both) | Works end to end on HF Jobs; config is 27 env vars + script edits; banking eval hardcoded |
| Evaluation | `cloud_generate_tool_eval.py`, bare_model_arena.py, deictic_suite.py, sweeps | Probe sets are banking-specific; the bare/guided/harness three-tier method is generic |
| Serving harness | `poc/retail-bank-customer-service-poc/` — 15,361 lines total: ~8k source + 7,207 lines of tests the extraction must carry (review: count tests) | Single-channel, single-domain; module-level singletons (`state.BANK`, model/adapter ids, router repo, BEST_OF_N bound at import) block any multi-config use; the safety architecture is the crown jewel |
| Router | `banking_conversation_router*` + DistilBERT heads (`train_conversation_router.py`, ~2.2k lines) | Generic architecture (multi-head intent/lane/entity), banking taxonomy; trained separately from the adapter with no cross-artifact compatibility gate |

**Core principle carried into every phase (proven across v7–v11):** the harness carries
safety, the tuning carries polish; a behaviour reaches the weights through repetition of a
mapping, and every dataset claim is enforced by a build-time gate, not a convention.

**Verified cost model:** ~$1.40 per 30-minute training run on rtx-pro-6000 ($2.75/h,
per-minute billing; five corroborating measured runs). Every phase's billed work is priced
from this before launch.

---

## Phase 0 — dataforge core (done / in flight)

- v9 mechanisms ported and pushed (`a191af9..2aeb72a`): banned-wording gate, duplicate/paired
  counterfactual dedup, teacher provenance hashes, batch checker, conversation rows.
- v11 mechanisms port (in flight): behaviour-curriculum builder (frames × subjects dosing,
  held-back validation subjects), field-invariant gates, probe-exclusion guard,
  evidence-aware unsupported-claim guard, fuzzy final dedup, allowed-use registry tagging.

Exit criteria: committed on master, tests green, README documents the new surface, worked
example exercises it, and a post-commit review pass (review: Phase 0 status must be judged
at commit time, not from a moving working tree).

## Phase 1 — `slm-harness`: extract the serving harness into a library

The POC's safety architecture becomes a reusable package; the banking POC becomes its
reference app. Extraction order follows dependency direction (verified: `model_service.py`
imports only `dialogue_state`, `mock_bank`, `response_policy`):

0. **Invert import-time configuration first** (review: blocker for every later phase).
   `state.py`'s module-level `BANK`, the model/adapter/router ids and `BEST_OF_N` bound at
   import become constructor-injected config objects. Two domains in one process is the
   acceptance test for this step, and nothing else proceeds until it passes.
1. **Core contracts:** `ToolSpec` (name, args schema, read/write class, renderer),
   `RouterDecision`, `GuardVerdict`, `ChannelTurn`.
2. **Guard pipeline** (from `response_policy.py`): evidence-aware zero-tool action-claim
   guard, banned-term/final-response checks, interrogative resume gate, mutation-intent
   constraints — each as a configurable guard object; banking regexes/policies move to
   config. The corpus regression pattern (`test_response_policy.py:668` — a guard must never
   reject its own training corpus) ships as a framework test helper.
3. **Dialogue core** (from `model_service.py` + `dialogue_state.py` + `entity_grounding.py`):
   route → guidance injection → decode → tool execution → validation → retry/fallback loop
   (wrong-tool retry, best-of-N validator selection, honest fallback) — parameterized by
   ToolSpecs + a Router + a ModelBackend.
4. **Model backends:** local NF4, ZeroGPU BF16, later vLLM/OpenAI-compatible endpoints.
5. **Router interface:** the DistilBERT multi-head router behind a `Router` protocol, plus an
   **LLM-routed mode as a first-class deliverable** — it is the only way a second domain runs
   without training a router, so the Phase-1 exit depends on it (review: was a parenthetical).
6. **Adversarial-input guards** (review: gap): today's guards validate model *output*;
   add validation of tool-result content and policy-corpus content against injection before
   it enters the prompt.

Exit criteria: banking POC runs on the library with **behavioural equivalence at
`BEST_OF_N=1`** — identical route/intent/action/tool_calls/response_path/fallback across the
sweep suites, and all 7,207 lines of existing POC tests pass unmodified. (Byte-identity is
unachievable: sweeps embed wall-clock seconds and best-of-N candidates 2+ decode unseeded at
temperature 0.7 — review.) Second exit: a toy second domain (3 tools) runs on config + data
alone, using the LLM-routed mode.

## Phase 2 — `forge-train`: the training and gating lane as a product

1. Config-file-driven runs (model, dataset revision, LoRA/lr/steps, multipliers, budget
   ceiling) replacing the 27-env-var + script-edit flow; the launcher refuses to start
   without a priced estimate from the measured cost model.
2. Gate framework: declarative gate sets — threshold (default the proven 0.95 floor, NOT the
   measured 1.0 — review), N-consecutive-passes upload rule (from the continuation lane),
   shadow suites, publish-from-checkpoint recovery (which today needs a post-gate bundle and
   cannot rescue a gate-failed run — carry that limit forward or fix it).
3. Eval harness: the three-tier method (bare probes / bare+guidance / in-harness) as a
   reusable evaluator; probe sets are data; every probe set auto-registers a probe-exclusion
   gate on the training corpus. **Probe-rot policy** (review: gap): probes are permanently
   load-bearing once excluded from training, so probe sets are versioned, additions create a
   new probe-set version rather than retroactively invalidating old datasets, and each
   dataset lock records the probe-set version it was gated against.
4. Provenance + compatibility chain (review: gap #17): dataset lock → job id → adapter
   revision → deploy pin, machine-checkable — and a **three-way compatibility gate**
   asserting router taxonomy ⊆ harness ToolSpecs ⊇ adapter tool vocabulary at deploy time
   (the likeliest production failure today is an ungated router/adapter/harness skew).
   First step: audit whether `data/sources/*.lock.json` already carries the fields to make
   this cheap.
5. Consolidate the six near-duplicate recovery scripts into the config-driven lane (review:
   this phase is a real refactor of ~14k lines, not "consolidation of proven pieces").

Exit criteria: the next banking retrain runs entirely through forge-train configs; a
from-scratch user fine-tunes a different base model with a config file and one command; the
compatibility gate blocks a deliberately skewed deploy in a test.

## Phase 3 — ASR: harden the live route, then close the loop (voice-robust data)

Reordered (review): legal/PII design decisions gate entry, because the transcript route
already exists and its only current safeguard is operator attestation.

**3-entry (before any code): the data-protection ground rules.** Lawful basis, consent
model, retention, residency/jurisdiction, teacher-LLM vendor exposure (schemas derived from
personal data sent to a third-party teacher is a vendor transfer — review), and the
**erasure story**: per-transcript provenance must let a withdrawn call be traced to the
frames/rows it induced, with rebuild+retrain as the documented (and priced) worst case.
These are design inputs, not Phase-5 documentation.

**3a. Voice-robust synthetic data (build new — review: nothing to "lift").** A disfluency /
filler / homophone / mis-transcription noise generator over synthetic turns (the
"frieze my card" class), as a dataforge transform with its own gates; the existing overlay
module contributes only the row schema (ASR metadata, timestamps, speaker fields).

**3b. Production ASR → similar synthetic data (the PII firewall, retrofitted onto the live
route).** The invariant is *no production span, value, or singleton pattern reaches a
dataset*. Layers, each a hard gate, revised per review:

1. **Closed-vocabulary extraction (the firewall).** The extractor emits ONLY values from a
   closed, enumerated vocabulary per field — intent enums, slot *types*, bucketed discourse
   shape — enforced by a schema validator that rejects any free-text or out-of-vocabulary
   value (review: "no wide string fields" was the wrong invariant — a 16-digit number or a
   rare merchant name fits in a narrow one; an `other` enum with free text is an escape
   hatch, so there is none).
2. **Quasi-identifier coarsening.** Turn counts, interruption counts, and outcomes are
   bucketed before storage; no per-call combination rarer than the k-floor below survives
   (review: discourse shape is a fingerprint).
3. **Minimum-support frame induction (k-anonymity floor).** A frame may only be induced
   from a schema cluster of ≥ k distinct calls (k configurable, default 20); singleton
   shapes are dropped, not generalized (review: the largest structural hole — a
   one-call-derived frame is a membership-inference channel).
4. **Verbatim-span containment gate, rarity-scoped.** A flat reverse n-gram gate is
   measurably unusable — at n=4 it rejects 99.8% of *legitimate* independently generated
   banking rows, because domain language is formulaic (review, measured). Replace with:
   (a) longest-common-substring detection vs the source corpus with a token-length
   threshold, and (b) an n-gram gate restricted to n-grams whose document frequency in the
   source corpus is 1 (unique to a single call) — rare content blocks, formulaic banking
   phrasing does not.
5. **PII detector sweep, upgraded.** Today's four US-centric regexes gain names/addresses/
   DOB/IBAN/non-US formats plus an actual NER pass (review: the "NER" half currently does
   not exist) — defense in depth, never the primary mechanism.
6. **Provenance without content, with controls.** Raw transcripts and the containment-gate
   index never enter the data repo; the gate runs in the operator's controlled environment,
   and the framework documents that the n-gram/LCS index is itself a PII asset with access
   controls (review: overlapping n-grams are stitchable back into sentences).
7. **Replace attestation with mechanism.** The existing `review.*` booleans stay as a
   human sign-off layer but no longer carry the guarantee alone.

Exit criteria: planted-PII/verbatim/singleton fixtures are provably rejected at every layer;
false-rejection rate on legitimate synthetic rows is measured and acceptable (<5%); and a
**memorization probe against the trained adapter** (extraction attempts for planted spans)
comes back clean — the dataset gate alone does not prove the model is clean (review).

## Phase 4 — omni-channel adapters

1. **Channel adapter contract:** ingress (text or ASR stream), egress (markdown tables vs
   short spoken sentences vs SMS-length), turn-taking (barge-in for voice), auth handoff.
2. **Adapters:** web chat (exists; add API/webhook), voice/IVR (ASR → dialogue core → TTS;
   Phase 3a data makes the SLM robust to ASR noise), length-constrained messaging.
3. **Per-channel guidance:** the TURN GUIDANCE mechanism becomes channel-aware (voice: no
   tables, confirm numbers digit-by-digit, shorter finals) — guidance is config.
4. **Channel-aware gates:** rendering validators per channel; sweep suites per adapter.

Depends on Phase 1 step 0 (config injection): two channels with different guidance/model
settings in one process is impossible until singletons are gone (review).

Exit criteria: the same deployed model serves the web POC and a voice loop demo from one
harness process, with channel-specific guidance and validators active.

## Phase 5 — packaging, docs, tutorial

1. Repo split/naming decision: `sft-dataforge` / `forge-train` / `slm-harness` — or a
   monorepo with three packages; decide at Phase 2 exit.
2. PyPI-ready packaging, semantic versioning, CI (tests + lint + example runs).
3. **Licensing workstream** (review: gap): base-model license terms, teacher-model ToS
   (training-a-model restrictions), and the per-transcript license field's actual semantics.
4. The tutorial that proves reusability: "SFT-tune a servicing agent for domain X" —
   taxonomy → seeds → gates → teacher pass → train → gate → deploy → harness — with the
   banking build as the worked reference.
5. Operator-facing PII documentation: what the framework guarantees mechanically, what the
   operator still owns (consent, retention, jurisdiction — designed in Phase 3-entry).

## Sequencing and effort

Order: Phase 1 step 0 unblocks Phases 1 and 4; Phase 3-entry (legal ground rules) gates all
of Phase 3; 3a and Phase 2 can run alongside Phase 1. Honest sizing (review): Phase 1 and
Phase 2 are both real refactors (15.4k and ~14k lines respectively), Phase 3b is the most
novel (the firewall needs its own adversarial review and a measured false-rejection budget),
Phase 4 is the smallest once step 0 exists. Every phase ends with a critic review and a
worked example, per the house protocol.

## Non-goals (for now)

- RLHF/DPO lanes, multi-adapter routing, non-banking taxonomies beyond the toy second domain.
- Ingesting real production transcripts before Phase 3's gates exist and the memorization
  probe passes on fixtures. NOTE (review): the *mechanism* to ingest transcripts is already
  shipped and runnable today; until Phase 3 lands, its safety rests on operator-attested
  review booleans — treat any real-transcript use before then as out of policy.
