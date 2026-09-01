# Improvement Register (audit of 2026-08-29)

Six independent read-only reviews — serving harness, data pipeline, training/eval lane,
sft-dataforge, cross-cutting security/provenance/docs, and the bare-probe gate — plus the
lead's own verification. Findings marked **verified** were reproduced by executing the code,
not inferred from reading it. Line references are as of commit `2a43086`.

---

## The one finding behind most of the others

**Every gate in this project is precise about detecting violations and loose about verifying
requirements, and no test asserts that a detector can fire.**

The same shape recurs in all six lanes:

| Lane | The gate that cannot fail |
| --- | --- |
| Data | 13 leakage/PII assertions are `== 0` against clean fixtures; replace every detector body with `return 0` and all 13 still pass |
| Data | `_assert_no_cross_split_leakage` keys on a value that embeds the split, so two splits can never collide — behind it, 32 of 35 alignment test rows share a 4-gram with train |
| Data | Router PII is report-only; an injected SSN, email and PAN yields `pii_matches = 9` and the build succeeds |
| Data | `banking_tool_sft_data.py` has no PII gate at all — a final containing a full PAN and customer ID passes |
| Bare-probe gate | Calibrated against one seed out of thirteen; the only seed that passed cleanly was that one (fixed, `cbda8ac`) |
| Dataforge | A split-name **typo** silently disarms any guard's split filter; the same module raises rigorously on a *field*-name typo |
| Dataforge | `manifest_extra` is spread last and can overwrite the governance `report` — the artifact the framework exists to produce |
| Harness | One prior `role=="tool"` message disables the zero-tool claim guard for the rest of the session |
| Harness | `joint_decision_accepted` defaults to `True`, so every route that omits it is treated as accepted |
| Train lane | `COREFERENCE_GATE_MINIMUM` is printed into the plan but never passed to the enforcer |
| Cross-cutting | The upstream lock check is off by default and its lock file does not exist; no test recomputes any committed digest |

What these gates certify is that nothing bad was *found*, which is far weaker than the claim
the release process treats them as making.

**Two changes fix the class rather than the instances:**

1. **Every detector gets an injection test.** One test per gate that plants the violation and
   asserts the build fails, naming the key. Effort M, and it would have caught at least six of
   the criticals above before they shipped.
2. **Absent means most-restrictive, everywhere.** A missing route action, an unknown split
   name, an unrecognised field, a report without the expected contract — each currently
   resolves to the permissive branch. Invert the defaults and raise on unknown names.

---

## P0 — correctness and safety, fix before extraction or further training

| # | Finding | Where | Effort |
| --- | --- | --- | --- |
| 1 | **From-scratch training is not reproducible, and the seed knob records determinism it does not provide.** `seed_training()` is never called by `run_remote_training()`; `SFTConfig` sets neither `seed=` nor `data_seed=`. `TRAINING_SEED` reaches only the mix path, which short-circuits when all multipliers are 1 (the default). The value still lands in the fingerprint. **This is the most plausible explanation for the v11→v12 behaviour churn** — not dosing. | `cloud_train_tool_sft.py:801`, `:855-882`, `:473-483`, `:603` | S |
| 2 | **Router-unavailable degrades to maximum tool authority** — all 9 tools including 4 mutations, no turn guidance, no entity grounding. Verified: `router=None` exposes `freeze_card/replace_card/dispute_transaction/cancel_transfer`. A genuine `uncertain` route is safe; only the degraded path fails open. | `app.py:601-620`, `local_app_service.py:529-546`, `model_service.py:1303-1304` | S |
| 3 | **One prior tool message disables the zero-tool claim guard for the whole session.** Verified: with a `role=="tool"` message in history, `"I've frozen your card ending 6101…"` passes; without it, correctly rejected. Fires on exactly the dangerous turn. A test pins the permissive behaviour. | `response_policy.py:499-504`, `tests/test_response_policy.py:841` | M |
| 4 | **No PII gate in the largest generator**; **router PII is report-only**; **cross-split leakage gate structurally cannot fire** (32/35 test rows share a 4-gram with train). | `banking_tool_sft_data.py:803`, `banking_conversation_router_data.py:244`, `banking_servicing_alignment_data.py:4738-4747` | S + M |
| 5 | **No test asserts any detector fires** (13 assertions, all `== 0` on clean fixtures). | `tests/test_banking_*_data.py` | M |
| 6 | **No test recomputes any committed dataset or lock digest**, and the upstream lock check is off by default with a missing lock file. One v4 lock records three digests matching nothing in the repo. | `prepare_servicing_alignment_data.py:18`, `data/sources/*.lock.json` | S |
| 7 | **Two "semantics unchanged" hashes validate a value against itself** — the tool-SFT hash excludes exactly the two messages it is recomputed after overwriting; the ASR digest compares a deepcopy against its source with the mutated field masked. | `banking_tool_sft_data.py:589-590`, `banking_asr_sft_data.py:229` | S / M |
| 8 | **Three fail-open paths in dataforge**: split-name typo disarms guards; `record_id_field` means two different things across modules and reports a clean batch; `manifest_extra` can overwrite the governance report. | `guards.py:454,698,327,761`, `teacher.py:176` vs `checks.py:231`, `emit.py:138-145` | S each |

## P1 — reproducibility, provenance, and cost safety

| # | Finding | Where | Effort |
| --- | --- | --- | --- |
| 9 | **The from-scratch fingerprint omits every hyperparameter that changes the model** — lr, steps, batch, grad-accum, seq-len, warmup, multipliers, source commit. Two materially different runs fingerprint identically, and resume will load a mismatched checkpoint. The continuation fingerprint already carries all of it. | `cloud_train_tool_sft.py:596-611` | S |
| 10 | **Publication is five non-atomic commits**; a partial failure leaves weights with no metadata, indistinguishable from a complete release, and the retry then dies on the non-empty-destination check. Both other lanes already use a single `create_commit`. | `cloud_train_tool_sft.py:1402-1433` | M |
| 11 | **The billed-job guard cannot gate spend**: the launcher submits after format validation only, the real guard runs inside the already-billed container, and `JOB_TIMEOUT` defaults to 5h on a $2.75/h flavour with no ceiling (`999h` validates). | `run_remote_training_job.sh:41,70-73,155` | S |
| 12 | **Re-running the canonical release pipeline deploys the wrong model** — deploy inherits the v8 defaults instead of the deployed v11 pins, and the pipeline cannot execute either paid stage (wrong dataset repo; destination already populated). | `run_release_pipeline.py:166-174,195,242-265` | S–M |
| 13 | **Gate wiring is asymmetric**: the bare-probe gate is train-lane only, so the continuation lane can publish without it; the coreference gates live in the continuation module and are reached through a circular-import shim. | `cloud_train_tool_sft.py:443-456,1333`, `cloud_continue_tool_sft.py:1409-1416` | M |
| 14 | **Nothing is enforced by machine** — no CI, no pre-commit, no hooks, against a public repo where `git add -A` would sweep in 12 untracked data directories. | repo root | S |

## P2 — reusability (the Phase-1/Phase-2 blockers)

| # | Finding | Where | Effort |
| --- | --- | --- | --- |
| 15 | **The real two-domains blocker is `MODEL_TOOLS`, not the singletons.** `_validate_tool_calls` computes `supported_names` from the module global and ignores its own `allowed_tools` argument, so a second domain's manifest is rejected by its own validator. Make this, not `state.BANK`, the acceptance test for Phase-1 step 0. | `model_service.py:995,991,1500,1504` | M |
| 16 | `_generation_plan` raises `IndexError` on any intent→tool map that is not the banking one — and `IndexError` escapes the `ValueError` handler. Latent today, guaranteed on the first new domain. | `model_service.py:1350-1353` | S |
| 17 | The `ToolSyntaxAdapter` seam is bypassed by the hot path: Granite tool syntax is hardcoded in prefill, close, echo-detection and the repair check outside the protocol. | `model_service.py:167,1437-1492,611` | M |
| 18 | Import-time model loading: `zero_gpu_runtime`'s module body loads a 9B model onto CUDA at import. | `zero_gpu_runtime.py:79-113` | M |
| 19 | Domain rules are fused into scoring mechanics and the eval generator imports the trainer — there is no seam to cut for a reusable evaluator. | `banking_tool_eval.py:575-622`, `cloud_generate_tool_eval.py:43` | M |
| 20 | Dataforge: five return shapes and three finding shapes across the guards layer; a `Finding` dataclass already exists and is unused. Unify **before** 1.0. | `guards.py:63,111,181,321,809` | M, breaking |
| 21 | Dataforge packaging is not publishable: no LICENSE, no `py.typed` (a mypy-clean library shipping invisible types), deprecated license form, no version/CHANGELOG/CI/CLI. | `pyproject.toml:11` | S–M |
| 22 | Neither dataforge normalizer applies Unicode NFC/NFKC — load-bearing for the planned ASR ingest, where mixed normalization forms are the norm. One `unicodedata.normalize` prefix repairs every downstream check. | `rows.py:53,64` | S |

## P3 — quality, cost, and hygiene

- The oracle eval contract writes the expected answer into the prompt (`{"const": <expected>}`) for two metrics gated at exactly 1.0, and oracle rows are not separated from non-oracle in the release gate. `cloud_generate_tool_eval.py:921-940` · M
- Two release-gate metrics are inert: `no_tool_faq_quality` is computed and reported but in no gate list; `executable_tool_success` hard-assigns `state_matches = True`. `banking_tool_eval.py:38-54,233-247` · S
- Both final-dedup gates measure prefix variety, not answer variety: 1101 of 1200 records share an answer core (302 distinct), and the fuzzy gate is unreachable below 199 chars while median length is 147. `banking_tool_sft_data.py:863-866,1173-1199` · M
- Banned-wording scan skips `tool` and `system` messages, which are in the training prompt: 190 trainable records carry banned terms inside tool-result content. `banking_tool_sft_data.py:868-870` · S
- `ACCOUNT_STATE_CLAIMS` false-closes on correctly-cited policy answers and can hard-fail the turn (`"There are no limits on internal transfers."` rejected). `response_policy.py:386-394` · M
- The pinned clarify template that the code says must never drift is stored verbatim in two places, with a test covering only one. `banking_servicing_alignment_data.py:1373,1562` · M
- The SequenceMatcher hoist is worth doing: measured 2.59s → 0.05s on 400 rows. `guards.py:391` · S
- The Space publishes its own demo credentials on the login screen. This is **deliberate** (a public demo inviting visitors to sign in as a fictional profile), but they are stored as a Space *secret* and validated for strength as if confidential. Pick one model. `app.py:73-81` · S
- ASR consent/PII-review attestations are caller-set booleans re-emitted as verified facts under `validation`. Must be resolved before any real-transcript use. `banking_asr_sft_data.py:221-228,279-289` · M
- `tests/test_repository_documentation.py` does existence/link checks only — zero value checks, so it caught none of the doc errors fixed at `2a43086`. · M

## Fixed after the audit

- **P0 #1 — from-scratch training now seeds** (`abad022`). `seed_training(TRAINING_SEED)` runs
  before the configs are built, and `SFTConfig` carries `seed=`/`data_seed=`. Two tests assert
  the wiring rather than the constant and were checked by mutation; a second, source-level test
  pins the same wiring where TRL's absence cannot make it skip.
- **P0 #4 and #5 (partial) — three PII gates can now fire** (`428bcb6`). The tool-SFT generator
  raises on any message carrying an email, SSN or long digit run (every message, not just the
  trainable pair); the router raises instead of reporting; and the shared card-number pattern's
  upper bound is gone in all five copies — `{12,19}\b` could not match a 22-digit run, so padding
  a card number evaded the gate. `tests/test_detector_injection.py` plants each violation and
  asserts rejection. The digit-run hole surfaced *because* the test was written first and failed.

- **P1 #11 — the launcher now prices a run before submitting it** (`4c526fe`). Worst-case cost is
  computed from flavour rate × timeout, printed, refused above `MAX_JOB_COST_USD` (default $5),
  and refused without `CONFIRM_SPEND=1`. The 5h default is worth $13.75 and no longer launches
  silently. `DRY_RUN=1` runs every check and stops — added because testing the gate reached a
  real submission that had to be cancelled ($0.18).
- **P1 #9 — the fingerprint now records what changes the weights** (`07b7df2`). Learning rate,
  steps, batch, accumulation, sequence length, warmup and the four mix multipliers. The two runs
  that used to fingerprint identically no longer do. This deliberately invalidates resume against
  checkpoints written under the narrower fingerprint.
- **P1 #10 — publication is one commit** (`b68515f`). A partial publish previously left weights
  with no provenance, indistinguishable from a finished release, and the retry died on the
  non-empty destination check.

### Eval contamination: diagnosed, fix written, blocked on teacher re-realization

Measured on the current corpus, with instruction boilerplate discounted: **33 of 35 test rows and
238 of 268 validation rows share task-content 4-grams with train.** The cause is structural rather
than accidental — the builders that feed the eval splits hardcode one question per record and
distinguish the splits with `_suffix()` alone, so `"When did the mailing-address case get
created{suffix}?"` becomes a train row, a validation row and a test row that differ only in a
trailing phrase. "Held out" means held out by five trailing words.

A fix was written and measured: per-split question stems for the eight records in the four
test-feeding builders, keeping the test wording byte-identical so the frozen fixtures do not move.
It takes test contamination from 33/35 to 9/35, and the nine residuals are all the realizer's
shared *instruction* suffix ("Tell me what you find and what happens next") rather than task
content. Frozen `test.jsonl`, `coreference-shadow.jsonl`, `granite-v7-shadow.jsonl` and
`screenshot-regression.jsonl` all verified byte-identical under it.

**It is blocked, and deliberately not applied.** Teacher realizations pin a hash of everything
except the final, so changing a train prompt invalidates them: `312 of 819` realization rows (38%)
belong to the eight affected records, and regeneration fails closed with *"alignment teacher rows
may edit final_response only"*. That guard is right. The options, none of which should be taken
silently:

1. **Re-run the teacher** on the 312 changed prompts and re-import. Correct, and needs a teacher
   pass plus the cost that implies.
2. **Drop the realizations** for those rows, reverting their finals to the authored templates.
   Free, but reintroduces template-shaped finals on ~8% of train — the exact regression the v9
   conversational-voice work existed to remove.
3. **Re-pin the hashes** while keeping the finals. Cheapest and **wrong**: the file would then
   assert the teacher saw prompts it never saw, which is the precise thing the hash exists to
   prevent.

The prepared patch is kept at `agent-tmp/scratch-v9/v12/READY-stem-fix.py` so whichever option is
chosen, the wording work does not have to be redone.

**Update (`cd6c787`): option 1 is now unblocked.** The prompt-realization layer exists — the
teacher may rewrite `user_content` as well as `final_response` under
`allow_prompt_realization=True`, with the structural hash still pinning tool calls, tool results,
grounding facts and split keys, and with rewritten prompts held to a 4-gram isolation invariant
against every held-out split. Editing the builders' stems is therefore no longer the right shape
for this fix: the builders stay deterministic and the wording variation arrives as realizations,
which keeps provenance honest instead of re-pinning it.

**Update (`ae4d894`): first pass applied.** Prompts and finals now travel in separate files with
separate teacher attribution, and 320 rows across eight record families have been rewritten.
Measured on alignment rows with instruction boilerplate discounted:

| | before | after |
| --- | --- | --- |
| test rows sharing task-content 4-grams with train | 38 / 40 | **14 / 40** |
| validation rows sharing with train | 112 / 128 | **70 / 128** |

Frozen fixtures byte-identical, row counts unchanged. The residue is the families not yet
rewritten — `clarify_`, `repair_`, `history_entity`, `tool_outcome`, `deictic_` — each of which
can be taken through the same pass. The isolation invariant is rarity-scoped: a gram identifies a
held-out row only when it belongs to a single eval family, so shared style directives do not count
as contamination.

**The corpus on disk is now unpublished and newer than `@8494c94f`.** It must be published before
anything trains on it.

Still open from #4/#5: the cross-split leakage gate (it keys on a value embedding the split, and
behind it 32 of 35 alignment test rows share a 4-gram with train — fixing the gate requires
reworking the per-split templates, which is a corpus change, not a gate change), and injection
coverage for the remaining leakage detectors.

## Already fixed during this audit

- Bare-probe gate rewritten (`cbda8ac`): failed open on all 10 evasion shapes and rejected 110 of
  280 correct trained finals. Now 0/455 false failures, 10/10 evasions caught, calibration test
  replays all 13 seeds.
- v13's regression verdict retracted (`87ec4c9`) — very likely a false positive from the broken gate.
- Three doc claims that contradicted their artifacts (`2a43086`): runbook split counts, the
  v11-vs-v12 corpus attribution, and a fossil router manifest digest.

## Recommended order

Findings 1 and 5–7 first: they are cheap, and until they land, no experiment result from this
lane can be trusted — including the behaviour-stability conclusions that motivated the v12 and
v13 runs. Then 2–4 and 8 (fail-open safety paths), then the P1 provenance block, and only then
the reusability work, which is otherwise built on unverified foundations.

**Do not spend on another training run until finding 1 is fixed.** Two of the three runs in this
arc differed only in a seed that had no effect.
