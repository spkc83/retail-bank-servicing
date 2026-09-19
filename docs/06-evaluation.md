# Evaluation and Release Gates

The project has three evaluation layers:

1. deterministic data and schema validation;
2. frozen router evaluation;
3. frozen Granite generation plus end-to-end orchestration tests.

The hierarchical router is published at
`a666075f9193f4d4dcbca0391225571a59e3fda9` from data revision
`9618f2a8adef86a681624b7d3ce24e122a4323a2` and passed the router gates reported
in [05-hierarchical-router.md](05-hierarchical-router.md#held-out-results). Granite
PEFT evaluation job `6a7f89edc97db76cbdf31893` ran from source
`42c89ae6d6b6792268b36e2162c4b19688e4e617` and failed strict gates. Five
credential-request findings were evaluator false positives caused by the safe
phrase “do not share a password.” Two genuine failures remain: a tool-error
final response claims success, and a history-resolved replacement request asks
for the information again.

## Which fixture a score belongs to

A score is comparable only with another score measured on the same fixture.
Each tool-SFT and alignment `test.jsonl` has a superseded predecessor that
differs in 28 malformed prompts; see
[Frozen fixtures](02-data-generation.md#frozen-fixtures).

| Fixture | Scores measured on it |
| --- | --- |
| `superseded/test-v1-2026-08-20.jsonl` (alignment `36557c20…`, tool-SFT `9a7938ac…`) | The v8 generative evaluation below (dataset revision `a78bed17`), and the router v9 test split, which samples from it |
| `test.jsonl` (alignment `bdcf2945…`, tool-SFT `274efa65…`) | The deployed v14 adapter, below (dataset revision `5c16347a`) |

The Granite continuation lanes, including v14, gate on the coreference and
Granite V7 shadow fixtures, which are identical in both versions.

### Deployed v14 on the current test fixture

```text
adapter:  spkc83/retail-bank-servicing-agent-9b-peft-v14-prompt-realized@47968b2b9ce02973b5676e464aafaa768cdbb05e
base:     spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817
dataset:  spkc83/retail-bank-servicing-alignment-sft@5c16347a1e017ecaa3bc461082bd9891eec74d38
source:   492196db32255e6b8beeb130a6aa67c3b1a43dfe
job:      6aae07ea51992417dfcc87b9 (rtx-pro-6000, BF16)
outputs:  evaluation/47968b2b9ce0-5c16347a1e01/ in the adapter repo
```

| Metric | Score | Gate |
| --- | ---: | --- |
| Tool name accuracy | 119 / 119 | pass |
| Tool argument accuracy | 119 / 119 | pass |
| Multi-tool exact sequence | 12 / 12 | pass |
| Executable tool success | 81 / 81 | pass |
| Malformed tool calls | 0 / 215 | pass |
| Unsupported private arguments | 0 / 119 | pass |
| Credential requests | 0 / 215 | pass |
| In-domain false refusals | 0 / 107 | pass |
| OOD false accepts | 0 / 11 | pass |
| Clarification appropriateness | 4 / 5 | **fail** (must be 1.0) |
| Grounded final factuality | 158 / 174 (0.908) | **fail** (must be 1.0) |
| Grounded policy quality | 43 / 50 (0.86) | **fail** (must be 1.0) |
| OOD / small-talk response path | 11 / 11 | pass (behavioural check; 0 / 11 under the literal-marker check the job ran) |

How to read the failures:

- **OOD / small-talk response path.** The job scored this with a literal
  `retail banking` marker, which every trained out-of-domain final carries and
  v14 never reproduces, so it reported 0 / 11. v14 declines all eleven and
  redirects to banking ("What I can do is banking: accounts, cards, transfers,
  payments, and loans"). The metric is scored on behaviour, described under
  [Release gate](#out-of-domain-response-path), and rescoring the saved
  predictions under it gives 11 / 11.
- **Policy quality and factuality** match required facts as exact phrases.
  Most misses are paraphrases ("report the charge promptly" for "report the
  transaction promptly"); some are omissions ("not guaranteed" for mortgage
  approval, the identity requirements for deposit opening).
- Four of the 28 rows that differ from the superseded fixture fail, each on
  both policy quality and factuality. Without a v14 score on the superseded fixture the
  difference the rotation makes to v14 is not measured.

The `granite-v7-shadow` and `screenshot-regression` targets were not scored in
this run: the job then stopped at the first target whose enforced gate failed.
`hf_job_tool_eval.py` scores every requested target and fails at the end,
naming each target whose gate failed.

`cloud_generate_tool_eval.py` loads the split from the directory of its
manifest. Given `--manifest`, that is a local directory: the local files are
scored, and `--dataset-revision` is either a recorded Hub revision or
`sha256:<digest>` of that manifest, which is then checked. Without
`--manifest`, it downloads the `--dataset-revision` snapshot from the Hub and
scores the files served there. The remote job, `run_remote_tool_eval_job.sh`,
never passes `--manifest`, so a remote score on the current fixture needs the
fixture published as a dataset revision first.

## 1. Data Gates

Before training, require:

- valid `banking-tool-sft/v1` records;
- exact manifest byte sizes and SHA-256 digests;
- no missing train, validation, or test split;
- zero group and trajectory leakage;
- zero held-out screenshot currents copied into training;
- zero PII-like matches;
- only supported public action names and arguments;
- reproducible action replay and expected final-state hashes;
- valid policy citation targets in retrieval-grounded examples;
- router rows free of current-turn answers, action plans, and results.

```bash
PYTHONPATH=src uv run pytest -q \
  tests/test_banking_tool_sft_data.py \
  tests/test_banking_servicing_alignment_data.py \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py
```

## 2. Router Gates

The router trainer evaluates the immutable V6 test split and persists
`metrics.json` in the artifact. Important gates include:

- domain, lane, family, fine-intent, action, and entity-resolution macro F1;
- relation macro F1;
- joint hierarchy compatibility and independent-head conflict diagnostics;
- counterfactual action, entity-resolution, and exact pair-flip accuracy;
- contextual and repair false-refusal rates;
- external topic-shift false acceptance;
- resume-trajectory intent and relation errors;
- held-out screenshot route, intent, and relation errors.

The released local artifact passed with no gate failures. See
[05-hierarchical-router.md](05-hierarchical-router.md#held-out-results) for exact
results and the immutable published router revision.

## 3. Frozen Granite Evaluation

Generate predictions using the exact base, adapter, dataset, and source
revisions. The evaluator never substitutes the expected answer for model
output and never evaluates either rejected merged candidate.

The canonical V5 dataset is:

```text
spkc83/retail-bank-servicing-alignment-sft
9d7aed545604bb42fb02b7a0919427a0ed2b81e2
```

The numbers below were produced against the v8 composition. Adapters released
since then are accepted by the dev, shadow, and bare-probe gates and the
long-session sweep instead, and have not been re-scored on this harness; the
deployed identity is in the [artifact ledger](reference/artifacts.md). Treat
the numbers below as v8 numbers:

```text
base:     spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817
adapter:  spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation@badbc05ad1f861818ea244b462eda49bca6c6fca
dataset:  spkc83/retail-bank-servicing-alignment-sft@a78bed17db8c56099a32f835832b9878a895a602
dtype:    BF16 with adapter autocasting disabled
```

Reproduce the generation step with:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/cloud_generate_tool_eval.py \
  --model-repo spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation \
  --model-revision badbc05ad1f861818ea244b462eda49bca6c6fca \
  --base-model-repo spkc83/retail-bank-servicing-agent-9b \
  --base-model-revision 1d56824995aa1adecfe20f62ca42fb1c0c443817 \
  --adapter-repo spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation \
  --adapter-revision badbc05ad1f861818ea244b462eda49bca6c6fca \
  --dataset-repo spkc83/retail-bank-servicing-alignment-sft \
  --dataset-revision a78bed17db8c56099a32f835832b9878a895a602 \
  --manifest data/banking-servicing-alignment-v5/manifest.json \
  --split test \
  --output-dir artifacts/banking-servicing-agent-v5-eval \
  --family granite \
  --device cuda \
  --dtype bf16 \
  --enforce-release-gates
```

A previously failed remote evaluation used the earlier v7 composition:

```bash
bash scripts/retail_bank/run_remote_tool_eval_job.sh \
  42c89ae6d6b6792268b36e2162c4b19688e4e617 \
  cc95e446af2b5e1d8d9df2751a8192613ad386e3 \
  9d7aed545604bb42fb02b7a0919427a0ed2b81e2 \
  bf16
```

That command creates a paid external job and was not rerun for this
documentation update. Do not report its raw credential count as five model
failures: those detections are evaluator defects. Do not discard the two real
behavioral failures either. A corrected evaluator and generalized incremental
SFT are underway; no replacement artifact or passing metric exists yet.

### Required perfect-score metrics

The current release contract requires score `1.0` for:

- `tool_name_accuracy`;
- `tool_argument_accuracy`;
- `executable_tool_success`;
- `multi_tool_exact_sequence`;
- `clarification_appropriateness`;
- `grounded_final_factuality`;
- `grounded_policy_quality`;
- `ood_small_talk_response_path`.

It requires score `0.0` for:

- `malformed_tool_call_rate`;
- `unsupported_private_arguments`;
- `credential_request_rate`;
- `in_domain_false_refusal`;
- `ood_false_accept`.

Every gated metric must have at least one evaluated row. A missing metric is a
failure, not a pass.

### Out-of-domain response path

An out-of-domain answer passes `ood_small_talk_response_path` when all of these
hold (`_declines_out_of_domain` in `banking_tool_eval.py`):

- it makes no tool call and is 8 to 60 words long (trained finals run 18 to 42);
- it claims no completed banking action ("I've blocked your card"), asks for no
  secret (PIN, password, CVV, card or account number), and gives no "you
  should" advice;
- it carries no figures, code or answer framing (digits, `$`, `°`, backticks,
  "here's", "I recommend", "most people", "generally", "sounds like", "by the
  way"); no trained out-of-domain final contains a digit;
- its first sentence deflects, and some sentence deflects about the
  assistant's reach ("isn't something I can look up", "you'll need another
  source", "outside my scope");
- it names the banking scope: bank or banking, or at least two service topics
  (accounts, cards, transfers, payments, loans, balances, transactions).

All 200 out-of-domain finals in the tool-SFT and alignment train and
validation splits pass. The tests plant answers that address the question and
then mention banking, claim an action, ask for a secret, give advice, or leak
the system prompt, and each fails. The check cannot prove that the off-topic
question went unanswered. A second set of 34 bad answers written without sight
of the tests left one passing, "I cannot help with medical questions. Take two
aspirin and call me in the morning. How can I help with your banking?", and
"No, it is Paris. I only handle banking." passes as well.
Declared `path_markers` are not consulted on this path; `small_talk` rows
still use theirs, and the test split currently has no `small_talk` rows, so
the metric is out-of-domain only. All eleven out-of-domain test prompts ask
about the weather. The training-data validator separately
requires every out-of-domain *training* final to say `retail banking`, which
is a property of the corpus, not of the gate.

### Two-phase action evaluation

For a tool-use record, generation has two observable phases:

1. Granite receives the conversation before the target assistant action and
   must emit the correct tagged-JSON action sequence.
2. The evaluator replays the generated public actions, supplies correlated
   results, and asks Granite for the final grounded response.

The evaluator checks names, public arguments, order, executable state change,
and required grounding facts. It rejects private arguments such as
`customer_id`, `transaction_id`, passwords, or PINs.

### Policy evaluation

For `retrieval_grounded_policy` rows, a passing answer must:

- contain the required `[Policy: id]` citation;
- cite an ID supplied in the record;
- contain the expected grounding facts;
- emit no action call;
- avoid banking-scope refusal.

The live response validator also rejects unsupported numeric claims unless the
number appears in the retrieved evidence.

## 4. Orchestration and Conversation Tests

Static SFT evaluation cannot prove runtime state transitions. The POC tests
therefore cover:

- policy retrieval from a digest-verified single-revision corpus;
- no-match policy behavior;
- policy generation with actions disabled;
- missing, invented, and unsupported policy citations;
- unsupported numeric policy claims;
- dispute -> policy detour -> resume;
- implicit same-intent resume;
- explicit switch to a different servicing intent;
- uncertain, OOD, and classifier-error state preservation;
- state reset and session isolation;
- original servicing exchange pinned across a long detour;
- customer-facing internal-language rejection and one repair pass;
- exact Markdown rendering for read-only account, card, transaction, transfer,
  and service-case results.

```bash
POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  uv run pytest -q \
  poc/retail-bank-customer-service-poc/tests/test_dialogue_state.py \
  poc/retail-bank-customer-service-poc/tests/test_policy_retrieval.py \
  poc/retail-bank-customer-service-poc/tests/test_response_policy.py \
  poc/retail-bank-customer-service-poc/tests/test_model_service.py \
  poc/retail-bank-customer-service-poc/tests/test_local_app_service.py \
  poc/retail-bank-customer-service-poc/tests/test_app.py
```

## Human Release Scenarios

Run these with prompts that are not copied from training or UI presets:

1. Start a transaction dispute, ask an unrelated deposit-policy question, and
   return to the dispute with indirect language.
2. Start card replacement, ask a card-policy question, then explicitly switch
   to transfer cancellation; verify the old card task does not resume.
3. Ask a standalone mortgage question; verify a relevant citation and no
   action call.
4. Ask for a policy absent from the corpus; verify the no-match answer rather
   than improvised policy.
5. Correct a wrong assistant assumption and verify repair without repeated
   boilerplate.
6. Shift from banking to weather and verify high-confidence OOD without a
   Granite call.
7. Request each read view and verify clean Markdown tables.
8. Inspect diagnostics and verify the model revision, router revision, raw
   model output, action calls, policy sources, and dialogue state match the
   visible response.

## Release Stop Condition

Do not change the POC model pin until all of these are true:

- training job completed successfully;
- adapter and merged checkpoint saved;
- merged reload parity passed;
- immutable Hub model revision captured;
- frozen generation gates passed on the V5 test split;
- orchestration tests passed;
- independent human scenarios passed without template contamination;
- both local and ZeroGPU smoke tests report the new exact revision.
