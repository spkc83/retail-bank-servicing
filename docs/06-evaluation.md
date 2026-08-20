# Evaluation and Release Gates

The project has three evaluation layers:

1. deterministic data and schema validation;
2. frozen router evaluation;
3. frozen Granite generation plus end-to-end orchestration tests.

The hierarchical router is published at
`dd5ea26674a0f9808d42110a9ee51a9af6762a76` from data revision
`b33c27170e27cdb11783704ede14f7d25f70625e` and passed the router gates reported
in [05-dual-head-router.md](05-dual-head-router.md#held-out-results). Granite
PEFT evaluation job `6a7f89edc97db76cbdf31893` ran from source
`42c89ae6d6b6792268b36e2162c4b19688e4e617` and failed strict gates. Five
credential-request findings were evaluator false positives caused by the safe
phrase “do not share a password.” Two genuine failures remain: a tool-error
final response claims success, and a history-resolved replacement request asks
for the information again.

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
[05-dual-head-router.md](05-dual-head-router.md#held-out-results) for exact
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

The active evaluation composition is:

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
