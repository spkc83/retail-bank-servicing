# Granite Servicing Alignment v4

This document describes the released continuation-SFT data used for the second
Granite training stage. It aligns the 8.79B agent with the same multi-turn
servicing behavior that the v4 router is expected to route.

- Corrected dataset revision:
  `0ce32f9c7a3edff227005e5b89b089947b87625a`
- Prompt-identical training revision:
  `fea8aa1cda716954eb7322325e2be25c9f570ea3`
- Released model revision:
  `spkc83/retail-bank-servicing-agent-9b` at
  `1d56824995aa1adecfe20f62ca42fb1c0c443817`

## Why the Generative Data Also Changes

The classifier can decide that a follow-up belongs in the banking
conversation, but it cannot make Granite resolve the follow-up correctly. The
generative model still needs examples that teach it to:

- connect pronouns and short replies to earlier visible turns;
- correct a previous assistant mistake without repeating it;
- select a tool after a clarification answer;
- use service-case `created_at`, status, type, and subject fields;
- move from banking to an external topic and back again;
- write a natural final answer from a tool result.

The router and Granite data therefore cover the same use-case boundaries but
have different labels. Router rows contain only information available before
routing. Granite SFT rows contain the later assistant tool call, correlated
tool result, and final answer because those are its training targets.

## Files

| File | Responsibility |
| --- | --- |
| [`banking_servicing_alignment_data.py`](../src/hello_slm/banking_servicing_alignment_data.py) | Builds deterministic alignment records, expands natural realizations, merges the released SFT base, and validates leakage and PII policy. |
| [`prepare_servicing_alignment_data.py`](../scripts/retail_bank/prepare_servicing_alignment_data.py) | Writes the composite splits, verifies the tracked release lock, and optionally publishes after explicit request. |
| [`banking-servicing-alignment-v4.lock.json`](../data/sources/banking-servicing-alignment-v4.lock.json) | Pins the base manifest, synthetic-bank snapshot, split counts, and exact composite split hashes. |
| [`retail-bank-servicing-alignment-sft.md`](../data_cards/retail-bank-servicing-alignment-sft.md) | Released dataset card. |
| [`test_banking_servicing_alignment_data.py`](../tests/test_banking_servicing_alignment_data.py) | Locks schema, coverage, split counts, held-out isolation, and digest-drift behavior. |

## Composite Dataset

Training only on a small correction set risks catastrophic forgetting. The
generator therefore copies the governed released corpus and appends the
alignment examples within the corresponding split:

| Split | Released base | Alignment addition | Composite total |
| --- | ---: | ---: | ---: |
| Train | 6,304 | 320 | 6,624 |
| Validation | 1,349 | 80 | 1,429 |
| Test | 1,347 | 27 | 1,374 |

The 320 training additions are 32 deterministic natural realizations of ten
behavior templates. Validation uses eight separately worded realizations per
template. Test uses four separately worded realizations plus three exact
captured regressions. The exact captured wording never appears in training.

All records retain the existing `banking-tool-sft/v1` contract. That means the
current Granite tokenizer, assistant-only loss mask, tool-wire adapter,
training worker, replay checks, and frozen evaluator can consume the composite
without a schema migration.

## Behavior Coverage

The augmentation contains:

- service-case contextual follow-ups using `list_service_cases`;
- grounded answers containing `created_at`, status, `case_type`, and subject;
- card anaphora followed by `replace_card` or `freeze_card`;
- clarification answers that supply `last4` or a public merchant description;
- agent-repair turns after an irrelevant or repeated answer;
- banking-to-external topic shifts with the governed OOD answer;
- external-to-banking shifts that return to a model-selected banking tool.

Tool calls contain only public arguments. Synthetic tool results match the
mock backend fields used by the POC.

### Worked remediation example

Observed failure shape:

```text
User: Show my service cases.
Assistant: You have a closed mailing-address update case.
User: When was that created?
Bad behavior: repeats the first answer or chooses an unrelated tool.
```

The Granite remediation record includes visible history, a model-owned
`list_service_cases` target when fresh customer data is needed, a correlated
synthetic result, and a final assistant target grounded in `created_at`.

The matching router row stops before the current assistant action:

```text
input: current user turn + prior visible user/assistant exchange
domain target: supported banking
capability target: service_cases
relation target: context_dependent
excluded: expected tool call, tool result, grounding facts, final answer
```

This pairing teaches two separate abilities. The router learns not to block the
follow-up. Granite learns how to answer it.

Training and test use separately worded variants. Copying the exact observed
sentence into training would make the regression easy without proving the
general behavior.

## Generate and Verify

From the repository root:

```bash
PYTHONPATH=src python \
  scripts/retail_bank/prepare_servicing_alignment_data.py
```

The command writes `data/banking-servicing-alignment-v4/`, validates every
record, verifies the released base split hashes, checks for PII-like content
and held-out leakage, and compares the composite split hashes with the tracked
lock. It fails rather than silently updating the lock.

Before a paid HF job, publish this dataset to a dataset repo revision:

```bash
PYTHONPATH=src python \
  scripts/retail_bank/prepare_servicing_alignment_data.py \
  --push-to-hub \
  --repo-id spkc83/retail-bank-servicing-alignment-sft
```

Run its focused tests:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_banking_servicing_alignment_data.py
```

## Released Granite Training Stage

The remediation model continued from the already tool-trained Granite weights
rather than restarting from the IBM base:

| Setting | Released value |
| --- | --- |
| Starting model | stage-1 tool-trained Granite checkpoint |
| Base family | `ibm-granite/granite-4.1-8b` |
| IBM base revision | `1504002f650e656a0a3789d99574df12e3e94ed0` |
| Architecture | Granite 4.1 8B decoder-only causal transformer |
| Adaptation | LoRA on attention and MLP projections |
| Dataset | `data/banking-servicing-alignment-v4/manifest.json` |
| Maximum sequence | 2,048 tokens |
| Training job | `spkc83/6a6ca6276b79c09949c1d6cb` |
| Runtime | about 18 minutes 59 seconds |
| Estimated cost | about `$0.87` |
| Train loss | `0.0069123295` |
| Eval loss | `0.0002181597` |
| Token accuracy | `0.999976121` |
| Publication target | `spkc83/retail-bank-servicing-agent-9b` |

The existing worker accepts this dataset without changing model architecture.
A safe dry-run plan for a reproduction is:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-servicing-alignment-v4/manifest.json \
  --base-model spkc83/retail-bank-agent-9b \
  --base-revision 085df3d089cfadd77424b548542da0390a54a23e \
  --learning-rate 2e-5 \
  --max-steps 500 \
  --output-dir artifacts/retail-bank-servicing-alignment-v4
```

That command remains a dry run unless the existing explicit remote-execution
guards are provided. A real 9B run still requires separate authorization for
paid compute.

## Rescore Correctness

The final public dataset revision is
`0ce32f9c7a3edff227005e5b89b089947b87625a`. The training job used revision
`fea8aa1cda716954eb7322325e2be25c9f570ea3`. The correction did not change the
rendered prompts, target tool calls, or target final responses used for
generation and scoring. `scripts/retail_bank/rescore_tool_eval.py` therefore
rescored prompt-equivalent rows against the existing predictions. This is not a
second generation run.

## Evaluation and Stop Condition

The released checkpoint was not release eligible merely because training
finished. It had to:

- retain the released tool-name, argument, executable-trajectory, FAQ, OOD,
  dependent-tool, and grounding gates;
- pass all 1,374 composite test records;
- pass the exact held-out servicing regressions;
- improve live multi-turn service-case, card-reference, clarification,
  repair, and topic-shift conversations;
- show the exact 9B revision and CUDA inference path in POC diagnostics.

The released model passed the exact frozen evaluation with 796/796 tool
names/arguments, 700/700 executable trajectories, 96/96 multi-tool sequences,
63/63 clarifications, 258/258 FAQ answers, 35/35 OOD paths, 1,141/1,141
grounded responses, and zero hard error metrics.

## Space Runtime Pins

The released Space should pin:

- `RETAIL_BANK_MODEL_ID=spkc83/retail-bank-servicing-agent-9b`
- `RETAIL_BANK_MODEL_REVISION=1d56824995aa1adecfe20f62ca42fb1c0c443817`
- `RETAIL_BANK_ROUTER_ID=spkc83/retail-bank-conversation-router`
- `RETAIL_BANK_ROUTER_REVISION=dd5ea26674a0f9808d42110a9ee51a9af6762a76`

Verify deployment by checking the diagnostic block in
`poc/retail-bank-customer-service-poc/app.py` for the exact model revision,
router revision, CUDA-backed model passes, generated tool calls, and
capability/relation router payload fields for each turn.
