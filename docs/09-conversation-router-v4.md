# Conversation Router v4

This document describes the released v4 classifier. It is published as
`spkc83/retail-bank-conversation-router` at revision
`9e090c0fa21cebbaa03a431a7ce61e656c0739fe` and trained from
`spkc83/retail-bank-conversation-router-data` revision
`e9a64a2e7f2b622d5412c15eac4618ceca2150da`.

## Why v4 Exists

The released v1 router classifies the current message first and adds history
only when hand-written lexical rules recognize a short or referential
follow-up. That creates three failure modes:

- natural follow-ups such as corrections and agent-repair turns can be blocked
  before Granite sees them;
- the 77 Banking77 intent labels do not match the POC's actual tools and
  servicing capabilities;
- training and serving use different contextual input shapes.

V4 removes the lexical pre-router. Every turn is classified once with the
current message and recent visible conversation in the same encoder input.

## What a Cross-Encoder Is

A cross-encoder receives both sides of a classification decision in one token
sequence. For this project, those sides are the current user message and the
recent dialogue:

```text
[CURRENT_USER]
When was that created?
[PREVIOUS_ASSISTANT]
You have a closed mailing-address update case.
[PREVIOUS_USER]
Show my recent service cases.
```

The transformer performs bidirectional self-attention across the entire
sequence. A token in `When was that created?` can therefore attend directly to
`mailing-address update case`, and the representation used by the classifier
contains their relationship.

A bi-encoder would encode the current message and history separately and
compare two fixed vectors. That is faster for large retrieval indexes, but it
loses token-level interaction and is unnecessary here because the POC
classifies one conversation at a time.

The current message is rendered first. Up to three complete, visible
user/assistant exchanges follow in newest-first order. Standard right-side
token truncation therefore removes the oldest context first and protects the
current and most recent language. Tool payloads and hidden tool-call messages
are not classifier input.

## Architecture

[`banking_conversation_router.py`](../src/hello_slm/banking_conversation_router.py)
defines one DistilBERT encoder and three task heads over its pooled first-token
representation:

```text
combined current + history
  -> shared DistilBERT cross-encoder
     -> domain head: external OOD or supported conversation
     -> capability head: coarse POC servicing area
     -> relation head: four independent probabilities
```

The coarse capabilities are:

- accounts;
- cards;
- card actions;
- transactions;
- transfers;
- service cases;
- FAQ;
- conversation.

They are diagnostics. They do not select a tool, authorize an action, supply
arguments, or enter the Granite system prompt.

Conversation relations are multi-label because they overlap:

- `context_dependent`;
- `agent_repair`;
- `topic_shift`;
- `clarification_answer`.

For example, “No, I meant the address case” can be context-dependent, an agent
repair, and a wrong-topic shift at the same time. Independent sigmoid outputs
represent that correctly; a four-way softmax would force a false choice. The
training loss uses capped positive-class weights so rare repair and
clarification labels are not overwhelmed by negative rows.

Each relation receives a validation-calibrated activation threshold. Threshold
selection allows a small near-optimal F1 tolerance only for
`context_dependent`, because missing a valid follow-up blocks the conversation.
The other labels use the lowest threshold that attains the exact best
validation F1. This keeps their validation precision intact while preserving
recall among tied cutoffs. The runtime exposes both raw probabilities and the
active multi-label relations; the separate rescue threshold still controls
only the conservative OOD policy.

## Routing Policy

The runtime calculates:

```text
banking_probability = softmax(domain_logits)[in_domain]
rescue_probability = max(
    context_dependent,
    agent_repair,
    clarification_answer,
)
```

It then applies three bands:

- `in_domain` when banking probability reaches the calibrated in-domain
  boundary;
- `out_of_domain` only when banking probability is below the calibrated OOD
  boundary and the rescue probability is below its boundary;
- `uncertain` otherwise.

Both `in_domain` and `uncertain` go to Granite. A confident external topic
shift can still be rejected because `topic_shift` is intentionally not a
rescue signal. This policy prefers an extra model call over a false refusal of
a valid conversational continuation.

### Numeric routing examples

The values are illustrative policy inputs, not captured predictions:

| Banking probability | Context/repair/clarification maximum | Result |
| ---: | ---: | --- |
| `0.73` | `0.09` | `in_domain` because banking reaches `0.50`. |
| `0.26` | `0.12` | `uncertain`; the middle band goes to Granite. |
| `0.06` | `0.58` | `uncertain`; relation rescue prevents an OOD block. |
| `0.06` | `0.18` | `out_of_domain`; both OOD conditions are satisfied. |

`topic_shift=0.90` does not enter `rescue_probability`. A weather follow-up can
therefore remain OOD even when the relation head correctly detects a shift.

An `uncertain` capability is set to `None` in the route result, though the top
three capability candidates remain visible as diagnostics. The application
does not convert those candidates into tool choices.

## Governed Data

[`banking_conversation_router_data.py`](../src/hello_slm/banking_conversation_router_data.py)
builds the released router dataset from:

- the existing split-isolated Granite SFT conversations for POC-aligned
  in-domain examples;
- checksum-pinned CLINC data for external OOD language;
- deterministic synthetic conversations for anaphora, typographical
  variation, clarification answers, corrections, agent repair, and topic
  shifts.

The router row is created at the instant before the current assistant reply.
It may contain only:

- prior visible user and assistant text;
- the current user text;
- classifier labels and source provenance.

It must not contain the current turn's expected tool call, tool result,
grounding facts, final assistant answer, or customer state. Those values would
leak the answer into classifier training.

Exact captured failure utterances are held-out regression rows only. They are
not copied into the training split.

| Split | Rows |
| --- | ---: |
| Train | 61,759 |
| Validation | 13,173 |
| Test | 15,466 |

The test split contains independently worded contextual, repair,
clarification, typo, and topic-shift examples in addition to the seven exact
captured regressions. Split-derived generalization rows use only held-out base
SFT and CLINC test records. Separately authored targeted validation and test
families use split-specific wording, and the seven exact captured utterances
remain test-only. None reuse training anchors.

The training split also contains 1,456 targeted, non-verbatim use-case
realizations for service-case references, standalone address requests, card
selection followed by replacement, repeated-answer repair, and wrong-topic
repair. Validation and test use separately worded versions.

Generate the local governed dataset:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_conversation_router_data.py
```

The output directory is `data/banking-conversation-router-v4/` and remains
untracked because the manifest, source lock, generator, and published dataset
revision are the reproducibility surfaces.

## Local Training

Training is local by default and uses the available CUDA device:

```bash
PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v4 \
  --output-dir artifacts/banking-conversation-router-v4
```

The script pins PyTorch 2.12.1 from the CUDA 12.6 wheel index because that
build still includes TITAN V (`sm_70`) kernels. Running it as `uv run python
...` would bypass the inline environment and may select an incompatible CUDA
13 build.

No Hugging Face Job is required. Publication is an explicit second mode:

```bash
HF_TOKEN=... PYTHONPATH=src uv run \
  scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v4 \
  --output-dir artifacts/banking-conversation-router-v4 \
  --publish \
  --data-revision DATASET_COMMIT_40_HEX
```

The script refuses to overwrite a non-empty artifact directory and requires an
immutable dataset revision for publication.

## Release Gates

The held-out test report blocks publication unless all of these pass:

| Metric | Gate |
| --- | ---: |
| Capability macro F1 | at least `0.85` |
| Relation macro F1 | at least `0.85` |
| In-domain false-refusal rate | at most `0.02` |
| OOD false-accept rate | at most `0.05` |
| Contextual false-refusal rate | at most `0.02` |
| Repair false-refusal rate | at most `0.01` |
| External topic-shift false-accept rate | at most `0.05` |
| Held-out captured-regression route error rate | exactly `0.00` |
| Held-out captured-regression capability error rate | exactly `0.00` |
| Held-out captured-regression relation error rate | exactly `0.00` |

The exact screenshot-derived regression set is test-only and must route every
case correctly before a Space revision changes.

### Released local-training result

The deterministic one-epoch TITAN V run produced
`artifacts/banking-conversation-router-v4/` and passed all release gates:

| Held-out test metric | Result |
| --- | ---: |
| Rows | `15,466` |
| Capability macro F1 | `0.997838` |
| Relation macro F1 | `0.998628` |
| In-domain false-refusal rate | `0.000167` |
| OOD false-accept rate | `0.012735` |
| Contextual false-refusal rate | `0.000105` |
| Repair false-refusal rate | `0.000000` |
| External topic-shift false-accept rate | `0.000778` |
| Captured-regression route/capability/relation errors | `0 / 0 / 0` |

These metrics are the release gate evidence for router revision
`9e090c0fa21cebbaa03a431a7ce61e656c0739fe`. The local artifact remains ignored
by Git; the published Hub artifact and tracked lock are the reproducibility
surfaces.

## POC Integration

[`router.py`](../poc/retail-bank-customer-service-poc/router.py) implements the
same model heads, input rendering, and route policy in the standalone Space.
It requires `RETAIL_BANK_ROUTER_REVISION` to be an immutable 40-character
artifact commit. The released revision is
`9e090c0fa21cebbaa03a431a7ce61e656c0739fe`.

A classifier exception produces a visible `classifier_error` response path
and blocks the Granite call. The POC therefore never presents a classifier
outage as an ordinary uncertain decision during v4 evaluation.

[`model_service.py`](../poc/retail-bank-customer-service-poc/model_service.py)
continues to send Granite the token-budgeted conversation history. The router
result remains outside the system prompt, so a wrong capability prediction
cannot anchor or authorize the 9B model.

## Focused Validation

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_banking_conversation_router.py \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py \
  tests/test_banking_conversation_router_training.py \
  poc/retail-bank-customer-service-poc/tests/test_router.py \
  poc/retail-bank-customer-service-poc/tests/test_app.py \
  poc/retail-bank-customer-service-poc/tests/test_model_service.py
```
