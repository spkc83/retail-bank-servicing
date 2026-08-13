# History-Aware Conversation Router

This guide covers the released CPU router: governed v4 data preparation,
DistilBERT cross-encoder training, threshold calibration, publication, and
serving behavior. The router does not select tools and does not supply tool
arguments to the Granite model.

The previous Banking77 intent router has been superseded for the POC runtime.
This file keeps its historical filename so existing documentation links remain
stable.

## Active Artifact IDs

| Artifact | Value | Owner |
| --- | --- | --- |
| Router repo | `spkc83/retail-bank-conversation-router` | [`poc/retail-bank-customer-service-poc/router.py`](../poc/retail-bank-customer-service-poc/router.py) |
| Router revision | `9e090c0fa21cebbaa03a431a7ce61e656c0739fe` | [`model card`](../model_cards/retail-bank-domain-intent-router.md) |
| Router dataset repo | `spkc83/retail-bank-conversation-router-data` | [`scripts/retail_bank/prepare_conversation_router_data.py`](../scripts/retail_bank/prepare_conversation_router_data.py) |
| Router dataset revision | `e9a64a2e7f2b622d5412c15eac4618ceca2150da` | [`data card`](../data_cards/retail-bank-router-training-data.md) |
| Base encoder | `distilbert/distilbert-base-uncased` | [`scripts/retail_bank/train_conversation_router.py`](../scripts/retail_bank/train_conversation_router.py) |
| Base encoder revision | `12040accade4e8a0f71eabdb258fecc2e7e948be` | [`scripts/retail_bank/train_conversation_router.py`](../scripts/retail_bank/train_conversation_router.py) |

The public dataset card is
[`data_cards/retail-bank-router-training-data.md`](../data_cards/retail-bank-router-training-data.md).
The public model card is
[`model_cards/retail-bank-domain-intent-router.md`](../model_cards/retail-bank-domain-intent-router.md).

## Architecture

[`scripts/retail_bank/train_conversation_router.py`](../scripts/retail_bank/train_conversation_router.py)
trains one shared DistilBERT cross-encoder with three heads:

- a binary domain head for supported retail banking vs out-of-domain;
- an eight-way servicing-capability head for diagnostics;
- a four-label sigmoid relation head for `context_dependent`, `agent_repair`,
  `topic_shift`, and `clarification_answer`.

The domain loss applies to every row. Capability loss applies to in-domain
servicing rows. Relation loss is multi-label and uses capped positive-class
weights so rare repair and clarification rows are not overwhelmed.

The runtime input format is:

```text
[CURRENT_USER]
{current user turn}
[PREVIOUS_ASSISTANT]
{most recent visible assistant response}
[PREVIOUS_USER]
{most recent visible user turn}
```

Up to three complete visible exchanges are included newest-first after the
current user turn. Tool payloads and hidden tool-call messages are excluded.

## Data Preparation

The preparation script is
[`scripts/retail_bank/prepare_conversation_router_data.py`](../scripts/retail_bank/prepare_conversation_router_data.py).
It builds split-isolated rows from:

- the governed synthetic SFT conversations for POC-aligned in-domain examples;
- checksum-pinned UCI CLINC150 data for external OOD language;
- deterministic synthetic contextual follow-ups, typo variants,
  clarification answers, corrections, agent-repair turns, and topic shifts.

Prepare and reproduce the released split digests:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_conversation_router_data.py
```

Outputs:

- `data/banking-conversation-router-v4/train.jsonl`;
- `data/banking-conversation-router-v4/validation.jsonl`;
- `data/banking-conversation-router-v4/test.jsonl`;
- `data/banking-conversation-router-v4/manifest.json`;
- `data/banking-conversation-router-v4/README.md`.

The prepared public dataset contains 61,759 train rows, 13,173 validation rows,
and 15,466 test rows. Exact captured POC failure utterances are held out in the
test split and are not copied into training.

## Training and Calibration

Train locally without publishing:

```bash
PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py
```

The script pins a CUDA 12.6 PyTorch build that supports TITAN V (`sm_70`) when
run through its inline `uv` environment. It calibrates:

- OOD banking boundary: `0.10`;
- in-domain boundary: `0.50`;
- relation rescue boundary: `0.40`;
- per-relation activation thresholds from validation probabilities.

Publication requires `--publish`, an immutable `--data-revision`, and
`HF_TOKEN`. Do not publish over an existing release without fresh frozen
evaluation evidence.

## Release Gates

The released router passed these held-out gates:

| Metric | Result |
| --- | ---: |
| Test rows | `15,466` |
| Capability macro F1 | `0.997838` |
| Relation macro F1 | `0.998628` |
| In-domain false-refusal rate | `0.000167` |
| OOD false-accept rate | `0.012735` |
| Contextual false-refusal rate | `0.000105` |
| Repair false-refusal rate | `0.000000` |
| External topic-shift false-accept rate | `0.000778` |
| Captured-regression route/capability/relation errors | `0 / 0 / 0` |

## Serving Boundaries

[`LearnedBankingRouter.from_hub`](../poc/retail-bank-customer-service-poc/router.py)
loads the pinned router revision from Hub, verifies the artifact manifest, and
serves without `trust_remote_code`.

Serving routes:

- banking probability `< 0.10` and no relation rescue: `out_of_domain`
- banking probability `>= 0.50`: `in_domain`
- middle range or relation rescue: `uncertain`

The classifier's capability and relation outputs are diagnostics only. They do
not enter the Granite prompt, select tools, or provide tool arguments. If the
router fails during normal serving, the POC reports `classifier_error` and does
not invoke the model for that turn.

### Worked threshold examples

The values below illustrate the released policy. They are not captured model
outputs for the example text.

| Banking | Highest rescue relation | Decision | What the application does |
| ---: | ---: | --- | --- |
| `0.91` | `0.05` | `in_domain` | Invoke Granite 9B. |
| `0.32` | `0.15` | `uncertain` | Invoke Granite 9B. |
| `0.08` | `0.72` | `uncertain` | Rescue the likely conversational follow-up and invoke Granite. |
| `0.04` | `0.11` | `out_of_domain` | Return the stock scope response; do not invoke Granite. |

The entire `0.10 <= banking < 0.50` band is uncertain. It does not require a
relation rescue. Rescue prevents a likely follow-up below `0.10` from being
rejected as OOD.

Example contextual input:

```text
[CURRENT_USER]
When was that created?
[PREVIOUS_ASSISTANT]
You have a closed mailing-address update case.
[PREVIOUS_USER]
Show my service cases.
```

The shared encoder lets “that” interact with “mailing-address update case.” The
relation head can mark the turn `context_dependent`, while the domain head
estimates whether the combined conversation is supported banking.

For background, see
[decision-threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)
and [selective classification](https://arxiv.org/abs/1705.08500).

## Stop Conditions

Stop before publication if:

- source digests do not match;
- generated split digests drift from
  [`data/sources/banking-conversation-router-v4.lock.json`](../data/sources/banking-conversation-router-v4.lock.json);
- cross-split duplicates or PII-like matches are nonzero;
- `HF_TOKEN` is unavailable for publish training;
- any release gate fails;
- the artifact manifest cannot verify every file;
- serving requires `trust_remote_code`.

Run the focused tests after any router change:

```bash
python -m pytest -q tests/test_banking_conversation_router.py \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py \
  tests/test_banking_conversation_router_training.py
```
