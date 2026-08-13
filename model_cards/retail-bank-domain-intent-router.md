---
base_model: distilbert/distilbert-base-uncased
datasets:
  - spkc83/retail-bank-conversation-router-data
library_name: transformers
license: apache-2.0
pipeline_tag: text-classification
tags:
  - banking
  - out-of-domain-detection
  - conversation-router
---

# Retail Bank Conversation Router

The released router is a DistilBERT cross-encoder with one shared encoder and
three heads:

- binary supported-banking/OOD domain head;
- coarse servicing-capability head for diagnostics;
- multi-label conversation-relation head for `context_dependent`,
  `agent_repair`, `topic_shift`, and `clarification_answer`.

Source and evaluation code:
https://github.com/spkc83/retail-bank-servicing

Live model-driven application:
https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc

## Artifact Identity

- Model repository: `spkc83/retail-bank-conversation-router`
- Model revision: `9e090c0fa21cebbaa03a431a7ce61e656c0739fe`
- Training-data repository: `spkc83/retail-bank-conversation-router-data`
- Training-data revision: `e9a64a2e7f2b622d5412c15eac4618ceca2150da`
- Base encoder revision:
  `12040accade4e8a0f71eabdb258fecc2e7e948be`
- Source revision:
  `475dc2b563ef87fa0c9aa597b0b0465d56d2ee0f`

## Held-Out Results

- Release eligible: `True`
- Test rows: `15,466`
- Capability macro F1: `0.997838`
- Relation macro F1: `0.998628`
- In-domain false-refusal rate: `0.000167`
- OOD false-accept rate: `0.012735`
- Contextual false-refusal rate: `0.000105`
- Repair false-refusal rate: `0.000000`
- External topic-shift false-accept rate: `0.000778`
- Captured-regression route/capability/relation errors: `0 / 0 / 0`

## Serving Policy

The router receives the current user turn and up to three complete visible prior
user/assistant exchanges in one encoder sequence. Tool payloads and hidden
tool-call messages are not classifier input.

Serving uses:

- banking probability `< 0.10` plus no relation rescue: `out_of_domain`;
- banking probability `>= 0.50`: `in_domain`;
- middle region or relation rescue: `uncertain`;
- `context_dependent`, `agent_repair`, and `clarification_answer` can rescue a
  low-domain follow-up from immediate OOD refusal;
- `topic_shift` is diagnostic and does not rescue external requests.

Both `in_domain` and `uncertain` continue to the 9B agent. Capability
predictions and relation probabilities are displayed only as diagnostics; they
do not enter the prompt, select a tool, or provide tool arguments.

### Threshold examples

| Banking | Rescue | Route | Generation |
| ---: | ---: | --- | --- |
| `0.75` | `0.08` | `in_domain` | 9B invoked |
| `0.27` | `0.15` | `uncertain` | 9B invoked |
| `0.07` | `0.62` | `uncertain` | 9B invoked |
| `0.03` | `0.09` | `out_of_domain` | no 9B call |

These values illustrate the released policy. They are not captured predictions
for a specific sentence.

## Data and Licenses

Classifier-only data combines the governed synthetic tool-use/SFT conversations
with UCI CLINC150 external OOD examples. The prepared dataset contains 61,759
training, 13,173 validation, and 15,466 test rows. Exact captured POC failures
are held out in test and are not copied into training.

## Serving Fallback

During normal serving, a classifier exception becomes `classifier_error`. The
POC returns its explicit failure response and does not call the 9B model for
that turn.

In explicit local test mode, `POC_SKIP_ROUTER_LOAD=1` leaves the router absent.
That test-only path returns `uncertain` and delegates the turn to the 9B model.
The router remains experimental, not a production authorization boundary.
