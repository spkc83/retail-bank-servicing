---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - banking
  - tool-calling
  - synthetic
  - conversational
pretty_name: Retail Bank Agent Tool-Use SFT
---

# Retail Bank Agent Tool-Use SFT

This dataset contains 9,000 deterministic, fictional retail-banking
conversations for supervised fine-tuning of a conversational tool-using model.

- Dataset: https://huggingface.co/datasets/spkc83/retail-bank-agent-sft
- Training revision:
  `183e7e1ed1aba9c3d7155e7b83b64dc854935055`
- Source: https://github.com/spkc83/retail-bank-servicing
- Released stage-2 model:
  https://huggingface.co/spkc83/retail-bank-servicing-agent-9b
- Public POC:
  https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc

## Splits

- Train: 6,304
- Validation: 1,349
- Frozen test: 1,347
- Corpus fingerprint:
  `2bb7a400ed2556b15c7e5eb6147668041b5deef8ae4f037f9e2e52295ff29ab5`
- Split seed: `711`

## Role In The Release Pipeline

This is the stage-1 tool-use SFT corpus. It teaches the Granite base model the
synthetic-bank tool wire, public tool schemas, tool-result grounding,
clarification, FAQ, OOD refusal, and multi-tool ordering. The released
servicing agent then receives a second SFT stage on
`spkc83/retail-bank-servicing-alignment-sft` for observed POC conversation and
tool-use failures.

## Coverage

The corpus covers all nine public synthetic-bank tools, successful and failed
tool results, clarification, general banking FAQ, hard-negative private-field
requests, out-of-domain refusal, multi-turn context, and ordered multi-tool
calls.

| Scenario family | Conversations |
|---|---:|
| Clarification | 333 |
| Conversation | 999 |
| Hard negative | 333 |
| Multi-turn | 1,665 |
| No-tool banking FAQ | 1,665 |
| OOD | 333 |
| Tool error | 666 |
| Tool success | 3,006 |

Every tool-bearing record was replayed against isolated deterministic
synthetic state before inclusion. Assistant tool-call and final-response tokens
are trainable; system, user, and tool-result tokens are context only.
The data validator rejects semantically empty final responses and asserts
path-specific content for clarification, FAQ, OOD, and hard-negative rows.

## Example Record Shape

```text
user context: Show my accounts and balances.
assistant target: call list_accounts({})
tool context: synthetic account rows and balances
assistant target: summarize account names, last four digits, and balances
expected: exact tool/arguments plus required grounding facts
```

The tool result is not a model target. At inference time it comes from the
backend. The model learns the assistant-owned call and grounded final response.

Related variants share split keys so a paraphrase of one state/template group
cannot appear in both training and frozen test data.

## Source and privacy policy

All included rows are self-authored synthetic data under MIT. External
classifier corpora are prepared by a separate pipeline and never enter the
generative SFT splits.

The dataset contains no real customers, credentials, accounts, or financial
events. It is for research demonstrations, not production banking.
