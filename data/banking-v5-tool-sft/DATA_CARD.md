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

This dataset contains 1,200 deterministic, fictional
retail-banking conversations for supervised fine-tuning of a conversational
tool-using model.

## Splits

- Train: 841
- Validation: 179
- Test: 180
- Corpus fingerprint: `5f1b6b391557158c551529fa4cbcd7e760d96c95b25f66daac641e11afd69247`
- Split seed: `711`

## Coverage

The corpus covers all nine public synthetic-bank tools, successful and failed
tool results, clarification, general banking FAQ, hard-negative private-field
requests, out-of-domain refusal, multi-turn context, and ordered multi-tool
calls.

- `clarification`: 41
- `conversation`: 123
- `hard_negative`: 41
- `multi_turn`: 207
- `ood`: 41
- `retrieval_grounded_policy`: 287
- `tool_error`: 82
- `tool_success`: 378

Every tool-bearing record was replayed against an isolated synthetic bank state
before inclusion. Assistant tool-call and final-response tokens are trainable;
system, user, and tool-result tokens are context only.

## Source policy

All included rows are self-authored synthetic data under MIT. External
classifier corpora are prepared by a separate pipeline and never enter these
generative splits.

This dataset contains no real customers, credentials, accounts, or financial
events. It is for a research demonstration, not production banking.
