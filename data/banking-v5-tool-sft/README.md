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

- Train: 838
- Validation: 181
- Test: 181
- Corpus fingerprint: `11f46022c528fefda50aaf07cc97e6bebe0fcc3be2f4c44728c27d834634601d`
- Split seed: `711`

## Coverage

The corpus covers all nine public synthetic-bank tools, successful and failed
tool results, clarification, general banking FAQ, hard-negative private-field
requests, out-of-domain refusal, multi-turn context, and ordered multi-tool
calls.

- `clarification`: 44
- `conversation`: 132
- `hard_negative`: 44
- `multi_turn`: 223
- `ood`: 44
- `retrieval_grounded_policy`: 220
- `tool_error`: 88
- `tool_success`: 405

Every tool-bearing record was replayed against an isolated synthetic bank state
before inclusion. Assistant tool-call and final-response tokens are trainable;
system, user, and tool-result tokens are context only.

## Source policy

All included rows are self-authored synthetic data under MIT. External
classifier corpora are prepared by a separate pipeline and never enter these
generative splits.

This dataset contains no real customers, credentials, accounts, or financial
events. It is for a research demonstration, not production banking.
