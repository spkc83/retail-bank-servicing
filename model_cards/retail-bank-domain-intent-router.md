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

# Retail Bank Conversation Router V5

This release is a state-aware DistilBERT cross-encoder with one shared encoder
and three learned heads:

- binary banking-domain/OOD classification;
- 12-way fine-intent classification;
- multi-label conversational relations, including explicit service resumption.

The application derives the servicing, policy, conversation, and other-banking
lanes from the fine intent. A deterministic dialogue-state layer—not the
classifier—preserves one pending servicing task across temporary policy or
social detours. Classifier outputs do not authorize tools or enter Granite's
generation prompt.

## Artifact Identity

- Model repository: `spkc83/retail-bank-conversation-router`
- Model revision: `c8f154266612e79afe20af8abef25761fa56d589`
- Training-data repository: `spkc83/retail-bank-conversation-router-data`
- Training-data revision: `8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc`
- Base encoder revision: `12040accade4e8a0f71eabdb258fecc2e7e948be`

## Frozen Test Results

- Release eligible: `true`
- Test rows: `6,171`
- Fine-intent macro F1: `0.990312`
- Relation macro F1: `0.996474`
- In-domain false-refusal rate: `0.000000`
- OOD false-accept rate: `0.007899`
- Resume intent and relation error rates: `0.000000 / 0.000000`
- State-conditioned route error rate: `0.000000`
- State-conditioned intent error rate: `0.000000`
- Runtime-transition error rate: `0.000000`
- Held-out social-generalization error rate: `0.000000`
- Held-out policy-follow-up-generalization error rate: `0.000000`
- State-conditioned false-resume rate: `0.000000`
- Captured-regression route, intent, and relation errors: `0 / 0 / 0`

The calibrated domain boundaries are `0.45` for OOD and `0.50` for in-domain;
the relation-rescue boundary is `0.40`. Mid-band and low-confidence intent
decisions do not mutate deterministic dialogue state.

## Input and Labels

Each input contains the current user turn, at most three complete recent
user/assistant exchanges, and a compact pre-turn dialogue-state header when a
servicing task is pending. Tool calls, tool results, expected answers, and
post-turn state are excluded.

Fine intents cover nine servicing operations plus `policy_knowledge`,
`conversation`, and `other_banking`. Relations are `context_dependent`,
`agent_repair`, `topic_shift`, `clarification_answer`, and
`resume_previous_service`.

See [`docs/05-hierarchical-router.md`](../docs/05-hierarchical-router.md) for the
complete architecture, transitions, calibration, data-generation strategy, and
reproduction commands.
