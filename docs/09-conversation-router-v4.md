# Hierarchical Router Architecture Deep Dive

The filename is retained for stable links. This page documents the active V6
router artifact format 4.

## Why a Hierarchy

A single flat intent label answers only “what does this resemble?” The POC
also needs to decide whether the turn belongs in scope, which orchestration
lane should run, whether conversation context changes the meaning, whether a
banking action is appropriate, and whether its target is usable.

V6 models those questions explicitly:

```text
domain -> lane -> family -> intent
                     + relations
                     + action disposition
                     + entity-resolution state
```

The hierarchy lets the runtime distinguish cases that share an intent but need
different behavior:

| Turn | Intent | Action | Entity state |
| --- | --- | --- | --- |
| “Replace card 4821.” | `replace_card` | `execute_tool` | `resolved` |
| “Replace my card.” with several cards | `replace_card` | `clarify` | `ambiguous` |
| “Replace that closed card anyway.” | `replace_card` | `converse` | `ineligible` |

## Shared Encoder and Seven Heads

One DistilBERT encoder processes the current turn, recent visible exchanges,
and bounded pre-turn state. Seven linear heads consume the same pooled vector:

1. domain: 3 categorical labels;
2. lane: 5 categorical labels;
3. family: 9 categorical labels;
4. intent: 12 categorical labels;
5. relation: 5 independent binary labels;
6. action: 5 categorical labels;
7. entity resolution: 5 categorical labels.

Training optimizes a weighted sum of those losses. Targeted and counterfactual
rows receive higher sample weights. The entity head also uses bounded
class-balanced weights because `ineligible` is intentionally rare.

## Legal Tuple Decoding

The categorical heads are not exposed as seven unrelated argmaxes. The joint
decoder constructs legal tuples from the canonical taxonomy. For each legal
tuple it adds the corresponding raw head scores and selects the maximum.

Conceptually:

```text
score(tuple) =
    domain_score[tuple.domain]
  + lane_score[tuple.lane]
  + family_score[tuple.family]
  + intent_score[tuple.intent]
  + action_score[tuple.action]
  + entity_score[tuple.entity_resolution]
```

Relations remain independently thresholded because several can be active.
OOD has a dedicated legal tuple with no intent, `refuse_ood`, and
`not_required` entity state.

This decoder enforces structural compatibility. It does not prove the semantic
prediction is correct, and it does not authorize a customer action.

## Route Boundary and Abstention

The domain support probability and contextual relation rescue score determine
`in_domain`, `out_of_domain`, or `uncertain` before the decoded tuple is
exposed. OOD suppresses downstream predictions and emits the safe OOD tuple.
Uncertain suppresses intent, lane, family, action, and entity outputs.

That suppression matters: an uncertain turn cannot accidentally inherit a
high action logit and expose a tool.

## Counterfactual Training

The dataset contains paired rows where wording is controlled while state or
conversation evidence changes. These examples teach the router to use context
rather than memorize a phrase.

Examples include:

- the same follow-up with one matching card versus several cards;
- a resume phrase with an active pending task versus no pending task;
- an explicit intent switch while old state remains present;
- a social or OOD detour during a banking conversation;
- eligible versus closed/ineligible entities.

The held-out metrics report 1.0 for action accuracy, entity-resolution
accuracy, and pair-flip accuracy across 32 counterfactual rows.

## From Router Tuple to Generation

The runtime contract is deliberately narrower than “classifier controls the
agent”:

```text
router tuple
  -> harness selects response lane
  -> execute_tool selects one compatible schema, never arguments
  -> Granite reads token-budgeted conversation
  -> Granite calls that schema with its chosen arguments or asks a question
  -> harness validates and executes against fictional state
```

For policy turns, the harness supplies retrieved evidence and no tools. For
conversation or clarification turns, it supplies action-specific instructions
and no tools. For OOD, it does not invoke Granite.

## Failure Boundaries

The design reduces several failure modes:

- incompatible flat-label predictions are repaired by joint decoding;
- stale dialogue state is countered by explicit intent-switch examples;
- missing/ambiguous/ineligible entities cannot reach an executable plan;
- tool selection is narrowed to one schema;
- tool arguments remain grounded in model-visible conversation;
- uncertain decisions abstain from tools;
- external OOD cannot be rescued by `topic_shift` alone.

It does not eliminate:

- errors outside the governed data distribution;
- incorrect Granite arguments within the selected schema;
- missing knowledge-base coverage;
- response-quality or grounding failures after a successful route;
- production authorization, fraud, privacy, or regulatory controls.

## Evidence

The release artifact is
`artifacts/banking-conversation-router-v6-hierarchical`. Its `metrics.json`
records:

- `release_eligible: true` and no failed gates;
- 0 hierarchy compatibility errors;
- 0 held-out route, intent, or relation regression errors;
- 0 trajectory runtime-transition errors;
- 0 in-domain false refusals;
- 0.003232 OOD false-accept rate;
- 0.996473 intent macro F1;
- 0.997865 action macro F1;
- 0.999060 entity-resolution macro F1.

Dataset identity:

```text
spkc83/retail-bank-conversation-router-data
073e61156885a8a2074c7254d76f00634058429a
```

Router identity after publication:

```text
spkc83/retail-bank-conversation-router
c0d71b433fd1eef510fce36f6308eb36e423e329
```
