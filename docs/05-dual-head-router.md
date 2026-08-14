# V5 Three-Head State-Aware Router

The filename is retained for stable links, but the active V5 router is not a
dual-head classifier. It is a shared DistilBERT cross-encoder with three heads:

1. two-class banking domain head;
2. 12-class fine-intent head;
3. five-label sigmoid conversation-relation head.

The broad runtime lane is derived from the fine intent. It is not learned by a
separate head.

## Artifact Status

| Item | Value |
| --- | --- |
| Router repo | `spkc83/retail-bank-conversation-router` |
| Router revision | `c8f154266612e79afe20af8abef25761fa56d589` |
| Router dataset repo | `spkc83/retail-bank-conversation-router-data` |
| Dataset revision | `8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc` |
| Local data | `data/banking-conversation-router-v5-social-policy-generalization-candidate5` |
| Local artifact | `artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5` |
| Base encoder | `distilbert/distilbert-base-uncased` |
| Base revision | `12040accade4e8a0f71eabdb258fecc2e7e948be` |
| Artifact format | 3 |
| Maximum input length | 256 tokens |
| Visible exchanges | at most 3 |

## Architecture

[`ConversationRouterModel`](../src/hello_slm/banking_conversation_router.py)
encodes the complete rendered sequence once. The `[CLS]` representation feeds
three independent linear heads:

```text
rendered current turn + history + prior state
                |
         DistilBERT encoder
                |
         pooled [CLS] vector
          /        |        \
     domain      intent    relations
      2-way      12-way     5 sigmoid
```

Domain and intent use softmax. Relations use independent sigmoid probabilities
because multiple relations can be active on one turn.

## Labels and Derived Lanes

| Fine intent | Derived lane | Expected servicing action, if applicable |
| --- | --- | --- |
| `view_accounts` | servicing | `list_accounts` |
| `view_cards` | servicing | `list_cards` |
| `freeze_card` | servicing | `freeze_card` |
| `replace_card` | servicing | `replace_card` |
| `view_transactions` | servicing | `list_transactions` |
| `dispute_transaction` | servicing | `dispute_transaction` |
| `view_transfers` | servicing | `list_transfers` |
| `cancel_transfer` | servicing | `cancel_transfer` |
| `view_service_cases` | servicing | `list_service_cases` |
| `policy_knowledge` | policy | actions disabled |
| `conversation` | conversation | none predetermined |
| `other_banking` | other banking | none predetermined |

The action column documents the state machine's completion expectation. It
does not authorize that action. Granite still chooses the action and public
arguments.

The relation labels are:

| Relation | Meaning |
| --- | --- |
| `context_dependent` | The current turn needs visible prior conversation or state. |
| `agent_repair` | The customer corrects or challenges the previous answer. |
| `topic_shift` | The customer changes topic, including a possible external shift. |
| `clarification_answer` | The customer answers a clarification request. |
| `resume_previous_service` | The customer returns from a policy detour to the pending servicing task. |

## State-Aware Input

The serving renderer in
[`router.py`](../poc/retail-bank-customer-service-poc/router.py) and the training
renderer use the same order:

```text
[PRIOR_DIALOGUE_STATE]
{canonical JSON}
[CURRENT_USER]
{current text}
[PREVIOUS_ASSISTANT]
{most recent assistant text}
[PREVIOUS_USER]
{most recent user text}
...
```

State is trusted application state from before the current turn. It includes
only the bounded pending-service anchor and detour flag. The router never sees
the current expected action or answer.

## Calibration and Route Policy

The published V5 artifact uses:

| Threshold | Value |
| --- | ---: |
| in-domain banking | 0.50 |
| OOD banking boundary | 0.45 |
| relation rescue | 0.40 |
| `context_dependent` active | 0.15 |
| `agent_repair` active | 0.75 |
| `topic_shift` active | 0.85 |
| `clarification_answer` active | 0.90 |
| `resume_previous_service` active | 0.20 |

Routing uses the maximum probability of `context_dependent`, `agent_repair`,
`clarification_answer`, and `resume_previous_service` as the rescue score:

```text
banking >= 0.50
  -> in_domain

banking < 0.45 and rescue < 0.40
  -> out_of_domain

otherwise
  -> uncertain
```

`topic_shift` is diagnostic and does not rescue an otherwise OOD turn. This is
important for sequences such as a banking conversation followed by “What is
the weather?”

## How Runtime Uses the Outputs

- `out_of_domain` returns the fixed banking-scope response without Granite.
- `classifier_error` returns the model-failure response without Granite.
- `in_domain` exposes the top intent and relations to the bounded dialogue
  state machine.
- `uncertain` still reaches Granite, but it does not mutate pending dialogue
  state because the fine intent is not accepted.
- a confident `policy_knowledge` intent chooses the retrieval-grounded policy
  lane;
- a confident servicing intent starts, continues, or replaces one pending
  servicing task;
- `resume_previous_service` can restore the pending servicing lane after a
  policy detour and pin the original exchange.

Intent labels remain outside the Granite prompt. The router cannot provide an
action name or arguments to Granite, and the state machine does not treat a
classification as authorization.

## Held-Out Results

The generalized router passed its release gates on 6,171 test rows:

| Metric | Result |
| --- | ---: |
| Intent macro F1 | 0.990312 |
| Relation macro F1 | 0.996474 |
| OOD false-accept rate | 0.007899 |
| In-domain false-refusal rate | 0 |
| Resume-trajectory intent error | 0 |
| Resume-trajectory relation error | 0 |
| State-conditioned route error | 0 |
| State-conditioned intent error | 0 |
| Runtime transition error | 0 |
| State-conditioned non-resume false-positive rate | 0 |
| Held-out route error | 0 |
| Held-out intent error | 0 |
| Held-out relation error | 0 |
| Held-out social-generalization error | 0 |
| Held-out policy-follow-up-generalization error | 0 |

These are governed synthetic and CLINC-based test results, not a claim about
production banking traffic.

## Generate and Train

```bash
PYTHONPATH=src uv run python scripts/retail_bank/prepare_conversation_router_data.py \
  --sft-dir data/banking-servicing-alignment-v5 \
  --output-dir data/banking-conversation-router-v5-social-policy-generalization-candidate5 \
  --source-lock data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json \
  --expected-release-lock data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json

PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v5-social-policy-generalization-candidate5 \
  --output-dir artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5
```

The final publication used the exact dataset revision:

```bash
ROUTER_DATA_REVISION=8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc

PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v5-social-policy-generalization-candidate5 \
  --output-dir artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5 \
  --publish \
  --data-revision "$ROUTER_DATA_REVISION" \
  --destination-id spkc83/retail-bank-conversation-router
```

The publication command changes external state and should be run only for an
artifact that has passed the local gates.

## Verify

```bash
PYTHONPATH=src uv run pytest -q \
  tests/test_banking_conversation_router.py \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py \
  tests/test_banking_conversation_router_training.py \
  poc/retail-bank-customer-service-poc/tests/test_router.py \
  poc/retail-bank-customer-service-poc/tests/test_dialogue_state.py
```
