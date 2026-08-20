# V6 Hierarchical Seven-Head Router

The filename is retained for stable links. The active router is neither dual
head nor three head. It is a DistilBERT cross-encoder with seven output heads
and a constrained joint decoder.

## Artifact Identity

| Item | Value |
| --- | --- |
| Router | `spkc83/retail-bank-conversation-router@dd5ea26674a0f9808d42110a9ee51a9af6762a76` |
| Dataset | `spkc83/retail-bank-conversation-router-data@b33c27170e27cdb11783704ede14f7d25f70625e` |
| Local artifact | `artifacts/banking-conversation-router-v8-first-turn-mutation` |
| Base | `distilbert/distilbert-base-uncased@12040accade4e8a0f71eabdb258fecc2e7e948be` |
| Format | 4 |
| Maximum input | 256 tokens; at most 3 visible exchanges |
| Selected epoch | 2 |
| Release status | eligible; no failed gates |

## Architecture

[`ConversationRouterModel`](../src/hello_slm/banking_conversation_router.py)
encodes the rendered turn once. Its pooled first-token representation feeds:

```text
                 shared DistilBERT encoder
                            |
                     pooled representation
       /          /         |        |       |        \          \
   domain       lane      family   intent  relation   action   entity resolution
   3-way        5-way      9-way   12-way  5 sigmoid  5-way        5-way
```

Domain, lane, family, intent, action, and entity resolution use categorical
logits. Relations use independent sigmoid logits because several relations may
be true on one turn.

## Canonical Taxonomy

[`banking_domain_taxonomy.py`](../src/hello_slm/banking_domain_taxonomy.py)
defines every legal intent hierarchy and action/entity combination.

| Intent | Domain / lane / family | Normal disposition |
| --- | --- | --- |
| `view_accounts` | banking / servicing / accounts | `execute_tool` -> `list_accounts` |
| `view_cards` | banking / servicing / cards | `execute_tool` -> `list_cards` |
| `freeze_card` | banking / servicing / cards | `execute_tool` -> `freeze_card` when resolved |
| `replace_card` | banking / servicing / cards | `execute_tool` -> `replace_card` when resolved |
| `view_transactions` | banking / servicing / transactions | `execute_tool` -> `list_transactions` |
| `dispute_transaction` | banking / servicing / transactions | `execute_tool` -> `dispute_transaction` when resolved |
| `view_transfers` | banking / servicing / transfers | `execute_tool` -> `list_transfers` |
| `cancel_transfer` | banking / servicing / transfers | `execute_tool` -> `cancel_transfer` when resolved |
| `view_service_cases` | banking / servicing / service_cases | `execute_tool` -> `list_service_cases` |
| `policy_knowledge` | banking / policy / policy | `retrieve_policy` |
| `conversation` | social / conversation / social | `converse` |
| `other_banking` | banking / other_banking / other_banking | `converse` |

Missing or ambiguous action targets produce `clarify`. Ineligible targets
produce `converse`. OOD produces `refuse_ood` and `not_required`.

## Joint Decoding

Independent argmax predictions can conflict, for example `cards` family with
`view_transactions`, or `execute_tool` with an ambiguous entity. V6 does not
pass those raw choices downstream.

`decode_v4_joint` enumerates legal taxonomy tuples, sums their head scores,
and returns the highest-scoring coherent tuple. It also records whether this
tuple differed from the independent head argmaxes. On the held-out set:

- raw independent-head incompatibility rate: `0.001018`;
- joint-decoder disagreement rate: `0.001221`;
- exposed hierarchy compatibility error rate: `0.0`.

The decoder constrains structure; it does not grant authorization or invent
tool arguments.

## State-Aware Input

Training and serving use the same rendering order:

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

State is trusted pre-turn application state. It contains at most one pending
servicing anchor plus the detour flag. Current-turn answers, target outputs,
tool results, and evaluation labels never enter this input.
The renderer omits the state header when both the pending task and detour are
absent; version-only/default state is not meaningful model context.

## Routing Thresholds

| Threshold | Value |
| --- | ---: |
| in-domain support | 0.50 |
| OOD support boundary | 0.45 |
| relation rescue | 0.40 |
| `context_dependent` | 0.40 |
| `agent_repair` | 0.60 |
| `topic_shift` | 0.70 |
| `clarification_answer` | 0.80 |
| `resume_previous_service` | 0.50 |

The route policy is:

```text
support >= 0.50                         -> in_domain
support < 0.45 and rescue < 0.40       -> out_of_domain
otherwise                               -> uncertain
```

`topic_shift` remains diagnostic and does not rescue an external turn.
Uncertain routes suppress intent, lane, family, action, and entity resolution;
the harness exposes no tools and asks Granite for one clarification.

## Action-Guided Generation

For a coherent in-domain tuple, the runtime consumes the action as follows:

| Router action | Harness behavior |
| --- | --- |
| `refuse_ood` | Return the fixed scope response; no Granite call. |
| `retrieve_policy` | Retrieve policy evidence and call Granite with no tools. |
| `execute_tool` | Map the intent to exactly one tool schema and expose only that schema. |
| `clarify` | Call Granite with no tools and ask one clarification. |
| `converse` | Call Granite with no tools for a concise natural response. |

The `generation_guidance_contract` is
`intent-selects-tool-schema-no-arguments-v1`. The router narrows the action
space but supplies no arguments. Granite chooses arguments from the visible
conversation and may ask for missing selectors rather than call the tool.

## Held-Out Results

The release passed all gates on 4,921 test rows:

| Metric | Result |
| --- | ---: |
| Domain macro F1 | 0.997533 |
| Lane macro F1 | 0.995155 |
| Family macro F1 | 0.994552 |
| Intent macro F1 | 0.996473 |
| Relation macro F1 | 0.961950 |
| Action macro F1 | 0.997865 |
| Entity-resolution macro F1 | 0.999060 |
| Exposed action macro F1 | 0.997996 |
| Exposed entity-resolution macro F1 | 0.999250 |
| OOD false-accept rate | 0.003232 |
| In-domain false-refusal rate | 0.000000 |
| Counterfactual action accuracy | 1.000000 |
| Counterfactual entity accuracy | 1.000000 |
| Counterfactual pair-flip accuracy | 1.000000 |
| Held-out route/intent/relation error | 0 / 0 / 0 |
| Trajectory runtime-transition error | 0.000000 |

These results measure governed synthetic and CLINC-based data. They are not a
production-traffic accuracy claim.

## Train and Verify

```bash
PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v8-first-turn-mutation \
  --output-dir artifacts/banking-conversation-router-v8-first-turn-mutation
```

```bash
PYTHONPATH=src uv run pytest -q \
  tests/test_banking_conversation_router.py \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py \
  tests/test_banking_conversation_router_training.py \
  poc/retail-bank-customer-service-poc/tests/test_router.py \
  poc/retail-bank-customer-service-poc/tests/test_dialogue_state.py
```
