# V6 Hierarchical Seven-Head Router

The router is the component that decides what a customer turn *is* before the
9B model is asked to do anything about it. It is a DistilBERT cross-encoder
with seven output heads and a constrained joint decoder, and it runs on CPU in
under a tenth of a second. Its decision narrows what the generator is allowed
to do — which single tool schema it may see, whether it should ask instead of
act, whether it should answer at all — but it never supplies tool arguments and
never authorises anything.

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

A first turn with no state renders to almost nothing:

```text
[CURRENT_USER]
Show my account balances.
```

A turn that resumes a task after a policy detour renders the whole context the
router needs to recognise the resumption. The customer asked to replace a card,
was asked which one, asked a policy question instead, and now says "go ahead":

```text
[PRIOR_DIALOGUE_STATE]
{"knowledge_detour_active":true,"pending_servicing":{"anchor_assistant_message":"Which card would you like to replace?","anchor_user_message":"Please replace my debit card.","intent":"replace_card","phase":"awaiting_customer"},"version":1}
[CURRENT_USER]
Okay, go ahead with the replacement.
[PREVIOUS_ASSISTANT]
A replacement card usually arrives within 7-10 business days. [Policy: card.replacement.us.v1]
[PREVIOUS_USER]
Actually, how long does a replacement take to arrive?
```

Nothing in the current turn names a card. The pending anchor is what lets the
router decode `replace_card` / `execute_tool` / `resolved` and raise
`resume_previous_service`, which the harness uses to restore the original
exchange into the generator's context.

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

## Worked Examples

Every row below is the shipped artifact's actual output for that input, run
through `LearnedBankingRouter.classify` and `router_diagnostic_fields`. The
last column is the tool schema the harness exposes to the 9B model as a result
— at most one, and often none.

| Customer turn | Context | domain / lane / intent | action / entity | Exposed tool |
| --- | --- | --- | --- | --- |
| "Show my account balances." | first turn | banking / servicing / `view_accounts` | `execute_tool` / `not_required` | `list_accounts` |
| "My card ending in 4821 was stolen. Freeze it." | first turn | banking / servicing / `freeze_card` | `execute_tool` / `resolved` | `freeze_card` |
| "Please replace my debit card." | first turn | banking / servicing / `replace_card` | `clarify` / `missing` | none |
| "Freeze that one." | after a card list showing one card | banking / servicing / `freeze_card` | `execute_tool` / `resolved` | `freeze_card` |
| "What is the timeline for a dispute review?" | first turn | banking / policy / `policy_knowledge` | `retrieve_policy` / `not_required` | none |
| "Actually, how long does a replacement take to arrive?" | mid card-replacement | banking / policy / `policy_knowledge` | `retrieve_policy` / `not_required` | none |
| "Okay, go ahead with the replacement." | pending `replace_card` anchor after a detour | banking / servicing / `replace_card` | `execute_tool` / `resolved` | `replace_card` |
| "No, I said the pending transfer, not a card." | assistant had asked about a card | banking / servicing / `cancel_transfer` | `clarify` / `ineligible` | none |
| "Hello, how are you?" | first turn | social / conversation / `conversation` | `converse` / `not_required` | none |
| "What is the weather tomorrow?" | first turn | out_of_domain | `refuse_ood` / `not_required` | none (no 9B call) |
| "Can you help me open a mortgage account?" | first turn | banking / policy / `policy_knowledge` | `retrieve_policy` / `not_required` | none |

Three of these show the parts of the design that matter most.

**A reference resolves against the visible conversation.** "Freeze that one."
carries no card number. The router sees the previous assistant turn listing a
single card ending 4821, raises `context_dependent` (0.99) and
`clarification_answer` (0.94), and decodes a *resolved* freeze. The harness
then exposes exactly one schema, `freeze_card`, and the 9B model is left to read
"4821" out of the conversation and supply it as the argument. The same words
after a list of two cards decode to `ambiguous`, which exposes no tool.

**A missing target asks instead of acting.** "Please replace my debit card."
is a mutation intent with no identified card. Entity resolution decodes
`missing`, which forces `clarify` and exposes nothing, so the generator can only
ask which card. Compare the first row: `view_accounts` is a read of everything,
so `not_required` is a legitimate state for `execute_tool` and `list_accounts`
is exposed.

**The joint decoder corrects the heads.** "No, I said the pending transfer, not
a card." is an agent-repair turn. The independent heads disagree with each
other here, and the joint decoder resolves them to `cancel_transfer` with an
`ineligible` target — the customer is correcting the assistant, not naming a
transfer — and the mutation-intent-cannot-converse constraint holds the action
at `clarify`. Both corrections are recorded in `constraint_diagnostics`:

```text
constraint:joint-decoder-resolved-independent-head-conflict
constraint:mutation-intent-cannot-converse
```

The mortgage question is worth reading as a boundary case rather than a
success. The router does not refuse it; it routes it as a policy question,
which is defensible — it is a banking question — and the harness then
retrieves policy, finds nothing in the corpus about mortgages, and answers with
the governed "no approved policy" response. The scope boundary is enforced one
layer down, by the absence of evidence, not by the router.

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

---

*This file keeps its historical name for stable links. The dual-head design it
once described is retired; its code remains in `banking_dual_head_router.py` and
`train_dual_head_router.py`.*
