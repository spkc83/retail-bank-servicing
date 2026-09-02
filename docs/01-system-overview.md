# System Overview

Harbor is a model-driven retail-bank customer-service assistant for the
fictional Harborlight Bank. The system combines a hierarchical CPU router, bounded
deterministic dialogue state, versioned policy retrieval, a PEFT-adapted
Granite 8.79B model, and session-isolated fictional bank records.

## Component Boundaries

| Component | Responsibility | Must not do |
| --- | --- | --- |
| V6 router | Jointly score domain, lane, family, fine intent, five conversation relations, action disposition, and entity resolution from the current turn, recent visible history, and prior dialogue state. | Provide action arguments, execute a tool, or write customer-facing text. |
| Dialogue state | Retain at most one pending servicing task and whether a policy detour is active. | Authorize an action or invent a summary of the earlier request. |
| Policy retriever | Return current versioned policy chunks through deterministic lexical scoring. | Generate prose or execute a customer action. |
| Granite | Converse, clarify, choose public banking actions and arguments, and write grounded responses. | Bypass the action schema or cite policy chunks that retrieval did not supply. |
| Action/backend boundary | Validate and execute supported calls against session-isolated fictional state. | Expose private internal identifiers in the chat. |
| Response policy | Render exact read tables; validate action facts, policy citations, and internal-language exclusions. | Replace ordinary model responses with hard-coded customer-service scripts. |

## Router Inputs and Outputs

The router is a shared DistilBERT cross-encoder with seven classification
heads:

1. **Domain head:** `out_of_domain`, `banking`, or `social`.
2. **Lane head:** orchestration path such as `servicing`, `policy`, or `conversation`.
3. **Family head:** product grouping such as `cards` or `transactions`.
4. **Intent head:** 12 mutually exclusive fine intents.
5. **Relation head:** five independent sigmoid labels.
6. **Action head:** `execute_tool`, `clarify`, `retrieve_policy`, `converse`, or `refuse_ood`.
7. **Entity-resolution head:** `resolved`, `missing`, `ambiguous`, `ineligible`, or `not_required`.

The rendered input can contain:

```text
[PRIOR_DIALOGUE_STATE]
{"knowledge_detour_active":true,"pending_servicing":{...},"version":1}
[CURRENT_USER]
Let's continue with that dispute.
[PREVIOUS_ASSISTANT]
Savings interest uses the disclosed APY ...
[PREVIOUS_USER]
First, how does savings interest work?
```

Only visible prior user/assistant text and trusted pre-turn state are included.
Current-turn action plans, results, expected answers, and hidden labels are not
router input.

The 12 intents are:

```text
view_accounts       view_cards          freeze_card
replace_card        view_transactions   dispute_transaction
view_transfers      cancel_transfer     view_service_cases
policy_knowledge    conversation        other_banking
```

The five relations are:

```text
context_dependent   agent_repair         topic_shift
clarification_answer                     resume_previous_service
```

At runtime a constrained joint decoder selects the highest-scoring legal
domain/lane/family/intent/action/entity tuple. Raw independent-head candidates
remain diagnostics; incompatible combinations do not reach the harness.

## Bounded Dialogue State

[`dialogue_state.py`](../poc/retail-bank-customer-service-poc/dialogue_state.py)
stores this state per authenticated user and session:

```json
{
  "version": 1,
  "pending_servicing": {
    "intent": "dispute_transaction",
    "anchor_user_message": "Dispute the North Harbor Market purchase.",
    "anchor_assistant_message": "Which purchase would you like to dispute?",
    "phase": "awaiting_user"
  },
  "knowledge_detour_active": true
}
```

This is intentionally not a general task stack. It retains one servicing task,
allows one policy detour, and resets after the expected action succeeds or the
session is reset. A confident different servicing intent replaces the pending
task. Low-confidence, uncertain, or OOD observations do not mutate it.

When the router activates `resume_previous_service`, the state machine chooses
the servicing lane and supplies the original user/assistant exchange as a
pinned context group. The state machine does not create a synthetic summary;
Granite sees the actual earlier exchange.

## Policy Question During a Servicing Task

Consider this sequence:

```text
Customer: Dispute the North Harbor Market purchase.
Harbor:    I can help. What would you like to confirm first?
Customer: How does a card dispute investigation work?
Harbor:    ... [Policy: card.dispute.us.v1]
Customer: Thanks. Continue with the dispute.
```

The V5 path is:

1. The first turn starts a pending `dispute_transaction` state.
2. The policy question receives intent `policy_knowledge`; the state records an
   active knowledge detour without discarding the dispute.
3. [`policy_retrieval.py`](../poc/retail-bank-customer-service-poc/policy_retrieval.py)
   retrieves matching chunks from
   [`policy_knowledge.json`](../poc/retail-bank-customer-service-poc/policy_knowledge.json).
4. Granite receives only the policy evidence and no banking action schemas.
5. The visible answer must cite an allowed chunk as `[Policy: chunk_id]`.
6. The resume turn restores the servicing lane and pins the original dispute
   exchange into Granite's token-budgeted context.
7. Granite may emit the supported `dispute_transaction` action. The pending
   state clears only after that action succeeds.

If retrieval finds no eligible chunk, the runtime returns a policy-not-found
response and does not ask Granite to improvise an answer.

## Servicing and Action Loop

For `execute_tool` turns, Granite receives the Harborlight system prompt,
token-budgeted conversation, and exactly one intent-compatible public action
schema selected by the router disposition. The full supported set is:

```text
list_accounts        list_cards          list_service_cases
list_transactions    list_transfers      freeze_card
replace_card         dispute_transaction cancel_transfer
```

For an accepted V6 `execute_tool` route, Granite supplies every argument and
must emit exactly one tagged JSON call:

```text
<tool_call>{"name":"list_transactions","arguments":{"limit":5}}</tool_call>
```

The harness validates the name, arguments, and types before execution. If the
first pass emits prose, it retries Granite once without inventing arguments. It
then executes the one valid call against session-isolated state and returns the
correlated result to a no-tools grounded-final pass. Repeated calls are rejected.
The bounded multi-call chain remains only for legacy V3 compatibility.

Successful read-only lists are rendered by the harness as Markdown tables from
exact result fields. Action responses remain Granite-authored, but the harness
checks essential facts such as card ending digits, transaction description,
recipient, and successful outcome.

## Customer-Experience Validation

The model prompt establishes the Harborlight Bank and Harbor identity, asks for
warm concise wording, and requires empathy for distress. A final validator
rejects internal vocabulary such as `synthetic`, `demo`, `mock`, `model`,
`router`, `GPU`, `CPU`, or `tool`. One tools-disabled repair pass may rewrite
the response while preserving authoritative action results or policy
citations. A second invalid answer fails closed to the generic model-failure
response.

Branding and fictional-data notices remain in the UI. They must not leak into
the assistant's ordinary customer-facing answers.

## Runtime Deployments

| Runtime | Model loading | UI |
| --- | --- | --- |
| Hugging Face Space | BF16 base-plus-PEFT Granite inside `@spaces.GPU(size="large", duration=90)` | Gradio |
| Local CUDA | bitsandbytes NF4 double quantization, FP16 compute, device `cuda:0` | Streamlit |

Both paths share the router, dialogue state, policy retriever, Granite action
service, response policy, Harborlight branding helpers, and fictional bank
backend.

The hierarchical state-conditioned router is published at
`dd5ea26674a0f9808d42110a9ee51a9af6762a76` from dataset revision
`b33c27170e27cdb11783704ede14f7d25f70625e`. Granite inference composes base
`spkc83/retail-bank-servicing-agent-9b@1d568249...` with PEFT adapter
`spkc83/retail-bank-servicing-agent-9b-peft-v14-prompt-realized@47968b2b...`
(published with `adapter_config.json` under `adapter/`; set `RETAIL_BANK_ADAPTER_SUBFOLDER=adapter`).
The previously deployed adapter was `spkc83/retail-bank-servicing-agent-9b-peft-v11-alignment@03a7b446...`.
Merged candidates are not used because they failed the unchanged
behavioral-parity gates. Evaluation job `6a7f89edc97db76cbdf31893` (v7
composition) then failed strict behavioral gates. A corrected evaluator and
generalized incremental SFT are underway; no passing result is claimed here.
