# System Overview

This repository builds a model-driven synthetic retail-bank service agent. The
active release has three runtime pieces:

- A CPU history-aware router for domain/OOD gating, servicing-capability
  diagnostics, and conversation-relation scoring.
- A Granite 8.79B generative agent fine-tuned with PEFT/LoRA.
- A synthetic SQLite banking backend wrapped by the Gradio/ZeroGPU POC.

All customer data is fictional. No file in this repository connects to a real
bank.

## Runtime Flow

```text
Authenticated synthetic customer
  -> CPU history-aware router
  -> high-confidence OOD: governed scope response
  -> in-domain or uncertain: ZeroGPU Granite 8.79B generation
  -> direct answer, clarification, or tagged-JSON tool call
  -> synthetic SQLite tool execution
  -> tool result returned to the model
  -> deterministic read table or validated model-authored response
```

The released architecture is documented across these files:

| Step | Code |
| --- | --- |
| Gradio event and OOD shortcut | [../poc/retail-bank-customer-service-poc/app.py](../poc/retail-bank-customer-service-poc/app.py) |
| History-aware router loading and prediction | [../poc/retail-bank-customer-service-poc/router.py](../poc/retail-bank-customer-service-poc/router.py) |
| Model/tool loop | [../poc/retail-bank-customer-service-poc/model_service.py](../poc/retail-bank-customer-service-poc/model_service.py) |
| ZeroGPU model loading and deterministic decoding | [../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| Synthetic bank state and tool execution | [../poc/retail-bank-customer-service-poc/mock_bank.py](../poc/retail-bank-customer-service-poc/mock_bank.py) |

## Component Responsibilities

### History-aware router

The router is a shared DistilBERT cross-encoder with:

- a binary supported-banking/OOD head;
- a coarse servicing-capability diagnostic head;
- a multi-label conversation-relation head.

The POC loads the artifact from `spkc83/retail-bank-conversation-router` at
revision `9e090c0fa21cebbaa03a431a7ce61e656c0739fe`. Its released thresholds
are recorded in
[the router model card](../model_cards/retail-bank-domain-intent-router.md):

- banking probability below `0.10` with no relation rescue: high-confidence OOD;
- banking probability at least `0.50`: in-domain;
- the middle region, or rescued relation turns: uncertain.

High-confidence OOD requests receive the static governed response from
[../poc/retail-bank-customer-service-poc/responses.py](../poc/retail-bank-customer-service-poc/responses.py).
Uncertain requests continue to the 8.79B model. Capability and relation outputs
are diagnostics only; they are not added to the prompt and they do not choose
tools.

### Granite generative agent

The POC loads `spkc83/retail-bank-servicing-agent-9b` at revision
`1d56824995aa1adecfe20f62ca42fb1c0c443817` by default in
[../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py).
The model uses deterministic generation:

- `do_sample=False`;
- max new tokens `512`;
- FP16 weights on CUDA in the live Space.

The model receives the system prompt, retained conversation history, and the
nine public tool schemas. It can answer directly or emit Granite tagged-JSON
tool calls.

### Tool loop

[../poc/retail-bank-customer-service-poc/model_service.py](../poc/retail-bank-customer-service-poc/model_service.py)
owns the model/tool protocol:

1. Build a token-budgeted prompt with an 8,192-token input budget.
2. Ask the model for a first response.
3. Parse `<tool_call>...</tool_call>` blocks when present.
4. Validate tool names, argument types, argument ranges, call IDs, and indexes.
5. Execute each tool against the session-isolated synthetic bank.
6. Append tool results to the conversation.
7. Ask the same model for the grounded final response.
8. Repeat only if the model emits another valid tool call, up to eight total
   tool calls.
9. Render successful read-list results directly from backend fields, or check
   essential action selectors and outcomes in the model answer.
10. If an action answer fails that check, allow one text-only repair with tools
    disabled; never repair a generated tool name or argument.

The runtime does not infer intent, repair malformed calls, rename tools, or fill
missing arguments. Deterministic list rendering presents exact tool data; it
does not infer an action or substitute another banking backend.

### Synthetic bank backend

[../poc/retail-bank-customer-service-poc/mock_bank.py](../poc/retail-bank-customer-service-poc/mock_bank.py)
creates a SQLite database per authenticated user/session pair. It supports:

- read tools: `list_accounts`, `list_cards`, `list_service_cases`,
  `list_transactions`, `list_transfers`;
- write tools: `cancel_transfer`, `dispute_transaction`, `freeze_card`,
  `replace_card`.

The seed records live in
[../poc/retail-bank-customer-service-poc/synthetic_bank.json](../poc/retail-bank-customer-service-poc/synthetic_bank.json).
Each browser session gets isolated state, so a write action in one demo session
does not modify another session.

## Worked Runtime Turn

Consider this two-turn conversation:

```text
User: Show my cards.
Assistant: You have an active card ending in 4821.
User: Replace the active one.
```

The router encodes the current turn with the visible exchange. If its banking
score is high, the route is `in_domain`. If the score is in the middle band or
the relation head rescues the follow-up, the route is `uncertain`.

Both routes invoke Granite. Granite receives token-budgeted conversation
history and the public tool schemas. It may emit:

```text
<tool_call>{"name":"replace_card","arguments":{"last4":"4821"}}</tool_call>
```

The runtime validates the call, executes it against the session SQLite state,
and returns the result to Granite. Granite then writes the visible answer.

If the banking score is below `0.10` and no relation rescue reaches `0.40`, the
route is `out_of_domain`; the stock scope response is returned with zero 9B
generation calls.

See [End-to-End Flow by Example](11-end-to-end-flow-by-example.md) for the same
behavior traced through data preparation, training, evaluation, and serving.

## Training-Time Counterparts

Runtime behavior mirrors the SFT and evaluation code:

| Runtime behavior | Training/evaluation counterpart |
| --- | --- |
| Public tool schema | `public_tool_manifest()` in [../src/hello_slm/banking_tool_sft_data.py](../src/hello_slm/banking_tool_sft_data.py) |
| Granite tagged-JSON parsing | [../src/hello_slm/banking_tool_wire.py](../src/hello_slm/banking_tool_wire.py) |
| Assistant-only targets | `ToolWireAdapter.render_training()` in [../src/hello_slm/banking_tool_wire.py](../src/hello_slm/banking_tool_wire.py) |
| Frozen tool/final-response scoring | [../src/hello_slm/banking_tool_eval.py](../src/hello_slm/banking_tool_eval.py) |
| Router architecture | [../src/hello_slm/banking_conversation_router.py](../src/hello_slm/banking_conversation_router.py) |

## Failure Behavior

The POC is explicit about failure boundaries:

- Explicit local router-skip mode marks the route uncertain. A normal-turn
  classifier exception reports `classifier_error` and does not invoke the 8.79B
  model, so a classifier outage cannot masquerade as a valid experiment.
- If ZeroGPU allocation or generation fails, the UI reports model
  unavailability.
- If the model emits malformed tool syntax or invalid tool arguments, the UI
  reports a model failure and exposes diagnostics.
- The application does not substitute a CPU-authored banking answer for a
  failed model response.

See [../poc/retail-bank-customer-service-poc/README.md](../poc/retail-bank-customer-service-poc/README.md)
for the public POC card and live verification expectations.
