# End-to-End Flow by Example

This guide follows concrete customer-service examples from product design to
data generation, fine-tuning, routing, tool execution, evaluation, and live
ZeroGPU inference.

Read this after [the system overview](01-system-overview.md). Use
[the runbook](08-end-to-end-runbook.md) when you are ready to execute the
pipeline.

## 1. The Mental Model

The system has two learned components with different jobs:

```text
conversation -> small classifier -> route decision
                              |
                              +-> high-confidence OOD: stock scope response
                              |
                              +-> in-domain or uncertain: Granite 9B
                                                       |
                                                       +-> direct answer
                                                       +-> clarification
                                                       +-> tool call -> SQLite
                                                                      |
                                                                      +-> result -> Granite 9B -> final answer
```

The classifier decides whether Granite should see the turn. Granite decides
what to say and whether to use a tool. The classifier's capability prediction
does not choose a tool or enter the generation prompt.

That separation matters. A classifier can recognize “card servicing,” but only
the generative model sees the tool schemas and learns how to emit a valid
`replace_card` call.

## 2. Start With a Behavior Contract

Before generating examples, describe the observable behavior. For a card
replacement request, the contract might be:

```yaml
user_goal: replace an active card
needed_context: which card the user means
allowed_tool: replace_card
public_arguments:
  last4: optional string
success_behavior: call the tool and summarize its result
ambiguous_behavior: ask which card
backend_error_behavior: explain the safe error returned by the tool
forbidden_behavior:
  - invent a replacement status
  - request a full card number
  - call an unknown tool
```

This contract becomes data scenarios, validators, evaluation cases, and live
test presets. It prevents training examples from drifting away from the actual
application interface.

## 3. Turn the Contract Into Scenario Families

One happy-path sentence is not enough. The same capability needs several
decision shapes:

| Scenario | User wording | Expected behavior |
| --- | --- | --- |
| Explicit request | “Replace card 4821.” | Call `replace_card(last4="4821")`. |
| Missing selector | “Replace my card.” | Ask which card if more than one is plausible. |
| Contextual selector | “Show my cards.” then “Replace the active one.” | Resolve the prior list and call the tool. |
| Tool error | “Replace card 9999.” | Call the tool, then explain the safe not-found result. |
| Hard negative | “Use the full card number from your database.” | Refuse the private-field request. |
| Topic shift | “Replace card 4821.” then “What is the weather?” | Route the second turn OOD. |

Variation should change wording, context, and backend state while preserving
the same product rule. Random paraphrases alone do not create new reasoning
coverage.

## 4. Build a Generative SFT Record

The stage-1 and stage-2 generative datasets use the
`banking-tool-sft/v1` record contract. This shortened example shows the parts
that affect learning and evaluation:

```json
{
  "record_id": "replace_active_card_example",
  "schema_version": "banking-tool-sft/v1",
  "messages": [
    {
      "role": "system",
      "content": "You are the fictional retail-bank service agent...",
      "loss": false
    },
    {
      "role": "user",
      "content": "Replace the active card ending in 4821.",
      "loss": false
    },
    {
      "role": "assistant",
      "content": null,
      "loss": true,
      "tool_calls": [
        {
          "id": "call_replace_0",
          "index": 0,
          "type": "function",
          "function": {
            "name": "replace_card",
            "arguments": {"last4": "4821"}
          }
        }
      ]
    },
    {
      "role": "tool",
      "name": "replace_card",
      "tool_call_id": "call_replace_0",
      "content": {
        "ok": true,
        "result": {
          "card": {"last4": "4821", "status": "replacement_pending"},
          "simulated": true
        }
      },
      "loss": false
    },
    {
      "role": "assistant",
      "content": "A replacement for the card ending in 4821 is pending.",
      "loss": true
    }
  ],
  "expected": {
    "requires_tool": true,
    "path": "tool_success",
    "ordered_calls": ["call_replace_0"],
    "tool_calls": [
      {"name": "replace_card", "arguments": {"last4": "4821"}}
    ],
    "grounding_facts": [
      "card.last4=4821",
      "card.status=replacement_pending"
    ]
  },
  "split_keys": {
    "scenario_family": "replace_card",
    "state_seed": "state-card-0042",
    "template_id": "replace-explicit-v1"
  },
  "provenance": {
    "source": "self-authored-synthetic",
    "license": "MIT"
  }
}
```

The `messages` array teaches the model's behavior. The `expected` block is for
validation and scoring; it is not inserted into the generation prompt.

The tool result is context, not a target. The model learns to emit the tool call
and final answer. It does not learn to fabricate the backend payload.

## 5. Render the Record for Granite

The raw JSON is not fed directly to the model. The tokenizer's Granite chat
template converts messages and tool schemas into model-specific tokens.

The training adapter then creates labels:

```text
system tokens       -> -100, ignored by loss
user tokens         -> -100, ignored by loss
assistant tool call -> token IDs, included in loss
tool-result tokens  -> -100, ignored by loss
assistant final     -> token IDs, included in loss
padding             -> -100, ignored by loss
```

This is assistant-only SFT. It teaches Granite what the assistant should
produce while keeping instructions, customer language, and database results as
conditioning context.

The adapter keeps a whole user-to-final-assistant chain. It will not truncate
between a generated tool call and its correlated tool result.

## 6. Build the Matching Router Record

The router is trained on what is visible before the current assistant response.
It must not see the target tool call, tool result, or final answer.

```json
{
  "text": "[CURRENT_USER]\nReplace the active one.\n[PREVIOUS_ASSISTANT]\nYou have an active card ending in 4821.\n[PREVIOUS_USER]\nShow my cards.",
  "current_text": "Replace the active one.",
  "history": [
    {"role": "user", "content": "Show my cards."},
    {"role": "assistant", "content": "You have an active card ending in 4821."}
  ],
  "domain_label": 1,
  "capability": "card_actions",
  "capability_label": 2,
  "relation_labels": [1, 0, 0, 0],
  "example_kind": "contextual_followup",
  "source": "self-authored-router-v4-use-case-alignment",
  "source_split": "train",
  "group_id": "card-0042|replace-active|realization-03"
}
```

`domain_label=1` means supported retail banking. The relation vector follows
the artifact's label order and marks this row as context-dependent.

The same customer behavior therefore creates two records:

| Learner | Input available | Target |
| --- | --- | --- |
| Router | current turn plus visible history | domain, capability, relations |
| Granite | system, history, tools, user turn, and later tool result | tool call and final answer |

## 7. Split Before You Expand

Closely related variants must stay in one split. Otherwise, the test set may be
only a paraphrase of training data.

This project uses stable grouping fields such as scenario family, state seed,
customer seed, and template ID. The group is assigned to train, validation, or
test before natural-language variants are expanded.

Bad split:

```text
train: “Replace the active card.”
test:  “Please replace my active card.”
```

Better split:

```text
train group: card replacement with one active and one frozen card
test group:  replacement after a prior card list with a different state seed
```

Validation supports model selection and threshold calibration. Frozen test data
is used only after choices are fixed.

## 8. Run the Two Granite SFT Stages

Stage 1 starts from the pinned pretrained IBM Granite checkpoint. It teaches
the complete synthetic-bank tool protocol and broad behavior families.

Stage 2 starts from the stage-1 checkpoint. It retains the entire stage-1 corpus
and adds targeted examples for failures observed in the POC.

```text
IBM Granite base
  -> stage 1: 9,000 general tool/conversation records
  -> stage-1 tool-trained checkpoint
  -> stage 2: base corpus + 427 servicing-remediation records
  -> released servicing agent
```

This is one training pipeline with two sequential SFT stages. Stage 2 is not a
separate product or an unrelated model.

The full base corpus remains in stage 2 to reduce catastrophic forgetting. A
small correction-only run could improve one behavior while damaging tool calls
that previously worked.

## 9. Train and Calibrate the Router

The router passes the combined current turn and history through one DistilBERT
cross-encoder. Three heads read the shared representation:

```text
shared encoder
  -> 2-class domain softmax
  -> 8-class capability softmax
  -> 4 independent relation sigmoids
```

The runtime uses two domain thresholds and one rescue threshold:

```text
if banking_probability >= 0.50:
    route = in_domain
elif banking_probability < 0.10 and rescue_probability < 0.40:
    route = out_of_domain
else:
    route = uncertain
```

The numbers below are illustrative inputs to the released policy, not recorded
predictions for the sample text:

| Banking probability | Rescue probability | Route | Runtime action |
| ---: | ---: | --- | --- |
| `0.86` | `0.08` | `in_domain` | Invoke Granite 9B. |
| `0.28` | `0.12` | `uncertain` | Invoke Granite 9B. |
| `0.08` | `0.62` | `uncertain` | Rescue the likely follow-up and invoke Granite 9B. |
| `0.04` | `0.10` | `out_of_domain` | Return the stock scope response. |

`uncertain` is a classifier abstention, not an application failure. The policy
prefers one extra model call over incorrectly blocking a valid follow-up.

## 10. Evaluate Before Publishing

For a tool-use record, frozen evaluation is model-owned but backend-safe:

1. Give Granite the prompt before the expected assistant tool call.
2. Parse the generated call without repairing it.
3. If it exactly matches the expected call, append the canonical tool result.
4. Ask Granite for the grounded final answer.
5. Score tool name, arguments, ordering, execution trajectory, and facts.

The evaluator does not execute live tools. It replays canonical results so the
same checkpoint and dataset revision produce comparable evidence.

For the replacement example, a release pass requires:

```text
generated tool name      == replace_card
generated arguments      == {"last4":"4821"}
call order               == [call_replace_0]
final answer contains    == last4 4821 and pending status
malformed/private fields == none
```

Loss and token accuracy alone cannot prove these behaviors. Exact behavioral
metrics are the release gate.

## 11. Live Inference Examples

### Example A: clear in-domain read

```text
User: Show my service cases.
Router: banking=0.93 -> in_domain
Granite pass 1: <tool_call>{"name":"list_service_cases","arguments":{}}</tool_call>
SQLite: returns the signed-in synthetic customer's cases
Granite pass 2: You have a closed mailing-address update case.
```

Diagnostics show two 9B passes, the exact model revision, CUDA device, generated
call, and tool result.

### Example B: uncertain contextual follow-up

```text
User: When was that created?
History: prior assistant described a mailing-address service case
Router: banking=0.28, context_dependent=0.67 -> uncertain
Runtime: invokes Granite because uncertain is an accepted route
Granite: uses retained conversation and may call list_service_cases
Final: answers from the tool result's created_at value
```

The numeric values illustrate the policy. Actual values appear in the live
diagnostics panel.

The capability prediction remains diagnostic. Granite receives normal retained
history, not a router-authored intent hint.

### Example C: mid-range without relation rescue

```text
User: Can you help with that charge?
Router: banking=0.31, rescue=0.18 -> uncertain
Runtime: invokes Granite
Granite: uses conversation history to answer, ask a clarification, or call a tool
```

The middle band is uncertain regardless of whether rescue is active. Rescue is
most important when banking probability falls below the OOD boundary.

### Example D: high-confidence OOD

```text
User: What will the weather be tomorrow?
Router: banking=0.03, rescue=0.06 -> out_of_domain
Runtime: returns the governed scope response
Granite calls: 0
```

### Example E: classifier failure

```text
Router raises an exception
Route: classifier_error
Runtime: returns the visible model-failure response
Granite calls: 0
```

A classifier exception is not treated as ordinary uncertainty. This keeps an
infrastructure failure visible during the experiment.

## 12. Adapting the Pipeline to Another Domain

For an airline assistant, do not begin by replacing “account” with “flight.”
Repeat the design process:

1. List supported user goals and unsupported boundaries.
2. Define public tools and schemas that match the real application contract.
3. Create backend states for success, ambiguity, empty results, and errors.
4. Generate direct, multi-turn, clarification, repair, and OOD scenarios.
5. Split by scenario/state groups before paraphrase expansion.
6. Render through the chosen base model's own chat and tool template.
7. Mask loss to assistant targets.
8. Train, calibrate, and evaluate with task-level metrics.
9. Feed live failures back as newly authored families, not copied test rows.

The durable asset is not a large pile of text. It is the alignment between the
application contract, training records, validators, frozen tests, and runtime
protocol.

## 13. Code Traceability

| Flow step | Primary implementation |
| --- | --- |
| Stage-1 record generation | [`banking_tool_sft_data.py`](../src/hello_slm/banking_tool_sft_data.py) |
| Stage-2 remediation generation | [`banking_servicing_alignment_data.py`](../src/hello_slm/banking_servicing_alignment_data.py) |
| Router record generation | [`banking_conversation_router_data.py`](../src/hello_slm/banking_conversation_router_data.py) |
| Granite rendering and masking | [`banking_tool_wire.py`](../src/hello_slm/banking_tool_wire.py) |
| Granite training | [`cloud_train_tool_sft.py`](../scripts/retail_bank/cloud_train_tool_sft.py) |
| Router training and calibration | [`train_conversation_router.py`](../scripts/retail_bank/train_conversation_router.py) |
| Frozen behavior scoring | [`banking_tool_eval.py`](../src/hello_slm/banking_tool_eval.py) |
| Runtime route gate | [`app.py`](../poc/retail-bank-customer-service-poc/app.py) |
| Model/tool loop | [`model_service.py`](../poc/retail-bank-customer-service-poc/model_service.py) |
| Synthetic tool backend | [`mock_bank.py`](../poc/retail-bank-customer-service-poc/mock_bank.py) |
| Canonical orchestration | [`run_release_pipeline.py`](../scripts/retail_bank/run_release_pipeline.py) |

## 14. Further Reading

See [the annotated learning references](reference/learning-resources.md) for
official documentation on chat templates, tool use, SFT, PEFT/LoRA, dataset
processing, classifier calibration, threshold tuning, and selective routing.
