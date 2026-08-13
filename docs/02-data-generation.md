# Data Generation

This guide explains how a product behavior becomes a validated training record.
It covers scenario design, generative SFT records, stage-2 remediation, router
records, split isolation, validation, and provenance.

For a single example traced through the whole system, read
[End-to-End Flow by Example](11-end-to-end-flow-by-example.md).

## The Three Dataset Surfaces

The repository has two learning lanes and three released dataset surfaces:

| Surface | Learner | Purpose |
| --- | --- | --- |
| Stage-1 tool-use SFT | Granite 9B | Broad conversation, tool syntax, grounding, clarification, FAQ, and OOD behavior. |
| Stage-2 servicing SFT | Granite 9B | Continue training on the full base corpus plus targeted POC-failure remediation. |
| Conversation-router data | DistilBERT router | Domain, servicing-capability, and conversation-relation classification. |

Do not mix the classifier rows into Granite SFT. Do not put target tool calls,
tool results, or final answers into router input.

The generative data is self-authored synthetic data under MIT. Checksum-pinned
CLINC data contributes only to classifier training. Historical Banking77 v1
scripts remain for reproducibility but are not the active POC router path.

## Design the Data From the End-User Experience

Fine-tuning data should describe decisions the assistant must make, not merely
contain domain vocabulary. Begin with the application contract:

1. What goals are supported?
2. Which goals need live customer data or an action?
3. Which public tool and arguments implement each goal?
4. When should the assistant answer, clarify, call a tool, or reject scope?
5. What backend success, empty, ambiguous, and error states can occur?
6. Which facts must the final answer preserve?

Example design for card replacement:

```yaml
goal: replace a card
tool: replace_card
public_argument: last4
clear_request: call the tool
missing_selector: ask which card when needed
contextual_request: resolve "the active one" from visible history
tool_error: explain the returned error without inventing success
private_request: never ask for or expose the full card number
```

Create scenario families around these branches. Wording variation is useful
only after the behavior, state, and expected result are correct.

### Coverage matrix example

| Dimension | Useful values |
| --- | --- |
| User wording | direct, polite, terse, typo, correction, pronoun |
| Conversation state | first turn, follow-up, clarification answer, topic shift |
| Backend state | one match, several matches, no match, already completed, error |
| Response path | direct FAQ, tool success, tool error, clarification, OOD |
| Tool pattern | no tool, one tool, ordered dependent tools |

Cross-product expansion should be intentional. Do not generate every possible
combination if most are meaningless or duplicate the same decision.

## Tool-Use SFT Data

The tool-use dataset trains the 8.79B model to converse, ask clarifying
questions, call tools, consume tool results, and write final answers.

Main files:

| File | Purpose |
| --- | --- |
| [../src/hello_slm/banking_tool_sft_data.py](../src/hello_slm/banking_tool_sft_data.py) | Generates records, validates invariants, writes manifests and cards. |
| [../scripts/retail_bank/prepare_tool_sft_data.py](../scripts/retail_bank/prepare_tool_sft_data.py) | CLI wrapper around the generator. |
| [../poc/retail-bank-customer-service-poc/synthetic_bank.json](../poc/retail-bank-customer-service-poc/synthetic_bank.json) | Seed customer/account/card/transaction/transfer/service-case state. |
| [../data/banking-v3-tool-sft/manifest.json](../data/banking-v3-tool-sft/manifest.json) | Local generated split manifest. |
| [../data/banking-v3-tool-sft/DATA_CARD.md](../data/banking-v3-tool-sft/DATA_CARD.md) | Local generated dataset card. |
| [../data_cards/retail-bank-agent-sft.md](../data_cards/retail-bank-agent-sft.md) | Public dataset card. |

### Generated split counts

The active public data card reports:

| Split | Records |
| --- | ---: |
| Train | 6,304 |
| Validation | 1,349 |
| Frozen test | 1,347 |
| Total | 9,000 |

The corpus fingerprint is
`2bb7a400ed2556b15c7e5eb6147668041b5deef8ae4f037f9e2e52295ff29ab5`.
The released split seed is `711`, which is also the generator default.

### Scenario coverage

The local generated card records these scenario-family counts:

| Scenario family | Conversations |
| --- | ---: |
| `tool_success` | 3,006 |
| `no_tool_banking_faq` | 1,665 |
| `multi_turn` | 1,665 |
| `conversation` | 999 |
| `tool_error` | 666 |
| `clarification` | 333 |
| `hard_negative` | 333 |
| `ood` | 333 |

The generator covers all nine public tools:

- `list_accounts`
- `list_cards`
- `list_service_cases`
- `list_transactions`
- `list_transfers`
- `cancel_transfer`
- `dispute_transaction`
- `freeze_card`
- `replace_card`

The canonical public tool schema is produced by `public_tool_manifest()` in
[../src/hello_slm/banking_tool_sft_data.py](../src/hello_slm/banking_tool_sft_data.py).
The same schema shape is used by the POC in
[../poc/retail-bank-customer-service-poc/model_service.py](../poc/retail-bank-customer-service-poc/model_service.py).

### Record structure

Each record contains:

- `record_id`: stable record identifier;
- `schema_version`: `banking-tool-sft/v1`;
- `messages`: system, user, assistant, and tool messages;
- `expected.ordered_calls`: expected tool-call IDs in order;
- `expected.tool_calls`: expected public tool names and arguments;
- `expected.requires_tool`: whether the row requires tool execution;
- `expected.path`: response path such as tool success, clarification, FAQ, or OOD;
- `expected.grounding_facts`: facts the final response must preserve;
- `split_keys`: stable values used for deterministic split assignment;
- `provenance.source`: `self-authored-synthetic`.

Tool-bearing records include assistant tool-call messages followed by correlated
tool-result messages. System, user, and tool-result messages are context only.
Assistant tool calls and final assistant responses are trainable.

### Worked record: account lookup

This shortened row is based on the generated `accounts_read` training record:

```json
{
  "record_id": "accounts_read",
  "schema_version": "banking-tool-sft/v1",
  "messages": [
    {
      "role": "system",
      "content": "You are the fictional retail-bank service agent...",
      "loss": false
    },
    {
      "role": "user",
      "content": "Please show the accounts available to me and their balances",
      "loss": false
    },
    {
      "role": "assistant",
      "content": null,
      "loss": true,
      "tool_calls": [
        {
          "id": "call_accounts_read_0",
          "index": 0,
          "type": "function",
          "function": {"name": "list_accounts", "arguments": {}}
        }
      ]
    },
    {
      "role": "tool",
      "name": "list_accounts",
      "tool_call_id": "call_accounts_read_0",
      "content": {
        "ok": true,
        "result": {
          "accounts": [
            {
              "name": "Main Checking",
              "last4": "1792",
              "available_balance_cents": 472825,
              "currency": "USD"
            }
          ]
        }
      },
      "loss": false
    },
    {
      "role": "assistant",
      "content": "Main Checking ending in 1792 has USD 4,728.25 available.",
      "loss": true
    }
  ],
  "expected": {
    "requires_tool": true,
    "path": "tool_success",
    "ordered_calls": ["call_accounts_read_0"],
    "tool_calls": [{"name": "list_accounts", "arguments": {}}],
    "grounding_facts": [
      "account.last4=1792",
      "account.balance=4,728.25"
    ]
  },
  "split_keys": {
    "scenario_family": "read_accounts",
    "state_seed": "state-accounts_read-0000",
    "template_id": "accounts-balance-v1"
  },
  "provenance": {
    "source": "self-authored-synthetic",
    "license": "MIT"
  }
}
```

The assistant tool call is a target because Granite must learn to produce it.
The tool result is context because the backend produces it at runtime. The final
assistant message is a target because Granite must learn grounded wording.

`expected.grounding_facts` lets evaluation check meaning without requiring one
exact sentence. `split_keys` keeps related state/template variants together.

### Worked record: clarification without a tool

```json
{
  "messages": [
    {"role": "user", "content": "Replace my card.", "loss": false},
    {
      "role": "assistant",
      "content": "Which card should I replace? You can give me its last four digits.",
      "loss": true
    }
  ],
  "expected": {
    "requires_tool": false,
    "path": "clarification",
    "ordered_calls": [],
    "tool_calls": []
  }
}
```

This row teaches a decision not to call a tool yet. A dataset containing only
successful calls encourages premature actions and invented arguments.

### Worked record: tool error

```text
user -> “Cancel my transfer to Taylor.”
assistant target -> cancel_transfer(recipient="Taylor")
tool context -> {"ok":false,"error":{"code":"transfer_not_pending"}}
assistant target -> explain that the transfer cannot be cancelled because it is not pending
```

Error examples teach the model to consume backend truth. They should not turn a
failed action into a success statement.

### How records become token labels

[`banking_tool_wire.py`](../src/hello_slm/banking_tool_wire.py) renders each
record with Granite's pinned chat template and assigns loss only to assistant
spans:

```text
system/user/tool-result/padding -> label -100
assistant tool call/final text  -> label equals token ID
```

The model therefore learns assistant behavior while conditioning on the user,
instructions, tool schemas, and tool results.

### Validation rules

`validate_records()` in
[../src/hello_slm/banking_tool_sft_data.py](../src/hello_slm/banking_tool_sft_data.py)
rejects records that violate release invariants, including:

- duplicate `record_id` values;
- unsupported provenance;
- duplicate normalized user text;
- unknown tools;
- unsupported tool arguments;
- missing or mismatched tool results;
- unstable tool-call IDs;
- semantically empty final responses;
- missing path-specific content for clarification, OOD, FAQ, and hard-negative
  rows.

The manifest validator, `validate_banking_tool_sft_manifest()`, re-reads split
files, checks record counts, and validates all records.

Validation is part of data generation, not a cleanup step after training. A
single broken correlation can teach the model that tool results belong to the
wrong call.

Example failures:

| Invalid row | Why it is rejected |
| --- | --- |
| Tool call ID is `call_1`, result references `call_2` | The model cannot learn a correlated tool exchange. |
| `replace_card` receives `account_id` | The argument is not in the public schema. |
| Final answer says “completed,” tool result says “pending” | The response contradicts backend truth. |
| Same normalized user text appears in multiple records | It inflates coverage with duplicates. |
| A test-only captured failure appears in training | It leaks the regression answer. |

### Split strategy and leakage control

Assign a scenario group to a split before expanding wording. Keep the customer
state, template family, and close variants in that split.

```text
group = scenario_family + state_seed + customer_seed + template_id
group -> deterministic hash -> train, validation, or test
```

Train teaches parameters. Validation selects checkpoints and thresholds. Test
is frozen evidence after those choices are complete.

Do not report test results while repeatedly editing data to pass the same test.
Turn a failure into a newly authored training family and preserve an independent
held-out realization.

### Validate dataset generation

From the repository root:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_tool_sft_data.py \
  --output-dir /tmp/retail-bank-tool-sft-check \
  --pilot-count 9000 \
  --split-seed 711
```

The command writes:

- `/tmp/retail-bank-tool-sft-check/train.jsonl`
- `/tmp/retail-bank-tool-sft-check/validation.jsonl`
- `/tmp/retail-bank-tool-sft-check/test.jsonl`
- `/tmp/retail-bank-tool-sft-check/manifest.json`
- `/tmp/retail-bank-tool-sft-check/preparation-report.json`
- `/tmp/retail-bank-tool-sft-check/README.md`
- `/tmp/retail-bank-tool-sft-check/DATA_CARD.md`

Use a smaller pilot when testing the generator quickly:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_tool_sft_data.py \
  --output-dir /tmp/retail-bank-tool-sft-smoke \
  --pilot-count 1200
```

`--pilot-count` must be at least the number of required base scenarios. The
full release uses `9000`. Use `data/banking-v3-tool-sft` as the output
directory only when intentionally refreshing the repository's generated local
copy.

### Optional teacher wording pass

The generator can export teacher-realization requests:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_tool_sft_data.py \
  --output-dir /tmp/retail-bank-tool-sft-teacher-check \
  --pilot-count 1200 \
  --export-teacher-requests /tmp/teacher-requests.jsonl
```

Teacher responses are wording-only. `import_teacher_realizations()` proves that
tool calls, tool results, expected ordered calls, final state hashes, grounding
facts, and split keys are unchanged by checking immutable hashes before and
after applying teacher text.

The published 9,000-row release did not use a teacher realization pass. If you
experiment with one, the realizer requires an explicitly selected model and an
immutable revision:

```bash
PYTHONPATH=src python scripts/retail_bank/realize_tool_sft_teacher.py \
  --input-requests /tmp/teacher-requests.jsonl \
  --output-responses /tmp/teacher-responses.jsonl \
  --model MODEL_REPOSITORY \
  --revision IMMUTABLE_40_CHARACTER_REVISION
```

Do not treat teacher-realized output as the released corpus unless it passes
the same invariants and is published under a new dataset revision.

Teacher generation is optional because a larger model can improve surface
variety while silently changing tool names, arguments, facts, or refusal scope.
This repo allows wording changes only after immutable structural fields match.

## Stage-2 Servicing-Remediation Data

Live POC testing exposed failures that broad stage-1 coverage did not fully
solve. Stage 2 keeps the complete stage-1 corpus and appends 427 targeted rows.

| Split | Stage-1 rows | Added remediation rows | Composite total |
| --- | ---: | ---: | ---: |
| Train | 6,304 | 320 | 6,624 |
| Validation | 1,349 | 80 | 1,429 |
| Test | 1,347 | 27 | 1,374 |

Example remediation chain:

```text
User: Show my service cases.
Assistant target: call list_service_cases
Tool context: a closed mailing-address update created at 2026-06-18T14:00:00Z
Assistant target: summarize the case
User: When was that created?
Assistant target: use visible context and the relevant tool-backed fact
```

The added rows target service-case references, card anaphora, clarification
answers, agent repair, and topic shifts. Exact captured wording stays test-only.

The full base corpus remains present to reduce catastrophic forgetting. A
correction-only dataset could fix service-case follow-ups while degrading
account, transfer, or transaction tools.

Generate the composite data with:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_servicing_alignment_data.py
```

See [Granite Servicing Alignment v4](10-servicing-alignment-v4.md) for the
released counts, hashes, and training relationship.

## Conversation-Router Data

The active v4 router dataset trains the CPU cross-encoder, not Granite. Each row
contains only information available at the moment a route decision is made.

Main files:

| File | Purpose |
| --- | --- |
| [../src/hello_slm/banking_conversation_router_data.py](../src/hello_slm/banking_conversation_router_data.py) | Builds leakage-safe history-aware rows and labels. |
| [../scripts/retail_bank/prepare_conversation_router_data.py](../scripts/retail_bank/prepare_conversation_router_data.py) | Downloads pinned external data and writes governed v4 splits. |
| [../data/banking-conversation-router-v4/manifest.json](../data/banking-conversation-router-v4/manifest.json) | Local generated router-data manifest. |
| [../data/sources/banking-conversation-router-v4.lock.json](../data/sources/banking-conversation-router-v4.lock.json) | Tracked release lock for sources and split digests. |
| [../data_cards/retail-bank-router-training-data.md](../data_cards/retail-bank-router-training-data.md) | Public router dataset card. |

### Sources

The preparation script combines:

- governed Granite SFT conversations for POC-aligned banking language;
- checksum-pinned UCI CLINC150 language for external OOD coverage;
- deterministic synthetic follow-ups, clarification answers, corrections,
  agent-repair turns, typos, and topic shifts.

It verifies source digests before writing splits. The former Banking77 v1
pipeline is retained as historical code but is not used by the released POC.

### Router split counts

The active router data card reports:

| Split | Rows |
| --- | ---: |
| Train | 61,759 |
| Validation | 13,173 |
| Test | 15,466 |

The prepared manifest records `pii_matches: 0` and `review_status:
automated-policy-pass`.

### Router record structure

Router examples use the text format from `render_router_input()` in
[../src/hello_slm/banking_conversation_router_data.py](../src/hello_slm/banking_conversation_router_data.py):

```text
[CURRENT_USER]
<current user turn>
[PREVIOUS_ASSISTANT]
<most recent visible assistant turn>
[PREVIOUS_USER]
<most recent visible user turn>
```

Up to three complete visible exchanges are included newest-first. The same
rendering policy is used by the POC.

Worked example:

```json
{
  "current_text": "When was that created?",
  "history": [
    {"role": "user", "content": "Show my service cases."},
    {
      "role": "assistant",
      "content": "You have a closed mailing-address update case."
    }
  ],
  "text": "[CURRENT_USER]\nWhen was that created?\n[PREVIOUS_ASSISTANT]\nYou have a closed mailing-address update case.\n[PREVIOUS_USER]\nShow my service cases.",
  "domain_label": 1,
  "capability": "service_cases",
  "capability_label": 5,
  "relation_labels": [1, 0, 0, 0],
  "example_kind": "contextual_followup",
  "source_split": "train",
  "group_id": "service-case-created-at|realization-04"
}
```

The relation vector follows the released label order:
`context_dependent`, `agent_repair`, `topic_shift`, and
`clarification_answer`.

The row must not contain the current turn's expected tool call, tool result,
grounding facts, or final response. Including them would leak the answer into
classifier training.

### Why history changes the label

“When was that created?” is ambiguous in isolation. With a prior service-case
answer it is supported banking and context-dependent. After a weather answer it
may be an external follow-up.

The cross-encoder sees current and prior text in one token sequence. It can
model token-level relationships rather than classifying the current sentence
alone.

### From probabilities to routes

The classifier produces probabilities. The runtime policy turns them into an
action:

| Example probabilities | Route | Action |
| --- | --- | --- |
| banking `0.82` | `in_domain` | Invoke Granite 9B. |
| banking `0.27`, rescue `0.18` | `uncertain` | Invoke Granite 9B. |
| banking `0.07`, rescue `0.61` | `uncertain` | Rescue the conversational turn and invoke Granite 9B. |
| banking `0.03`, rescue `0.08` | `out_of_domain` | Return the stock scope response. |

These numbers illustrate the released thresholds. They are not recorded model
outputs for the sample utterances.

### Validate router-data generation

From the repository root:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_conversation_router_data.py \
  --output-dir /tmp/retail-bank-conversation-router-v4
```

The command writes:

- `/tmp/retail-bank-conversation-router-v4/train.jsonl`
- `/tmp/retail-bank-conversation-router-v4/validation.jsonl`
- `/tmp/retail-bank-conversation-router-v4/test.jsonl`
- `/tmp/retail-bank-conversation-router-v4/manifest.json`
- `/tmp/retail-bank-conversation-router-v4/README.md`

By default, the script compares produced source and split digests with the
tracked v4 release lock. A mismatch is evidence that source data, generation
logic, or split assignment changed and must be reviewed.

## Preparing Data for a New End-User Domain

Use this sequence for another assistant:

1. Write the supported and OOD capability boundary.
2. Define public tools from the application API, not from model preferences.
3. Build deterministic synthetic backend states.
4. Author scenario families for direct, multi-turn, clarification, repair,
   tool-error, and OOD behavior.
5. Assign groups to splits before paraphrase expansion.
6. Validate schemas, correlations, facts, PII policy, provenance, and leakage.
7. Render with the exact base model chat template and verify assistant labels.
8. Train a small pilot and inspect behavior-level errors.
9. Expand coverage where the error taxonomy shows a real gap.
10. Freeze test data before the release candidate is selected.

More data is useful only when it adds coverage or robustness. Ten thousand near
duplicates can be less useful than several hundred carefully varied tool,
state, error, and conversation trajectories.

## Speech-recognition robustness overlays

Speech recognition adds filler words, truncation, punctuation loss, acoustic
confusions, and uncertainty. Do not use raw ASR text as both the input and the
label. Instead, bind each reviewed customer transcript to an existing validated
semantic SFT record and change only its latest user message.

Example:

```text
validated user text:  Cancel the River Consulting transfer.
reviewed ASR text:    can you cancel the river consulting transfer uh please
preserved target:     cancel_transfer({"recipient":"River Consulting"})
preserved outcome:    River Consulting transfer is cancelled
```

If ASR changes the recipient or loses the action, `semantic_match` must not be
approved. The row needs correction, a different semantic source, or a
clarification target. ASR alternatives, timestamps, confidence, audio hash,
model revision, consent, license, and reviewer identity remain provenance and
metadata; they are not inserted into the chat messages.

The implementation, full record contract, runnable example, quality gates,
and training handoff are documented in
[15-asr-to-sft-pipeline.md](15-asr-to-sft-pipeline.md).

## References

- [Transformers chat templates](https://huggingface.co/docs/transformers/en/chat_templating)
  explains model-specific message rendering.
- [Transformers tool use](https://huggingface.co/docs/transformers/en/chat_extras)
  explains tool schemas and tool-call messages.
- [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) documents
  conversational SFT formats, assistant-only loss, and PEFT integration.
- [Hugging Face Datasets processing](https://huggingface.co/docs/datasets/en/process)
  covers dataset transforms, splits, and shards.
- [PEFT LoRA guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
  explains the low-rank adapter method used by the Granite stages.

See [Learning Resources](reference/learning-resources.md) for calibration,
selective classification, evaluation, and primary-paper references.

## Tests

Focused data tests live in:

- [../tests/test_banking_tool_sft_data.py](../tests/test_banking_tool_sft_data.py)
- [../tests/test_banking_tool_sft_release.py](../tests/test_banking_tool_sft_release.py)
- [../tests/test_banking_conversation_router_data.py](../tests/test_banking_conversation_router_data.py)
- [../tests/test_banking_conversation_router_preparation.py](../tests/test_banking_conversation_router_preparation.py)
- [../tests/test_banking_servicing_alignment_data.py](../tests/test_banking_servicing_alignment_data.py)
- [../tests/test_banking_asr_sft_data.py](../tests/test_banking_asr_sft_data.py)

Run them from the repository root:

```bash
python -m pytest -q \
  tests/test_banking_tool_sft_data.py \
  tests/test_banking_tool_sft_release.py \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py \
  tests/test_banking_servicing_alignment_data.py \
  tests/test_banking_asr_sft_data.py
```
