# V5 Data Generation

V5 uses three related datasets. Generate them in order because each downstream
dataset records the digest of its upstream source.

| Dataset | Consumer | Local directory | Published revision |
| --- | --- | --- | --- |
| Base tool SFT | Granite behavior foundation | `data/banking-v5-tool-sft` | local governed source for the composite dataset |
| Composite servicing alignment | Granite V5 continuation SFT and in-domain router examples | `data/banking-servicing-alignment-v5` | `spkc83/retail-bank-servicing-alignment-sft@40a0b68b9f746131ffff32a83e077fd7e4a344d1` |
| Conversation router | CPU DistilBERT cross-encoder | `data/banking-conversation-router-v5-social-policy-generalization-candidate5` | `spkc83/retail-bank-conversation-router-data@8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc` |

Do not mix classifier labels into Granite input. Do not put the current turn's
expected action, action result, or final answer into router text.

## 1. Base Tool-Use SFT

[`banking_tool_sft_data.py`](../src/hello_slm/banking_tool_sft_data.py)
creates deterministic, self-authored records for:

- direct banking conversation and clarification;
- nine supported read and write actions;
- successful and failed action results;
- multi-turn and dependent-action sequences;
- banking policy questions with citation targets;
- banking-scope refusal for external OOD prompts;
- Harborlight Bank branding, concise tables, empathy, and customer-facing
  wording.

Generate it with:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/prepare_tool_sft_data.py \
  --output-dir data/banking-v5-tool-sft
```

The committed manifest reports 838 training, 181 validation, and 181 test
records.

### Record structure

Each JSONL row has these top-level fields:

| Field | Purpose |
| --- | --- |
| `record_id` | Stable scenario identifier. |
| `schema_version` | `banking-tool-sft/v1`. |
| `messages` | Ordered system, user, assistant, and action-result messages. |
| `expected` | Evaluation truth: path, calls, grounding facts, and final state hash. |
| `metadata` | Split, scenario family, customer fixture, and training eligibility. |
| `split_keys` | Grouping keys that keep related realizations in one split. |
| `provenance` | Generator, source, license, and optional teacher metadata. |
| `validation` | Acceptance result, replay hash, and action-manifest hash. |

A simplified account-read record looks like this:

```json
{
  "messages": [
    {"role": "system", "content": "You are Harbor ...", "loss": false},
    {"role": "user", "content": "Show my accounts and balances.", "loss": false},
    {
      "role": "assistant",
      "content": null,
      "loss": true,
      "tool_calls": [{
        "id": "call_accounts_read_0",
        "index": 0,
        "type": "function",
        "function": {"name": "list_accounts", "arguments": {}}
      }]
    },
    {
      "role": "tool",
      "name": "list_accounts",
      "tool_call_id": "call_accounts_read_0",
      "content": {"ok": true, "result": {"accounts": ["..."]}},
      "loss": false
    },
    {
      "role": "assistant",
      "content": "| Account | Ending | Available | Current |\n|---|---:|---:|---:|\n...",
      "loss": true
    }
  ],
  "expected": {
    "path": "tool_success",
    "requires_tool": true,
    "tool_calls": [{"name": "list_accounts", "arguments": {}}],
    "grounding_facts": ["accounts.count=2"]
  },
  "metadata": {"scenario_family": "read_accounts", "split": "train"}
}
```

`loss: true` marks the assistant tokens that the trainer should learn. System,
user, and action-result tokens remain context and receive ignored labels.
`expected` is not fed to Granite; evaluation uses it after generation.

## 2. Composite Servicing-Alignment SFT

[`banking_servicing_alignment_data.py`](../src/hello_slm/banking_servicing_alignment_data.py)
adds V5 scenarios to the base corpus and keeps the complete base set in every
corresponding split. The augmentation targets the observed failure space:

- card references and clarification answers;
- service-case follow-ups;
- agent repair after wrong or repetitive answers;
- banking and external topic shifts;
- retrieval-grounded policy answers;
- policy detours during a servicing task;
- explicit return to the pending servicing task;
- held-out screenshot regressions in evaluation only.

Generate the composite dataset with:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/prepare_servicing_alignment_data.py \
  --base-sft-dir data/banking-v5-tool-sft \
  --output-dir data/banking-servicing-alignment-v5
```

The committed manifest reports:

| Split | Base rows | V5 alignment rows | Composite rows |
| --- | ---: | ---: | ---: |
| train | 838 | 384 | 1,222 |
| validation | 181 | 96 | 277 |
| test | 181 | 35 | 216 |

### Policy-answer example

A policy record supplies authoritative context as a non-loss system message
and trains only the cited assistant answer:

```json
{
  "messages": [
    {"role": "system", "content": "You are Harbor ...", "loss": false},
    {
      "role": "system",
      "content": "[Policy: deposit.account.opening.us.v1] Deposit account opening: ...",
      "loss": false
    },
    {"role": "user", "content": "How would I open a savings account?", "loss": false},
    {
      "role": "assistant",
      "content": "Choose the product, review disclosures, complete identity verification, and provide opening funds when required [Policy: deposit.account.opening.us.v1].",
      "loss": true
    }
  ],
  "expected": {
    "path": "retrieval_grounded_policy",
    "requires_tool": false,
    "policy_citations": ["deposit.account.opening.us.v1"]
  }
}
```

The runtime policy corpus uses a separately versioned JSON file and may use
different chunk IDs as that corpus evolves. Training teaches the behavior:
answer only from supplied evidence and cite the exact supplied ID.

### Policy-detour and resume example

The alignment data also teaches this sequence:

```text
user:      Dispute the North Harbor Market purchase.
assistant: I can help with that dispute. What would you like to know first?
user:      First, how does savings interest work?
assistant: ... [Policy: deposit.savings.interest.us.v1]
user:      Thanks. Continue with that dispute.
assistant: <tool_call>{"name":"dispute_transaction",...}</tool_call>
tool:      {"ok":true,"result":{"transaction":{...}}}
assistant: I resumed your earlier request and opened the dispute ...
```

This teaches Granite to act naturally after the harness has resolved the lane
and pinned the original servicing exchange. It does not make Granite the owner
of dialogue-state transitions.

## 3. State-Aware Router Data

[`prepare_conversation_router_data.py`](../scripts/retail_bank/prepare_conversation_router_data.py)
combines leakage-safe in-domain examples derived from the composite SFT data
with checksum-pinned CLINC150 external OOD examples. It adds deterministic
contextual follow-ups, repairs, topic shifts, clarification answers, and resume
trajectories.

```bash
PYTHONPATH=src uv run python scripts/retail_bank/prepare_conversation_router_data.py \
  --sft-dir data/banking-servicing-alignment-v5 \
  --output-dir data/banking-conversation-router-v5-social-policy-generalization-candidate5 \
  --source-lock data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json \
  --expected-release-lock data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json
```

The V5 source lock is
[`data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json`](../data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json).
The generalized regeneration produced 19,363 training, 5,056 validation, and
6,171 test rows. The manifest SHA-256 is
`9cb527bdc337ce4da06e391f1d1e341da80092ab1ac46bf619bd33947f7a3608`.
The train, validation, and test SHA-256 values are, respectively,
`1e67741213b2ee48a61b6aa20be485f9f634850434637f533f928a858e1572f5`,
`4df22958f9519355204bcc2910a2874ead44425644056165133126042abcdafa`, and
`6af19f8079ff07c087d692ae4c331c55ef33adcdbcd316aa425e866452bd5d97`.
The added
state-conditioned negatives prevent a stale pending task from overriding an
explicit current-turn intent, policy question, social turn, or OOD shift.

### Router record example

```json
{
  "current_text": "Let's continue with the original request.",
  "domain_label": 1,
  "intent": "view_accounts",
  "intent_label": 0,
  "lane": "servicing",
  "relation_labels": [1, 0, 0, 0, 1],
  "prior_dialogue_state": {
    "version": 1,
    "knowledge_detour_active": true,
    "pending_servicing": {
      "intent": "view_accounts",
      "anchor_user_message": "Show me my account balances.",
      "anchor_assistant_message": "Which accounts should I include?",
      "phase": "awaiting_user"
    }
  },
  "history": [
    {"role": "user", "content": "How is available balance different from current balance?"},
    {"role": "assistant", "content": "I can explain the applicable policy."}
  ]
}
```

The relation vector follows this fixed order:

```text
context_dependent, agent_repair, topic_shift,
clarification_answer, resume_previous_service
```

## Split and Leakage Rules

The generators enforce these invariants:

- all realizations from one `split_group` stay in one split;
- complete resume trajectories stay in one split;
- validation and test records are not sampled from training rows;
- held-out screenshot currents do not appear exactly in composite training;
- generated records contain no email, SSN-like, or long payment-card patterns;
- manifests store split counts, byte sizes, SHA-256 digests, schema versions,
  generator versions, and allowed uses;
- the router never sees target actions, action results, expected fields, or
  final current-turn answers.

These controls prevent direct row leakage. They do not prove broad real-world
generalization. Keep live human tests and independently authored evaluation
prompts separate from generator templates and POC presets.

## Data Verification

```bash
PYTHONPATH=src uv run pytest -q \
  tests/test_banking_tool_sft_data.py \
  tests/test_banking_servicing_alignment_data.py \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py
```

The two published dataset revisions were verified on the Hub as:

```text
spkc83/retail-bank-servicing-alignment-sft
  40a0b68b9f746131ffff32a83e077fd7e4a344d1

spkc83/retail-bank-conversation-router-data
  8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc
```

Both generative data and live retrieval use canonical policy corpus revision
`sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a`.
This identity match is a strict release condition: an SFT record derived from a
different policy revision must not be mixed into the corrected training job.
