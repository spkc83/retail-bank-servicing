---
title: Retail Bank Customer Service POC
emoji: 🏦
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
python_version: 3.10
app_file: app.py
pinned: false
suggested_hardware: zero-a10g
models:
  - spkc83/retail-bank-servicing-agent-9b-peft
  - spkc83/retail-bank-servicing-agent-9b
  - spkc83/retail-bank-conversation-router
datasets:
  - spkc83/retail-bank-servicing-alignment-sft
  - spkc83/retail-bank-conversation-router-data
short_description: Harborlight Bank model-driven 8.79B customer-service POC.
---

# Harborlight Bank Customer-Service POC

This authenticated POC tests Harbor, a model-driven retail-bank assistant,
through Gradio/ZeroGPU and a local Streamlit interface. It combines:

- a PEFT-adapted Granite 8.79B generator;
- a state-aware CPU router with domain, 12 fine-intent, and five relation
  heads;
- one bounded pending servicing task with policy-detour/resume behavior;
- deterministic retrieval from a versioned policy knowledge base;
- nine Granite-selected banking actions over fictional session data;
- response grounding, exact read tables, policy citations, and customer-facing
  language validation.

All profiles and records are fictional. Nothing connects to a real bank or can
perform a real transaction.

## V5 Status

| Component | Revision/status |
| --- | --- |
| Generalized router `spkc83/retail-bank-conversation-router` | `c8f154266612e79afe20af8abef25761fa56d589` |
| Generalized router data `spkc83/retail-bank-conversation-router-data` | `8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc` |
| Canonical-policy Granite SFT data `spkc83/retail-bank-servicing-alignment-sft` | `40a0b68b9f746131ffff32a83e077fd7e4a344d1` |
| Policy corpus | `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a` |
| Granite Stage-2 base | `spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| Granite V5 PEFT adapter | `spkc83/retail-bank-servicing-agent-9b-peft@cc95e446af2b5e1d8d9df2751a8192613ad386e3`; bundle commit `b4269445ce7b2b943d2d9531102166bf8840a074` |
| Strict evaluation | job `6a7f89edc97db76cbdf31893` failed; replacement evaluator/SFT underway |

Both runtimes load the exact Stage-2 base and attach the immutable BF16 LoRA
adapter without merging. Merged FP16/BF16 candidates failed unchanged parity
gates. The PEFT candidate also failed strict evaluation: five credential flags
were evaluator false positives and two genuine behavioral failures remain.
Use the UI for diagnosis only; it is not an approved deployment.

## Request Flow

```text
authenticated Harborlight customer + conversation + prior dialogue state
  -> V5 CPU cross-encoder
     -> high-confidence OOD: fixed banking-scope answer; Granite not called
     -> accepted/uncertain: bounded state transition
        -> policy: retrieve versioned chunks; Granite called without actions
           -> require [Policy: chunk_id]
        -> conversation/servicing: Granite may answer or emit tagged JSON
           -> validate and execute against session-isolated fictional SQLite
           -> exact read table or validated Granite-authored action answer
  -> reject internal implementation language; allow one text-only repair
```

The router intent never enters Granite's prompt and never authorizes an action.
Granite chooses public actions and arguments. The state machine only tracks one
pending service task, preserves it through a policy detour, and pins the actual
original exchange when the customer returns.

## Policy Detour Example

```text
Customer: Dispute the North Harbor Market purchase.
Harbor:    I can help. What would you like to confirm first?
Customer: How does a card dispute investigation work?
Harbor:    ... [Policy: card.dispute.us.v1]
Customer: Thanks. Continue with the dispute.
Harbor:    <tool_call>{"name":"dispute_transaction",...}</tool_call>
```

The policy turn receives retrieved text and no banking actions. The final turn
receives the original dispute exchange as pinned context. The pending task
clears only after `dispute_transaction` succeeds.

## Supported Actions

```text
list_accounts        list_cards          list_service_cases
list_transactions    list_transfers      freeze_card
replace_card         dispute_transaction cancel_transfer
```

Granite emits tagged JSON such as:

```text
<tool_call>{"name":"list_transactions","arguments":{"limit":5}}</tool_call>
```

The harness validates the schema, executes the call, and returns the correlated
result. Read-only result lists render as Markdown tables from exact fields.
Action answers remain Granite-authored and must preserve essential result
facts without exposing private internal IDs.

## Confirming Model Generation

Open **Technical details** in Gradio or **Experiment diagnostics** in
Streamlit. A model-authored turn records:

- one or more model-pass labels and raw outputs;
- prompt and output SHA-256 values;
- input token count;
- runtime/CUDA device;
- exact base, adapter-bundle, and router revisions;
- action calls/results or policy source IDs;
- response path and bounded dialogue state.

If the turn records zero model passes, Granite did not generate it. Expected
zero-pass paths are high-confidence OOD, classifier failure, and policy
no-match.

## Static POC Authentication

Both apps accept exactly two usernames:

```text
alex.demo
maya.demo
```

Set `DEMO_AUTH_JSON` to a JSON object with different passwords of at least 12
characters:

```bash
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-strong-value-1","maya.demo":"replace-with-strong-value-2"}'
```

Authentication only selects a fictional profile and isolates session state. It
is not a production security design.

## Local Streamlit on a 12GB CUDA GPU

From the development repository root:

```bash
uv run scripts/retail_bank/run_local_streamlit.py
```

Open <http://127.0.0.1:8501>. If `DEMO_AUTH_JSON` is unset, the local login page
shows these local-only defaults:

```text
alex.demo / alex-local-demo
maya.demo / maya-local-demo
```

The local runtime quantizes the pinned base with bitsandbytes NF4 double
quantization, then attaches the pinned adapter. It prefers:

```text
artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5
```

and downloads the final published V5 router only when the
local artifact is absent.
Streamlit caches the runtime, router, and controller across normal reruns.

Useful overrides:

```text
LOCAL_STREAMLIT_PORT
LOCAL_ROUTER_ARTIFACT_DIR
RETAIL_BANK_MODEL_ID
RETAIL_BANK_MODEL_REVISION
RETAIL_BANK_BASE_MODEL_ID
RETAIL_BANK_BASE_MODEL_REVISION
RETAIL_BANK_ADAPTER_ID
RETAIL_BANK_ADAPTER_REVISION
RETAIL_BANK_ROUTER_ID
RETAIL_BANK_ROUTER_REVISION
HF_TOKEN
```

The launcher refuses `LOCAL_POC_SKIP_MODEL_LOAD=1` or
`LOCAL_POC_SKIP_ROUTER_LOAD=1` during an ordinary start. Those flags are for
tests only.

## Hugging Face Gradio/ZeroGPU

The Space loads the Stage-2 base in BF16, attaches the BF16 LoRA adapter, and
executes each chat turn inside:

```python
@spaces.GPU(size="large", duration=90)
```

The hidden `zero_gpu_probe` event uses a 30-second GPU allocation to prove user
code entered the worker and to report model/device metadata. The Gradio queue
uses a concurrency limit of one because the full packed model and mutable
session loop are intentionally low-throughput POC infrastructure.

The deploy helper requires exact base, adapter, and router revisions and uploads only an
allowlist of application files:

```bash
ADAPTER_REVISION=cc95e446af2b5e1d8d9df2751a8192613ad386e3
ROUTER_REVISION=c8f154266612e79afe20af8abef25761fa56d589

PYTHONPATH=src uv run python scripts/retail_bank/deploy_zero_gpu_space.py \
  --space-id spkc83/retail-bank-servicing-poc \
  --model-id spkc83/retail-bank-servicing-agent-9b-peft \
  --model-revision "$ADAPTER_REVISION" \
  --base-model-id spkc83/retail-bank-servicing-agent-9b \
  --base-model-revision 1d56824995aa1adecfe20f62ca42fb1c0c443817 \
  --adapter-id spkc83/retail-bank-servicing-agent-9b-peft \
  --adapter-revision "$ADAPTER_REVISION" \
  --model-dtype bf16 \
  --router-id spkc83/retail-bank-conversation-router \
  --router-revision "$ROUTER_REVISION"
```

The command prints a plan and proves the four PEFT identity fields are complete.
Do not add `--execute --allow-publish` for `cc95e446...`: strict evaluation
failed. Deployment remains pending a replacement artifact and passing result.

## Reset and Session Isolation

**Reset demo** or **Start over** clears:

- visible and internal conversation;
- pending servicing task;
- policy-detour flag;
- fictional bank changes for that session.

Another authenticated browser session has a different session ID and does not
share the mutable account state or bounded dialogue state.

## Run Tests

From the development repository root, without loading the 8.79B model or Hub
router:

```bash
POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  uv run pytest -q poc/retail-bank-customer-service-poc/tests
```

The suite covers auth, router loading and thresholds, policy retrieval and
citations, dialogue detours/resume/switch/reset, context pinning, action wire,
grounding, internal-language repair, exact read tables, local controller,
Gradio UI state, Streamlit helpers, and runtime metadata.

## Source and Artifacts

- Space: <https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc>
- POC source: <https://github.com/spkc83/retail-bank-servicing-poc>
- Model-development source: <https://github.com/spkc83/retail-bank-servicing>
- Granite model repo: <https://huggingface.co/spkc83/retail-bank-servicing-agent-9b>
- Router: <https://huggingface.co/spkc83/retail-bank-conversation-router>
- Granite V5 data: <https://huggingface.co/datasets/spkc83/retail-bank-servicing-alignment-sft>
- Router V5 data: <https://huggingface.co/datasets/spkc83/retail-bank-conversation-router-data>
