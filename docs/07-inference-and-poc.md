# Inference and ZeroGPU POC

This page explains the active authenticated Gradio POC in
[`poc/retail-bank-customer-service-poc/`](../poc/retail-bank-customer-service-poc/).
It documents the current Granite 9B model path, the CPU history-aware router,
the model-owned tool loop, session state, diagnostics, local tests, and the
Space deployment surface.

Everything in the POC is synthetic. It has no connection to a real bank, cannot
access real accounts, and must not receive credentials, full account numbers,
payment-card details, or real customer data.

## Active Artifacts

The POC uses the released artifacts below. Do not substitute branch names such
as `main` where these revisions are required.

| Role | Repository | Immutable revision | Evidence |
| --- | --- | --- | --- |
| Generative agent | `spkc83/retail-bank-servicing-agent-9b` | `1d56824995aa1adecfe20f62ca42fb1c0c443817` | [`zero_gpu_runtime.py`](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py), [`model card`](../model_cards/retail-bank-agent-9b.md) |
| Agent base | `ibm-granite/granite-4.1-8b` | `1504002f650e656a0a3789d99574df12e3e94ed0` | [`configs/banking-tool-sft-granite.toml`](../configs/banking-tool-sft-granite.toml), [`model card`](../model_cards/retail-bank-agent-9b.md) |
| Tool-use SFT dataset | `spkc83/retail-bank-agent-sft` | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` | [`data card`](../data_cards/retail-bank-agent-sft.md), [`model card`](../model_cards/retail-bank-agent-9b.md) |
| Servicing-remediation SFT dataset | `spkc83/retail-bank-servicing-alignment-sft` | `0ce32f9c7a3edff227005e5b89b089947b87625a` | [`data card`](../data_cards/retail-bank-servicing-alignment-sft.md), [`model card`](../model_cards/retail-bank-agent-9b.md) |
| History-aware router | `spkc83/retail-bank-conversation-router` | `9e090c0fa21cebbaa03a431a7ce61e656c0739fe` | [`router.py`](../poc/retail-bank-customer-service-poc/router.py), [`router card`](../model_cards/retail-bank-domain-intent-router.md) |
| Router dataset | `spkc83/retail-bank-conversation-router-data` | `e9a64a2e7f2b622d5412c15eac4618ceca2150da` | [`train_conversation_router.py`](../scripts/retail_bank/train_conversation_router.py), [`router card`](../model_cards/retail-bank-domain-intent-router.md) |

For the complete artifact ledger, see
[`docs/reference/artifacts.md`](reference/artifacts.md).

## Runtime Flow

The public app is [`app.py`](../poc/retail-bank-customer-service-poc/app.py).
At a high level, one authenticated chat turn follows this path:

```text
Gradio authenticated user
  -> CPU history-aware router
  -> high-confidence OOD returns the governed stock response
  -> in-domain or uncertain turn enters one ZeroGPU event
  -> Granite 9B either answers directly or emits tagged-JSON tool calls
  -> generated calls execute against session-isolated SQLite
  -> tool results return to Granite 9B
  -> exact read lists render as tables; other grounded answers are validated
```

The router does not choose tools or write arguments. The runtime budgets
context, validates generated tool syntax, executes the synthetic backend,
renders successful list results from exact records, validates action wording,
and records diagnostics.

## Static Authentication

Authentication is loaded by
[`auth.py`](../poc/retail-bank-customer-service-poc/auth.py). The app accepts
exactly two usernames:

- `alex.demo`
- `maya.demo`

Passwords come from the `DEMO_AUTH_JSON` environment variable. The value must be
a JSON object with exactly those two usernames, different passwords, and each
password must contain at least 12 characters.

Example local value:

```bash
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-more"}'
```

The auth layer only selects one of the two synthetic customer records and the
Gradio session hash. It is not a bank identity provider.

## Router Loading and Thresholds

The Space loads `spkc83/retail-bank-conversation-router` at revision
`9e090c0fa21cebbaa03a431a7ce61e656c0739fe`.
[`router.py`](../poc/retail-bank-customer-service-poc/router.py) verifies the
artifact manifest, loads the shared DistilBERT encoder plus domain,
servicing-capability, and conversation-relation heads, and uses the artifact's
calibrated policy.

The released artifact uses these boundaries:

- banking probability `< 0.10` plus no relation rescue: `out_of_domain`
- banking probability `>= 0.50`: `in_domain`
- banking probability from `0.10` through `< 0.50`, or relation rescue:
  `uncertain`
- relation rescue boundary: `0.40`

Capability and relation outputs are diagnostics only. An unavailable router in
explicit local test mode produces an `uncertain` test route; a classifier
exception during a normal model turn produces a visible `classifier_error` and
the 9B generator is not invoked.

### What happens to `uncertain`

`uncertain` is an accepted generation route. It falls through to the same
`ConversationalBankingAgent.run_turn()` path as `in_domain`.

Only two router outcomes stop generation:

- `out_of_domain`: return the governed scope response;
- `classifier_error`: return the visible failure response.

Example policy inputs:

```text
banking=0.31, rescue=0.18 -> uncertain -> Granite runs
banking=0.07, rescue=0.65 -> uncertain -> Granite runs
banking=0.03, rescue=0.09 -> out_of_domain -> Granite does not run
```

The values illustrate the thresholds. Actual probabilities are displayed in
the diagnostics panel.

## Model Loading

ZeroGPU model loading is isolated in
[`zero_gpu_runtime.py`](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py).
By default it loads:

- model ID: `spkc83/retail-bank-servicing-agent-9b`
- model revision: `1d56824995aa1adecfe20f62ca42fb1c0c443817`
- dtype: `torch.float16`
- device: CUDA
- generation: deterministic, `do_sample=False`

The model ID and revision can be overridden with:

```bash
export RETAIL_BANK_MODEL_ID=spkc83/retail-bank-servicing-agent-9b
export RETAIL_BANK_MODEL_REVISION=1d56824995aa1adecfe20f62ca42fb1c0c443817
```

For local tests that should not load the 9B model, set:

```bash
export POC_SKIP_MODEL_LOAD=1
```

When `POC_SKIP_MODEL_LOAD=1`, the module installs a local `spaces.GPU`
decorator stub and leaves the tokenizer/model unset. Tests can then validate
routing, auth, state, parsing, and UI plumbing without downloading the model.

## Local Streamlit NF4 Boundary

The repository also provides a local-only Streamlit path for the 12GB TITAN V:

```bash
uv run scripts/retail_bank/run_local_streamlit.py
```

The launcher executes
[`streamlit_app.py`](../poc/retail-bank-customer-service-poc/streamlit_app.py)
with a CUDA 12.6 PyTorch build containing `sm_70`. Unlike the Space runtime,
[`local_gpu_runtime.py`](../poc/retail-bank-customer-service-poc/local_gpu_runtime.py)
loads the pinned merged checkpoint using bitsandbytes NF4 double quantization
and FP16 compute. Quantization occurs at load time and is not published or
saved as another model.

The Streamlit layer delegates each accepted turn to
[`LocalBankingController`](../poc/retail-bank-customer-service-poc/local_app_service.py).
That controller calls the same CPU router and `ConversationalBankingAgent` used
by the POC design. Granite still owns direct responses, clarifications, tool
selection, tool arguments, and action wording. The shared harness—not the
UI—renders read-list tables and may request one text-only repair for an
ungrounded action answer.

Resource ownership differs from browser conversation state:

- `st.cache_resource` retains one router, NF4 runtime, and controller across
  Streamlit reruns;
- a generation lock serializes access to the single TITAN V;
- `st.session_state` retains the authenticated username, browser-session key,
  canonical conversation, and diagnostics;
- SQLite state remains isolated by username and browser-session key.

The local path first checks
`artifacts/banking-conversation-router-v4-release` and verifies its manifest.
It downloads the same immutable router revision from the Hub only when that
local release artifact is unavailable. `LOCAL_ROUTER_ARTIFACT_DIR` can point to
another complete verified copy.

The local login page displays two local-only default accounts. A valid
`DEMO_AUTH_JSON` environment value overrides those defaults. This does not
change the Space authentication contract.

Local diagnostics identify the execution boundary as `Local CUDA / NF4`, not
`ZeroGPU large`, and expose the model/revision, CUDA device, quantization,
allocated VRAM, route probabilities, model passes, generated calls, and tool
results.

## ZeroGPU Boundary

The model turn is registered in
[`app.py`](../poc/retail-bank-customer-service-poc/app.py) with:

```python
@spaces.GPU(size="large", duration=90)
def run_model_turn(...):
    ...
```

The whole route-plus-generation turn is queued as a Gradio event. If ZeroGPU
allocation or model generation fails, the app returns
[`MODEL_FAILURE_RESPONSE`](../poc/retail-bank-customer-service-poc/responses.py).
It does not synthesize a Python-authored banking answer.

## Tool Loop

The agent loop is implemented in
[`model_service.py`](../poc/retail-bank-customer-service-poc/model_service.py).
The model receives one system message plus the public tool manifest. The system
message instructs the already-authenticated synthetic-bank agent to use tools
for customer-specific records or actions and never ask for private IDs,
passwords, PINs, or credentials.

The active public tools are:

| Tool | Purpose | Arguments |
| --- | --- | --- |
| `list_accounts` | List accounts and balances | none |
| `list_cards` | List cards and statuses | none |
| `list_service_cases` | List recent service cases | none |
| `list_transactions` | List recent account transactions | optional `limit` from 1 to 20 |
| `list_transfers` | List transfers and statuses | none |
| `freeze_card` | Freeze a card | optional `last4` |
| `replace_card` | Request card replacement | optional `last4` |
| `dispute_transaction` | Dispute one transaction | optional `description` |
| `cancel_transfer` | Cancel one pending transfer | optional `recipient` |

Granite emits tagged JSON:

```text
<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>
```

The runtime parses the tag, validates the JSON object, checks the tool name and
argument schema, executes the call, appends a correlated tool result, and asks
the same model for the next response. The model may emit another tool call after
seeing a tool result. The loop stops when the model emits a normal assistant
response.

After the loop:

- if every executed call is a successful read-list tool, `response_policy.py`
  renders Markdown tables directly from the backend result;
- otherwise, essential action selectors and outcomes are checked against the
  tool envelopes;
- a failed action-answer check permits exactly one `final_repair_1` model pass
  with tools disabled and immutable tool events supplied as data;
- malformed tool syntax, wrong tool names, and wrong arguments are never
  automatically repaired.

Limits:

- maximum input budget: 8,192 tokens
- maximum generation per pass: 512 tokens
- maximum tool calls per turn: 8

Unsupported tool names, duplicate call IDs, out-of-order indexes, malformed
JSON, and invalid argument types raise protocol errors. Backend errors are
returned to the model as safe tool-result envelopes.

### Worked tool turn

```text
1. User: Show my cards.
2. Router: in_domain or uncertain.
3. Granite: <tool_call>{"name":"list_cards","arguments":{}}</tool_call>
4. Runtime: validates the call and executes list_cards in the session database.
5. Tool result: one active card ending in 4821.
6. Runtime: render the exact card fields as a Markdown table.
7. Diagnostics: two model passes, CUDA device, exact revision, call, and result.
```

On the next turn, “Replace the active one” is routed with visible history. The
same 9B model receives the retained interaction group and can emit
`replace_card(last4="4821")`.

See [End-to-End Flow by Example](11-end-to-end-flow-by-example.md) for data and
training records that correspond to this runtime exchange.

## Conversation Budget

The budgeter is
[`select_token_budgeted_context`](../poc/retail-bank-customer-service-poc/model_service.py).
It keeps complete interaction groups:

- user message
- assistant tool call message, when present
- correlated tool result messages, when present
- final assistant response

The latest current interaction and system message are retained first. Then the
newest complete prior groups are added while the rendered prompt and tool
definitions fit within 8,192 input tokens. A tool chain is never split across
the context boundary.

The app reserves 512 new tokens for each model pass.

The runtime does not promote browser/session transcript dictionaries into a
trusted system-memory block. Prior context remains ordinary role-tagged model
messages, and only complete interaction groups are retained.

## Synthetic SQLite State

State setup is in
[`state.py`](../poc/retail-bank-customer-service-poc/state.py). The seed data is
[`synthetic_bank.json`](../poc/retail-bank-customer-service-poc/synthetic_bank.json).

`SessionBankRegistry` in
[`mock_bank.py`](../poc/retail-bank-customer-service-poc/mock_bank.py) creates
one SQLite database per `(username, Gradio session hash)` pair. By default,
files are written under:

```text
/tmp/retail-bank-servicing-poc
```

Override the directory with:

```bash
export POC_SESSION_DB_DIR=/tmp/my-retail-bank-poc
```

Session behavior:

- each session starts from deterministic synthetic JSON records;
- sessions expire after 7,200 seconds;
- at most 32 sessions are retained;
- write tools update only the session database;
- reset reseeds the current user's session database.

The sidebar snapshot is rendered from the current session database, so write
tools such as `freeze_card` and `cancel_transfer` are visible immediately after
the model turn completes.

## Diagnostics

Diagnostics are rendered by
[`_render_diagnostics`](../poc/retail-bank-customer-service-poc/app.py). They
are part of the proof that a turn used the active model path.

The current v4 branch panel shows:

- route: `in_domain`, `uncertain`, `out_of_domain`, or `classifier_error`
- in-domain and OOD probabilities
- whether complete visible conversation context was applied
- conversation-relation probabilities and router failure details
- response path, such as `direct_answer`, `base_tool_rendered`,
  `base_tool_repaired`, or `base_tool_chain`
- top diagnostic servicing-capability candidates
- generated tool calls and public arguments
- tool-result success or safe error code
- model pass labels, input-token counts, prompt SHA-256 values, raw output
  SHA-256 values, raw outputs, runtime device, and CUDA device name
- generation call count
- model ID and exact model revision
- `SPACE_COMMIT_SHA`, when provided by the Space runtime
- visible response SHA-256

A successful live model turn should show:

- `Model: spkc83/retail-bank-servicing-agent-9b`
- `Exact model revision: 1d56824995aa1adecfe20f62ca42fb1c0c443817`
- `Registered execution boundary: ZeroGPU large`
- a CUDA runtime device for model passes

High-confidence OOD diagnostics intentionally show no model passes.

## Local Tests

Run repository tests from the repo root:

```bash
python -m pytest -q tests
```

Run the POC tests without loading the 9B model or router artifact:

```bash
cd poc/retail-bank-customer-service-poc
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-more"}'
export POC_SKIP_MODEL_LOAD=1
export POC_SKIP_ROUTER_LOAD=1
python -m pytest -q tests
```

Run static checks from the repo root:

```bash
ruff check .
MYPYPATH=src mypy src scripts tests
uv lock --check
```

Local POC tests validate auth, router behavior, model-service protocol handling,
SQLite state, Gradio app behavior, and the ZeroGPU test stub. They do not prove
a live GPU generation unless the skip variables are removed in a Space or other
CUDA-capable environment.

## Space Deployment Surface

The Space app files are in
[`poc/retail-bank-customer-service-poc/`](../poc/retail-bank-customer-service-poc/):

- [`README.md`](../poc/retail-bank-customer-service-poc/README.md): Space card
  front matter and public operating notes
- [`app.py`](../poc/retail-bank-customer-service-poc/app.py): Gradio app
- [`zero_gpu_runtime.py`](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py):
  model load and deterministic generation
- [`requirements.txt`](../poc/retail-bank-customer-service-poc/requirements.txt):
  Space dependencies
- [`auth.py`](../poc/retail-bank-customer-service-poc/auth.py): demo auth
- [`router.py`](../poc/retail-bank-customer-service-poc/router.py): CPU router
- [`model_service.py`](../poc/retail-bank-customer-service-poc/model_service.py):
  model-owned tool loop
- [`response_policy.py`](../poc/retail-bank-customer-service-poc/response_policy.py):
  deterministic tables and grounded action-answer checks
- [`mock_bank.py`](../poc/retail-bank-customer-service-poc/mock_bank.py): SQLite
  synthetic backend
- [`state.py`](../poc/retail-bank-customer-service-poc/state.py): session
  registry setup
- [`synthetic_bank.json`](../poc/retail-bank-customer-service-poc/synthetic_bank.json):
  seed data

Deploying to the public Space is an external production action. Do not run a
deployment command without explicit authorization for that deployment. Before a
deployment, verify the local POC tests and set the Space secret `DEMO_AUTH_JSON`
to the exact two demo users. The canonical deploy stage uploads only allowlisted
application files, persists the exact model and router revisions as Space
variables, records the returned Space commit in `SPACE_COMMIT_SHA`, triggers a
rebuild through those Hub updates, and waits for the rebuilt runtime:

```bash
PYTHONPATH=src python scripts/retail_bank/run_release_pipeline.py \
  --stage deploy \
  --execute \
  --allow-publish
```

The deploy helper does not request deletion of existing Hub files. Any cleanup
of an accidental remote cache is a separate destructive operation and requires
separate authorization.

The active public Space is:

```text
https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc
```

After deployment, call the authenticated `/zero_gpu_probe` endpoint first. It
must enter application code and report the exact model/router revisions plus a
CUDA device. Then run live read, write, multi-tool, clarification, FAQ, OOD, and
multi-turn cases. The diagnostics panel must show the active model revision and
CUDA-backed generation for model-handled turns.
