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
  - spkc83/retail-bank-servicing-agent-9b
  - spkc83/retail-bank-conversation-router
datasets:
  - spkc83/retail-bank-servicing-alignment-sft
  - spkc83/retail-bank-conversation-router-data
short_description: Model-driven 8.79B synthetic retail-bank service agent.
---

# Retail Bank Customer Service POC

This authenticated POC tests whether a history-aware OOD/capability/relation
router plus a two-stage SFT 8.79B Granite model can provide natural multi-turn
customer service and operate a synthetic retail-bank backend.

Everything is fictional. The application has no connection to a bank and
cannot access real accounts or perform real transactions.

## Released v4 stack

The Space pins the history-aware cross-encoder router and servicing-aligned
Granite model at immutable Hub revisions. The router has domain,
servicing-capability, and multi-label conversation-relation heads. Capability
predictions remain diagnostics and never enter the Granite prompt or choose
tools.

## Live artifacts

- Application: https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc
- POC source: https://github.com/spkc83/retail-bank-servicing-poc
- Model-development source: https://github.com/spkc83/retail-bank-servicing
- Model: https://huggingface.co/spkc83/retail-bank-servicing-agent-9b
- Tool-use dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-servicing-alignment-sft
- History-aware classifier:
  https://huggingface.co/spkc83/retail-bank-conversation-router

## Runtime

```text
Authenticated synthetic customer and session transcript
  → one managed ZeroGPU chat event
  → CPU history-aware router
  → high-confidence OOD: governed scope response
  → in-domain or uncertain: 8.79B model generation
  → direct answer, clarification, or tagged-JSON tool calls
  → generated calls execute against session-isolated synthetic SQLite
  → exact read lists render as tables; other grounded answers are validated
```

The router's capability and relation outputs are diagnostic metadata. They do
not enter the generation prompt, select a tool, or supply arguments. The 8.79B
model owns greetings, conversation, clarification, tool choice, public
arguments, and action wording. The runtime budgets context, parses and validates
the tagged-JSON wire format, invokes the named mock function, renders successful
read-list results from exact fields, validates action wording, and records
diagnostics.

The live generation prompt and iterative model → tool → model protocol match
the SFT corpus and frozen evaluator. A first-pass answer without a tool call is
returned directly. Read-list presentation is host-rendered. An action answer
that omits or contradicts an essential selector or outcome gets at most one
text-only repair pass with tools disabled. Tool calls and arguments are never
automatically repaired.

### Route examples

```text
high banking score          → in_domain → run the 8.79B model
middle banking score        → uncertain → run the 8.79B model
low banking + relation cue  → uncertain → run the 8.79B model
low banking + no rescue     → OOD       → stock scope response
classifier exception        → error     → visible failure response
```

`uncertain` is deliberate abstention by the classifier, not a model failure.
The actual probabilities, thresholds, route, and generation-call count appear
in the diagnostics panel.

## Conversation context

The application stores complete valid interaction groups: user messages,
assistant tool calls, correlated tool results, and final model responses.

Each inference uses an 8,192-token input budget and reserves 512 tokens for
generation. The current interaction and system instructions are retained
first; newest complete prior interaction groups are then added while they fit.
A tool chain is never split across the context boundary.

The harness does not promote browser/session transcript dictionaries into the
trusted system prompt. Prior context remains ordinary role-tagged messages.

## Synthetic tools

- list accounts, cards, transactions, transfers, and service cases;
- freeze or replace a card;
- dispute a transaction by merchant description;
- cancel a pending transfer by recipient.

Calls execute in generated order. Schema or backend errors return to the model
as tool results so it can explain the outcome conversationally.

Example:

```text
User: Show my cards.
Granite: list_cards({})
SQLite: active card ending in 4821
Harness: renders the exact card result as a Markdown table.
```

## Proving model inference

The diagnostics panel exposes:

- exact model repository and immutable revision;
- runtime and CUDA device;
- response path and model-call count;
- raw `base`, `grounded_final`, and iterative tool-follow-up outputs;
- generated tool names and public arguments;
- correlated tool results;
- prompt and output SHA-256 values for every model pass.

The authenticated `/zero_gpu_probe` API performs no generation. It returns
only after a managed ZeroGPU worker enters application code and reports the
packed model's device plus exact model and router revisions. If it fails in
ZeroGPU `worker_init`, the failure is in GPU allocation before the chat or model
code runs.

A successful live turn is counted as 8.79B inference only when diagnostics show
`spkc83/retail-bank-servicing-agent-9b` at revision
`1d56824995aa1adecfe20f62ca42fb1c0c443817` and a CUDA device. Preset prompts
are evaluation cases, not hard-coded routes.

The pinned checkpoint passed all 1,374 frozen test conversations: 796/796 tool
names and arguments, 700/700 executable trajectories, 96/96 exact dependent
multi-tool sequences, 63/63 appropriate clarifications, 258/258 banking FAQs,
35/35 OOD paths, and 1,141/1,141 grounded facts. It produced zero
parse failures, malformed calls, private arguments, credential requests,
in-domain false refusals, or OOD false accepts.

If ZeroGPU allocation or generation fails, the UI reports model
unavailability. It does not substitute a Python-generated banking answer.

## Authentication

The demo usernames are `alex.demo` and `maya.demo`. Passwords come from the
Space's write-only `DEMO_AUTH_JSON` secret and are displayed on the login page
for public testing. Authentication only selects isolated synthetic records.

## Local Streamlit on TITAN V

[`streamlit_app.py`](streamlit_app.py) is a separate local UI. It does not
import or modify the Space's Gradio/ZeroGPU boundary. It reuses the same:

- pinned Granite 9B checkpoint and tokenizer;
- history-aware CPU router;
- `ConversationalBankingAgent` model-to-tool-to-model loop;
- synthetic SQLite backend and two demo customers;
- 8,192-token conversation budget and complete interaction groups.

[`local_gpu_runtime.py`](local_gpu_runtime.py) quantizes linear weights during
loading with bitsandbytes NF4, double quantization, and FP16 compute. It does
not create or save a second quantized model artifact. The published checkpoint
remains unchanged in the Hugging Face cache.

Launch from the repository root:

```bash
uv run scripts/retail_bank/run_local_streamlit.py
```

Then open <http://127.0.0.1:8501>. The launcher pins a CUDA 12.6 PyTorch wheel
that contains `sm_70` kernels for the TITAN V. It also pins Streamlit,
Transformers, bitsandbytes, and Accelerate independently of the Space
requirements.

The local-only default credentials are:

```text
alex.demo / alex-local-demo
maya.demo / maya-local-demo
```

Override both with the same authenticated POC contract when needed:

```bash
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-more"}'
uv run scripts/retail_bank/run_local_streamlit.py
```

Optional launch settings:

```bash
export LOCAL_STREAMLIT_PORT=8502
export POC_SESSION_DB_DIR=/tmp/retail-bank-local-streamlit
```

The first run may download roughly 16GB of pinned model files. With a warm
cache, loading NF4 weights on the TITAN V can still take several minutes.
Streamlit `cache_resource` factories retain one router, runtime, and controller
across page reruns; browser identity and conversation remain session-local.
Generation is serialized because the local machine has one GPU.

The local app prefers the verified release router at
`artifacts/banking-conversation-router-v4-release`. If it is absent, it
downloads the same immutable Hub revision. Set `LOCAL_ROUTER_ARTIFACT_DIR` to
use another verified local copy.

Successful model inference is proven by diagnostics showing:

- model `spkc83/retail-bank-servicing-agent-9b`;
- revision `1d56824995aa1adecfe20f62ca42fb1c0c443817`;
- runtime `cuda:0` and `NVIDIA TITAN V`;
- weight quantization `bitsandbytes-nf4-double`;
- at least one model pass with prompt/output hashes and raw output;
- generated tool calls and correlated results when a tool was used;
- execution boundary `Local CUDA / NF4`.

`LOCAL_POC_SKIP_MODEL_LOAD=1` and `LOCAL_POC_SKIP_ROUTER_LOAD=1` exist only for
automated tests. The launcher refuses to start with either flag unless
`--allow-test-skip` is explicitly supplied, preventing a test environment from
silently masquerading as real local inference.

Troubleshooting:

- `CUDA is unavailable`: verify `nvidia-smi` and the active NVIDIA driver.
- missing `sm_70`: launch through the provided UV script instead of an
  unrelated CUDA 13 PyTorch environment.
- slow first page: wait for the pinned model download and NF4 packing; do not
  repeatedly refresh during the initial load.
- port already in use: set `LOCAL_STREAMLIT_PORT` to another local port.

## Local verification

```bash
python -m pip install -r requirements.txt
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-chars"}'
export POC_SKIP_MODEL_LOAD=1
export POC_SKIP_ROUTER_LOAD=1
pytest -q
ruff check .
```

Local Streamlit tests do not load the 9B checkpoint:

```bash
python -m pytest -q tests/test_local_gpu_runtime.py \
  tests/test_local_app_service.py tests/test_streamlit_app.py
```

Release verification additionally requires live ZeroGPU read, write,
multi-tool, clarification, FAQ, OOD, and multi-turn cases with the exact model
revision visible in diagnostics.
