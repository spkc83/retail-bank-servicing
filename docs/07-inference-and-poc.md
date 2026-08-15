# V6 Inference and POC

The same orchestration design serves two interfaces:

- Gradio on Hugging Face ZeroGPU;
- Streamlit on a local CUDA GPU.

Both use the V6 hierarchical CPU router, the Granite PEFT composition, bounded
dialogue state, the versioned policy corpus, and fictional session-isolated
bank data.

## Runtime Pins

| Component | Identity |
| --- | --- |
| Router | `spkc83/retail-bank-conversation-router@7f6a0e77ad231233702039560ced007fdc68bd74` |
| Granite PEFT release | `spkc83/retail-bank-servicing-agent-9b-peft@cc95e446af2b5e1d8d9df2751a8192613ad386e3` |
| Granite adapter bundle | `b4269445ce7b2b943d2d9531102166bf8840a074` |
| Granite Stage-2 base | `spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| Policy corpus | `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a` |

The loader pins this immutable commit and rejects a mutable branch name.

## Turn Lifecycle

1. Resolve the authenticated fictional profile and browser session.
2. Canonicalize visible conversation history.
3. Classify the current text with up to three recent exchanges and trusted
   pre-turn dialogue state.
4. Joint-decode domain, lane, family, intent, action, and entity resolution;
   independently threshold the relation head.
5. Apply the route/action contract:
   - OOD: return the governed scope response without Granite;
   - uncertain: no tools, one natural clarification;
   - policy: retrieve evidence, no tools, require valid citations;
   - servicing: expose one intent-compatible tool schema when executable;
   - clarify/converse: no tools and action-specific generation guidance.
6. Let Granite choose tool arguments from conversation context.
7. Validate and execute up to eight action passes against fictional state.
8. Render exact list results as Markdown tables or validate action-grounded
   prose.
9. Record router tuple, raw candidates, model passes, tool results, hashes,
   devices, and immutable revisions in diagnostics.

## Router Outputs at Runtime

| Output | Consumer |
| --- | --- |
| `route` / `domain` | Scope gate and abstention behavior. |
| `lane` | Servicing, policy, conversation, or other-banking orchestration. |
| `family` | Diagnostic product hierarchy. |
| `intent` | Dialogue-state continuity and single-tool schema selection. |
| `active_relations` | Repair, topic-change, clarification, and resume transitions. |
| `action` | Refuse, retrieve, execute, clarify, or converse generation plan. |
| `entity_resolution` | Blocks execution for missing, ambiguous, or ineligible targets. |
| candidate probabilities | Diagnostics only; they do not bypass the joint decision. |

## Action-Guided Harness

The harness does not send all nine tools on every turn. If the router emits
`execute_tool`, `_generation_plan` maps the intent to one public schema:

```text
replace_card -> expose replace_card only
view_transactions -> expose list_transactions only
cancel_transfer -> expose cancel_transfer only
```

The system guidance names the allowed tool but supplies no arguments. Granite
must derive selectors such as a card ending or transfer recipient from the
conversation. If required selectors are absent, Granite should ask one
clarifying question.

For `missing`, `ambiguous`, or `ineligible` entity resolution, the harness
exposes no tool. This reduces invalid tool choice while preserving Granite as
the natural-language and argument-selection component.

## Policy Detours and Intent Changes

The dialogue state stores one pending servicing task. A policy question can
temporarily activate a knowledge detour without discarding that task. A later
`resume_previous_service` relation restores the task and pins its original
user/assistant exchange into model context.

An explicit new intent takes precedence over stale state. For example, after
a card-replacement clarification, “Actually, show my transfers” produces the
transfer hierarchy and replaces the pending route instead of being interpreted
as a card answer.

## Token-Budgeted Conversation

[`select_token_budgeted_context`](../poc/retail-bank-customer-service-poc/model_service.py)
keeps complete interaction groups instead of truncating individual messages.
The default model-input budget is 8,192 tokens. A resumed task pins its actual
anchor exchange so ordinary oldest-first trimming cannot remove it.

The router sees only up to three recent exchanges plus bounded state. Granite
receives a larger token-budgeted history because generation needs more natural
conversation context than classification.

## Confirming Granite Generated a Response

Open **Technical details** in Gradio or **Experiment diagnostics** in
Streamlit. A Granite-authored turn records at least one model pass with:

- pass label and raw output;
- prompt and output SHA-256;
- input-token count;
- runtime and CUDA device;
- exact base, adapter, and router identities;
- exposed action schema, calls, and results;
- response path and policy sources.

Zero model passes means Granite did not generate the response. Expected
zero-pass paths include OOD, classifier failure, and policy no-match.

## Local Streamlit

```bash
uv run scripts/retail_bank/run_local_streamlit.py
```

Open <http://127.0.0.1:8501>. The local runtime loads the Granite base with
bitsandbytes NF4 double quantization and attaches the immutable adapter. It
prefers:

```text
artifacts/banking-conversation-router-v6-hierarchical
```

Local-only default credentials are:

```text
alex.demo / alex-local-demo
maya.demo / maya-local-demo
```

Set stronger values with:

```bash
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-strong-value-1","maya.demo":"replace-with-strong-value-2"}'
uv run scripts/retail_bank/run_local_streamlit.py
```

Relevant overrides are `LOCAL_STREAMLIT_PORT`,
`LOCAL_ROUTER_ARTIFACT_DIR`, `RETAIL_BANK_MODEL_ID`,
`RETAIL_BANK_MODEL_REVISION`, `RETAIL_BANK_BASE_MODEL_ID`,
`RETAIL_BANK_BASE_MODEL_REVISION`, `RETAIL_BANK_ADAPTER_ID`,
`RETAIL_BANK_ADAPTER_REVISION`, `RETAIL_BANK_ROUTER_ID`,
`RETAIL_BANK_ROUTER_REVISION`, and `HF_TOKEN`.

## Gradio and ZeroGPU

The model event uses `@spaces.GPU(size="large", duration=90)`. An authenticated
chat smoke confirms that user code entered a ZeroGPU worker and generated a
response. The Gradio queue uses concurrency one for this low-traffic POC.

Plan a deployment with the immutable router placeholder replaced:

```bash
ADAPTER_REVISION=cc95e446af2b5e1d8d9df2751a8192613ad386e3
ROUTER_REVISION=7f6a0e77ad231233702039560ced007fdc68bd74

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

Without `--execute --allow-publish`, the helper prints and validates a plan.
Execution uploads only allowlisted POC files and persists exact runtime pins.

The current Space source/pin deployment is
`f018cad020a17e33be59992035c1418c4cf91a01`. Its runtime remains **PAUSED**;
the current OAuth token receives HTTP 401 when attempting a restart. Treat the
Space as deployed but unavailable, not READY, until authentication is repaired
and a remote smoke test passes.

## Verify

```bash
POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  uv run pytest -q poc/retail-bank-customer-service-poc/tests
```

Skip flags are test-only. Ordinary local launches and the deployed Space must
load the real model and router.
