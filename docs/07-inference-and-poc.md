# V5 Inference and POC

The POC has two interfaces over the same orchestration design:

- Gradio on Hugging Face ZeroGPU;
- Streamlit on a local CUDA GPU.

Both present Harborlight Bank and Harbor, use two fictional authenticated
profiles, load the released V5 CPU router, preserve bounded dialogue state,
retrieve from the same policy corpus, run Granite through the same action
service, and expose developer diagnostics separately from the chat.

## Runtime Pins

| Component | ID | Revision/status |
| --- | --- | --- |
| Generalized router | `spkc83/retail-bank-conversation-router` | `c8f154266612e79afe20af8abef25761fa56d589` |
| Granite PEFT release | `spkc83/retail-bank-servicing-agent-9b-peft` | `cc95e446af2b5e1d8d9df2751a8192613ad386e3`; adapter bundle commit `b4269445...` |
| Granite base | `spkc83/retail-bank-servicing-agent-9b` | `1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| Model loading mode | PEFT base plus adapter | BF16 on ZeroGPU; NF4-quantized base plus adapter locally |
| Policy corpus | `policy_knowledge.json` | `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a` |

Both runtimes pin all four PEFT composition fields. They load the base first,
attach the adapter with `autocast_adapter_dtype=False`, and expose
`model_loading_mode: peft_adapter` in diagnostics. The final PEFT revision
`cc95e446...` is the model identity; `b4269445...` is the adapter-files bundle
commit recorded in its release metadata. Strict
evaluation failed, so these pins describe a testable candidate, not an
approved deployment.

## Turn Lifecycle

1. The UI obtains the authenticated username and session ID.
2. The runtime canonicalizes visible conversation history.
3. The router receives the current text, up to three recent visible exchanges,
   and the trusted pre-turn dialogue state.
4. OOD and classifier-error routes return direct governed responses without a
   Granite call.
5. An accepted fine intent and relations update the bounded state machine.
6. The policy lane retrieves versioned chunks and calls Granite with banking
   actions disabled.
7. Other accepted/uncertain turns call Granite with normal action schemas.
8. The action loop validates and executes up to eight calls.
9. Read-only lists are rendered as Markdown tables. Other answers are checked
   against action results.
10. Policy answers require allowed `[Policy: id]` citations.
11. All customer-facing answers pass the internal-language validator; one
    tools-disabled repair is allowed.
12. A successful action matching the pending servicing intent clears state.
13. The UI records route scores, state, policy sources, action calls/results,
    model-pass hashes, raw output, device, and immutable revisions in the
    diagnostics panel.

## Routing and State Behavior

| Router result | State behavior | Granite behavior |
| --- | --- | --- |
| `out_of_domain` | unchanged | not called |
| `classifier_error` | unchanged | not called |
| `uncertain` | unchanged | normal model turn |
| in-domain `conversation` | unchanged | direct conversation or clarification |
| in-domain servicing intent | start, continue, or replace one pending task | normal action-capable turn |
| in-domain `policy_knowledge` | preserve pending task and activate detour | retrieved policy context; actions disabled |
| `resume_previous_service` after detour | deactivate detour and restore servicing lane | original servicing exchange pinned into context |

The router intent does not enter the model prompt. It influences only the
bounded orchestration lane and state transition. Granite still chooses whether
to call an action and supplies its public arguments.

## Token-Budgeted Conversation

[`select_token_budgeted_context`](../poc/retail-bank-customer-service-poc/model_service.py)
keeps complete interaction groups rather than truncating individual messages.
The default input budget is 8,192 tokens. During a resumed task, the actual
original user/assistant exchange is a pinned group and survives ordinary
oldest-first trimming.

This approach supports long conversations without sending unlimited history.
It also prevents the state machine from inventing a summary that Granite never
saw.

## Policy Retrieval and Generation

[`PolicyKnowledgeBase`](../poc/retail-bank-customer-service-poc/policy_retrieval.py)
loads a schema-versioned JSON corpus, recomputes its SHA-256 revision, rejects
tampering, and ranks chunks through deterministic weighted lexical overlap.
Each match includes product, jurisdiction, effective dates, score, revision,
and citation.

For a mortgage question, a retrieved chunk might be:

```text
[Policy: mortgage.opening.us.v1]
Customers may begin a mortgage application online or with a mortgage specialist ...
```

Granite receives the passage and no action schemas. The response validator
requires at least one returned citation, rejects invented IDs, rejects internal
implementation language, and rejects numeric claims absent from the evidence.

If retrieval finds no match, the application returns the policy-not-found
response and does not ask Granite to answer from memory.

## Action-Capable Granite Turn

Normal servicing calls pass the nine public action schemas. Granite can answer
directly or emit:

```text
<tool_call>{"name":"cancel_transfer","arguments":{"recipient":"River Consulting"}}</tool_call>
```

The harness validates the wire format and executes the action against the
authenticated session's fictional SQLite state. The next Granite pass sees the
correlated result. Dependent actions are executed one pass at a time so later
arguments can depend on earlier results.

The harness renders successful read-only actions as exact tables. It validates
action answers for essential facts and private-ID leakage. It never substitutes
a CPU-authored servicing answer when Granite fails.

## Proving Granite Generated the Response

Open **Technical details** in Gradio or **Experiment diagnostics** in
Streamlit. For every Granite pass, the panel records:

- label such as `base`, `grounded_final`, `policy_grounded`, or a repair pass;
- input token count;
- prompt SHA-256;
- raw output and raw-output SHA-256;
- runtime device and CUDA device;
- exact base, adapter-bundle, and router revisions;
- emitted actions and execution results;
- response path, policy sources, and dialogue state;
- visible-response SHA-256.

A turn with zero recorded model passes was not generated by Granite. This is
expected for high-confidence OOD, classifier failure, or policy no-match.

## Local Streamlit

Start the local app from the repository root:

```bash
uv run scripts/retail_bank/run_local_streamlit.py
```

Open <http://127.0.0.1:8501>. The launcher installs its pinned runtime through
the UV script metadata, refuses test skip flags unless explicitly allowed, and
loads:

- the pinned Granite base with bitsandbytes NF4 double quantization and the
  pinned LoRA adapter attached without merging;
- `artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5`
  when present, otherwise the published generalized router;
- the versioned policy JSON;
- the same fictional bank state and controller used by tests.

Local default credentials are displayed on the login page:

```text
alex.demo / alex-local-demo
maya.demo / maya-local-demo
```

Override them with a JSON object containing exactly those two usernames and
different passwords of at least 12 characters:

```bash
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-strong-value-1","maya.demo":"replace-with-strong-value-2"}'
uv run scripts/retail_bank/run_local_streamlit.py
```

Useful local overrides:

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

## Gradio and ZeroGPU

[`app.py`](../poc/retail-bank-customer-service-poc/app.py) runs one queued model
event at a time. The model turn uses
`@spaces.GPU(size="large", duration=90)`. A separate 30-second probe confirms
that user code entered a ZeroGPU worker and reports the packed model/device
metadata.

The Space requires `DEMO_AUTH_JSON` as a variable or secret. It must define
exactly `alex.demo` and `maya.demo`, use different passwords, and use at least
12 characters per password.

Plan a deployment without publishing:

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

The helper uploads only allowlisted POC files and persists the exact base,
adapter, router, dtype, and Space commit identities as Space variables. The
command above only prints a plan. Do not add `--execute --allow-publish`: the
current PEFT candidate failed strict evaluation. Deployment remains pending a
new evaluated artifact.

## Reset and Isolation

Each authenticated browser session has independent conversation and bounded
dialogue state. The fictional bank registry isolates mutable records by
username and session ID. **Start over** or **Reset demo** clears the complete
conversation, pending task, detour flag, and session bank changes.

Static authentication is only a POC profile selector. It is not a production
identity, authorization, or security design.

## POC Verification

Run without loading Granite or downloading the router:

```bash
POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  uv run pytest -q poc/retail-bank-customer-service-poc/tests
```

The skip flags are test-only. The local launcher refuses them during an
ordinary start, and production Space configuration should not set them.
