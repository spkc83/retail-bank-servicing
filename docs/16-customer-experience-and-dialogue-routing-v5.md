# Customer Experience and Dialogue Routing V5

This document is the implementation checklist for the next retail-banking
release. It corrects an observed training-data failure, adds grounded policy
answers, and makes conversational focus explicit without turning the router
into a workflow engine.

## Why V5 Is Needed

The prompt `Can you help me open a mortgage account?` is currently both a POC
preset and an SFT prompt. Its target answer is repeated 333 times across the
generated train, validation, and test files. A live response that matches that
target is produced by Granite, but it is learned near-verbatim from the SFT
corpus. It is not evidence that the model generalized to an unseen policy
question.

The current router also stops at classification. Its domain output can block a
confident external query, but its capability and relation outputs are only
diagnostics. No application state records the servicing task that was active
before a customer asks a temporary policy question.

V5 addresses both problems.

## Target Experience

The customer-facing experience uses a fictional bank identity and keeps the
prototype disclosure outside the conversation. Normal assistant replies must
not mention implementation details such as `synthetic`, `demo`, `mock`,
`backend`, model size, GPU type, router, or tool calls.

The assistant should:

- introduce itself using the fictional bank and assistant names;
- identify checking, savings, card, transfer, and servicing products clearly;
- acknowledge distress before helping with fraud, lost cards, failed payments,
  or similar events;
- use concise Markdown tables for account and transaction lists;
- answer policy questions only from the active knowledge-base revision;
- preserve an unfinished servicing task during a policy detour;
- return to that task when the customer asks to continue;
- keep account actions model-selected and schema-validated.

## Consolidated Change List

### 1. Correct the Granite training contract

- Replace the customer-facing SFT system prompt with the fictional bank and
  assistant identity.
- Remove prototype terminology from trainable assistant targets and tool
  descriptions. Internal record IDs and test usernames may retain technical
  names because they are not customer-facing targets.
- Replace static, no-tool policy answers with evidence-grounded policy-answer
  records.
- Generate multiple answer realizations instead of copying one answer hundreds
  of times.
- Add empathy, concise explanations, product naming, and table-format examples.
- Remove every exact POC preset from the training corpus.

### 2. Add a versioned policy knowledge base

Each policy chunk contains:

```json
{
  "chunk_id": "mortgage.application.overview.us.v1",
  "title": "Mortgage application overview",
  "product": "mortgage",
  "jurisdiction": "US",
  "effective_from": "2026-01-01",
  "effective_to": null,
  "text": "...",
  "corpus_revision": "sha256:..."
}
```

The retriever returns only approved chunks from one immutable corpus revision.
It must support an explicit no-match result and must expose the selected chunk
IDs in diagnostics.

Granite receives the retrieved chunks as authoritative policy context with
account-action tools disabled. Its answer must cite only returned chunk IDs. A
policy-specific validator rejects missing, invented, stale, or unsupported
citations and allows one evidence-preserving repair pass.

### 3. Refine the multi-head router

Retain the shared DistilBERT cross-encoder and three semantic heads:

```text
current turn + recent visible history + prior dialogue-state header
  -> shared DistilBERT encoder
     -> domain head
     -> fine-grained intent head
     -> multi-label relation head
```

The domain head remains `in_scope` versus `external_ood` with calibrated
in-domain, uncertain, and OOD bands.

The fine-grained intent labels are:

- `view_accounts`;
- `view_cards`;
- `freeze_card`;
- `replace_card`;
- `view_transactions`;
- `dispute_transaction`;
- `view_transfers`;
- `cancel_transfer`;
- `view_service_cases`;
- `policy_knowledge`;
- `conversation`;
- `other_banking`.

The application derives the broad lane from the intent label. There is no
separate lane head whose output could contradict the intent head.

The relation labels are:

- `context_dependent`;
- `agent_repair`;
- `topic_shift`;
- `clarification_answer`;
- `resume_previous_service`.

There is no learned transition head. A transition is a deterministic comparison
between calibrated current-turn observations and the prior dialogue state.

### 4. Add a bounded dialogue-state machine

The runtime stores at most one pending servicing task:

```json
{
  "version": 1,
  "pending_servicing": {
    "intent": "dispute_transaction",
    "anchor_user_message": "I need to dispute a purchase.",
    "anchor_assistant_message": "Which transaction should I look for?",
    "phase": "awaiting_user"
  },
  "knowledge_detour_active": true
}
```

State rules:

- A confident servicing intent starts or replaces the pending task.
- A same-intent follow-up keeps it active.
- A policy intent preserves the pending task and activates a knowledge detour.
- A valid resume relation clears the detour and pins the original servicing
  exchange into Granite's token-budgeted context.
- A successful completing action clears the pending task.
- A confident different servicing intent replaces the old task; it must not
  later resume unexpectedly.
- OOD, social, uncertain, and classifier-failure turns do not mutate task state.
- Router output never supplies tool arguments or authorizes an action.

This deliberately avoids an unlimited task stack. More than one suspended
servicing task is outside the V5 requirement and should be added only after
held-out evidence demonstrates a need.

### 5. Harden the shared inference harness

- Route policy turns through retrieval and Granite policy generation.
- Keep Granite tool selection for customer-specific reads and actions.
- Pin the pending task's original exchange during resumption.
- Preserve existing tool schema, call-count, execution, and grounding checks.
- Add hard validation for internal-language leakage and policy citations.
- Keep tone, empathy, and brand as measured generation qualities; do not replace
  normal replies with hard-coded scripts.
- Continue host-rendering read results as tables while reporting honestly that
  Granite selected the tool and the application rendered verified data.

### 6. Update both POC surfaces

- Apply one shared fictional-bank identity to Gradio and Streamlit.
- Show a subtle, separate prototype/test-data notice outside chat.
- Present account types and product information in customer-friendly cards and
  tables.
- Collapse technical diagnostics and expose model revision, CUDA execution,
  raw Granite output, response path, router scores, dialogue state, and policy
  citations.
- Reset dialogue state together with the synthetic bank session.
- Use an accurate provenance label for each response: direct Granite generation,
  retrieval-grounded Granite generation, or Granite-selected/host-rendered data.

### 7. Replace contaminated evaluation cases

The frozen V5 evaluation corpus is authored independently and is never merged
into SFT. Entire trajectories and paraphrase families remain in one split.
Preparation fails on exact or fuzzy overlap with training prompts, final
answers, templates, or POC presets.

Required held-out conversations include:

- dispute start -> policy question -> resume -> dispute creation;
- card replacement -> fee policy -> resume -> replacement;
- pending dispute -> explicit switch to card freeze, with no stale dispute call;
- standalone policy question, with no phantom servicing task;
- long policy detour beyond the normal visible-history window -> resume;
- OOD and social detours that preserve but do not mutate the pending task;
- policy no-match and stale-policy cases;
- account/transaction reads with exact table schemas;
- paired counterfactual conversations with identical wording and different tool
  results.

Hard release gates include:

- zero customer-visible internal-language leaks;
- zero invalid policy citations;
- zero stale or abandoned-intent tool calls;
- exact pending-task transition accuracy on frozen trajectories;
- perfect table-schema and grounded-fact checks;
- no regression in existing OOD, tool name, argument, executable trajectory,
  clarification, and action-grounding gates;
- empathy and brand compliance on every applicable row without exact-string
  matching.

## Training and Release Sequence

1. Generate and validate the new Granite SFT corpus locally.
2. Generate the router corpus from pre-response dialogue state only.
3. Run contamination, PII, schema, replay, retrieval, and frozen-evaluation
   checks locally.
4. Train and calibrate the router on the local TITAN V.
5. Publish immutable dataset revisions.
6. Run incremental LoRA/PEFT SFT from the released V4 Granite checkpoint on a
   Hugging Face GPU Job with Trackio, checkpoints, Hub persistence, and a
   five-hour timeout cap.
7. Run exact frozen generation and executable tool/retrieval evaluation against
   the new model revision.
8. Publish model and router revisions only if every gate passes.
9. Pin both POCs to the immutable revisions, deploy the ZeroGPU Space, and
   verify live CUDA/model/retrieval evidence.

## Stop Condition

V5 is complete only when an unseen policy question is answered from a cited KB
chunk, a pending servicing task survives that detour and resumes correctly, the
new model produces customer-facing branded and empathetic language without
prototype terminology, all frozen gates pass, and both local and HF POCs expose
verifiable response provenance.
