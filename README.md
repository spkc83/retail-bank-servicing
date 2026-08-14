# Harborlight Retail-Bank Servicing Agent

This repository implements one V5 conversational banking pipeline:

- an IBM Granite 8.79B generative agent adapted with PEFT/LoRA;
- a CPU DistilBERT cross-encoder with three heads: banking domain, 12 fine
  intents, and five independent conversation relations;
- deterministic, bounded dialogue state for one pending servicing task and an
  optional policy detour;
- a versioned policy knowledge base used for retrieval-grounded answers;
- Gradio/ZeroGPU and Streamlit interfaces branded as **Harborlight Bank**, with
  **Harbor** as the assistant.

All customer and bank records are fictional. The POC does not connect to a
real bank.

## V5 Release Status

| Component | Repository or local path | Immutable revision/status |
| --- | --- | --- |
| Generalized V5 router | [spkc83/retail-bank-conversation-router](https://huggingface.co/spkc83/retail-bank-conversation-router) | `c8f154266612e79afe20af8abef25761fa56d589` |
| Generalized V5 router data | [spkc83/retail-bank-conversation-router-data](https://huggingface.co/datasets/spkc83/retail-bank-conversation-router-data) | `8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc` |
| Canonical-policy Granite SFT data | [spkc83/retail-bank-servicing-alignment-sft](https://huggingface.co/datasets/spkc83/retail-bank-servicing-alignment-sft) | `40a0b68b9f746131ffff32a83e077fd7e4a344d1` |
| Canonical policy corpus | `poc/retail-bank-customer-service-poc/policy_knowledge.json` | `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a` |
| Granite V5 PEFT adapter | `spkc83/retail-bank-servicing-agent-9b-peft` | release revision `cc95e446af2b5e1d8d9df2751a8192613ad386e3`; adapter bundle commit `b4269445ce7b2b943d2d9531102166bf8840a074` |
| Granite evaluation | PEFT base-plus-adapter composition | job `6a7f89edc97db76cbdf31893` failed strict gates; replacement work underway |

Training job `6a7f79531f5885ae605b96cc` completed 750 steps from Stage-2 base
`1d568249...`. The release keeps the trained BF16 LoRA adapter separate because
both merged FP16/BF16 candidates failed unchanged behavioral-parity gates.
Evaluation job `6a7f89edc97db76cbdf31893` loaded that exact composition from
source `42c89ae...` but failed strict gates. Five credential flags were
evaluator false positives; the two genuine failures were a false success claim
after an action error and redundant clarification for a history-resolved card
replacement. A corrected evaluator and generalized incremental SFT are
underway; deployment remains blocked.

## Request Flow

```text
authenticated customer turn + recent visible history + prior dialogue state
  -> V5 CPU router
     -> domain head: banking or out of domain
     -> intent head: one of 12 fine intents
     -> relation head: zero or more of 5 conversation relations
  -> high-confidence OOD: fixed banking-scope response; Granite is not called
  -> accepted turn: deterministic dialogue-state transition
     -> policy lane: retrieve versioned policy chunks
        -> Granite, with banking actions disabled
        -> require citations such as [Policy: mortgage.opening.us.v1]
     -> servicing/conversation lane: Granite may answer or emit a tagged-JSON action
        -> execute against session-isolated fictional bank state
        -> render exact read tables or validate the grounded action response
  -> reject internal implementation language; allow one tools-disabled repair
```

The classifier observes history and prior state, but its intent prediction
does not enter the Granite prompt and does not authorize an action. Granite
selects public banking actions and arguments. The dialogue-state machine only
tracks continuity: it can preserve one pending servicing task across a policy
question and pin the original user/assistant exchange when that task resumes.

## Documentation

Read the active V5 path in this order:

1. [System overview](docs/01-system-overview.md)
2. [Data generation](docs/02-data-generation.md)
3. [Granite and PEFT](docs/03-model-and-peft.md)
4. [Training and recovery](docs/04-training-and-recovery.md)
5. [Three-head state-aware router](docs/05-dual-head-router.md)
6. [Evaluation](docs/06-evaluation.md)
7. [Inference and POC](docs/07-inference-and-poc.md)
8. [End-to-end V5 runbook](docs/08-end-to-end-runbook.md)
9. [Code and file map](docs/reference/file-map.md)

The older [conversation-router V4](docs/09-conversation-router-v4.md) and
[servicing-alignment V4](docs/10-servicing-alignment-v4.md) documents are
retained only as superseded historical references. They are not active
instructions.

## Local Setup

```bash
uv sync --extra dev --extra scale
```

Generate the three V5 datasets in dependency order:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/prepare_tool_sft_data.py \
  --output-dir data/banking-v5-tool-sft

PYTHONPATH=src uv run python scripts/retail_bank/prepare_servicing_alignment_data.py \
  --base-sft-dir data/banking-v5-tool-sft \
  --output-dir data/banking-servicing-alignment-v5

PYTHONPATH=src uv run python scripts/retail_bank/prepare_conversation_router_data.py \
  --sft-dir data/banking-servicing-alignment-v5 \
  --output-dir data/banking-conversation-router-v5-social-policy-generalization-candidate5 \
  --source-lock data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json \
  --expected-release-lock data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json
```

Train the V5 router locally:

```bash
PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v5-social-policy-generalization-candidate5 \
  --output-dir artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5
```

Run the local Streamlit POC on a compatible CUDA GPU:

```bash
uv run scripts/retail_bank/run_local_streamlit.py
```

The local runtime loads Granite with bitsandbytes NF4 double quantization and
prefers the generalized V5 router artifact under
`artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5`.
See the
[POC README](poc/retail-bank-customer-service-poc/README.md) for credentials,
environment overrides, and diagnostics.

## Verification

```bash
PYTHONPATH=src uv run pytest -q \
  tests/test_banking_tool_sft_data.py \
  tests/test_banking_servicing_alignment_data.py \
  tests/test_banking_conversation_router.py \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py \
  tests/test_banking_conversation_router_training.py

POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  uv run pytest -q poc/retail-bank-customer-service-poc/tests
```

Remote training, publication, and deployment require credentials and incur
external side effects. Follow the guarded commands in
[training and recovery](docs/04-training-and-recovery.md) and capture every new
immutable revision before continuing downstream.

## License

Repository code and the self-authored generative corpus are MIT licensed. The
router also uses the pinned CLINC150 source described in the V5 source lock;
review upstream terms before redistribution.
