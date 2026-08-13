# Retail Bank Servicing Agent Developer Docs

These docs explain the released Granite PEFT retail-bank servicing agent for a
junior developer. They describe the current code, cards, scripts, released Hub
artifacts, evaluation evidence, and ZeroGPU deployment path.

## What This Repository Builds

The repository builds a synthetic retail-bank customer-service demonstration
with two model components:

- an 8.79B parameter Granite generative agent trained in two SFT stages:
  initial IBM Granite base-tool SFT, followed by v4 servicing remediation SFT
  for observed conversation and tool-use failures;
- a CPU history-aware DistilBERT cross-encoder router that detects supported
  banking requests, emits servicing-capability diagnostics, and scores
  conversation relations such as context dependence and agent repair.

The generative agent owns normal conversation, clarification, tool selection,
public tool arguments, and final response wording. The router does not select
tools and does not provide arguments to the model. See
[01-system-overview.md](01-system-overview.md) for the request flow.

## Public Artifacts

| Artifact | Location | Revision |
| --- | --- | --- |
| 8.79B servicing agent | `spkc83/retail-bank-servicing-agent-9b` | `1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| Stage-1 Granite tool-use checkpoint | `spkc83/retail-bank-agent-9b` | `085df3d089cfadd77424b548542da0390a54a23e` |
| Initial tool-use SFT dataset | `spkc83/retail-bank-agent-sft` | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` |
| Servicing-remediation SFT dataset | `spkc83/retail-bank-servicing-alignment-sft` | `0ce32f9c7a3edff227005e5b89b089947b87625a` |
| Prompt-identical training data revision | `spkc83/retail-bank-servicing-alignment-sft` | `fea8aa1cda716954eb7322325e2be25c9f570ea3` |
| History-aware router | `spkc83/retail-bank-conversation-router` | `9e090c0fa21cebbaa03a431a7ce61e656c0739fe` |
| Router dataset | `spkc83/retail-bank-conversation-router-data` | `e9a64a2e7f2b622d5412c15eac4618ceca2150da` |
| Public POC Space | `spkc83/retail-bank-servicing-poc` | runtime diagnostics expose the Space commit |

The same artifact IDs appear in the root [README](../README.md), the model
cards under [../model_cards](../model_cards), and the data cards under
[../data_cards](../data_cards).

## Read In This Order

1. [System overview](01-system-overview.md)

   Learn the component boundaries, runtime request path, and where each part
   lives in the repository.

2. [Data generation](02-data-generation.md)

   Learn how the governed synthetic tool-use dataset, v4 servicing-remediation
   data, and router data are generated, and which files prove provenance.

3. [Model and PEFT](03-model-and-peft.md)

   Learn the Granite base model identity, two-stage LoRA training path,
   assistant-only masking, Granite tool wire, and merged/adapted release layout.

4. [Instruction Fine-Tuning and PEFT Design](12-instruction-fine-tuning-and-peft.md)

   Build intuition for instruction SFT, causal loss, assistant-only masking,
   LoRA updates, trainable versus frozen weights, two-stage adaptation, and the
   merged inference checkpoint through concrete examples.

5. [Training, continuation, and recovery](04-training-and-recovery.md)

   Learn the guarded local and paid-job paths, checkpoint resume, continuation,
   rescore, recovery, merge parity, and publication gates.

6. [Conversation router](05-dual-head-router.md)

   Learn the shared DistilBERT cross-encoder, domain/capability/relation heads,
   governed data, calibration, release gates, and serving thresholds.

7. [Frozen evaluation](06-evaluation.md)

   Learn the two-phase frozen evaluation contract, exact rescore correctness,
   and metrics needed for release.

8. [Inference and ZeroGPU POC](07-inference-and-poc.md)

   Learn model loading, token-budgeted history, static demo auth, the
   model-owned tool loop, synthetic SQLite state, and inference diagnostics.

9. [End-to-end runbook](08-end-to-end-runbook.md)

   Follow the complete install, data, training, rescore, router, evaluation,
   local POC, and deployment sequence.

10. [Conversation Router v4](09-conversation-router-v4.md)

   Understand the released cross-encoder, leakage-safe data contract,
   multi-label conversation relations, local training path, and rollout gate.

11. [Granite Servicing Alignment v4](10-servicing-alignment-v4.md)

    Understand the composite continuation-SFT data, use-case alignment,
    prompt-equivalent corrected dataset revision, and release stop condition.

12. [End-to-End Flow by Example](11-end-to-end-flow-by-example.md)

    Follow a concrete servicing request from behavior design through SFT and
    router records, masking, training, evaluation, runtime routing, tools, and
    final response generation.

13. [Leakage-Controlled Counterfactual Evaluation](13-counterfactual-evaluation.md)

    Build the evaluation-only post-training suite, understand paired unseen
    facts and contamination gates, run the pinned 9B model in 4-bit locally,
    and interpret the strict benchmark result.

14. [Questions and Answers](14-questions-and-answers.md)

    Find concise, example-driven answers to recurring implementation questions,
    including how the SFT `messages` array becomes model input during training
    and inference.

15. [ASR Output to Granite Fine-Tuning Data](15-asr-to-sft-pipeline.md)

    Convert reviewed speech-recognition utterances into leakage-safe,
    trainer-compatible tool-use overlays while preserving validated semantic
    targets, provenance, and split groups.

Use the [file map](reference/file-map.md) to jump from concepts to code and the
[artifact ledger](reference/artifacts.md) for immutable revisions and hashes.
The [learning resources](reference/learning-resources.md) annotate official
documentation and primary papers behind the design.
The [data-leakage audit](reference/data-leakage-audit.md) explains why the
released score is a protocol regression result rather than a clean
generalization benchmark.

## Repository Map

| Path | Purpose |
| --- | --- |
| [../configs/banking-tool-sft-granite.toml](../configs/banking-tool-sft-granite.toml) | Granite PEFT training configuration. |
| [../data/banking-v3-tool-sft](../data/banking-v3-tool-sft) | Generated local copy of the initial tool-use SFT dataset. |
| [../data/banking-servicing-alignment-v4](../data/banking-servicing-alignment-v4) | Generated local composite servicing-remediation SFT dataset. |
| [../data/banking-conversation-router-v4](../data/banking-conversation-router-v4) | Generated local history-aware router data. |
| [../data/banking-counterfactual-eval-v1](../data/banking-counterfactual-eval-v1) | Evaluation-only counterfactual benchmark with unseen SFT/POC facts. |
| [../data/sources](../data/sources) | Tracked source locks for governed data preparation. |
| [../examples/asr](../examples/asr) | Synthetic reviewed ASR input examples for the overlay pipeline. |
| [../data_cards](../data_cards) | Dataset documentation. |
| [../model_cards](../model_cards) | Model documentation. |
| [../poc/retail-bank-customer-service-poc](../poc/retail-bank-customer-service-poc) | Gradio/ZeroGPU customer-service POC. |
| [../scripts/retail_bank](../scripts/retail_bank) | Data, training, recovery, evaluation, rescore, and Hub job entry points. |
| [../src/hello_slm](../src/hello_slm) | Shared package code for data, tool wire, evaluator, and router. |
| [../tests](../tests) | Repository regression tests. |

The active scripts use the non-versioned `scripts/retail_bank` path. The
retained `data/banking-v3-tool-sft` name is a published dataset-schema
identifier, not an alternate model implementation.

## Local Setup

Use Python 3.11 or newer for the repository package.

```bash
python -m pip install -e '.[dev]'
```

Install the larger training stack only when you need local or remote model
training helpers. The optional extra is named `scale` in
[../pyproject.toml](../pyproject.toml).

## Common Verification Commands

Run the targeted repository checks from the root directory.

```bash
python -m pytest -q \
  tests/test_banking_tool_wire.py \
  tests/test_banking_tool_sft_worker.py \
  tests/test_banking_conversation_router.py
```

Run the POC checks without loading the 8.79B model or router:

```bash
POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  pytest -q poc/retail-bank-customer-service-poc/tests
```

Broader release checks are listed in the root [README](../README.md). Remote
Hugging Face Jobs and live ZeroGPU inference also require credentials, GPU
allocation, and exact revisions from the relevant script arguments.
