# Harborlight Retail-Bank Agent Documentation

These documents describe the active V5 implementation. A junior developer can
use them to reproduce the data, train the CPU router, run or recover Granite
PEFT training, evaluate immutable artifacts, and start either POC interface.

## Current Artifact Ledger

| Component | Repository | Revision/status |
| --- | --- | --- |
| Generalized three-head router | `spkc83/retail-bank-conversation-router` | `c8f154266612e79afe20af8abef25761fa56d589` |
| Generalized router dataset | `spkc83/retail-bank-conversation-router-data` | `8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc` |
| Canonical-policy Granite SFT dataset | `spkc83/retail-bank-servicing-alignment-sft` | `40a0b68b9f746131ffff32a83e077fd7e4a344d1` |
| Canonical policy corpus | `policy_knowledge.json` | `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a` |
| Granite V5 PEFT adapter | `spkc83/retail-bank-servicing-agent-9b-peft` | release `cc95e446af2b5e1d8d9df2751a8192613ad386e3`; bundle `b4269445ce7b2b943d2d9531102166bf8840a074`; base `1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| Frozen evaluation | exact PEFT composition | job `6a7f89edc97db76cbdf31893` failed strict gates; replacement work underway |

Training completed, but unchanged behavioral-parity gates rejected both
merged FP16 and BF16 candidates. The active model is therefore the exact
Stage-2 base plus immutable BF16 LoRA adapter. Strict evaluation failed: five
credential findings were evaluator false positives and two genuine behavioral
failures remain. This documentation does not claim a passing result or approve
deployment.

## Read in This Order

1. [System overview](01-system-overview.md) — components, trust boundaries, and
   one request from UI to response.
2. [Data generation](02-data-generation.md) — the V5 tool SFT, servicing
   alignment, router data, split isolation, and concrete records.
3. [Model and PEFT](03-model-and-peft.md) — Granite adaptation, assistant-only
   loss, tagged-JSON actions, and the published V5 checkpoint.
4. [Training and recovery](04-training-and-recovery.md) — local validation,
   guarded RTX PRO 6000 training, checkpoints, and publication.
5. [Three-head router](05-dual-head-router.md) — state-aware cross-encoder,
   labels, calibration, metrics, and runtime use. The filename is retained for
   stable links; the V5 router has three heads.
6. [Evaluation](06-evaluation.md) — router gates, frozen Granite evaluation,
   conversation trajectories, policy citations, and contamination checks.
7. [Inference and POC](07-inference-and-poc.md) — Gradio/ZeroGPU, local
   Streamlit, dialogue state, policy retrieval, action loop, and diagnostics.
8. [End-to-end runbook](08-end-to-end-runbook.md) — the reproducible sequence
   and stop conditions.
9. [File map](reference/file-map.md) — concepts mapped to implementation files.

The [V4 router](09-conversation-router-v4.md) and
[V4 servicing-alignment](10-servicing-alignment-v4.md) documents are retained
as explicitly superseded history. Do not use their artifact revisions,
dataset directories, thresholds, or commands for V5 work.

## V5 Repository Map

| Path | Purpose |
| --- | --- |
| [`../data/banking-v5-tool-sft`](../data/banking-v5-tool-sft) | Base tool-use, direct-answer, policy, and OOD SFT records. |
| [`../data/banking-servicing-alignment-v5`](../data/banking-servicing-alignment-v5) | Composite Granite V5 continuation corpus. |
| [`../data/banking-conversation-router-v5-social-policy-generalization-candidate5`](../data/banking-conversation-router-v5-social-policy-generalization-candidate5) | Generalized router train, validation, and test rows. |
| [`../data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json`](../data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json) | Pinned CLINC source and generalized split digests. |
| [`../artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5`](../artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5) | Local artifact published as router revision `c8f15426...`. |
| [`../poc/retail-bank-customer-service-poc`](../poc/retail-bank-customer-service-poc) | Shared Gradio/ZeroGPU and Streamlit implementation. |
| [`../scripts/retail_bank`](../scripts/retail_bank) | Data, router, Granite training, evaluation, and deployment entry points. |
| [`../src/hello_slm`](../src/hello_slm) | Reusable generators, schemas, model definition, and evaluation helpers. |

## Safe Local Checks

```bash
uv sync --extra dev --extra scale

PYTHONPATH=src uv run pytest -q \
  tests/test_banking_tool_sft_data.py \
  tests/test_banking_servicing_alignment_data.py \
  tests/test_banking_conversation_router.py \
  tests/test_banking_conversation_router_data.py

POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  uv run pytest -q poc/retail-bank-customer-service-poc/tests
```

Paid jobs and Hub publication are intentionally absent from this quick check.
Use the guarded runbook only when exact source, data, base-model, and
destination revisions have been recorded.
