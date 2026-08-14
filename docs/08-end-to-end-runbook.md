# End-to-End V5 Runbook

This is the active reproducible sequence. It intentionally uses the individual
V5 scripts because the retained release-orchestrator configuration still
describes the superseded V4 sequence.

Stop at any failed gate. Do not publish a downstream revision based on a failed
or ambiguous upstream artifact.

## 0. Record the Starting Identities

For the current V5 continuation:

```text
source commit:
  75b56ffff45e75ffbee11c0e0552dc35ae124d21

Granite base:
  spkc83/retail-bank-servicing-agent-9b
  1d56824995aa1adecfe20f62ca42fb1c0c443817

V5 Granite dataset:
  spkc83/retail-bank-servicing-alignment-sft
  40a0b68b9f746131ffff32a83e077fd7e4a344d1

Canonical policy corpus:
  sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a

V5 router dataset:
  spkc83/retail-bank-conversation-router-data
  8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc

V5 router:
  spkc83/retail-bank-conversation-router
  c8f154266612e79afe20af8abef25761fa56d589

Granite PEFT composition:
  base: spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817
  PEFT release: spkc83/retail-bank-servicing-agent-9b-peft@cc95e446af2b5e1d8d9df2751a8192613ad386e3
  adapter bundle commit: b4269445ce7b2b943d2d9531102166bf8840a074
```

Granite job `6a7f79531f5885ae605b96cc` completed. Merged FP16 and BF16 candidates
failed unchanged behavioral-parity gates, so the accepted release is the exact
base-plus-adapter PEFT composition above. Evaluation job
`6a7f89edc97db76cbdf31893` ran from source
`42c89ae6d6b6792268b36e2162c4b19688e4e617` and failed strict gates. Five
credential flags are evaluator false positives; two genuine behavioral
failures remain. A corrected evaluator and generalized incremental SFT are
underway.

## 1. Install and Validate the Repository

```bash
uv sync --extra dev --extra scale
uv lock --check
uv run ruff check .
```

Run the focused V5 tests:

```bash
PYTHONPATH=src uv run pytest -q \
  tests/test_banking_tool_sft_data.py \
  tests/test_banking_servicing_alignment_data.py \
  tests/test_banking_conversation_router.py \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py \
  tests/test_banking_conversation_router_training.py \
  tests/test_banking_tool_sft_worker.py \
  tests/test_banking_tool_sft_job.py

POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  uv run pytest -q poc/retail-bank-customer-service-poc/tests
```

Stop if any data, router, policy, state, action, or response-policy test fails.

## 2. Generate the V5 Data

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

Check the manifests:

```bash
python -m json.tool data/banking-v5-tool-sft/manifest.json >/dev/null
python -m json.tool data/banking-servicing-alignment-v5/manifest.json >/dev/null
python -m json.tool data/banking-conversation-router-v5-social-policy-generalization-candidate5/manifest.json >/dev/null
python -m json.tool data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json >/dev/null
```

Expected row counts:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Base tool SFT | 838 | 181 | 181 |
| Composite servicing alignment | 1,222 | 277 | 216 |
| Conversation router, generalized corpus | 19,363 | 5,056 | 6,171 |

Stop if counts, digests, leakage checks, or PII checks drift unexpectedly.

## 3. Publish and Pin Data

The current published revisions are already pinned:

```text
servicing alignment: 40a0b68b9f746131ffff32a83e077fd7e4a344d1
router data:         8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc
```

For a future changed dataset, publish once, then obtain the immutable commit
before training. The servicing generator supports explicit publication:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/prepare_servicing_alignment_data.py \
  --base-sft-dir data/banking-v5-tool-sft \
  --output-dir data/banking-servicing-alignment-v5 \
  --push-to-hub \
  --repo-id spkc83/retail-bank-servicing-alignment-sft
```

Publish router data with the Hub CLI:

```bash
hf upload spkc83/retail-bank-conversation-router-data \
  data/banking-conversation-router-v5-social-policy-generalization-candidate5 . \
  --type dataset \
  --commit-message "Publish V5 state-aware router data"
```

These commands change external state and were not rerun for this documentation
update. Never pass `main` as a training identity; capture the returned commit.

## 4. Train and Publish the CPU Router

Local training:

```bash
PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v5-social-policy-generalization-candidate5 \
  --output-dir artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5
```

Require `release_eligible: true` and an empty `release_gate_failures` list in
`artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5/metrics.json`
before publication.

The final published router is:

```text
c8f154266612e79afe20af8abef25761fa56d589
```

For a future changed artifact, publish with the new exact dataset revision:

```bash
ROUTER_DATA_REVISION=8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc

PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v5-social-policy-generalization-candidate5 \
  --output-dir artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5 \
  --publish \
  --data-revision "$ROUTER_DATA_REVISION" \
  --destination-id spkc83/retail-bank-conversation-router
```

## 5. Validate the Granite Worker

Dry-run plan:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-servicing-alignment-v5/manifest.json \
  --base-model spkc83/retail-bank-servicing-agent-9b \
  --base-revision 1d56824995aa1adecfe20f62ca42fb1c0c443817 \
  --output-dir artifacts/banking-servicing-agent-v5
```

Offline pipeline smoke:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-servicing-alignment-v5/manifest.json \
  --output-dir /tmp/harbor-granite-v5-smoke \
  --run-tiny-smoke
```

## 6. Reproduce Granite V5 Training

The active paid job was submitted with:

```bash
BASE_MODEL=spkc83/retail-bank-servicing-agent-9b \
BASE_REVISION=1d56824995aa1adecfe20f62ca42fb1c0c443817 \
DATASET_REPO=spkc83/retail-bank-servicing-alignment-sft \
HF_HUB_DEST=spkc83/retail-bank-servicing-agent-9b \
MAX_STEPS=750 \
LEARNING_RATE=2e-5 \
GRADIENT_ACCUMULATION_STEPS=2 \
CHECKPOINT_EVERY=250 \
TRACKIO_PROJECT=retail-bank-agent-v5 \
TRACKIO_RUN_NAME=granite-v5-grounded-dialogue-75b56ff \
bash scripts/retail_bank/run_remote_training_job.sh \
  75b56ffff45e75ffbee11c0e0552dc35ae124d21 \
  40a0b68b9f746131ffff32a83e077fd7e4a344d1
```

Inspect the completed job:

```bash
hf jobs inspect 6a7f79531f5885ae605b96cc
hf jobs logs 6a7f79531f5885ae605b96cc --tail 200
```

If it stops after a compatible checkpoint, use the checkpoint path as the
third launcher argument. See [04-training-and-recovery.md](04-training-and-recovery.md).

The unchanged parity gates rejected both merged candidates. This is a valid
stop for merged publication, not permission to weaken the gates. Finalize and
publish the validated adapter through
`scripts/retail_bank/hf_job_finalize_tool_sft_peft.py` instead.

## 7. Run Frozen Granite Evaluation

Evaluate the accepted PEFT composition:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/cloud_generate_tool_eval.py \
  --model-repo spkc83/retail-bank-servicing-agent-9b-peft \
  --model-revision cc95e446af2b5e1d8d9df2751a8192613ad386e3 \
  --base-model-repo spkc83/retail-bank-servicing-agent-9b \
  --base-model-revision 1d56824995aa1adecfe20f62ca42fb1c0c443817 \
  --adapter-repo spkc83/retail-bank-servicing-agent-9b-peft \
  --adapter-revision cc95e446af2b5e1d8d9df2751a8192613ad386e3 \
  --dataset-repo spkc83/retail-bank-servicing-alignment-sft \
  --dataset-revision 40a0b68b9f746131ffff32a83e077fd7e4a344d1 \
  --manifest data/banking-servicing-alignment-v5/manifest.json \
  --split test \
  --output-dir artifacts/banking-servicing-agent-v5-eval \
  --family granite \
  --device cuda \
  --dtype bf16 \
  --enforce-release-gates
```

Run the human scenarios in [06-evaluation.md](06-evaluation.md#human-release-scenarios).
The current frozen run failed, so stop here for release purposes. Local testing
below is diagnostic only until a replacement artifact passes every gate.

## 8. Test Locally

Pin the exact composition without editing source:

```bash
export RETAIL_BANK_MODEL_ID=spkc83/retail-bank-servicing-agent-9b-peft
export RETAIL_BANK_MODEL_REVISION=cc95e446af2b5e1d8d9df2751a8192613ad386e3
export RETAIL_BANK_BASE_MODEL_ID=spkc83/retail-bank-servicing-agent-9b
export RETAIL_BANK_BASE_MODEL_REVISION=1d56824995aa1adecfe20f62ca42fb1c0c443817
export RETAIL_BANK_ADAPTER_ID=spkc83/retail-bank-servicing-agent-9b-peft
export RETAIL_BANK_ADAPTER_REVISION=cc95e446af2b5e1d8d9df2751a8192613ad386e3
export RETAIL_BANK_ROUTER_ID=spkc83/retail-bank-conversation-router
export RETAIL_BANK_ROUTER_REVISION=c8f154266612e79afe20af8abef25761fa56d589

uv run scripts/retail_bank/run_local_streamlit.py
```

Verify diagnostics show the exact base, adapter-bundle, and router revisions,
`cuda:0`, quantization, raw model passes, actions, policy sources, and dialogue
state.

## 9. Deploy ZeroGPU

Plan first:

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

This plan is provided to verify the future deployment shape. Do not add
`--execute --allow-publish` for `cc95e446...`: it failed strict evaluation.
Deployment remains pending a new generalized adapter and corrected passing
evaluation.

## Final Stop Condition

The end-to-end pipeline is complete only when:

- all three local V5 datasets regenerate and validate;
- the published dataset revisions are captured;
- the V5 router remains release eligible and pinned;
- Granite training completes and merged candidates are accepted or explicitly
  rejected by unchanged parity gates;
- the validated immutable base-plus-adapter composition is captured;
- frozen model and orchestration gates pass;
- local Streamlit uses that exact revision successfully;
- ZeroGPU uses that exact revision successfully;
- diagnostics prove the base/adapter/router/Space identities and model-generated
  response path.
