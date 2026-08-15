# Granite V5 Training and Recovery

This runbook covers the guarded V5 Granite continuation job. The CPU router is
trained separately and is documented in [05-dual-head-router.md](05-dual-head-router.md).

## Current Job

| Field | Value |
| --- | --- |
| Job | `6a7f79531f5885ae605b96cc` |
| Status represented by these docs | completed |
| Hardware | `rtx-pro-6000` |
| Job timeout | five hours |
| Optimizer wall-clock limit | 14,400 seconds |
| Source commit | `75b56ffff45e75ffbee11c0e0552dc35ae124d21` |
| Dataset revision | `9d7aed545604bb42fb02b7a0919427a0ed2b81e2` |
| Policy corpus revision | `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a` |
| Base model revision | `1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| Maximum steps | 750 |
| Learning rate | `2e-5` |
| Gradient accumulation | 2 |
| Checkpoint interval | 250 |
| Trackio project | `retail-bank-agent-v5` |
| Training steps | 750 |
| Training loss | `0.13014758` |
| Evaluation loss | `0.3200804` |
| Token accuracy | `0.96240348` |
| Published PEFT candidate | release revision `cc95e446af2b5e1d8d9df2751a8192613ad386e3` |
| Adapter bundle commit | `b4269445ce7b2b943d2d9531102166bf8840a074` |

The release composes base `1d568249...` with the BF16 LoRA adapter at
`cc95e446...`. Both merged candidates were rejected by the unchanged
behavioral-parity gates, so neither merged FP16 nor merged BF16 weights are an
active artifact. Evaluation job `6a7f89edc97db76cbdf31893` ran against the
exact base-plus-adapter composition and failed strict gates. A corrected
evaluator and generalized incremental SFT are underway.

## Prerequisites

Before launching a new job:

1. Run the local data and worker tests.
2. Publish the exact V5 dataset and capture its immutable revision.
3. Push the source commit referenced by the bootstrap URL.
4. Confirm the base model and revision are immutable and downloadable.
5. Confirm `hf auth whoami` returns the intended account.
6. Make `HF_TOKEN` available as a Hugging Face Job secret, never a command-line
   literal.
7. Use a persistent Hub bucket so checkpoints survive job termination.

Verify local inputs:

```bash
hf auth whoami

PYTHONPATH=src uv run pytest -q \
  tests/test_banking_tool_sft_worker.py \
  tests/test_banking_tool_sft_job.py \
  tests/test_banking_tool_sft_continuation.py \
  tests/test_banking_tool_sft_release.py \
  tests/test_banking_tool_sft_peft_release.py \
  tests/test_banking_servicing_alignment_data.py
```

## Inspect the Plan Without Training

The worker defaults to dry-run mode:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-servicing-alignment-v5/manifest.json \
  --base-model spkc83/retail-bank-servicing-agent-9b \
  --base-revision 1d56824995aa1adecfe20f62ca42fb1c0c443817 \
  --output-dir artifacts/banking-servicing-agent-v5
```

The printed plan includes precision, LoRA target modules, record counts,
assistant-only masking, checkpoint cadence, merge behavior, and the remote
execution guard. It does not load the 8.79B model or publish anything.

Use the offline tiny smoke to validate tokenization and checkpoint metadata:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-servicing-alignment-v5/manifest.json \
  --output-dir /tmp/harbor-granite-v5-smoke \
  --run-tiny-smoke
```

## Submit a Guarded Hugging Face Job

[`run_remote_training_job.sh`](../scripts/retail_bank/run_remote_training_job.sh)
validates exact source, dataset, and base revisions, checks that the bootstrap
URL resolves, requests an RTX PRO 6000, applies a five-hour timeout, mounts the
persistent artifact bucket, and forwards `HF_TOKEN` as a secret.

For the currently submitted V5 configuration, the equivalent launcher inputs
are:

```bash
SOURCE_COMMIT=75b56ffff45e75ffbee11c0e0552dc35ae124d21
DATASET_REVISION=9d7aed545604bb42fb02b7a0919427a0ed2b81e2

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
  "$SOURCE_COMMIT" "$DATASET_REVISION"
```

This command creates a paid external job. The documentation records it for
reproduction; it was not re-submitted while editing these files.

The bootstrap script
[`hf_job_tool_sft.py`](../scripts/retail_bank/hf_job_tool_sft.py) downloads the
exact GitHub source and Hub dataset revisions into an isolated job directory,
then invokes the worker with all three execution guards:

```text
--execute-remote
--allow-remote-execution
RETAIL_BANK_ALLOW_REMOTE_TOOL_SFT=banking-v5-grounded-dialogue-sft
```

## Monitor and Capture Evidence

```bash
hf jobs inspect 6a7f79531f5885ae605b96cc
hf jobs logs 6a7f79531f5885ae605b96cc --tail 200
```

Record:

- job state and duration;
- source, dataset, and base revisions;
- Trackio run name and final train/eval metrics;
- actual step and wall-clock stop reason;
- checkpoint paths;
- adapter and merged artifact paths;
- merge/reload parity result;
- published adapter-bundle and metadata revisions.

A successful optimizer exit is not a release. Merge, reload parity, upload,
and frozen evaluation must also succeed.

## Checkpoints and Resume

The worker saves trainer checkpoints and a project fingerprint. The fingerprint
binds the base model/revision, dataset manifest, tokenizer template, precision,
LoRA configuration, sequence length, and optimization settings. Resume rejects
an incompatible fingerprint.

Pass a checkpoint path as the optional third launcher argument:

```bash
bash scripts/retail_bank/run_remote_training_job.sh \
  75b56ffff45e75ffbee11c0e0552dc35ae124d21 \
  9d7aed545604bb42fb02b7a0919427a0ed2b81e2 \
  /data/retail-bank-agent-9b-75b56fff/checkpoint-000250
```

Use the same environment overrides as the original job. Changing the dataset,
base revision, template, or LoRA shape is a new run, not a resume.

## Finalization and Publication

After SFT, the worker produced an adapter and attempted merged exports:

1. saves the adapter and tokenizer;
2. merges the adapter into the base with `safe_merge=True`;
3. saves merged FP16 weights;
4. reloads the merged checkpoint;
5. compares generation behavior with the adapter-backed model using unchanged
   behavioral-parity gates.

For this run, steps 2-5 rejected both FP16 and BF16 merged candidates. The
adapter itself remained valid. The dedicated finalizer
[`hf_job_finalize_tool_sft_peft.py`](../scripts/retail_bank/hf_job_finalize_tool_sft_peft.py):

1. verifies the selected step and training fingerprint;
2. verifies LoRA rank, alpha, dropout, target modules, base revision, dataset
   revision, template hash, and BF16 precision;
3. hashes the adapter and tokenizer bundle;
4. publishes the adapter files atomically;
5. records the immutable adapter-bundle commit separately from the final PEFT
   release revision.

The resulting identities are:

```text
base:             spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817
PEFT release:     spkc83/retail-bank-servicing-agent-9b-peft@cc95e446af2b5e1d8d9df2751a8192613ad386e3
adapter bundle:   b4269445ce7b2b943d2d9531102166bf8840a074
adapter SHA-256:  043b22c5... (full digest in release metadata)
```

The final PEFT release revision is passed to PEFT. Evaluation and deployment
must pin all four composition fields: base ID/revision and adapter ID/revision.

## Failure Recovery Table

| Failure | Evidence to inspect | Recovery |
| --- | --- | --- |
| Source bootstrap fails | job log before dependency install | Verify the exact GitHub commit and raw bootstrap URL; submit a new job after source is reachable. |
| Dataset download or digest fails | dataset repo/revision and manifest path | Verify `40a0b68...` and the manifest; do not bypass digest checks. |
| CUDA OOM | final allocation and batch settings | Resume from the latest compatible checkpoint with a smaller batch or QLoRA; treat the configuration change as a new fingerprint when required. |
| Wall-clock stop | last step and saved checkpoint | Resume from the persistent checkpoint using the original fingerprint. |
| Merge fails | adapter path and merge traceback | Preserve the adapter; rerun the explicit merge/recovery helper against the same base revision. |
| Merged behavioral parity fails | parity inputs and outputs | Reject the merged checkpoint; publish the validated unmerged adapter only if its independent release checks pass. |
| Adapter upload fails | complete validated adapter directory and Hub error | Retry publication from the preserved adapter; do not retrain solely for an upload failure. |
| Frozen evaluation fails | per-record prediction report | Keep the existing POC model pin; fix data/model behavior and produce a new immutable revision. |

No cleanup command is required for any recovery path. Preserve checkpoints and
logs until the release is complete.
