# Granite V5 Training and Recovery

This runbook covers the guarded V5 Granite continuation job. The CPU router is
trained separately and is documented in [05-hierarchical-router.md](05-hierarchical-router.md).

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
URL resolves, requests an RTX PRO 6000, applies the `JOB_TIMEOUT` cap (default
five hours), mounts the persistent artifact bucket, and forwards `HF_TOKEN` as a
secret. It also requires `HF_HUB_DEST` to be set explicitly and to differ from
`BASE_MODEL`, so a from-scratch run can no longer overwrite the repository it
trains from.

### The spend gate

Every launch is priced before it is submitted. The launcher converts
`JOB_TIMEOUT` to hours, multiplies by `GPU_HOURLY_USD` (default `2.75`, the
rtx-pro-6000 rate), prints the worst case, and then refuses the run twice over:
above `MAX_JOB_COST_USD` (default `5.00`) it exits, and without `CONFIRM_SPEND=1`
it exits. `DRY_RUN=1` performs every validation and prints the exact `hf jobs`
command without submitting it.

```bash
DRY_RUN=1 CONFIRM_SPEND=1 JOB_TIMEOUT=80m ... \
  bash scripts/retail_bank/run_remote_training_job.sh <commit> <dataset-rev>
# Billed job: rtx-pro-6000 at $2.75/h, timeout 80m
# Worst case if it runs to the timeout: $3.67 (ceiling $5.00)
```

This exists because the worker's own guards run inside a container the job is
already paying for. A mistyped timeout — `50h` for `5h` — passed every format
check and would have been a four-figure mistake that nothing on the launching
side caught. Price with `DRY_RUN=1` first; it costs nothing and prints the same
number the real launch will.

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
reproduction; it was not re-submitted while editing these files. It is kept
verbatim as history and would now be refused: its `HF_HUB_DEST` equals
`BASE_MODEL`, which the launcher rejects with exit code 2.

The bootstrap script
[`hf_job_tool_sft.py`](../scripts/retail_bank/hf_job_tool_sft.py) downloads the
exact GitHub source and Hub dataset revisions into an isolated job directory,
then invokes the worker with all three execution guards:

```text
--execute-remote
--allow-remote-execution
RETAIL_BANK_ALLOW_REMOTE_TOOL_SFT=banking-v5-grounded-dialogue-sft
```

## V9 From-Scratch Guarded Run

The from-scratch Stage-2 lane carries the same destination, wall-clock, mix, and
behavioural guards as the continuation lane. One run against the V9 dataset,
publishing a new adapter repository:

```bash
HF_HUB_DEST=spkc83/retail-bank-servicing-agent-9b-peft-v9-scratch \
JOB_TIMEOUT=80m \
MAX_STEPS=2000 \
MAX_TRAIN_SECONDS=3600 \
SKIP_MERGE_ADAPTER=1 \
OUTPUT_PREFIX=/data/retail-bank-agent-9b-v9-scratch-<commit8> \
POSITIVE_MULTIPLIER=2 \
AMBIGUITY_MULTIPLIER=3 \
POLICY_FAQ_MULTIPLIER=1 \
TOOL_OUTCOME_MULTIPLIER=2 \
scripts/retail_bank/run_remote_training_job.sh \
  <commit> 0f99604ac5f9366828e90fd46a6343cebb72f1a5
```

### What each guard does

- `HF_HUB_DEST` is mandatory and must differ from `BASE_MODEL`. The launcher
  exits 2 before calling `hf`, the bootstrap raises `ValueError` before any
  network call, and the worker raises `RuntimeError` before loading the
  tokenizer.
- Before training starts, the worker refuses a destination repository that
  already contains files, so a second run cannot silently replace a release.
  The same check runs again immediately before upload.
- `JOB_TIMEOUT` caps the whole job (setup, training, gates, merge, upload).
  `MAX_TRAIN_SECONDS` caps only optimizer work, leaving the remaining budget
  for the gates and the upload.
- The dataset repository must be exactly
  `spkc83/retail-bank-servicing-alignment-sft` and the revision must be an exact
  40-character lowercase SHA. This is validated before the base weights load.
- The four multipliers weight the training mix. They default to `1/1/1/1`, which
  is the unweighted manifest order. When any is above `1`, the worker masks the
  final assistant turn of every coreference positive and rebuilds the mix with
  the continuation worker's `build_continuation_mix`, seeded by `TRAINING_SEED`.
  The resulting stats land in `training_result.json` under `training_mix`.
- After `trainer.evaluate()` and `trainer.save_model()`, but before merging or
  any upload, the worker runs two greedy behavioural gates: the dev gate over
  the validation records carrying `metadata.coreference_pair_id`, and the shadow
  gate over the manifest's non-trainable `coreference-shadow` split. Each of
  `positive_tool_argument_accuracy`, `ambiguity_accuracy`, and
  `pair_flip_accuracy` must be at least `0.95`. Reports are written to
  `<output_dir>/behavioral-evaluations/dev-step-<step>.json` and
  `shadow-step-<step>.json`, and the metrics are recorded in
  `training_result.json` as `coreference_behavioral_gate` and
  `shadow_coreference_behavioral_gate`. A failing gate raises before any upload,
  leaving the adapter and both reports on the job bucket for diagnosis.
- A third gate then runs the **bare probes**
  ([`banking_bare_probe_gate.py`](../src/hello_slm/banking_bare_probe_gate.py)):
  the guidance-free behaviours — declining a poem, refusing to read out a PIN,
  not claiming a balance it cannot see, not fabricating a delivery it did not
  perform. Every prompt is asked with the TURN GUIDANCE stripped, which is what
  makes it a weight-level measurement rather than a prompt-following one. It
  writes `behavioral-evaluations/bare-probes-step-<step>.json`, records
  `bare_probe_behavioral_gate` in `training_result.json`, and blocks the upload
  exactly as the coreference gates do. It exists because the v12 run held every
  coreference gate at 1.0 while two bare-probe behaviours regressed, so the
  coreference gates alone cannot certify a release.
- `SKIP_MERGE_ADAPTER=1` skips the FP16 merge and its reload-parity check. The
  published repository root then holds the LoRA adapter itself rather than
  merged weights; the `adapter/` subdirectory holds the same files either way,
  so downstream loaders can always read the PEFT weights from `adapter/`.

`MAX_STEPS` and `LEARNING_RATE` defaults are unchanged; the line above sets
`MAX_STEPS` explicitly for this run.

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
