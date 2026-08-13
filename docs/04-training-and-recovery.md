# Granite PEFT Training and Recovery

This runbook covers the active IBM Granite 8.79B PEFT lane: local checks,
stage-1 base-tool SFT, stage-2 servicing-remediation SFT, exact evaluation,
rescore, recovery boundaries, and final publication. The training configuration
is [`configs/banking-tool-sft-granite.toml`](../configs/banking-tool-sft-granite.toml).
The published model card is
[`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md).

## Active Artifact IDs

Use immutable 40-character revisions for every paid or published run. Branch
names such as `main` are rejected by the job entry points.

| Artifact | Value | Owner |
| --- | --- | --- |
| IBM base model | `ibm-granite/granite-4.1-8b` | [`configs/banking-tool-sft-granite.toml`](../configs/banking-tool-sft-granite.toml) |
| IBM base revision | `1504002f650e656a0a3789d99574df12e3e94ed0` | [`configs/banking-tool-sft-granite.toml`](../configs/banking-tool-sft-granite.toml) |
| Stage-1 tool-use checkpoint | `spkc83/retail-bank-agent-9b` at `085df3d089cfadd77424b548542da0390a54a23e` | [`release config`](../configs/retail-bank-release.toml) |
| Released model repo | `spkc83/retail-bank-servicing-agent-9b` | [`model card`](../model_cards/retail-bank-agent-9b.md) |
| Released weights revision | `1d56824995aa1adecfe20f62ca42fb1c0c443817` | [`model card`](../model_cards/retail-bank-agent-9b.md) |
| Initial tool-use dataset | `spkc83/retail-bank-agent-sft` at `183e7e1ed1aba9c3d7155e7b83b64dc854935055` | [`data card`](../data_cards/retail-bank-agent-sft.md) |
| Corrected servicing dataset | `spkc83/retail-bank-servicing-alignment-sft` at `0ce32f9c7a3edff227005e5b89b089947b87625a` | [`data card`](../data_cards/retail-bank-servicing-alignment-sft.md) |
| Prompt-identical training dataset | `spkc83/retail-bank-servicing-alignment-sft` at `fea8aa1cda716954eb7322325e2be25c9f570ea3` | [`data card`](../data_cards/retail-bank-servicing-alignment-sft.md) |
| Source revision used for release | `475dc2b563ef87fa0c9aa597b0b0465d56d2ee0f` | [`artifact ledger`](reference/artifacts.md) |
| Remote-job bootstrap source | `1da0bdc1cdcc5a0e1c5ce137c32384d927c1948b` | [`release config`](../configs/retail-bank-release.toml) |
| Servicing-remediation training job | `spkc83/6a6ca6276b79c09949c1d6cb` | [`model card`](../model_cards/retail-bank-agent-9b.md) |
| Exact evaluation job | `spkc83/6a6caac1a00abefd4b289b14` | [`model card`](../model_cards/retail-bank-agent-9b.md) |

## Bucket Retention Policy

The private `spkc83/jobs-artifacts` bucket is durable working storage for active
jobs, but it is not the release source of truth. Published Hub repositories are
authoritative for the model, router, datasets, and evaluation reports.

On 2026-07-31, obsolete job files were reduced from 290 files
(`449,461,595,301` bytes) to 58 files (`1,252,559,272` bytes). The retained set
preserves selected recovery adapters, trainer state, and provenance JSON. Failed
runs, superseded checkpoints, duplicate merged weights, temporary merge files,
optimizer state from non-selected runs, and bucket copies of already-published
evaluation outputs were removed.

New training and evaluation runs must use a new output prefix. Intermediate
files may be removed after publication and verification, but the selected
recovery adapter, trainer state, and run metadata must remain until that release
is formally retired.

## What Trains

The worker
[`scripts/retail_bank/cloud_train_tool_sft.py`](../scripts/retail_bank/cloud_train_tool_sft.py)
loads a pinned Granite-family model and trains a BF16 LoRA adapter with TRL
`SFTTrainer`.

| Setting | Value |
| --- | --- |
| PEFT stack | BF16 LoRA over Granite attention and MLP projections |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| LoRA rank / alpha / dropout | `32` / `64` / `0.05` |
| Maximum sequence length | `2048` |
| Outer HF Jobs timeout | `5h` |
| Remote GPU flavor | `rtx-pro-6000` |

The release has two Granite SFT stages:

1. Stage 1 starts from `ibm-granite/granite-4.1-8b` revision
   `1504002f650e656a0a3789d99574df12e3e94ed0` and trains the initial 9,000-row
   synthetic tool-use corpus. It teaches the tagged-JSON tool wire, tool-result
   grounding, clarification, FAQ, OOD refusal, and multi-tool ordering.
2. Stage 2 continues from the stage-1 tool-trained checkpoint and trains the
   composite v4 servicing-remediation corpus. It exists because POC testing
   exposed failures in service-case follow-ups, card anaphora, clarification
   answers, agent repair, and topic shifts.

### Choose the correct execution path

| Situation | Start point | Operation | Example |
| --- | --- | --- | --- |
| Rebuild the full release | Pinned IBM Granite base | Run stage 1, then stage 2 | New infrastructure or independent reproduction. |
| Improve a released behavior | Pinned stage-1 or released checkpoint | Continue SFT with base plus remediation data | Add robust service-case follow-ups. |
| Export failed after training | Retained completed adapter | Export-only recovery | Merge/upload failed after the trainer completed. |
| Training stopped mid-run | Matching checkpoint and fingerprint | Resume training | Worker interruption after checkpoint 500. |

If adapter state proves step 500 completed, resume may continue training. If
training completed but only Hub upload failed, export recovery must not call
`trainer.train` again.

The fingerprint prevents an invalid recovery, such as resuming a stage-2
adapter with a different chat template or dataset revision.

## Local Preflight

Run these before any paid job:

```bash
python -m pytest -q tests/test_banking_tool_sft_data.py \
  tests/test_banking_tool_wire.py \
  tests/test_banking_tool_sft_job.py \
  tests/test_banking_tool_sft_worker.py \
  tests/test_banking_tool_eval.py \
  tests/test_banking_tool_eval_runner.py
```

Check the stage-2 training plan without downloading 9B weights, launching a
job, merging, or pushing:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-servicing-alignment-v4/manifest.json \
  --base-model spkc83/retail-bank-agent-9b \
  --base-revision 085df3d089cfadd77424b548542da0390a54a23e \
  --hub-dest spkc83/retail-bank-servicing-agent-9b \
  --family granite \
  --learning-rate 2e-5 \
  --max-steps 500 \
  --checkpoint-every 100 \
  --dry-run
```

Run the local one-step smoke:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
  --run-tiny-smoke \
  --dry-run
```

The tiny smoke uses small offline stand-ins. It writes local smoke artifacts and
verifies assistant-label tokens, checkpoint metadata, adapter output paths,
final output paths, and merge/reload parity. It does not prove the 8.79B model
can train on GPU.

## Paid HF Jobs

Paid execution is launched through
[`scripts/retail_bank/run_remote_training_job.sh`](../scripts/retail_bank/run_remote_training_job.sh),
which submits
[`scripts/retail_bank/hf_job_tool_sft.py`](../scripts/retail_bank/hf_job_tool_sft.py)
with:

- `--flavor rtx-pro-6000`;
- `--timeout 5h`;
- `--secrets HF_TOKEN`;
- `--volume hf://buckets/spkc83/jobs-artifacts:/data`;
- exact source and dataset revisions.

The `HF_TOKEN` secret must have read access to the dataset and base, and write
access to the destination model repository. Do not put tokens on the command
line.

The released stage-2 training job was `spkc83/6a6ca6276b79c09949c1d6cb`. It ran
for about 18 minutes 59 seconds and cost about `$0.87`. Final reported metrics:

| Metric | Value |
| --- | ---: |
| Training loss | `0.0069123295` |
| Evaluation loss | `0.0002181597` |
| Token accuracy | `0.999976121` |

## Rescore and Evaluation

The exact evaluation job was `spkc83/6a6caac1a00abefd4b289b14`. It evaluated
1,374 frozen records and passed every hard gate.

The corrected dataset revision
`0ce32f9c7a3edff227005e5b89b089947b87625a` is prompt-identical to the training
revision `fea8aa1cda716954eb7322325e2be25c9f570ea3` for generation and scoring.
The helper `scripts/retail_bank/rescore_tool_eval.py` rescored equivalent rows
against the existing predictions. This is not a second generation run.

## Rebuild on Clean Infrastructure

The repository retains two separate, reproducible training lanes. Do not
confuse the servicing-remediation run with foundation pretraining:

1. Stage 1 starts from the immutable pretrained IBM Granite checkpoint. English
   and general language ability come from this base model, not from random
   initialization in this repository.
2. Stage 2 starts from the stage-1 tool-trained checkpoint and uses the
   composite servicing-remediation dataset.

On a new machine or cloud account, reproduce into a new destination repository
unless the goal is an explicitly authorized replacement release. Every
downstream evaluation and deployment command must consume captured
40-character revisions, never `main`.

[`scripts/retail_bank/run_release_pipeline.py`](../scripts/retail_bank/run_release_pipeline.py)
is the canonical orchestration entry point. It contains data preparation,
stage-1 Granite tool-use SFT from the pinned IBM checkpoint, stage-2 remediation
SFT from the captured stage-1 revision, router training, exact evaluation, and
ZeroGPU deployment:

```bash
PYTHONPATH=src python scripts/retail_bank/run_release_pipeline.py --stage all
```

The command above prints the complete plan without executing it. Execute one
stage at a time with `--execute` and the applicable `--allow-paid` and
`--allow-publish` guards. After a publishing stage, record the new immutable
revision in `configs/retail-bank-release.toml` before executing its downstream
consumer.

## Recovery Boundaries

Use export recovery only when a training job completed and wrote a retained
adapter to the bucket, but publication or final export failed. Recovery is
export-only: it does not call `trainer.train`.

Recovery cross-checks persisted metadata against the inspected training job:

- parent model revision;
- dataset revision;
- base model and base revision;
- training source commit;
- output root;
- completed step count;
- training-job artifact time window.

It tries FP16-native merged output before any fallback and publishes only after
unchanged parity gates pass. This is covered by
[`tests/test_banking_tool_sft_export_recovery.py`](../tests/test_banking_tool_sft_export_recovery.py).

## Stop Conditions

Stop before paid training if any local preflight fails, if the dry-run guard
says remote execution is not intentionally enabled, or if the source, dataset,
or base revision is not exact. Stop during or after remote work if:

- checkpoint metadata is missing or mismatched;
- validation loss or token accuracy is unavailable;
- merge/reload parity fails;
- the job cannot write to `/data`;
- Hub upload cannot record exact published revisions;
- frozen evaluation in [`docs/06-evaluation.md`](06-evaluation.md) fails a
  release gate.

Do not use output repair, deterministic tool planning, or branch-name revisions
to make a checkpoint appear releasable.

## Expected Outputs

Training writes under the mounted `/data` output root:

- adapter files under `adapter/`;
- merged root checkpoint under the selected merged subdirectory;
- checkpoint metadata under `checkpoint-*` or `checkpoints/step-*`;
- `training_result.json`;
- merge parity report files;
- Hub evidence files including exact revision records.

The published model repository root contains merged FP16 weights. The unmerged
adapter remains under `adapter/`, as documented in
[`model_cards/retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md).
