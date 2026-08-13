# Artifact Ledger

This page records the active immutable artifacts for the retail-bank servicing
agent, history-aware router, datasets, local manifests, paid job outputs, and
runtime defaults.

## Published Repositories

| Artifact | Repository | Immutable revision | Source |
| --- | --- | --- | --- |
| Granite servicing agent | `spkc83/retail-bank-servicing-agent-9b` | `1d56824995aa1adecfe20f62ca42fb1c0c443817` | [`model card`](../../model_cards/retail-bank-agent-9b.md), [`zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| Agent published evaluation head | `spkc83/retail-bank-servicing-agent-9b` | `214fc0d9e143e4fa7b658de1993113562b90958a` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Agent base model | `ibm-granite/granite-4.1-8b` | `1504002f650e656a0a3789d99574df12e3e94ed0` | [`configs/banking-tool-sft-granite.toml`](../../configs/banking-tool-sft-granite.toml), [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Stage-1 Granite tool-use checkpoint | `spkc83/retail-bank-agent-9b` | `085df3d089cfadd77424b548542da0390a54a23e` | [`release config`](../../configs/retail-bank-release.toml) |
| Initial tool-use SFT dataset | `spkc83/retail-bank-agent-sft` | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` | [`data card`](../../data_cards/retail-bank-agent-sft.md) |
| Corrected servicing-remediation dataset | `spkc83/retail-bank-servicing-alignment-sft` | `0ce32f9c7a3edff227005e5b89b089947b87625a` | [`data card`](../../data_cards/retail-bank-servicing-alignment-sft.md) |
| Prompt-identical training dataset revision | `spkc83/retail-bank-servicing-alignment-sft` | `fea8aa1cda716954eb7322325e2be25c9f570ea3` | [`data card`](../../data_cards/retail-bank-servicing-alignment-sft.md) |
| History-aware router | `spkc83/retail-bank-conversation-router` | `9e090c0fa21cebbaa03a431a7ce61e656c0739fe` | [`router card`](../../model_cards/retail-bank-domain-intent-router.md), [`router.py`](../../poc/retail-bank-customer-service-poc/router.py) |
| Router dataset | `spkc83/retail-bank-conversation-router-data` | `e9a64a2e7f2b622d5412c15eac4618ceca2150da` | [`data card`](../../data_cards/retail-bank-router-training-data.md) |
| Public Space | `spkc83/retail-bank-servicing-poc` | Space commit is exposed at runtime as `SPACE_COMMIT_SHA` | [`app.py`](../../poc/retail-bank-customer-service-poc/app.py), [`README.md`](../../poc/retail-bank-customer-service-poc/README.md) |

## Agent Model Details

| Field | Value |
| --- | --- |
| Model repository | `spkc83/retail-bank-servicing-agent-9b` |
| Immutable weights revision | `1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| Published evaluation head | `214fc0d9e143e4fa7b658de1993113562b90958a` |
| Base model | `ibm-granite/granite-4.1-8b` |
| Base revision | `1504002f650e656a0a3789d99574df12e3e94ed0` |
| Source revision | `475dc2b563ef87fa0c9aa597b0b0465d56d2ee0f` |
| Initial dataset revision | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` |
| Corrected remediation dataset revision | `0ce32f9c7a3edff227005e5b89b089947b87625a` |
| Prompt-identical training dataset revision | `fea8aa1cda716954eb7322325e2be25c9f570ea3` |
| Parameters | 8,791,592,960 |
| Tool format | Granite native tagged JSON |
| Remote-job bootstrap source | `1da0bdc1cdcc5a0e1c5ce137c32384d927c1948b` |

### How to follow one released artifact chain

Read the ledger as a dependency chain:

```text
IBM base revision
  -> stage-1 dataset revision
  -> stage-1 checkpoint revision
  -> stage-2 dataset revision
  -> servicing-model weights revision
  -> evaluation head revision
  -> Space runtime model variable
```

The live Space must load model revision
`1d56824995aa1adecfe20f62ca42fb1c0c443817`, not the mutable `main` branch.
Evaluation head `214fc0d9e143e4fa7b658de1993113562b90958a` describes
that model and its pinned corrected dataset revision.

Changing any upstream revision creates a new experiment. Reusing a downstream
metric after such a change would break provenance.

## Paid Job Records

These are the job records for the released artifacts. Do not start new paid jobs
unless explicitly authorized.

| Purpose | Job ID | Evidence |
| --- | --- | --- |
| Servicing-remediation SFT training | `spkc83/6a6ca6276b79c09949c1d6cb` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Exact frozen tool/final-response evaluation | `spkc83/6a6caac1a00abefd4b289b14` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |

The servicing-remediation training run took about 18 minutes 59 seconds and
cost about `$0.87`. It reported training loss `0.0069123295`, evaluation loss
`0.0002181597`, and token accuracy `0.999976121`.

## Private Job-Bucket Retention

Bucket: `spkc83/jobs-artifacts`

The bucket is a private, restartable job workspace. It is not read by the
public Space and it is not the authoritative location for published model,
router, dataset, or evaluation artifacts.

| State | Files | Logical size |
| --- | ---: | ---: |
| Before 2026-07-31 cleanup | 290 | 449,461,595,301 bytes |
| After cleanup | 58 | 1,252,559,272 bytes |
| Removed | 232 | about 448.2 GB |

The retained set preserves the selected recovery checkpoint adapter, trainer
state, and provenance JSON. Failed runs, superseded checkpoints, duplicate
merged weights, temporary merge files, and bucket copies of already-published
evaluation outputs were removed. New runs must write a new prefix and apply the
same publish-verify-retain policy.

## Released Evaluation

The final released score is from evaluation job
`spkc83/6a6caac1a00abefd4b289b14`.

| Metric | Result | Source |
| --- | ---: | --- |
| Frozen test conversations | 1,374 | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Tool names and arguments | `796/796` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Executable trajectories | `700/700` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Dependent multi-tool sequences | `96/96` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Clarifications | `63/63` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Banking FAQ answers | `258/258` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| OOD paths | `35/35` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Grounded factual responses | `1,141/1,141` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Malformed calls, unsupported/private arguments, credential requests, in-domain false refusals, OOD false accepts | `0` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |

The corrected dataset revision
`0ce32f9c7a3edff227005e5b89b089947b87625a` is prompt-identical to the training
revision `fea8aa1cda716954eb7322325e2be25c9f570ea3` for generation and scoring.
The final report is therefore a rescore of equivalent prompts, not a second
generation run. The rescore helper is `scripts/retail_bank/rescore_tool_eval.py`.

The released metric is an in-generator protocol regression result. It is not a
leakage-free generalization claim. The local audit found shared POC facts,
template families, and targets between training and test. See
[`data-leakage-audit.md`](data-leakage-audit.md).

## Local Counterfactual Evaluation

The clean project-SFT/POC acceptance set is tracked locally and is not a
published training dataset.

| Field | Value |
| --- | --- |
| Manifest | [`data/banking-counterfactual-eval-v1/manifest.json`](../../data/banking-counterfactual-eval-v1/manifest.json) |
| Manifest SHA-256 | `3737655087d79136f719ecf7f251d6a79f89f28392684bb3059cdab243a97b5b` |
| Test rows | `18` |
| Counterfactual pairs | `5` |
| Test JSONL SHA-256 | `d0b60766a659496fa1809c6a8124027172ebce22081db087fedfca4e2b71c6f6` |
| Training allowed | `false` |
| Preparation audit | pass |

The preparation pass proves only that the tracked rows satisfy the local
contamination and pair contracts. A model result exists only when a run emits
predictions, a report, and metadata under
`artifacts/banking-counterfactual-eval-v1`. See
[`13-counterfactual-evaluation.md`](../13-counterfactual-evaluation.md).

## Initial Tool-Use SFT Dataset

Published repository: `spkc83/retail-bank-agent-sft`

Published revision:
`183e7e1ed1aba9c3d7155e7b83b64dc854935055`

Corpus fingerprint:
`2bb7a400ed2556b15c7e5eb6147668041b5deef8ae4f037f9e2e52295ff29ab5`

Local manifest:
[`data/banking-v3-tool-sft/manifest.json`](../../data/banking-v3-tool-sft/manifest.json)

| Split | Records | Local SHA-256 |
| --- | ---: | --- |
| train | 6,304 | `8d92fa0ab1d39875f0c4d918bc5aeaf670f71bf660a01a0a376cb4edc1cced53` |
| validation | 1,349 | `a8c7871b33689fce026ea570ad0a8a90a609cde232a89486e5437b028279e6d3` |
| test | 1,347 | `76b485fa507d56002f12b556f100fd842c77146804cf49be3426be031cc692c0` |

## Servicing-Remediation SFT Dataset

Published repository: `spkc83/retail-bank-servicing-alignment-sft`

Corrected published revision:
`0ce32f9c7a3edff227005e5b89b089947b87625a`

Prompt-identical training revision:
`fea8aa1cda716954eb7322325e2be25c9f570ea3`

Local manifest:
[`data/banking-servicing-alignment-v4/manifest.json`](../../data/banking-servicing-alignment-v4/manifest.json)

| Split | Initial base | Remediation additions | Composite total |
| --- | ---: | ---: | ---: |
| train | 6,304 | 320 | 6,624 |
| validation | 1,349 | 80 | 1,429 |
| test | 1,347 | 27 | 1,374 |

## Router Artifact

Published repository: `spkc83/retail-bank-conversation-router`

Published revision:
`9e090c0fa21cebbaa03a431a7ce61e656c0739fe`

Training-data revision:
`e9a64a2e7f2b622d5412c15eac4618ceca2150da`

Router code:
[`poc/retail-bank-customer-service-poc/router.py`](../../poc/retail-bank-customer-service-poc/router.py)

| Field | Value |
| --- | ---: |
| Capability macro F1 | `0.997838` |
| Relation macro F1 | `0.998628` |
| In-domain false-refusal rate | `0.000167` |
| OOD false-accept rate | `0.012735` |
| Contextual false-refusal rate | `0.000105` |
| Repair false-refusal rate | `0.000000` |
| External topic-shift false-accept rate | `0.000778` |
| Captured-regression route/capability/relation errors | `0 / 0 / 0` |
| OOD banking boundary | `0.10` |
| Serving in-domain boundary | `0.50` |
| Relation rescue boundary | `0.40` |

## Router Dataset

Published repository: `spkc83/retail-bank-conversation-router-data`

Published revision:
`e9a64a2e7f2b622d5412c15eac4618ceca2150da`

Release lock:
[`data/sources/banking-conversation-router-v4.lock.json`](../../data/sources/banking-conversation-router-v4.lock.json)

| Split | Rows | Local SHA-256 |
| --- | ---: | --- |
| train | 61,759 | `8289533eb3df841c215bd4ea6e7f216c1b0fd988ad49dfd0fb78a13ad795b4e8` |
| validation | 13,173 | `ecde083032ee1dbd692190d4dcc08815c43f1459f255aaa3fd685ccad974df18` |
| test | 15,466 | `e4d70f0adccf0615bf79b1034203b76d0986c09d58259c31a2e2ea24a5d4931f` |

## Runtime Artifact Defaults

| Runtime field | Default | Source |
| --- | --- | --- |
| `RETAIL_BANK_MODEL_ID` | `spkc83/retail-bank-servicing-agent-9b` | [`zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| `RETAIL_BANK_MODEL_REVISION` | `1d56824995aa1adecfe20f62ca42fb1c0c443817` | [`zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| `RETAIL_BANK_ROUTER_ID` | `spkc83/retail-bank-conversation-router` | [`router.py`](../../poc/retail-bank-customer-service-poc/router.py) |
| `RETAIL_BANK_ROUTER_REVISION` | `9e090c0fa21cebbaa03a431a7ce61e656c0739fe` | [`router.py`](../../poc/retail-bank-customer-service-poc/router.py) |
| `INPUT_TOKEN_BUDGET` | `8192` | [`model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) |
| `MAX_NEW_TOKENS` | `512` | [`model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) |
| `MAX_TOOL_CALLS` | `8` | [`model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) |
| Demo usernames | `alex.demo`, `maya.demo` | [`auth.py`](../../poc/retail-bank-customer-service-poc/auth.py) |
| Session database directory | `/tmp/retail-bank-servicing-poc` unless `POC_SESSION_DB_DIR` is set | [`state.py`](../../poc/retail-bank-customer-service-poc/state.py) |

## Source Commit For A New Run

The released job source revision is `475dc2b563ef87fa0c9aa597b0b0465d56d2ee0f`.
For a new paid run, commit and push the intended source state, then obtain its
immutable revision with:

```bash
git rev-parse HEAD
```

Paid launchers require that value to be an exact 40-character commit and verify
that the remote job script exists at the same revision before starting work.
