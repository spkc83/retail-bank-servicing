# End-to-End Runbook

This runbook walks a new developer through the released retail-bank servicing
pipeline: local setup, data preparation, Granite SFT stages, exact evaluation,
router training, ZeroGPU POC validation, and deployment checks.

Paid Hugging Face Jobs steps are marked clearly. Run them only after explicit
authorization, working credentials, and budget approval. This document records
the released v4 facts; it does not imply that new paid jobs should be started.

## 1. Install

From the repository root:

```bash
python -m pip install -e ".[dev,scale]"
```

The top-level package metadata is in [`pyproject.toml`](../pyproject.toml). The
POC has its own dependency set in
[`poc/retail-bank-customer-service-poc/pyproject.toml`](../poc/retail-bank-customer-service-poc/pyproject.toml)
and [`requirements.txt`](../poc/retail-bank-customer-service-poc/requirements.txt).

For POC-only work:

```bash
cd poc/retail-bank-customer-service-poc
python -m pip install -r requirements.txt
python -m pip install pytest ruff
```

## 2. Verify the Repo Locally

Run the focused repository tests:

```bash
python -m pytest -q tests
```

Run the POC tests without loading the 9B model or router:

```bash
cd poc/retail-bank-customer-service-poc
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-more"}'
export POC_SKIP_MODEL_LOAD=1
export POC_SKIP_ROUTER_LOAD=1
python -m pytest -q tests
```

Run static checks from the repository root:

```bash
ruff check .
MYPYPATH=src mypy src scripts tests
uv lock --check
```

If a command fails, fix that stage before moving downstream.

## 3. Prepare Stage-1 Tool-Use SFT Data

The stage-1 SFT data script is
[`scripts/retail_bank/prepare_tool_sft_data.py`](../scripts/retail_bank/prepare_tool_sft_data.py).

Run:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_tool_sft_data.py \
  --output-dir data/banking-v3-tool-sft \
  --pilot-count 9000 \
  --split-seed 711
```

Current split identity:

| Split | Records | SHA-256 |
| --- | ---: | --- |
| train | 6,304 | `8d92fa0ab1d39875f0c4d918bc5aeaf670f71bf660a01a0a376cb4edc1cced53` |
| validation | 1,349 | `a8c7871b33689fce026ea570ad0a8a90a609cde232a89486e5437b028279e6d3` |
| test | 1,347 | `76b485fa507d56002f12b556f100fd842c77146804cf49be3426be031cc692c0` |

Published dataset revision:
`183e7e1ed1aba9c3d7155e7b83b64dc854935055`.

## 4. Prepare Stage-2 Servicing-Remediation Data

The stage-2 data script is
[`scripts/retail_bank/prepare_servicing_alignment_data.py`](../scripts/retail_bank/prepare_servicing_alignment_data.py).
It copies the full stage-1 corpus and appends targeted remediation rows in the
matching split.

Run:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_servicing_alignment_data.py
```

Current composite split identity:

| Split | Initial base | Remediation additions | Composite total |
| --- | ---: | ---: | ---: |
| train | 6,304 | 320 | 6,624 |
| validation | 1,349 | 80 | 1,429 |
| test | 1,347 | 27 | 1,374 |

Released dataset revisions:

- corrected public revision:
  `0ce32f9c7a3edff227005e5b89b089947b87625a`
- prompt-identical training revision:
  `fea8aa1cda716954eb7322325e2be25c9f570ea3`

The two revisions are prompt-identical for generation and scoring.

## 5. Granite SFT Stages

Stage 1 starts from `ibm-granite/granite-4.1-8b` revision
`1504002f650e656a0a3789d99574df12e3e94ed0` and trains the initial synthetic
tool-use corpus.

Stage 2 starts from the stage-1 tool-trained checkpoint and trains the
servicing-remediation composite. The released stage-2 output is:

- repo: `spkc83/retail-bank-servicing-agent-9b`
- immutable weights revision: `1d56824995aa1adecfe20f62ca42fb1c0c443817`
- published evaluation head: `214fc0d9e143e4fa7b658de1993113562b90958a`
- source revision: `475dc2b563ef87fa0c9aa597b0b0465d56d2ee0f`
- training job: `spkc83/6a6ca6276b79c09949c1d6cb`
- runtime: about 18 minutes 59 seconds
- estimated cost: about `$0.87`
- train loss: `0.0069123295`
- eval loss: `0.0002181597`
- token accuracy: `0.999976121`

Inspect the stage-2 training plan without allocating a GPU:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-servicing-alignment-v4/manifest.json \
  --base-model spkc83/retail-bank-agent-9b \
  --base-revision 085df3d089cfadd77424b548542da0390a54a23e \
  --hub-dest spkc83/retail-bank-servicing-agent-9b \
  --family granite \
  --learning-rate 2e-5 \
  --max-steps 500 \
  --dry-run
```

The canonical entry point prints the entire ordered release plan without side
effects:

```bash
PYTHONPATH=src python scripts/retail_bank/run_release_pipeline.py --stage all
```

It owns data preparation, both Granite SFT stages, router training, exact
evaluation, and deployment. Execute one stage at a time because every
publishing stage creates an immutable revision that must be captured in
`configs/retail-bank-release.toml` before the downstream stage starts. Paid and
publishing stages also require their explicit safety flags.

## 6. Exact Evaluation and Rescore

Run the frozen evaluator against the released model and corrected dataset:

```bash
export MODEL_REPO=spkc83/retail-bank-servicing-agent-9b
export DATASET_REPO=spkc83/retail-bank-servicing-alignment-sft
bash scripts/retail_bank/run_remote_tool_eval_job.sh \
  475dc2b563ef87fa0c9aa597b0b0465d56d2ee0f \
  1d56824995aa1adecfe20f62ca42fb1c0c443817 \
  0ce32f9c7a3edff227005e5b89b089947b87625a
```

The released evaluation job was `spkc83/6a6caac1a00abefd4b289b14`.

Final metrics:

| Slice | Result |
| --- | ---: |
| Frozen test conversations | `1,374` |
| Tool names and arguments | `796/796` |
| Executable tool trajectories | `700/700` |
| Exact dependent multi-tool sequences | `96/96` |
| Appropriate clarifications | `63/63` |
| Banking FAQ answers | `258/258` |
| OOD response paths | `35/35` |
| Grounded factual responses | `1,141/1,141` |
| Hard error metrics | `0` |

Use `scripts/retail_bank/rescore_tool_eval.py` only when the corrected dataset
is prompt-identical to the generated-prediction dataset. The v4 final result
meets that condition, so the rescore is valid and is not a second generation
run.

## 7. Prepare and Train the Router

The history-aware router data script is
[`scripts/retail_bank/prepare_conversation_router_data.py`](../scripts/retail_bank/prepare_conversation_router_data.py).

Run:

```bash
PYTHONPATH=src python scripts/retail_bank/prepare_conversation_router_data.py
```

Split identity:

| Split | Rows | SHA-256 |
| --- | ---: | --- |
| train | 61,759 | `8289533eb3df841c215bd4ea6e7f216c1b0fd988ad49dfd0fb78a13ad795b4e8` |
| validation | 13,173 | `ecde083032ee1dbd692190d4dcc08815c43f1459f255aaa3fd685ccad974df18` |
| test | 15,466 | `e4d70f0adccf0615bf79b1034203b76d0986c09d58259c31a2e2ea24a5d4931f` |

Train locally without publishing:

```bash
PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py
```

Released router:

- repo: `spkc83/retail-bank-conversation-router`
- revision: `9e090c0fa21cebbaa03a431a7ce61e656c0739fe`
- data revision: `e9a64a2e7f2b622d5412c15eac4618ceca2150da`
- capability macro F1: `0.997838`
- relation macro F1: `0.998628`
- captured-regression route/capability/relation errors: `0 / 0 / 0`

## 8. Run the POC Locally

```bash
cd poc/retail-bank-customer-service-poc
export DEMO_AUTH_JSON='{"alex.demo":"replace-with-12-chars","maya.demo":"replace-with-12-more"}'
export RETAIL_BANK_MODEL_ID=spkc83/retail-bank-servicing-agent-9b
export RETAIL_BANK_MODEL_REVISION=1d56824995aa1adecfe20f62ca42fb1c0c443817
export RETAIL_BANK_ROUTER_ID=spkc83/retail-bank-conversation-router
export RETAIL_BANK_ROUTER_REVISION=9e090c0fa21cebbaa03a431a7ce61e656c0739fe
python app.py
```

Local CPU-only testing can skip model and router loading with
`POC_SKIP_MODEL_LOAD=1` and `POC_SKIP_ROUTER_LOAD=1`, but that does not prove
live 9B inference.

## 9. Deploy the ZeroGPU POC

Deployment to `spkc83/retail-bank-servicing-poc` is an external production
action. Do not run a deployment command without explicit authorization for that
deployment.

The Space must pin:

- model `spkc83/retail-bank-servicing-agent-9b`;
- exact model revision `1d56824995aa1adecfe20f62ca42fb1c0c443817`;
- router `spkc83/retail-bank-conversation-router`;
- exact router revision `9e090c0fa21cebbaa03a431a7ce61e656c0739fe`;
- `DEMO_AUTH_JSON` with exactly `alex.demo` and `maya.demo`.

After the secret exists, execute the guarded canonical deployment stage:

```bash
PYTHONPATH=src python scripts/retail_bank/run_release_pipeline.py \
  --stage deploy \
  --execute \
  --allow-publish
```

[`deploy_zero_gpu_space.py`](../scripts/retail_bank/deploy_zero_gpu_space.py)
uploads an allowlist that excludes tests, virtual environments, bytecode, and
hidden caches. It then persists the exact model, router, and returned Space
commit revisions as runtime variables and waits for the resulting Hub-triggered
rebuild to start. It does not delete remote files; remote cleanup remains a
separately authorized operation.

After deployment, require the authenticated `/zero_gpu_probe` to enter user code
and report CUDA plus the pinned revisions. Then run live read, write, multi-tool,
clarification, FAQ, OOD, and multi-turn cases. The diagnostics panel must show
the active model revision, active router revision, and CUDA-backed generation
for model-handled turns.

## One Scenario Across the Runbook

Use “Show my service cases” followed by “When was that created?” as a mental
checkpoint at every stage:

| Stage | Evidence to inspect |
| --- | --- |
| Stage-1 data | A valid `list_service_cases` call, correlated result, and grounded final answer. |
| Stage-2 data | A separately authored contextual `created_at` follow-up. |
| Granite training | Assistant-only labels cover the call and final responses. |
| Router data | Current turn and visible history exist; target call/result do not. |
| Router calibration | Clear banking routes in-domain; ambiguous context may route uncertain, not OOD. |
| Frozen evaluation | Tool, arguments, trajectory, and `created_at` fact pass. |
| Live POC | Diagnostics show the pinned 9B model, CUDA, context applied, call, result, and final response. |

If a stage cannot produce its expected evidence, stop there. A later green
deployment cannot compensate for invalid data or evaluation.

The conceptual trace is expanded in
[End-to-End Flow by Example](11-end-to-end-flow-by-example.md).

## Stop Conditions

Stop before release if:

- any source, model, dataset, or router revision is not an exact immutable
  40-character commit;
- generated split digests drift from tracked locks;
- any local test, lint, typecheck, or lock check fails;
- paid-job guards are missing;
- training metrics are unavailable;
- evaluation uses a non-prompt-equivalent rescore;
- any exact frozen gate fails;
- the POC diagnostics cannot prove the model and router revisions used for live
  generation.
