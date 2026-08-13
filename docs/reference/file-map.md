# File Map

This map links the current active retail-bank agent workflow to repo files.

## Root

| Path | Purpose |
| --- | --- |
| [`../../README.md`](../../README.md) | Project overview, active public artifacts, runtime summary, verification commands. |
| [`../../pyproject.toml`](../../pyproject.toml) | Top-level package metadata, dev and scale dependencies, pytest, mypy, and ruff settings. |
| [`../../uv.lock`](../../uv.lock) | Locked top-level Python environment. |
| [`../../Makefile`](../../Makefile) | Repo command shortcuts, when used. |

## Active Docs

| Path | Purpose |
| --- | --- |
| [`../01-system-overview.md`](../01-system-overview.md) | Component boundaries plus a worked routed tool turn. |
| [`../02-data-generation.md`](../02-data-generation.md) | Scenario design, full record examples, split isolation, validation, stage-2 remediation, and router rows. |
| [`../03-model-and-peft.md`](../03-model-and-peft.md) | Granite identity, assistant-only targets, LoRA intuition, and release layout. |
| [`../04-training-and-recovery.md`](../04-training-and-recovery.md) | Two-stage training, resume/recovery choices, paid jobs, and stop conditions. |
| [`../05-dual-head-router.md`](../05-dual-head-router.md) | Current three-head history-aware router, worked thresholds, and serving boundaries. |
| [`../06-evaluation.md`](../06-evaluation.md) | Two-phase frozen evaluation with a worked tool-call example. |
| [`../07-inference-and-poc.md`](../07-inference-and-poc.md) | Detailed POC inference, routing, auth, tool loop, diagnostics, and deployment guide. |
| [`../08-end-to-end-runbook.md`](../08-end-to-end-runbook.md) | Install-to-data-to-training-to-eval-to-POC runbook. |
| [`../09-conversation-router-v4.md`](../09-conversation-router-v4.md) | Released history-aware cross-encoder, leakage-safe data, local training, release gates, and POC integration. |
| [`../10-servicing-alignment-v4.md`](../10-servicing-alignment-v4.md) | Composite Granite continuation-SFT design, use-case coverage, safe training plan, and release stop condition. |
| [`../11-end-to-end-flow-by-example.md`](../11-end-to-end-flow-by-example.md) | One example traced from behavior contract to live inference. |
| [`../12-instruction-fine-tuning-and-peft.md`](../12-instruction-fine-tuning-and-peft.md) | Example-driven instruction SFT, assistant masking, LoRA, two-stage adaptation, merging, and inference design. |
| [`../13-counterfactual-evaluation.md`](../13-counterfactual-evaluation.md) | Leakage-controlled paired benchmark, local 4-bit execution, evidence, and interpretation. |
| [`../15-asr-to-sft-pipeline.md`](../15-asr-to-sft-pipeline.md) | Reviewed ASR normalization, semantic overlays, validation gates, split inheritance, and training handoff. |
| [`artifacts.md`](artifacts.md) | Immutable model, dataset, job, and split identity ledger. |
| [`data-leakage-audit.md`](data-leakage-audit.md) | Granite train/test/POC contamination audit, interpretation limits, and clean benchmark requirements. |
| [`learning-resources.md`](learning-resources.md) | Annotated official documentation and primary-paper references. |

## Published Cards

| Path | Purpose |
| --- | --- |
| [`../../data_cards/retail-bank-agent-sft.md`](../../data_cards/retail-bank-agent-sft.md) | Published tool-use SFT dataset card. |
| [`../../data_cards/retail-bank-router-training-data.md`](../../data_cards/retail-bank-router-training-data.md) | Published router-training dataset card. |
| [`../../data_cards/retail-bank-servicing-alignment-sft.md`](../../data_cards/retail-bank-servicing-alignment-sft.md) | Released composite Granite servicing-alignment dataset card. |
| [`../../model_cards/retail-bank-agent-9b.md`](../../model_cards/retail-bank-agent-9b.md) | Published Granite agent model card and released evaluation results. |
| [`../../model_cards/retail-bank-domain-intent-router.md`](../../model_cards/retail-bank-domain-intent-router.md) | Published three-head conversation-router card and serving thresholds. |

## Configuration

| Path | Purpose |
| --- | --- |
| [`../../configs/banking-tool-sft-granite.toml`](../../configs/banking-tool-sft-granite.toml) | Active Granite BF16 LoRA training configuration. |
| [`../../configs/retail-bank-release.toml`](../../configs/retail-bank-release.toml) | Immutable identities and destinations consumed by the canonical release pipeline. |

## Local Data

| Path | Purpose |
| --- | --- |
| [`../../data/banking-v3-tool-sft/manifest.json`](../../data/banking-v3-tool-sft/manifest.json) | Local tool-use SFT split manifest. |
| [`../../data/banking-v3-tool-sft/train.jsonl`](../../data/banking-v3-tool-sft/train.jsonl) | Local SFT training split. |
| [`../../data/banking-v3-tool-sft/validation.jsonl`](../../data/banking-v3-tool-sft/validation.jsonl) | Local SFT validation split. |
| [`../../data/banking-v3-tool-sft/test.jsonl`](../../data/banking-v3-tool-sft/test.jsonl) | Local frozen SFT test split. |
| [`../../data/banking-v3-tool-sft/README.md`](../../data/banking-v3-tool-sft/README.md) | Local SFT dataset README/card. |
| [`../../data/banking-conversation-router-v4/manifest.json`](../../data/banking-conversation-router-v4/manifest.json) | Active local history-aware router dataset manifest. |
| [`../../data/sources/banking-conversation-router-v4.lock.json`](../../data/sources/banking-conversation-router-v4.lock.json) | Released cross-encoder source and deterministic prepared-split lock. |
| [`../../data/sources/banking-servicing-alignment-v4.lock.json`](../../data/sources/banking-servicing-alignment-v4.lock.json) | Released composite Granite data base-manifest and split-digest lock. |

## Source Package

| Path | Purpose |
| --- | --- |
| [`../../src/hello_slm/banking_tool_sft_data.py`](../../src/hello_slm/banking_tool_sft_data.py) | Tool-use SFT data generator, public tool manifest, validators. |
| [`../../src/hello_slm/banking_tool_wire.py`](../../src/hello_slm/banking_tool_wire.py) | Tool-wire adapter used by training and evaluation. |
| [`../../src/hello_slm/banking_tool_eval.py`](../../src/hello_slm/banking_tool_eval.py) | Frozen tool/final-response evaluator. |
| [`../../src/hello_slm/banking_conversation_router.py`](../../src/hello_slm/banking_conversation_router.py) | Released shared-encoder, three-head classifier model. |
| [`../../src/hello_slm/banking_conversation_router_data.py`](../../src/hello_slm/banking_conversation_router_data.py) | Leakage-safe history rendering and deterministic v4 classifier split builder. |
| [`../../src/hello_slm/banking_servicing_alignment_data.py`](../../src/hello_slm/banking_servicing_alignment_data.py) | Composite Granite SFT alignment generator and validation policy. |
| [`../../src/hello_slm/banking_asr_sft_data.py`](../../src/hello_slm/banking_asr_sft_data.py) | Reviewed ASR-to-SFT overlay builder, provenance checks, and manifest writer. |
| [`../../src/hello_slm/config.py`](../../src/hello_slm/config.py) | Canonical JSON and config helpers. |

## Scripts

| Path | Purpose |
| --- | --- |
| [`../../scripts/retail_bank/prepare_tool_sft_data.py`](../../scripts/retail_bank/prepare_tool_sft_data.py) | CLI wrapper for tool-use SFT data preparation. |
| [`../../scripts/retail_bank/prepare_conversation_router_data.py`](../../scripts/retail_bank/prepare_conversation_router_data.py) | Released history-aware router data preparation and digest verification. |
| [`../../scripts/retail_bank/train_conversation_router.py`](../../scripts/retail_bank/train_conversation_router.py) | Local cross-encoder training, calibration, test gates, artifact writing, and optional publication. |
| [`../../scripts/retail_bank/prepare_servicing_alignment_data.py`](../../scripts/retail_bank/prepare_servicing_alignment_data.py) | Released composite Granite data preparation, lock verification, and optional explicit publication. |
| [`../../scripts/retail_bank/prepare_asr_sft_data.py`](../../scripts/retail_bank/prepare_asr_sft_data.py) | Convert reviewed ASR utterances into trainer-compatible SFT overlays. |
| [`../../scripts/retail_bank/cloud_train_tool_sft.py`](../../scripts/retail_bank/cloud_train_tool_sft.py) | Guarded local/remote Granite tool-SFT worker. |
| [`../../scripts/retail_bank/hf_job_tool_sft.py`](../../scripts/retail_bank/hf_job_tool_sft.py) | Hugging Face Jobs bootstrap for paid Granite SFT. |
| [`../../scripts/retail_bank/run_remote_training_job.sh`](../../scripts/retail_bank/run_remote_training_job.sh) | Paid Granite training job launcher. |
| [`../../scripts/retail_bank/cloud_generate_tool_eval.py`](../../scripts/retail_bank/cloud_generate_tool_eval.py) | Frozen prediction generation and scoring worker. |
| [`../../scripts/retail_bank/hf_job_tool_eval.py`](../../scripts/retail_bank/hf_job_tool_eval.py) | Hugging Face Jobs bootstrap for paid frozen eval. |
| [`../../scripts/retail_bank/run_remote_tool_eval_job.sh`](../../scripts/retail_bank/run_remote_tool_eval_job.sh) | Paid frozen eval job launcher. |
| [`../../scripts/retail_bank/rescore_tool_eval.py`](../../scripts/retail_bank/rescore_tool_eval.py) | Reproducible prompt-equivalence proof and persisted-prediction rescore. |
| [`../../scripts/retail_bank/run_release_pipeline.py`](../../scripts/retail_bank/run_release_pipeline.py) | Canonical data, two-stage SFT, router, evaluation, and deployment orchestrator. |
| [`../../scripts/retail_bank/deploy_zero_gpu_space.py`](../../scripts/retail_bank/deploy_zero_gpu_space.py) | Guarded allowlist upload, exact runtime-pin persistence, and Space readiness helper. |
| [`../../scripts/retail_bank/hf_job_remerge_tool_sft.py`](../../scripts/retail_bank/hf_job_remerge_tool_sft.py) | Merge recovery helper used by release validation. |
| [`../../scripts/retail_bank/hf_job_merge_parity.py`](../../scripts/retail_bank/hf_job_merge_parity.py) | Merge parity helper. |

## POC

| Path | Purpose |
| --- | --- |
| [`../../poc/retail-bank-customer-service-poc/README.md`](../../poc/retail-bank-customer-service-poc/README.md) | Hugging Face Space card and public POC docs. |
| [`../../poc/retail-bank-customer-service-poc/app.py`](../../poc/retail-bank-customer-service-poc/app.py) | Gradio app, routing, ZeroGPU event, diagnostics, UI. |
| [`../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) | Granite model/tokenizer loading, token counting, deterministic generation. |
| [`../../poc/retail-bank-customer-service-poc/model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) | Model-owned tool loop, prompt budgeting, tool parsing, validation, execution trace. |
| [`../../poc/retail-bank-customer-service-poc/response_policy.py`](../../poc/retail-bank-customer-service-poc/response_policy.py) | Deterministic read-table rendering and bounded grounded-answer validation/repair input. |
| [`../../poc/retail-bank-customer-service-poc/router.py`](../../poc/retail-bank-customer-service-poc/router.py) | Released CPU cross-encoder loading, artifact verification, history rendering, three heads, and calibrated routing. |
| [`../../poc/retail-bank-customer-service-poc/auth.py`](../../poc/retail-bank-customer-service-poc/auth.py) | Static demo auth loader. |
| [`../../poc/retail-bank-customer-service-poc/mock_bank.py`](../../poc/retail-bank-customer-service-poc/mock_bank.py) | Session-isolated SQLite synthetic bank backend and tool implementation. |
| [`../../poc/retail-bank-customer-service-poc/state.py`](../../poc/retail-bank-customer-service-poc/state.py) | Session registry initialization. |
| [`../../poc/retail-bank-customer-service-poc/responses.py`](../../poc/retail-bank-customer-service-poc/responses.py) | Stock OOD and model-failure responses. |
| [`../../poc/retail-bank-customer-service-poc/synthetic_bank.json`](../../poc/retail-bank-customer-service-poc/synthetic_bank.json) | Synthetic customer seed records. |
| [`../../poc/retail-bank-customer-service-poc/requirements.txt`](../../poc/retail-bank-customer-service-poc/requirements.txt) | Space dependency list. |
| [`../../poc/retail-bank-customer-service-poc/pyproject.toml`](../../poc/retail-bank-customer-service-poc/pyproject.toml) | POC package metadata and local test settings. |

## Tests

| Path | Purpose |
| --- | --- |
| [`../../tests/test_banking_tool_sft_data.py`](../../tests/test_banking_tool_sft_data.py) | SFT data-generation tests. |
| [`../../tests/test_banking_tool_wire.py`](../../tests/test_banking_tool_wire.py) | Tool-wire adapter tests. |
| [`../../tests/test_banking_tool_eval.py`](../../tests/test_banking_tool_eval.py) | Evaluator tests. |
| [`../../tests/test_banking_tool_eval_runner.py`](../../tests/test_banking_tool_eval_runner.py) | Frozen eval runner and launcher tests. |
| [`../../tests/test_banking_counterfactual_eval_data.py`](../../tests/test_banking_counterfactual_eval_data.py) | Counterfactual determinism, contamination, pair, manifest, and benchmark-gate tests. |
| [`../../tests/test_banking_router_preparation.py`](../../tests/test_banking_router_preparation.py) | Router data preparation tests. |
| [`../../tests/test_banking_router_training.py`](../../tests/test_banking_router_training.py) | Router training and release-gate tests. |
| [`../../tests/test_banking_dual_head_router.py`](../../tests/test_banking_dual_head_router.py) | Shared router behavior tests. |
| [`../../tests/test_banking_conversation_router.py`](../../tests/test_banking_conversation_router.py) | Released three-head model tests. |
| [`../../tests/test_banking_conversation_router_data.py`](../../tests/test_banking_conversation_router_data.py) | History rendering, split isolation, leakage, and held-out data tests. |
| [`../../tests/test_banking_conversation_router_preparation.py`](../../tests/test_banking_conversation_router_preparation.py) | Source-lock and deterministic preparation tests. |
| [`../../tests/test_banking_conversation_router_training.py`](../../tests/test_banking_conversation_router_training.py) | Routing policy, metric, calibration, and release-gate tests. |
| [`../../tests/test_banking_servicing_alignment_data.py`](../../tests/test_banking_servicing_alignment_data.py) | Granite alignment coverage, composite counts, held-out isolation, and lock-drift tests. |
| [`../../tests/test_banking_asr_sft_data.py`](../../tests/test_banking_asr_sft_data.py) | ASR overlay semantics, provenance, split, digest, and fail-closed validation tests. |
| [`../../tests/test_banking_tool_sft_worker.py`](../../tests/test_banking_tool_sft_worker.py) | Tool SFT worker tests. |
| [`../../tests/test_banking_tool_eval_rescore.py`](../../tests/test_banking_tool_eval_rescore.py) | Prompt equivalence, coverage, and rescore release-gate tests. |
| [`../../tests/test_release_pipeline.py`](../../tests/test_release_pipeline.py) | Canonical stage order, immutable input, and execution-guard tests. |
| [`../../tests/test_deploy_zero_gpu_space.py`](../../tests/test_deploy_zero_gpu_space.py) | Space upload allowlist, exact pin persistence, and destructive-operation exclusion tests. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_auth.py`](../../poc/retail-bank-customer-service-poc/tests/test_auth.py) | POC auth tests. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_router.py`](../../poc/retail-bank-customer-service-poc/tests/test_router.py) | POC router tests. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_model_service.py`](../../poc/retail-bank-customer-service-poc/tests/test_model_service.py) | POC model-service and tool-loop tests. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_mock_bank.py`](../../poc/retail-bank-customer-service-poc/tests/test_mock_bank.py) | SQLite synthetic backend tests. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_app.py`](../../poc/retail-bank-customer-service-poc/tests/test_app.py) | Gradio app behavior tests. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/tests/test_zero_gpu_runtime.py) | ZeroGPU skip/runtime helper tests. |
