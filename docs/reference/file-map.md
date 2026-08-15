# V6 Code and File Map

This map links the active hierarchical-router design to implementation files.

## Taxonomy, Data, and Model

| File | Responsibility |
| --- | --- |
| [`../../src/hello_slm/banking_domain_taxonomy.py`](../../src/hello_slm/banking_domain_taxonomy.py) | Canonical domain/lane/family/intent hierarchy, action labels, entity states, tool compatibility, and label validation. |
| [`../../src/hello_slm/banking_conversation_router.py`](../../src/hello_slm/banking_conversation_router.py) | Shared encoder, seven heads, V4 joint decision type, constrained decoder, and compatibility helpers. |
| [`../../src/hello_slm/banking_conversation_router_data.py`](../../src/hello_slm/banking_conversation_router_data.py) | Router row construction, state trajectories, targeted/counterfactual examples, split isolation, and reports. |
| [`../../scripts/retail_bank/prepare_conversation_router_data.py`](../../scripts/retail_bank/prepare_conversation_router_data.py) | V6 dataset CLI, CLINC checksum validation, leakage/PII gates, manifests, locks, and data card. |
| [`../../scripts/retail_bank/train_conversation_router.py`](../../scripts/retail_bank/train_conversation_router.py) | Seven-head training, weighting, calibration, exact joint-decoder evaluation, release gates, artifact save, and optional publication. |

## Active Data and Artifact

| Path | Responsibility |
| --- | --- |
| [`../../data/banking-conversation-router-v6-hierarchical`](../../data/banking-conversation-router-v6-hierarchical) | 16,693/4,061/4,895 train/validation/test rows. |
| [`../../data/sources/banking-conversation-router-v6-hierarchical.lock.json`](../../data/sources/banking-conversation-router-v6-hierarchical.lock.json) | Pinned external source and prepared split digests. |
| [`../../artifacts/banking-conversation-router-v6-hierarchical/router_config.json`](../../artifacts/banking-conversation-router-v6-hierarchical/router_config.json) | Format-4 labels, thresholds, guidance contract, and immutable dataset revision. |
| [`../../artifacts/banking-conversation-router-v6-hierarchical/metrics.json`](../../artifacts/banking-conversation-router-v6-hierarchical/metrics.json) | Training/calibration history, held-out metrics, counterfactual tests, and release status. |
| [`../../artifacts/banking-conversation-router-v6-hierarchical/manifest.json`](../../artifacts/banking-conversation-router-v6-hierarchical/manifest.json) | Artifact digests and release eligibility. |
| [`../../artifacts/banking-conversation-router-v6-hierarchical/classifier_heads.safetensors`](../../artifacts/banking-conversation-router-v6-hierarchical/classifier_heads.safetensors) | Seven learned classifier heads. |

Published identities:

```text
spkc83/retail-bank-conversation-router-data
80c0edfea84b341d2ee4092f5c4a4bbb05405e40

spkc83/retail-bank-conversation-router
7f6a0e77ad231233702039560ced007fdc68bd74
```

## POC Orchestration

| File | Responsibility |
| --- | --- |
| [`../../poc/retail-bank-customer-service-poc/router.py`](../../poc/retail-bank-customer-service-poc/router.py) | Artifact verification, state-aware rendering, seven-head inference, route boundary, joint decoding, and diagnostics. |
| [`../../poc/retail-bank-customer-service-poc/model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) | Token-budgeted context, action-guided single-tool plan, Granite loop, policy generation, and model-pass traces. |
| [`../../poc/retail-bank-customer-service-poc/dialogue_state.py`](../../poc/retail-bank-customer-service-poc/dialogue_state.py) | Pending servicing task, policy detour, resume, intent switch, completion, and reset. |
| [`../../poc/retail-bank-customer-service-poc/policy_retrieval.py`](../../poc/retail-bank-customer-service-poc/policy_retrieval.py) | Policy corpus verification, ranking, and citation metadata. |
| [`../../poc/retail-bank-customer-service-poc/response_policy.py`](../../poc/retail-bank-customer-service-poc/response_policy.py) | Exact read tables, action grounding, policy citations, internal-language checks, and repair prompts. |
| [`../../poc/retail-bank-customer-service-poc/local_app_service.py`](../../poc/retail-bank-customer-service-poc/local_app_service.py) | UI-independent local controller and diagnostic rendering. |
| [`../../poc/retail-bank-customer-service-poc/app.py`](../../poc/retail-bank-customer-service-poc/app.py) | Gradio/ZeroGPU UI and session orchestration. |
| [`../../poc/retail-bank-customer-service-poc/streamlit_app.py`](../../poc/retail-bank-customer-service-poc/streamlit_app.py) | Local Streamlit UI, auth, presets, and diagnostics. |
| [`../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) | BF16 base-plus-adapter loading and generation on ZeroGPU. |
| [`../../poc/retail-bank-customer-service-poc/local_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/local_gpu_runtime.py) | NF4 base plus adapter loading, CUDA validation, and generation. |
| [`../../poc/retail-bank-customer-service-poc/mock_bank.py`](../../poc/retail-bank-customer-service-poc/mock_bank.py) | Fictional session-isolated bank state and action execution. |
| [`../../scripts/retail_bank/deploy_zero_gpu_space.py`](../../scripts/retail_bank/deploy_zero_gpu_space.py) | Allowlisted Space upload, immutable runtime variables, and readiness wait. |
| [`../../scripts/retail_bank/run_local_streamlit.py`](../../scripts/retail_bank/run_local_streamlit.py) | Pinned local launcher. |

## Granite Data and Training

| File | Responsibility |
| --- | --- |
| [`../../src/hello_slm/banking_tool_sft_data.py`](../../src/hello_slm/banking_tool_sft_data.py) | Base tool-use and conversational SFT generation. |
| [`../../src/hello_slm/banking_servicing_alignment_data.py`](../../src/hello_slm/banking_servicing_alignment_data.py) | Composite servicing-alignment scenarios used by Granite and router derivation. |
| [`../../scripts/retail_bank/prepare_tool_sft_data.py`](../../scripts/retail_bank/prepare_tool_sft_data.py) | Base SFT CLI. |
| [`../../scripts/retail_bank/prepare_servicing_alignment_data.py`](../../scripts/retail_bank/prepare_servicing_alignment_data.py) | Composite alignment CLI and optional publication. |
| [`../../scripts/retail_bank/cloud_train_tool_sft.py`](../../scripts/retail_bank/cloud_train_tool_sft.py) | Granite LoRA/QLoRA worker and checkpoints. |
| [`../../scripts/retail_bank/hf_job_finalize_tool_sft_peft.py`](../../scripts/retail_bank/hf_job_finalize_tool_sft_peft.py) | Unmerged adapter validation and publication. |
| [`../../scripts/retail_bank/cloud_generate_tool_eval.py`](../../scripts/retail_bank/cloud_generate_tool_eval.py) | Frozen generative evaluation. |

## Primary Tests

| Test | Evidence |
| --- | --- |
| [`../../tests/test_banking_conversation_router.py`](../../tests/test_banking_conversation_router.py) | Model, taxonomy, joint decoder, and compatibility behavior. |
| [`../../tests/test_banking_conversation_router_data.py`](../../tests/test_banking_conversation_router_data.py) | V6 rows, hierarchy, counterfactuals, and split isolation. |
| [`../../tests/test_banking_conversation_router_preparation.py`](../../tests/test_banking_conversation_router_preparation.py) | Source validation, manifest, lock, leakage, and publication data. |
| [`../../tests/test_banking_conversation_router_training.py`](../../tests/test_banking_conversation_router_training.py) | Seven-head loss, calibration, metrics, artifact, and release gates. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_router.py`](../../poc/retail-bank-customer-service-poc/tests/test_router.py) | Runtime artifact loading, joint tuple, thresholds, and diagnostics. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_model_service.py`](../../poc/retail-bank-customer-service-poc/tests/test_model_service.py) | Single-tool guidance, tool loop, policy lane, context, and traces. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_dialogue_state.py`](../../poc/retail-bank-customer-service-poc/tests/test_dialogue_state.py) | Detour, resume, intent switch, completion, and reset. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_local_app_service.py`](../../poc/retail-bank-customer-service-poc/tests/test_local_app_service.py) | End-to-end local orchestration behavior. |
