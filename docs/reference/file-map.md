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
| [`../../data/banking-conversation-router-v8-first-turn-mutation`](../../data/banking-conversation-router-v8-first-turn-mutation) | 20,439/4,158/4,921 train/validation/test rows. |
| [`../../data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json`](../../data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json) | Pinned external source and prepared split digests. |
| [`../../artifacts/banking-conversation-router-v8-first-turn-mutation/router_config.json`](../../artifacts/banking-conversation-router-v8-first-turn-mutation/router_config.json) | Format-4 labels, thresholds, guidance contract, and immutable dataset revision. |
| [`../../artifacts/banking-conversation-router-v8-first-turn-mutation/metrics.json`](../../artifacts/banking-conversation-router-v8-first-turn-mutation/metrics.json) | Training/calibration history, held-out metrics, counterfactual tests, and release status. |
| [`../../artifacts/banking-conversation-router-v8-first-turn-mutation/manifest.json`](../../artifacts/banking-conversation-router-v8-first-turn-mutation/manifest.json) | Artifact digests and release eligibility. |
| [`../../artifacts/banking-conversation-router-v8-first-turn-mutation/classifier_heads.safetensors`](../../artifacts/banking-conversation-router-v8-first-turn-mutation/classifier_heads.safetensors) | Seven learned classifier heads. |

Published identities:

```text
spkc83/retail-bank-conversation-router-data
b33c27170e27cdb11783704ede14f7d25f70625e

spkc83/retail-bank-conversation-router
dd5ea26674a0f9808d42110a9ee51a9af6762a76
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
| [`../../scripts/retail_bank/inproc_long_session_sweep.py`](../../scripts/retail_bank/inproc_long_session_sweep.py) | In-process long-session sweep: drives `LocalBankingController.run_turn` directly, one runtime load, a fresh session per case, no browser. |

## Granite Data and Training

| File | Responsibility |
| --- | --- |
| [`../../src/hello_slm/banking_tool_sft_data.py`](../../src/hello_slm/banking_tool_sft_data.py) | Base tool-use and conversational SFT generation. |
| [`../../src/hello_slm/banking_servicing_alignment_data.py`](../../src/hello_slm/banking_servicing_alignment_data.py) | Composite servicing-alignment scenarios used by Granite and router derivation. |
| [`../../scripts/retail_bank/prepare_tool_sft_data.py`](../../scripts/retail_bank/prepare_tool_sft_data.py) | Base SFT CLI. |
| [`../../scripts/retail_bank/prepare_servicing_alignment_data.py`](../../scripts/retail_bank/prepare_servicing_alignment_data.py) | Composite alignment CLI and optional publication. |
| [`../../poc/retail-bank-customer-service-poc/model_router.py`](../../poc/retail-bank-customer-service-poc/model_router.py) | The SLM as routing classifier, producing the learned router's decision shape so the harness is unchanged. |
| [`../../poc/retail-bank-customer-service-poc/taxonomy.py`](../../poc/retail-bank-customer-service-poc/taxonomy.py) | Canonical label sets, resolved once for both routing implementations. |
| [`../../scripts/retail_bank/compare_routing_classifiers.py`](../../scripts/retail_bank/compare_routing_classifiers.py) | Scores both classifiers on the router's held-out split. See [the comparison](../17-routing-classifier-comparison.md). |
| [`../../scripts/retail_bank/check_corpora_reproduce.py`](../../scripts/retail_bank/check_corpora_reproduce.py) | Runs the regeneration commands documented in `02-data-generation.md` and compares every split by SHA-256. Part of `make verify`. |
| [`../../scripts/retail_bank/measure_split_contamination.py`](../../scripts/retail_bank/measure_split_contamination.py) | Reports identifying-4-gram overlap and nearest-neighbour similarity for a corpus split, and documents why the first metric misleads. |
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
