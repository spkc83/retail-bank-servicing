# V5 Code and File Map

This map covers the active V5 path. The V4 design documents remain available
only as superseded history.

## Documentation

| File | Purpose |
| --- | --- |
| [`../01-system-overview.md`](../01-system-overview.md) | Component boundaries and complete request flow. |
| [`../02-data-generation.md`](../02-data-generation.md) | V5 SFT and router data, examples, split isolation, and revisions. |
| [`../03-model-and-peft.md`](../03-model-and-peft.md) | Granite assistant-only SFT, LoRA, action wire, and published V5 model. |
| [`../04-training-and-recovery.md`](../04-training-and-recovery.md) | Guarded RTX PRO 6000 training, checkpoints, resume, merge, and publication. |
| [`../05-dual-head-router.md`](../05-dual-head-router.md) | Active three-head state-aware router. The filename is retained for stable links. |
| [`../06-evaluation.md`](../06-evaluation.md) | Data, router, Granite, orchestration, and human release gates. |
| [`../07-inference-and-poc.md`](../07-inference-and-poc.md) | Local and ZeroGPU inference, state, retrieval, actions, and diagnostics. |
| [`../08-end-to-end-runbook.md`](../08-end-to-end-runbook.md) | Reproducible active V5 sequence. |
| [`../09-conversation-router-v4.md`](../09-conversation-router-v4.md) | Superseded V4 router history; not active instructions. |
| [`../10-servicing-alignment-v4.md`](../10-servicing-alignment-v4.md) | Superseded V4 alignment history; not active instructions. |
| [`../12-instruction-fine-tuning-and-peft.md`](../12-instruction-fine-tuning-and-peft.md) | Background explanation of instruction SFT and PEFT. |
| [`../13-counterfactual-evaluation.md`](../13-counterfactual-evaluation.md) | Evaluation-only counterfactual design retained as supporting methodology. |
| [`../14-questions-and-answers.md`](../14-questions-and-answers.md) | Focused implementation questions and answers. |
| [`../15-asr-to-sft-pipeline.md`](../15-asr-to-sft-pipeline.md) | ASR transcript conversion into reviewed SFT overlays. |

## V5 Data

| File or directory | Purpose |
| --- | --- |
| [`../../data/banking-v5-tool-sft`](../../data/banking-v5-tool-sft) | Generated base Granite tool-use, policy, conversation, and OOD SFT. |
| [`../../data/banking-v5-tool-sft/manifest.json`](../../data/banking-v5-tool-sft/manifest.json) | Base split counts, digests, schema, and action-manifest identity. |
| [`../../data/banking-servicing-alignment-v5`](../../data/banking-servicing-alignment-v5) | Composite base plus V5 servicing-alignment corpus. |
| [`../../data/banking-servicing-alignment-v5/manifest.json`](../../data/banking-servicing-alignment-v5/manifest.json) | Composite counts, base-manifest digest, path counts, and split digests. |
| [`../../data/banking-conversation-router-v5-social-policy-generalization-candidate5`](../../data/banking-conversation-router-v5-social-policy-generalization-candidate5) | Exact generalized router train, validation, and test rows. |
| [`../../data/banking-conversation-router-v5-social-policy-generalization-candidate5/manifest.json`](../../data/banking-conversation-router-v5-social-policy-generalization-candidate5/manifest.json) | Router labels, source pins, counts, leakage report, and split digests. |
| [`../../data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json`](../../data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json) | CLINC archive/member hashes and prepared generalized split hashes. |
| [`../../data/banking-counterfactual-eval-v1`](../../data/banking-counterfactual-eval-v1) | Separate evaluation-only counterfactual corpus. |
| [`../../examples/asr`](../../examples/asr) | Reviewed ASR examples for the optional overlay pipeline. |

Published V5 data identities:

```text
spkc83/retail-bank-servicing-alignment-sft
  40a0b68b9f746131ffff32a83e077fd7e4a344d1

spkc83/retail-bank-conversation-router-data
  8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc
```

## Reusable Source Modules

| File | Purpose |
| --- | --- |
| [`../../src/hello_slm/banking_tool_sft_data.py`](../../src/hello_slm/banking_tool_sft_data.py) | V5 base SFT generation, schema validation, split controls, policy targets, and fixtures. |
| [`../../src/hello_slm/banking_servicing_alignment_data.py`](../../src/hello_slm/banking_servicing_alignment_data.py) | Composite V5 alignment generation, policy detours/resumes, held-out regressions, and leakage checks. |
| [`../../src/hello_slm/banking_tool_wire.py`](../../src/hello_slm/banking_tool_wire.py) | Granite chat-template rendering, tagged-JSON actions, and assistant-only label masks. |
| [`../../src/hello_slm/banking_tool_eval.py`](../../src/hello_slm/banking_tool_eval.py) | Frozen action/final-response metrics and release gates. |
| [`../../src/hello_slm/banking_conversation_router.py`](../../src/hello_slm/banking_conversation_router.py) | Shared-encoder three-head router model. |
| [`../../src/hello_slm/banking_conversation_router_data.py`](../../src/hello_slm/banking_conversation_router_data.py) | State/history rendering, fine-intent and relation labels, and deterministic router examples. |
| [`../../src/hello_slm/banking_asr_sft_data.py`](../../src/hello_slm/banking_asr_sft_data.py) | Optional reviewed-ASR overlay conversion. |
| [`../../src/hello_slm/banking_counterfactual_eval_data.py`](../../src/hello_slm/banking_counterfactual_eval_data.py) | Evaluation-only counterfactual generation. |
| [`../../src/hello_slm/config.py`](../../src/hello_slm/config.py) | Canonical JSON and SHA-256 helpers. |

## Data and Training Scripts

| File | Purpose |
| --- | --- |
| [`../../scripts/retail_bank/prepare_tool_sft_data.py`](../../scripts/retail_bank/prepare_tool_sft_data.py) | CLI for `data/banking-v5-tool-sft`. |
| [`../../scripts/retail_bank/prepare_servicing_alignment_data.py`](../../scripts/retail_bank/prepare_servicing_alignment_data.py) | CLI for composite `data/banking-servicing-alignment-v5` and optional publication. |
| [`../../scripts/retail_bank/prepare_conversation_router_data.py`](../../scripts/retail_bank/prepare_conversation_router_data.py) | CLI used to produce the governed generalized candidate-5 router corpus. |
| [`../../scripts/retail_bank/train_conversation_router.py`](../../scripts/retail_bank/train_conversation_router.py) | Local three-head router training, calibration, gates, artifact save, and optional publication. |
| [`../../scripts/retail_bank/cloud_train_tool_sft.py`](../../scripts/retail_bank/cloud_train_tool_sft.py) | Guarded Granite V5 BF16 LoRA/QLoRA worker, checkpointing, merge parity, and upload. |
| [`../../scripts/retail_bank/hf_job_tool_sft.py`](../../scripts/retail_bank/hf_job_tool_sft.py) | Hugging Face Job bootstrap for exact source and dataset revisions. |
| [`../../scripts/retail_bank/hf_job_finalize_tool_sft_peft.py`](../../scripts/retail_bank/hf_job_finalize_tool_sft_peft.py) | Validates and atomically publishes an unmerged BF16 LoRA adapter when merged candidates fail parity. |
| [`../../scripts/retail_bank/run_remote_training_job.sh`](../../scripts/retail_bank/run_remote_training_job.sh) | RTX PRO 6000, five-hour, persistent-bucket job launcher. |
| [`../../scripts/retail_bank/cloud_generate_tool_eval.py`](../../scripts/retail_bank/cloud_generate_tool_eval.py) | Frozen Granite prediction generation and release-gate enforcement. |
| [`../../scripts/retail_bank/hf_job_tool_eval.py`](../../scripts/retail_bank/hf_job_tool_eval.py) | Hugging Face Job bootstrap for frozen evaluation. |
| [`../../scripts/retail_bank/run_remote_tool_eval_job.sh`](../../scripts/retail_bank/run_remote_tool_eval_job.sh) | Paid frozen-evaluation launcher. |
| [`../../scripts/retail_bank/evaluate_tool_model.py`](../../scripts/retail_bank/evaluate_tool_model.py) | Static prediction rescore CLI. |
| [`../../scripts/retail_bank/rescore_tool_eval.py`](../../scripts/retail_bank/rescore_tool_eval.py) | Prompt-equivalence and persisted-prediction rescore helper. |
| [`../../scripts/retail_bank/prepare_asr_sft_data.py`](../../scripts/retail_bank/prepare_asr_sft_data.py) | Reviewed ASR overlay CLI. |
| [`../../scripts/retail_bank/deploy_zero_gpu_space.py`](../../scripts/retail_bank/deploy_zero_gpu_space.py) | Allowlisted Space upload, immutable runtime variables, and readiness wait. |
| [`../../scripts/retail_bank/run_local_streamlit.py`](../../scripts/retail_bank/run_local_streamlit.py) | Pinned local Streamlit launcher. |

[`../../scripts/retail_bank/run_release_pipeline.py`](../../scripts/retail_bank/run_release_pipeline.py)
is retained but its current configuration describes the superseded V4
sequence. The active V5 runbook uses the individual scripts above.

## Router Artifact

| File | Purpose |
| --- | --- |
| [`../../artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5/manifest.json`](../../artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5/manifest.json) | Release eligibility and per-file digests. |
| [`../../artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5/router_config.json`](../../artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5/router_config.json) | Format 3 labels, input format, calibrated thresholds, and base identity. |
| [`../../artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5/metrics.json`](../../artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5/metrics.json) | Calibration, generalized social/policy gates, held-out metrics, and regression predictions. |
| [`../../artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5/model.safetensors`](../../artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5/model.safetensors) | DistilBERT encoder weights. |
| [`../../artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5/classifier_heads.safetensors`](../../artifacts/banking-conversation-router-v5-social-policy-generalization-candidate5/classifier_heads.safetensors) | Domain, fine-intent, and relation head weights. |

Published V5 router revision:

```text
spkc83/retail-bank-conversation-router
c8f154266612e79afe20af8abef25761fa56d589
```

## POC Runtime

| File | Purpose |
| --- | --- |
| [`../../poc/retail-bank-customer-service-poc/app.py`](../../poc/retail-bank-customer-service-poc/app.py) | Gradio UI, ZeroGPU event, state payload, routing, policy lane, reset, and diagnostics. |
| [`../../poc/retail-bank-customer-service-poc/streamlit_app.py`](../../poc/retail-bank-customer-service-poc/streamlit_app.py) | Local CUDA UI, authentication, presets, sidebar, and diagnostics popover. |
| [`../../poc/retail-bank-customer-service-poc/local_app_service.py`](../../poc/retail-bank-customer-service-poc/local_app_service.py) | UI-independent local orchestration controller. |
| [`../../poc/retail-bank-customer-service-poc/router.py`](../../poc/retail-bank-customer-service-poc/router.py) | V5 artifact verification, state-aware input rendering, inference, thresholds, and derived lane. |
| [`../../poc/retail-bank-customer-service-poc/dialogue_state.py`](../../poc/retail-bank-customer-service-poc/dialogue_state.py) | One-pending-task state machine, policy detour, resume anchor, completion, registry, and reset. |
| [`../../poc/retail-bank-customer-service-poc/policy_retrieval.py`](../../poc/retail-bank-customer-service-poc/policy_retrieval.py) | Corpus validation, digest verification, deterministic retrieval, and citation metadata. |
| [`../../poc/retail-bank-customer-service-poc/policy_knowledge.json`](../../poc/retail-bank-customer-service-poc/policy_knowledge.json) | Versioned policy chunks used at inference. |
| [`../../poc/retail-bank-customer-service-poc/model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) | Granite prompt, token budgeting, pinned resume context, action loop, policy generation, and traces. |
| [`../../poc/retail-bank-customer-service-poc/response_policy.py`](../../poc/retail-bank-customer-service-poc/response_policy.py) | Exact read tables, action grounding, policy citations/numbers, internal-language checks, and repair prompts. |
| [`../../poc/retail-bank-customer-service-poc/branding.py`](../../poc/retail-bank-customer-service-poc/branding.py) | Harborlight/Harbor names, CSS, account labels, and response provenance. |
| [`../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) | Pinned BF16 base-plus-adapter PEFT loading, chat-template tokenization, generation, and composition metadata. |
| [`../../poc/retail-bank-customer-service-poc/local_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/local_gpu_runtime.py) | NF4 base loading plus pinned adapter attachment, CUDA validation, generation, locking, and metadata. |
| [`../../poc/retail-bank-customer-service-poc/mock_bank.py`](../../poc/retail-bank-customer-service-poc/mock_bank.py) | Session-isolated fictional SQLite records and action execution. |
| [`../../poc/retail-bank-customer-service-poc/synthetic_bank.json`](../../poc/retail-bank-customer-service-poc/synthetic_bank.json) | Fictional seed profiles and account state. |
| [`../../poc/retail-bank-customer-service-poc/auth.py`](../../poc/retail-bank-customer-service-poc/auth.py) | Two-profile static POC authentication validation. |
| [`../../poc/retail-bank-customer-service-poc/responses.py`](../../poc/retail-bank-customer-service-poc/responses.py) | OOD, policy-no-match, and model-failure responses. |

## Active Tests

| Test group | Main evidence |
| --- | --- |
| [`../../tests/test_banking_tool_sft_data.py`](../../tests/test_banking_tool_sft_data.py) | V5 SFT schema, records, leakage, tables, policy, and branding. |
| [`../../tests/test_banking_servicing_alignment_data.py`](../../tests/test_banking_servicing_alignment_data.py) | Composite V5 counts, scenarios, and held-out isolation. |
| [`../../tests/test_banking_conversation_router_data.py`](../../tests/test_banking_conversation_router_data.py) | State-aware rows, labels, split/trajectory isolation. |
| [`../../tests/test_banking_conversation_router_training.py`](../../tests/test_banking_conversation_router_training.py) | Three-head training, calibration, gates, artifact, and publication guards. |
| [`../../tests/test_banking_tool_sft_peft_release.py`](../../tests/test_banking_tool_sft_peft_release.py) | Adapter fingerprint validation, atomic publication, bundle identity, and metadata commit separation. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_dialogue_state.py`](../../poc/retail-bank-customer-service-poc/tests/test_dialogue_state.py) | Detour, resume, switch, completion, reset, and state serialization. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_policy_retrieval.py`](../../poc/retail-bank-customer-service-poc/tests/test_policy_retrieval.py) | Corpus digest, dates, ranking, no-match, and citations. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_model_service.py`](../../poc/retail-bank-customer-service-poc/tests/test_model_service.py) | Action loop, policy generation, repairs, pinned context, and traces. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_local_app_service.py`](../../poc/retail-bank-customer-service-poc/tests/test_local_app_service.py) | End-to-end local routing, policy detour/resume, reset, and diagnostics. |
| [`../../poc/retail-bank-customer-service-poc/tests/test_app.py`](../../poc/retail-bank-customer-service-poc/tests/test_app.py) | Gradio/ZeroGPU orchestration and UI-state behavior. |
