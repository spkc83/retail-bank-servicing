# Artifact and Revision Ledger

Use immutable revisions for every model, dataset, and deployed runtime. A local
path identifies reproducible files; a Hub commit identifies the published
artifact.

## Active Conversation Router

| Artifact | Identity |
| --- | --- |
| Hub model | `spkc83/retail-bank-conversation-router@dd5ea26674a0f9808d42110a9ee51a9af6762a76` |
| Hub dataset | `spkc83/retail-bank-conversation-router-data@b33c27170e27cdb11783704ede14f7d25f70625e` |
| Local model | `artifacts/banking-conversation-router-v8-first-turn-mutation` |
| Local dataset | `data/banking-conversation-router-v8-first-turn-mutation` |
| Source lock | `data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json` |
| Base encoder | `distilbert/distilbert-base-uncased@12040accade4e8a0f71eabdb258fecc2e7e948be` |
| ZeroGPU Space source/pins | `spkc83/retail-bank-servicing-poc@a227df8d40934e6d3c1be31d49a49c4f20dcc81d` |
| ZeroGPU runtime | RUNNING; authenticated model-generation smoke pending |

### Dataset files

| File | Rows | SHA-256 |
| --- | ---: | --- |
| `train.jsonl` | 20,439 | `c838134cdecc22723fda887c1dd561329ab5cac2c72eabc2de484c54a4d4f733` |
| `validation.jsonl` | 4,158 | `5491dcbe64ef5c4d7a15d440076ef9964a3767a0adb94d0c4edbb33ecc3c2168` |
| `test.jsonl` | 4,921 | `135e2c16962a19c2752b85ca626e83e067eaa9222ff7e1b9029bbdbe681584e8` |

The dataset manifest SHA-256 recorded in the model is
`caae2209063beb9370d0f3a6fc166e4c35658fafdd2420b21e5920c6c9e90de5`.

### Router artifact files

| File | Purpose |
| --- | --- |
| `router_config.json` | Format-4 labels, thresholds, input format, base and dataset identity. |
| `metrics.json` | Training history, calibration, held-out metrics, and release decision. |
| `manifest.json` | Release eligibility and per-file sizes/digests. |
| `model.safetensors` | Shared DistilBERT encoder weights. |
| `classifier_heads.safetensors` | Domain, lane, family, intent, relation, action, and entity-resolution weights. |
| `config.json` | Hugging Face encoder configuration. |
| `tokenizer.json`, `tokenizer_config.json` | Pinned tokenizer files. |
| `README.md` | Generated model card. |

Release evidence: selected epoch 2, `release_eligible: true`, empty
`release_gate_failures`, and zero exposed hierarchy-compatibility errors.

## Granite Composition

| Component | Identity |
| --- | --- |
| PEFT release (deployed) | `spkc83/retail-bank-servicing-agent-9b-peft-v10-longctx@055ce38af4595b1e139a9e9baea8e0c53cba7c2e`; adapter_config.json lives under `adapter/`, so the runtime needs `RETAIL_BANK_ADAPTER_SUBFOLDER=adapter` |
| PEFT candidate (not deployed) | `spkc83/retail-bank-servicing-agent-9b-peft-v11-alignment@03a7b44633fadab7ad672b009925cc68b52494d4` — v11 policy-alignment retrain (job `6a908078`, dev and shadow gates 1.0, dataset `@b5ec0489`, source `6e53ebf`); same `adapter/` subfolder convention |
| PEFT release (last evaluated) | `spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation@badbc05ad1f861818ea244b462eda49bca6c6fca` |
| Adapter bundle | `b4269445ce7b2b943d2d9531102166bf8840a074` |
| Stage-2 base | `spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| SFT dataset | `spkc83/retail-bank-servicing-alignment-sft@a78bed17db8c56099a32f835832b9878a895a602` |
| Policy corpus | `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a` |

### Granite SFT dataset files

Digests below are the conversational-voice regeneration described in
[Granite SFT corpora](../02-data-generation.md#granite-sft-corpora). This is a
local regeneration, not yet published: the Hub SFT dataset revision above stays
`a78bed17db8c56099a32f835832b9878a895a602` until the corpora are republished,
and it remains what the v8 adapter cites.

`data/banking-v5-tool-sft`

| File | Rows | SHA-256 |
| --- | ---: | --- |
| `train.jsonl` | 841 | `b723dabbe44b5148cd729f723ee236141f03e202bc60de013c8b263eee6aea6c` |
| `validation.jsonl` | 179 | `e7c7ca152a2376b0f95c5e4bd495db437a53a02d637258698b28060f2f062573` |
| `test.jsonl` (frozen) | 180 | `9a7938ac5e5dfdc5e176de9d599debdd7c0e7a02fa70ce8f585541b68e03618c` |

`data/banking-servicing-alignment-v5`

| File | Rows | SHA-256 |
| --- | ---: | --- |
| `train.jsonl` | 3,267 | `9ec916e0dfc1fbd52d7f091a9ca7cb82f1b5038566bd649e9b9d92a6809e7c34` |
| `validation.jsonl` | 397 | `3cf683aed3432a2632187cbc71079cf61b5092687b421cee6dacc0f54818dbda` |
| `test.jsonl` (frozen) | 215 | `36557c20e13f9ab292d6310df0732d6ba9cdf9a7fa6ffef42ee2e3ef4f289811` |

The frozen `test.jsonl` digests, and the alignment `coreference-shadow.jsonl`,
`granite-v7-shadow.jsonl`, and `screenshot-regression.jsonl` files, are
unchanged by the regeneration.

The runtime loads the base and attaches the adapter without merging: BF16 on
ZeroGPU and NF4-quantized base plus adapter locally.

## Identity Rules

- Never deploy a branch name such as `main` as a model identity.
- Publish data before training and record its commit in `router_config.json`.
- Verify artifact file digests before loading.
- Persist exact router, base, adapter, and Space commits in diagnostics.
- Treat a new dataset or model commit as a new release candidate; rerun gates.
