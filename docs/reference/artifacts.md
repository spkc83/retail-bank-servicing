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
| ZeroGPU Space source/pins | `spkc83/retail-bank-servicing-poc@bab5b2237b22814ede6a76c6c5ac2a1354097d44` |
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
| PEFT release | `spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation@badbc05ad1f861818ea244b462eda49bca6c6fca` |
| Adapter bundle | `b4269445ce7b2b943d2d9531102166bf8840a074` |
| Stage-2 base | `spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| SFT dataset | `spkc83/retail-bank-servicing-alignment-sft@a78bed17db8c56099a32f835832b9878a895a602` |
| Policy corpus | `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a` |

The runtime loads the base and attaches the adapter without merging: BF16 on
ZeroGPU and NF4-quantized base plus adapter locally.

## Identity Rules

- Never deploy a branch name such as `main` as a model identity.
- Publish data before training and record its commit in `router_config.json`.
- Verify artifact file digests before loading.
- Persist exact router, base, adapter, and Space commits in diagnostics.
- Treat a new dataset or model commit as a new release candidate; rerun gates.
