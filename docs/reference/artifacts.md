# Artifact and Revision Ledger

Use immutable revisions for every model, dataset, and deployed runtime. A local
path identifies reproducible files; a Hub commit identifies the published
artifact.

## Active Conversation Router

| Artifact | Identity |
| --- | --- |
| Hub model | `spkc83/retail-bank-conversation-router@7f6a0e77ad231233702039560ced007fdc68bd74` |
| Hub dataset | `spkc83/retail-bank-conversation-router-data@80c0edfea84b341d2ee4092f5c4a4bbb05405e40` |
| Local model | `artifacts/banking-conversation-router-v6-hierarchical` |
| Local dataset | `data/banking-conversation-router-v6-hierarchical` |
| Source lock | `data/sources/banking-conversation-router-v6-hierarchical.lock.json` |
| Base encoder | `distilbert/distilbert-base-uncased@12040accade4e8a0f71eabdb258fecc2e7e948be` |
| ZeroGPU Space source/pins | `spkc83/retail-bank-servicing-poc@2ec64ceacc390f5619d246fbca60dcca67f4a83f` |
| ZeroGPU runtime | RUNNING; authenticated model-generation smoke passed |

### Dataset files

| File | Rows | SHA-256 |
| --- | ---: | --- |
| `train.jsonl` | 16,693 | `10b6d4316719f4cd9f162d9faef36a3c5f264bf7a3a0ae759a2c5562993032f3` |
| `validation.jsonl` | 4,061 | `cb9660b696d1b9d4ee81922c0fc042861c8cae9a55a9d0760b02ad98b94646f9` |
| `test.jsonl` | 4,895 | `58bc7c10dd11797988e987afd17bdd9f5e0f79b2dbc2a17ed2ea33a8cb176b68` |

The dataset manifest SHA-256 recorded in the model is
`0886dd8037e59d73b41c4ee60cde57dc865c85494309accd821d8a423681da11`.

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
| PEFT release | `spkc83/retail-bank-servicing-agent-9b-peft@cc95e446af2b5e1d8d9df2751a8192613ad386e3` |
| Adapter bundle | `b4269445ce7b2b943d2d9531102166bf8840a074` |
| Stage-2 base | `spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| SFT dataset | `spkc83/retail-bank-servicing-alignment-sft@40a0b68b9f746131ffff32a83e077fd7e4a344d1` |
| Policy corpus | `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a` |

The runtime loads the base and attaches the adapter without merging: BF16 on
ZeroGPU and NF4-quantized base plus adapter locally.

## Identity Rules

- Never deploy a branch name such as `main` as a model identity.
- Publish data before training and record its commit in `router_config.json`.
- Verify artifact file digests before loading.
- Persist exact router, base, adapter, and Space commits in diagnostics.
- Treat a new dataset or model commit as a new release candidate; rerun gates.
