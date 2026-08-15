# Artifact and Revision Ledger

Use immutable revisions for every model, dataset, and deployed runtime. A local
path identifies reproducible files; a Hub commit identifies the published
artifact.

## Active Conversation Router

| Artifact | Identity |
| --- | --- |
| Hub model | `spkc83/retail-bank-conversation-router@c0d71b433fd1eef510fce36f6308eb36e423e329` |
| Hub dataset | `spkc83/retail-bank-conversation-router-data@073e61156885a8a2074c7254d76f00634058429a` |
| Local model | `artifacts/banking-conversation-router-v6-hierarchical` |
| Local dataset | `data/banking-conversation-router-v6-hierarchical` |
| Source lock | `data/sources/banking-conversation-router-v6-hierarchical.lock.json` |
| Base encoder | `distilbert/distilbert-base-uncased@12040accade4e8a0f71eabdb258fecc2e7e948be` |
| ZeroGPU Space source/pins | `spkc83/retail-bank-servicing-poc@2ec64ceacc390f5619d246fbca60dcca67f4a83f` |
| ZeroGPU runtime | RUNNING; authenticated model-generation smoke passed |

### Dataset files

| File | Rows | SHA-256 |
| --- | ---: | --- |
| `train.jsonl` | 16,720 | `a03a35a384a80d39c10455d32d28a20a902f5ac11da5d18a279c1604fe96e38f` |
| `validation.jsonl` | 4,077 | `83ecf3f221ec03d858a99fbdb1f0cebd5172157444a296418529674b62f0a7b3` |
| `test.jsonl` | 4,913 | `64a0e7e6c6c0016979116da7e08a75f01fafb603e70b39caa0c8467345df61d9` |

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
| PEFT release | `spkc83/retail-bank-servicing-agent-9b-peft@cc95e446af2b5e1d8d9df2751a8192613ad386e3` |
| Adapter bundle | `b4269445ce7b2b943d2d9531102166bf8840a074` |
| Stage-2 base | `spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| SFT dataset | `spkc83/retail-bank-servicing-alignment-sft@9d7aed545604bb42fb02b7a0919427a0ed2b81e2` |
| Policy corpus | `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a` |

The runtime loads the base and attaches the adapter without merging: BF16 on
ZeroGPU and NF4-quantized base plus adapter locally.

## Identity Rules

- Never deploy a branch name such as `main` as a model identity.
- Publish data before training and record its commit in `router_config.json`.
- Verify artifact file digests before loading.
- Persist exact router, base, adapter, and Space commits in diagnostics.
- Treat a new dataset or model commit as a new release candidate; rerun gates.
