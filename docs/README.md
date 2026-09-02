# Harborlight Retail-Bank Agent Documentation

These documents describe the active V6 hierarchical conversation router and
the Granite PEFT servicing POC. A new developer can regenerate the governed
router data, train and evaluate the CPU model, inspect the orchestration
contract, and run either interface.

## Current Artifact Ledger

| Component | Identity/status |
| --- | --- |
| Hierarchical router | `spkc83/retail-bank-conversation-router@dd5ea26674a0f9808d42110a9ee51a9af6762a76` |
| Router dataset | `spkc83/retail-bank-conversation-router-data@b33c27170e27cdb11783704ede14f7d25f70625e` |
| Local router artifact | `artifacts/banking-conversation-router-v8-first-turn-mutation`; release eligible |
| Router dataset rows | train 20,439; validation 4,158; test 4,921 |
| Granite SFT dataset | `spkc83/retail-bank-servicing-alignment-sft@a649b7664844e029fddbb993917f9e58f0bddf93` (this checkout; `@ce0d4429` is the same data one commit earlier, before the card was added, and is the revision v14 trained on; v11, deployed until 2026-09-01, was trained on `@b5ec0489f96cf783a0bc993bc29898c6e9b35ba5`) |
| Granite PEFT adapter (deployed) | `spkc83/retail-bank-servicing-agent-9b-peft-v14-prompt-realized@47968b2b9ce02973b5676e464aafaa768cdbb05e`; `RETAIL_BANK_ADAPTER_SUBFOLDER=adapter`. First run to pass all three training gates, including bare probes 11/11; deployed 2026-09-01 |
| Granite PEFT adapter (previously deployed) | `spkc83/retail-bank-servicing-agent-9b-peft-v11-alignment@03a7b44633fadab7ad672b009925cc68b52494d4` |
| Granite PEFT adapter (last evaluated) | `spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation@badbc05ad1f861818ea244b462eda49bca6c6fca` |
| Granite Stage-2 base | `spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| ZeroGPU Space source/pins | `spkc83/retail-bank-servicing-poc@2a6501b6d5029d1e1991f7444c9f352eef31b000`; authenticated chat smoke pending |

## Read in This Order

1. [System overview](01-system-overview.md)
2. [Data generation](02-data-generation.md)
3. [Granite and PEFT](03-model-and-peft.md)
4. [Training and recovery](04-training-and-recovery.md)
5. [V6 hierarchical router](05-hierarchical-router.md)
6. [Evaluation](06-evaluation.md)
7. [Inference and POC](07-inference-and-poc.md)
8. [End-to-end runbook](08-end-to-end-runbook.md)
9. [Router architecture deep dive](09-conversation-router-v4.md)
10. [Who should decide the turn?](17-routing-classifier-comparison.md)
11. [File map](reference/file-map.md)

The router documentation retains two historical filenames for stable links;
the pages themselves describe V6 format 4 and its seven heads.

## Active Repository Map

| Path | Purpose |
| --- | --- |
| `data/banking-v5-tool-sft` | Governed Granite tool-use SFT corpus. |
| `data/banking-servicing-alignment-v5` | Governed Granite alignment source used to derive in-domain router examples. |
| `data/banking-conversation-router-v8-first-turn-mutation` | V6 train, validation, and test rows. |
| `data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json` | Source and prepared-split digests. |
| `artifacts/banking-conversation-router-v8-first-turn-mutation` | Format-4 model, heads, tokenizer, configuration, metrics, and manifest. |
| `src/hello_slm/banking_domain_taxonomy.py` | Canonical hierarchy and legal action/entity combinations. |
| `poc/retail-bank-customer-service-poc` | Shared Gradio/ZeroGPU and Streamlit runtime. |
| `scripts/retail_bank` | Data, training, evaluation, and deployment entry points. |

### Granite SFT dataset digests

Digests of the corpus **as it sits on disk**, which is the v12 corpus plus the
prompt-realization passes ([details](02-data-generation.md#prompt-realization)).
It is published at `@ce0d4429`, and the digests below are the ones that
revision carries. Two older revisions to keep distinct from it: `@8494c94f` is
the v12 corpus before the prompt passes, and `@b5ec0489` is what the
previously deployed v11 adapter was trained on. Regenerating this checkout reproduces `@ce0d4429`
only.

| File | Rows | SHA-256 |
| --- | ---: | --- |
| `data/banking-v5-tool-sft/train.jsonl` | 841 | `b723dabbe44b5148cd729f723ee236141f03e202bc60de013c8b263eee6aea6c` |
| `data/banking-v5-tool-sft/validation.jsonl` | 179 | `e7c7ca152a2376b0f95c5e4bd495db437a53a02d637258698b28060f2f062573` |
| `data/banking-servicing-alignment-v5/train.jsonl` | 3,959 | `a1f6f3f4a0c5da106bc049ba8660c22e235a48efd9a206080f2dc439d64d5d95` |
| `data/banking-servicing-alignment-v5/validation.jsonl` | 447 | `cab8b527c0124c7290e53a4192fb504bcbb63dfbcaef05baf90a58ffbcc4763f` |

## Safe Local Checks

```bash
python -m json.tool data/banking-conversation-router-v8-first-turn-mutation/manifest.json >/dev/null
python -m json.tool artifacts/banking-conversation-router-v8-first-turn-mutation/metrics.json >/dev/null

PYTHONPATH=src uv run pytest -q \
  tests/test_banking_conversation_router.py \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py \
  tests/test_banking_conversation_router_training.py

POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  uv run pytest -q poc/retail-bank-customer-service-poc/tests
```

Hub upload and Space deployment change external state. Use the runbook only
after recording the immutable dataset and router revisions.
