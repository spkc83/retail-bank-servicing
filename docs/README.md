# Harborlight Retail-Bank Agent Documentation

These documents describe the active V6 hierarchical conversation router and
the Granite PEFT servicing POC. A new developer can regenerate the governed
router data, train and evaluate the CPU model, inspect the orchestration
contract, and run either interface.

## Current Artifact Ledger

| Component | Identity/status |
| --- | --- |
| Hierarchical router | `spkc83/retail-bank-conversation-router@36920330d2502dfcf4d60572eadf1e3e71cd23fa` |
| Router dataset | `spkc83/retail-bank-conversation-router-data@2b8a8d92b2ab65b9f6bc7c7d1efbd3dc7c482029` |
| Local router artifact | `artifacts/banking-conversation-router-v6-hierarchical`; release eligible |
| Router dataset rows | train 16,693; validation 4,061; test 4,895 |
| Granite SFT dataset | `spkc83/retail-bank-servicing-alignment-sft@9d7aed545604bb42fb02b7a0919427a0ed2b81e2` |
| Granite PEFT adapter | `spkc83/retail-bank-servicing-agent-9b-peft@cc95e446af2b5e1d8d9df2751a8192613ad386e3` |
| Granite Stage-2 base | `spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| ZeroGPU Space source/pins | `spkc83/retail-bank-servicing-poc@2ec64ceacc390f5619d246fbca60dcca67f4a83f`; authenticated chat smoke passed |

## Read in This Order

1. [System overview](01-system-overview.md)
2. [Data generation](02-data-generation.md)
3. [Granite and PEFT](03-model-and-peft.md)
4. [Training and recovery](04-training-and-recovery.md)
5. [V6 hierarchical router](05-dual-head-router.md)
6. [Evaluation](06-evaluation.md)
7. [Inference and POC](07-inference-and-poc.md)
8. [End-to-end runbook](08-end-to-end-runbook.md)
9. [Router architecture deep dive](09-conversation-router-v4.md)
10. [File map](reference/file-map.md)

The router documentation retains two historical filenames for stable links;
the pages themselves describe V6 format 4 and its seven heads.

## Active Repository Map

| Path | Purpose |
| --- | --- |
| `data/banking-servicing-alignment-v5` | Governed Granite alignment source used to derive in-domain router examples. |
| `data/banking-conversation-router-v6-hierarchical` | V6 train, validation, and test rows. |
| `data/sources/banking-conversation-router-v6-hierarchical.lock.json` | Source and prepared-split digests. |
| `artifacts/banking-conversation-router-v6-hierarchical` | Format-4 model, heads, tokenizer, configuration, metrics, and manifest. |
| `src/hello_slm/banking_domain_taxonomy.py` | Canonical hierarchy and legal action/entity combinations. |
| `poc/retail-bank-customer-service-poc` | Shared Gradio/ZeroGPU and Streamlit runtime. |
| `scripts/retail_bank` | Data, training, evaluation, and deployment entry points. |

## Safe Local Checks

```bash
python -m json.tool data/banking-conversation-router-v6-hierarchical/manifest.json >/dev/null
python -m json.tool artifacts/banking-conversation-router-v6-hierarchical/metrics.json >/dev/null

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
