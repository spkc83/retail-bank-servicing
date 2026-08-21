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
| Granite SFT dataset | `spkc83/retail-bank-servicing-alignment-sft@a78bed17db8c56099a32f835832b9878a895a602` |
| Granite PEFT adapter | `spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation@badbc05ad1f861818ea244b462eda49bca6c6fca` |
| Granite Stage-2 base | `spkc83/retail-bank-servicing-agent-9b@1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| ZeroGPU Space source/pins | `spkc83/retail-bank-servicing-poc@bab5b2237b22814ede6a76c6c5ac2a1354097d44`; authenticated chat smoke pending |

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
| `data/banking-v5-tool-sft` | Governed Granite tool-use SFT corpus. |
| `data/banking-servicing-alignment-v5` | Governed Granite alignment source used to derive in-domain router examples. |
| `data/banking-conversation-router-v8-first-turn-mutation` | V6 train, validation, and test rows. |
| `data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json` | Source and prepared-split digests. |
| `artifacts/banking-conversation-router-v8-first-turn-mutation` | Format-4 model, heads, tokenizer, configuration, metrics, and manifest. |
| `src/hello_slm/banking_domain_taxonomy.py` | Canonical hierarchy and legal action/entity combinations. |
| `poc/retail-bank-customer-service-poc` | Shared Gradio/ZeroGPU and Streamlit runtime. |
| `scripts/retail_bank` | Data, training, evaluation, and deployment entry points. |

### Granite SFT dataset digests

Current local digests after the conversational-voice regeneration
([details](02-data-generation.md#granite-sft-corpora)). This is a local
regeneration, not yet published; the Hub SFT dataset revision stays
`a78bed17db8c56099a32f835832b9878a895a602`.

| File | Rows | SHA-256 |
| --- | ---: | --- |
| `data/banking-v5-tool-sft/train.jsonl` | 841 | `d31a45ea3896158043e0838211d989573efaccbc6fd4b3b03e052bc7775999a1` |
| `data/banking-v5-tool-sft/validation.jsonl` | 179 | `f83810e5b51670bdff6f6ccbded37bae8eced2a065ee5db61ce64848012492aa` |
| `data/banking-servicing-alignment-v5/train.jsonl` | 3,043 | `68f3f859d624ac62e56597c32c962d7eea47e4c81113eb781de20475bbf7c432` |
| `data/banking-servicing-alignment-v5/validation.jsonl` | 397 | `7b61f14bcafcafe79499b46ced27e51ddec5a4b11957f63722be36a544265a0d` |

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
