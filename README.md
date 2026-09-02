# Harborlight Retail-Bank Servicing Agent

This repository implements a model-driven retail-bank customer-service POC:

- IBM Granite 8.79B adapted with PEFT/LoRA for banking conversation and tool use;
- a CPU DistilBERT cross-encoder with seven routing heads;
- a constrained joint decoder that emits a coherent domain-to-action decision;
- bounded dialogue state for a pending servicing task and policy detours;
- policy retrieval and nine banking actions over fictional customer data;
- Gradio/ZeroGPU and local Streamlit interfaces branded as **Harborlight Bank**.

All customers, accounts, cards, transfers, transactions, and cases are
fictional. The POC does not connect to a real bank.

## Active Router Release

| Component | Identity |
| --- | --- |
| Router | `spkc83/retail-bank-conversation-router@dd5ea26674a0f9808d42110a9ee51a9af6762a76` |
| Router data | `spkc83/retail-bank-conversation-router-data@b33c27170e27cdb11783704ede14f7d25f70625e` |
| Local artifact | `artifacts/banking-conversation-router-v8-first-turn-mutation` |
| Local data | `data/banking-conversation-router-v8-first-turn-mutation` |
| Base encoder | `distilbert/distilbert-base-uncased@12040accade4e8a0f71eabdb258fecc2e7e948be` |
| Artifact format | 4 |
| Release gate | `release_eligible: true`; no failures |
| ZeroGPU Space source/pins | `spkc83/retail-bank-servicing-poc@a227df8d40934e6d3c1be31d49a49c4f20dcc81d`; runtime running |
| Deployed Granite adapter | `spkc83/retail-bank-servicing-agent-9b-peft-v11-alignment@03a7b44633fadab7ad672b009925cc68b52494d4` (subfolder `adapter`) |

The Space is running with the exact base, adapter, and router revisions
above. Authenticated chat smoke is pending.

## Request Flow

```text
current turn + up to 3 visible exchanges + trusted prior dialogue state
  -> V6 hierarchical CPU router
     -> domain -> lane -> family -> intent
     -> conversation relations
     -> action disposition + entity-resolution state
     -> constrained joint decoder selects one legal tuple
  -> out_of_domain: fixed scope response; Granite is not called
  -> uncertain: Granite receives no tools and asks a clarification
  -> retrieve_policy: retrieve governed policy; Granite receives no tools
  -> execute_tool: expose the single tool schema mapped from the intent
     -> Granite must call it exactly once and supplies all arguments
     -> one bounded retry if Granite emits prose instead of the required call
     -> execute against fictional session state
     -> remove tools for the grounded-final pass
     -> render exact read tables or validate the grounded response
  -> clarify/converse: Granite receives no tools and follows turn guidance
```

The classifier does not generate tool arguments and does not execute a tool.
For an accepted `execute_tool` decision, the harness exposes exactly one
intent-compatible schema and requires exactly one call before execution.
Granite chooses the arguments from conversation context; unresolved targets
are routed to `clarify` without exposing a tool.

## Seven-Head Taxonomy

| Head | Examples | Runtime purpose |
| --- | --- | --- |
| Domain | `out_of_domain`, `banking`, `social` | Scope boundary |
| Lane | `servicing`, `policy`, `conversation` | Orchestration path |
| Family | `cards`, `transactions`, `service_cases` | Product grouping |
| Intent | `replace_card`, `view_transactions`, `policy_knowledge` | Fine request |
| Relation | `context_dependent`, `agent_repair`, `topic_shift`, ... | Multi-turn meaning |
| Action | `execute_tool`, `clarify`, `retrieve_policy`, `converse`, `refuse_ood` | Generation disposition |
| Entity resolution | `resolved`, `missing`, `ambiguous`, `ineligible`, `not_required` | Whether an action target is usable |

The heads share one encoder. The joint decoder enumerates legal taxonomy
tuples and prevents incompatible independent head predictions from reaching
the harness.

## Documentation

Read these in order:

1. [System overview](docs/01-system-overview.md)
2. [Data generation](docs/02-data-generation.md)
3. [Granite and PEFT](docs/03-model-and-peft.md)
4. [Training and recovery](docs/04-training-and-recovery.md)
5. [Hierarchical router](docs/05-hierarchical-router.md)
6. [Evaluation](docs/06-evaluation.md)
7. [Inference and POC](docs/07-inference-and-poc.md)
8. [End-to-end runbook](docs/08-end-to-end-runbook.md)
9. [Router architecture deep dive](docs/09-conversation-router-v4.md)
10. [Who should decide the turn?](docs/17-routing-classifier-comparison.md)
11. [Use-case coverage](docs/18-corpus-usecase-coverage.md)
12. [File map](docs/reference/file-map.md)

The filenames `05-hierarchical-router.md` and `09-conversation-router-v4.md` are
retained for stable links. Their content documents the active V6 router.

## Reproduce the Router

```bash
uv sync --extra dev --extra scale

PYTHONPATH=src uv run python scripts/retail_bank/prepare_conversation_router_data.py \
  --sft-dir data/banking-servicing-alignment-v5 \
  --output-dir data/banking-conversation-router-v8-first-turn-mutation \
  --source-lock data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json \
  --expected-release-lock data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json

PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v8-first-turn-mutation \
  --output-dir artifacts/banking-conversation-router-v8-first-turn-mutation
```

Expected split counts are 20,439 training, 4,158 validation, and 4,921 test
rows. Training must produce `release_eligible: true` and an empty
`release_gate_failures` list in `metrics.json`.

## Run and Verify

```bash
uv run scripts/retail_bank/run_local_streamlit.py
```

```bash
PYTHONPATH=src uv run pytest -q \
  tests/test_banking_conversation_router.py \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py \
  tests/test_banking_conversation_router_training.py

POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  uv run pytest -q poc/retail-bank-customer-service-poc/tests
```

See the [POC README](poc/retail-bank-customer-service-poc/README.md) for local
credentials, environment overrides, ZeroGPU deployment, and diagnostics.

## License

Repository code and self-authored data are MIT licensed. The router also uses
the checksum-pinned CLINC150 source recorded in the source lock; review its
upstream terms before redistribution.
