# End-to-End V6 Router Runbook

This runbook reproduces, publishes, and deploys the active hierarchical router.
Stop at the first failed prerequisite or release gate.

## 0. Record Immutable Inputs

```text
Router data:
  spkc83/retail-bank-conversation-router-data
  b33c27170e27cdb11783704ede14f7d25f70625e

Router base:
  distilbert/distilbert-base-uncased
  12040accade4e8a0f71eabdb258fecc2e7e948be

Router release:
  spkc83/retail-bank-conversation-router
  dd5ea26674a0f9808d42110a9ee51a9af6762a76

Granite PEFT (deployed):
  spkc83/retail-bank-servicing-agent-9b-peft-v14-prompt-realized
  47968b2b9ce02973b5676e464aafaa768cdbb05e
  adapter subfolder: adapter

Granite PEFT (last evaluated):
  spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation
  badbc05ad1f861818ea244b462eda49bca6c6fca

Granite Stage-2 base:
  spkc83/retail-bank-servicing-agent-9b
  1d56824995aa1adecfe20f62ca42fb1c0c443817
```

This is the immutable router commit returned by publication.

## 1. Install and Validate

`make verify` runs the whole gate in one command: lockfile, ruff, both test suites, and
`check_corpora_reproduce.py`, which re-runs the regeneration commands documented in
[Data generation](02-data-generation.md) and compares every split by SHA-256. There is no
hosted CI, so this is the enforcement — run it before pushing, not after.

```bash
uv sync --extra dev --extra scale
uv lock --check
uv run ruff check .

PYTHONPATH=src uv run pytest -q \
  tests/test_banking_conversation_router.py \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py \
  tests/test_banking_conversation_router_training.py

POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
  uv run pytest -q poc/retail-bank-customer-service-poc/tests
```

Stop if taxonomy, dataset, joint decoder, artifact, harness, or POC tests fail.

## 2. Generate the Dataset

```bash
PYTHONPATH=src uv run python scripts/retail_bank/prepare_conversation_router_data.py \
  --sft-dir data/banking-servicing-alignment-v5 \
  --output-dir data/banking-conversation-router-v8-first-turn-mutation \
  --source-lock data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json \
  --expected-release-lock data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json
```

Validate the JSON and expected counts:

```bash
python -m json.tool data/banking-conversation-router-v8-first-turn-mutation/manifest.json >/dev/null
python -m json.tool data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json >/dev/null

python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("data/banking-conversation-router-v8-first-turn-mutation/manifest.json").read_text()
)
actual = manifest["report"]["split_counts"]
expected = {"train": 20439, "validation": 4158, "test": 4921}
assert actual == expected, (actual, expected)
assert manifest["report"]["pii_matches"] == 0
assert all(value == 0 for value in manifest["report"]["leakage"].values() if isinstance(value, int))
print(actual)
PY
```

Stop if counts, digests, leakage checks, or PII checks differ.

> **The committed router corpus is a frozen release artifact.** It is pinned to
> the deployed router and derives from the alignment corpus, which has changed
> since it was built, so at HEAD `--expected-release-lock` reports a train-split
> digest drift. That is expected: `check_corpora_reproduce.py` lists this
> corpus under `FROZEN_RELEASE_ARTIFACTS` and enforces reproducibility for the
> base and alignment corpora only. Do not regenerate it into the tracked
> directory; see section 4 for why a rebuilt router is not releasable.

## 3. Confirm the Published Dataset

The training identity is already pinned:

```text
spkc83/retail-bank-conversation-router-data
b33c27170e27cdb11783704ede14f7d25f70625e
```

For a future dataset, upload its directory and capture the returned commit:

```bash
hf upload spkc83/retail-bank-conversation-router-data \
  data/banking-conversation-router-v8-first-turn-mutation . \
  --type dataset \
  --commit-message "Publish V6 hierarchical router data"
```

Do not train against `main`; use the immutable commit.

## 4. Train Locally

```bash
PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v8-first-turn-mutation \
  --output-dir artifacts/banking-conversation-router-v8-first-turn-mutation
```

Check the gate:

```bash
python - <<'PY'
import json
from pathlib import Path

metrics = json.loads(
    Path("artifacts/banking-conversation-router-v8-first-turn-mutation/metrics.json").read_text()
)
assert metrics["release_eligible"] is True
assert metrics["release_gate_failures"] == []
assert metrics["test"]["hierarchy_compatibility_error_rate"] == 0.0
print({
    "selected_epoch": metrics["selected_epoch"],
    "intent_macro_f1": metrics["test"]["intent_macro_f1"],
    "action_macro_f1": metrics["test"]["action_macro_f1"],
    "entity_resolution_macro_f1": metrics["test"]["entity_resolution_macro_f1"],
})
PY
```

The release selected epoch 2 and passed every gate.

> **A rebuild at HEAD is release eligible, but it is not the shipped router.**
> Retraining from the shipped corpus reproduces the shipped metrics exactly: the
> trainer is deterministic on one GPU at seed 7401. Retraining from the corpus HEAD
> derived before 2026-09-03 failed five gates, and the cause was not the deictic
> families (they only compound it). It was terminal punctuation acting as the
> domain label. Teacher-realized banking prompts end in `?` or `.`, CLINC
> out-of-domain lines are bare, and the encoder is uncased, so on first turns the
> mark predicted the domain almost perfectly (1,470 punctuated banking rows against
> 24 bare; 4,043 bare OOD rows against 31 punctuated). A router trained on that
> corpus scored "Could you mark Bright Meadow Electronics for dispute" at 0.01
> banking and the same words with a `?` at 1.00, and lost both held-out repair
> fixture turns, which are bare, for the same reason, stably across three seeds. The
> shipped router escaped only because 129 template-mangled, unpunctuated banking
> prompts ("Can you what information is needed for a card dispute") were still in
> its train split; the frozen test splits carry 31 of them to this day.
>
> Three derivation changes fix it, all in
> [`banking_conversation_router_data.py`](../src/hello_slm/banking_conversation_router_data.py)
> and described in [Data generation](02-data-generation.md#derivation-guards):
> the retired-realizer shape is filtered from every router split (28 records, 84
> derived test rows; the alignment fixtures stay byte-identical), a first-turn
> phrasing family adds the plain first ask in question, modal and greeting-led
> form for every servicing intent (+498 train, +99 validation), and a surface-form
> pass rewrites a fixed share of train and validation rows into the other
> punctuation form so the mark carries no signal. The result is
> `data/banking-conversation-router-v9-surface-form`. A router trained on it at
> seed 7401 clears every gate: repair and held-out regression at 0.0, in-domain
> false refusal 0.0024, OOD false accept 0.0145 (shipped: 0.0061), intent macro-F1
> 0.994 (shipped: 0.997). It is a local candidate. The shipped router stays pinned
> and deployed until the candidate is published deliberately through section 5.
>
> Excluding the deictic curricula is still **not** a fix: it drops train to 16,483
> and `load_governed_data` then refuses outright, because the counterfactual
> action/entity pairs every split requires come from that curriculum.

## 5. Publish the Router

Publication requires the exact dataset revision:

```bash
ROUTER_DATA_REVISION=b33c27170e27cdb11783704ede14f7d25f70625e

PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v8-first-turn-mutation \
  --output-dir artifacts/banking-conversation-router-v8-first-turn-mutation \
  --publish \
  --data-revision "$ROUTER_DATA_REVISION" \
  --destination-id spkc83/retail-bank-conversation-router
```

Capture the returned commit. For this release it is
`dd5ea26674a0f9808d42110a9ee51a9af6762a76`. Re-read the published
`router_config.json` and confirm `format_version: 4` plus the exact data
revision.

## 6. Local Behavioral Smoke

```bash
export RETAIL_BANK_ROUTER_ID=spkc83/retail-bank-conversation-router
export RETAIL_BANK_ROUTER_REVISION=dd5ea26674a0f9808d42110a9ee51a9af6762a76
uv run scripts/retail_bank/run_local_streamlit.py
```

Exercise at least:

1. direct account listing -> `execute_tool` / `list_accounts`;
2. ambiguous card replacement -> `clarify`, no tool;
3. policy detour and resume -> `retrieve_policy`, then original servicing task;
4. explicit intent change -> new intent replaces stale state;
5. weather question -> `refuse_ood`, zero Granite passes;
6. social greeting -> `converse`, Granite pass, no tools.

Inspect diagnostics for the seven-head tuple, joint-decoder constraint notes,
exposed tool schema, model passes, and immutable revisions.

## 7. Plan and Execute ZeroGPU Deployment

```bash
ADAPTER_REVISION=47968b2b9ce02973b5676e464aafaa768cdbb05e
ROUTER_REVISION=dd5ea26674a0f9808d42110a9ee51a9af6762a76

PYTHONPATH=src uv run python scripts/retail_bank/deploy_zero_gpu_space.py \
  --space-id spkc83/retail-bank-servicing-poc \
  --model-id spkc83/retail-bank-servicing-agent-9b-peft-v14-prompt-realized \
  --model-revision "$ADAPTER_REVISION" \
  --base-model-id spkc83/retail-bank-servicing-agent-9b \
  --base-model-revision 1d56824995aa1adecfe20f62ca42fb1c0c443817 \
  --adapter-id spkc83/retail-bank-servicing-agent-9b-peft-v14-prompt-realized \
  --adapter-revision "$ADAPTER_REVISION" \
  --adapter-subfolder adapter \
  --model-dtype bf16 \
  --router-id spkc83/retail-bank-conversation-router \
  --router-revision "$ROUTER_REVISION" \
  --best-of-n 2
```

Review the plan, then repeat with `--execute --allow-publish`. The current
Space source commit is
`2a6501b6d5029d1e1991f7444c9f352eef31b000`; the pins it carries are in the
[artifact ledger](reference/artifacts.md). The Space runs on `zero-a10g` and reports `SLEEPING`
between requests — the normal ZeroGPU idle state, not an outage.
Authenticated chat smoke on ZeroGPU is pending -- the credentials are held in the
`DEMO_AUTH_JSON` Space secret and are not readable through the API.

## Stop Conditions

Do not publish or deploy when:

- the data revision is mutable or differs from the artifact configuration;
- any split digest, leakage, or PII gate fails;
- `release_eligible` is false or failures are non-empty;
- hierarchy compatibility is nonzero;
- the runtime exposes more than the single intent-compatible tool on an
  executable turn;
- diagnostics do not identify the exact router/base/adapter composition.
