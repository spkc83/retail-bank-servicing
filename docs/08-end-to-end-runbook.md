# End-to-End V6 Router Runbook

This runbook reproduces, publishes, and deploys the active hierarchical router.
Stop at the first failed prerequisite or release gate.

## 0. Record Immutable Inputs

```text
Router data:
  spkc83/retail-bank-conversation-router-data
  80c0edfea84b341d2ee4092f5c4a4bbb05405e40

Router base:
  distilbert/distilbert-base-uncased
  12040accade4e8a0f71eabdb258fecc2e7e948be

Router release:
  spkc83/retail-bank-conversation-router
  7f6a0e77ad231233702039560ced007fdc68bd74

Granite PEFT:
  spkc83/retail-bank-servicing-agent-9b-peft
  cc95e446af2b5e1d8d9df2751a8192613ad386e3

Granite Stage-2 base:
  spkc83/retail-bank-servicing-agent-9b
  1d56824995aa1adecfe20f62ca42fb1c0c443817
```

This is the immutable router commit returned by publication.

## 1. Install and Validate

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
  --output-dir data/banking-conversation-router-v6-hierarchical \
  --source-lock data/sources/banking-conversation-router-v6-hierarchical.lock.json \
  --expected-release-lock data/sources/banking-conversation-router-v6-hierarchical.lock.json
```

Validate the JSON and expected counts:

```bash
python -m json.tool data/banking-conversation-router-v6-hierarchical/manifest.json >/dev/null
python -m json.tool data/sources/banking-conversation-router-v6-hierarchical.lock.json >/dev/null

python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("data/banking-conversation-router-v6-hierarchical/manifest.json").read_text()
)
actual = manifest["report"]["split_counts"]
expected = {"train": 16693, "validation": 4061, "test": 4895}
assert actual == expected, (actual, expected)
assert manifest["report"]["pii_matches"] == 0
assert all(value == 0 for value in manifest["report"]["leakage"].values() if isinstance(value, int))
print(actual)
PY
```

Stop if counts, digests, leakage checks, or PII checks differ.

## 3. Confirm the Published Dataset

The training identity is already pinned:

```text
spkc83/retail-bank-conversation-router-data
80c0edfea84b341d2ee4092f5c4a4bbb05405e40
```

For a future dataset, upload its directory and capture the returned commit:

```bash
hf upload spkc83/retail-bank-conversation-router-data \
  data/banking-conversation-router-v6-hierarchical . \
  --type dataset \
  --commit-message "Publish V6 hierarchical router data"
```

Do not train against `main`; use the immutable commit.

## 4. Train Locally

```bash
PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v6-hierarchical \
  --output-dir artifacts/banking-conversation-router-v6-hierarchical
```

Check the gate:

```bash
python - <<'PY'
import json
from pathlib import Path

metrics = json.loads(
    Path("artifacts/banking-conversation-router-v6-hierarchical/metrics.json").read_text()
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

## 5. Publish the Router

Publication requires the exact dataset revision:

```bash
ROUTER_DATA_REVISION=80c0edfea84b341d2ee4092f5c4a4bbb05405e40

PYTHONPATH=src uv run scripts/retail_bank/train_conversation_router.py \
  --dataset-dir data/banking-conversation-router-v6-hierarchical \
  --output-dir artifacts/banking-conversation-router-v6-hierarchical \
  --publish \
  --data-revision "$ROUTER_DATA_REVISION" \
  --destination-id spkc83/retail-bank-conversation-router
```

Capture the returned commit. For this release it is
`7f6a0e77ad231233702039560ced007fdc68bd74`. Re-read the published
`router_config.json` and confirm `format_version: 4` plus the exact data
revision.

## 6. Local Behavioral Smoke

```bash
export RETAIL_BANK_ROUTER_ID=spkc83/retail-bank-conversation-router
export RETAIL_BANK_ROUTER_REVISION=7f6a0e77ad231233702039560ced007fdc68bd74
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
ADAPTER_REVISION=cc95e446af2b5e1d8d9df2751a8192613ad386e3
ROUTER_REVISION=7f6a0e77ad231233702039560ced007fdc68bd74

PYTHONPATH=src uv run python scripts/retail_bank/deploy_zero_gpu_space.py \
  --space-id spkc83/retail-bank-servicing-poc \
  --model-id spkc83/retail-bank-servicing-agent-9b-peft \
  --model-revision "$ADAPTER_REVISION" \
  --base-model-id spkc83/retail-bank-servicing-agent-9b \
  --base-model-revision 1d56824995aa1adecfe20f62ca42fb1c0c443817 \
  --adapter-id spkc83/retail-bank-servicing-agent-9b-peft \
  --adapter-revision "$ADAPTER_REVISION" \
  --model-dtype bf16 \
  --router-id spkc83/retail-bank-conversation-router \
  --router-revision "$ROUTER_REVISION"
```

Review the plan, then repeat with `--execute --allow-publish`. The current
Space source/pin deployment is
`f018cad020a17e33be59992035c1418c4cf91a01`. The runtime remains PAUSED because
the current OAuth token receives HTTP 401 on restart. Do not claim READY or run
remote behavioral smokes until authentication is repaired and the Space starts.

## Stop Conditions

Do not publish or deploy when:

- the data revision is mutable or differs from the artifact configuration;
- any split digest, leakage, or PII gate fails;
- `release_eligible` is false or failures are non-empty;
- hierarchy compatibility is nonzero;
- the runtime exposes more than the single intent-compatible tool on an
  executable turn;
- diagnostics do not identify the exact router/base/adapter composition.
