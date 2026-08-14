#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 SOURCE_COMMIT MODEL_REVISION DATASET_REVISION [fp16|bf16]" >&2
  exit 2
fi

source_commit="$1"
model_revision="$2"
dataset_revision="$3"
dtype="${4:-fp16}"
model_repo="${MODEL_REPO:-spkc83/retail-bank-servicing-agent-9b}"
dataset_repo="${DATASET_REPO:-spkc83/retail-bank-servicing-alignment-sft}"
script_url="https://raw.githubusercontent.com/spkc83/retail-bank-servicing/${source_commit}/scripts/retail_bank/hf_job_tool_eval.py"

for revision_name in source_commit model_revision dataset_revision; do
  revision_value="${!revision_name}"
  if [[ ! "$revision_value" =~ ^[0-9a-f]{40}$ ]]; then
    echo "${revision_name} must be an exact 40-character lowercase Git commit." >&2
    exit 2
  fi
done

if [[ "$dtype" != "fp16" && "$dtype" != "bf16" ]]; then
  echo "dtype must be fp16 or bf16" >&2
  exit 2
fi

if ! curl --fail --silent --head "$script_url" >/dev/null 2>&1; then
  echo "Could not resolve bootstrap script: ${script_url}" >&2
  exit 2
fi

hf jobs uv run \
  --flavor rtx-pro-6000 \
  --timeout 2h \
  --secrets HF_TOKEN \
  --volume hf://buckets/spkc83/jobs-artifacts:/data \
  --label project=retail-bank-agent-v5-eval \
  --label model="${model_revision:0:8}" \
  "$script_url" \
  --source-commit "$source_commit" \
  --model-repo "$model_repo" \
  --model-revision "$model_revision" \
  --dataset-repo "$dataset_repo" \
  --dataset-revision "$dataset_revision" \
  --dtype "$dtype" \
  --output-dir "/data/retail-bank-agent-eval-${model_revision:0:8}-${dataset_revision:0:8}"
