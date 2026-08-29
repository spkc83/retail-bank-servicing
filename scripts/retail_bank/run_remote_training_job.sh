#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 SOURCE_COMMIT DATASET_REVISION [RESUME_FROM]" >&2
  echo "optional env overrides:"
  echo "  BASE_MODEL (default: spkc83/retail-bank-servicing-agent-9b)"
  echo "  BASE_REVISION (exact 40-char SHA)"
  echo "  BASE_FAMILY (default: granite)"
  echo "  CONFIRMATION_TOKEN (default: banking-v5-grounded-dialogue-sft)"
  echo "  DATASET_REPO (default: spkc83/retail-bank-servicing-alignment-sft)"
  echo "  HF_HUB_DEST (REQUIRED, must differ from BASE_MODEL)"
  echo "  JOB_TIMEOUT (default: 5h; whole number followed by s, m, or h)"
  echo "  SKIP_MERGE_ADAPTER (set to 1 to publish the adapter without merging)"
  echo "  POSITIVE_MULTIPLIER / AMBIGUITY_MULTIPLIER (mix weights, defaults 1 / 1)"
  echo "  POLICY_FAQ_MULTIPLIER / TOOL_OUTCOME_MULTIPLIER (mix weights, defaults 1 / 1)"
  echo "  MANIFEST_PATH (optional path, forwarded to hf_job_tool_sft.py)"
  echo "  OUTPUT_PREFIX (default: /data/retail-bank-agent-9b-\${SOURCE_COMMIT_PREFIX})"
  echo "  MAX_STEPS (default: 3000)"
  echo "  MAX_TRAIN_SECONDS (default: 14400)"
  echo "  GRADIENT_ACCUMULATION_STEPS (default: 2)"
  echo "  CHECKPOINT_EVERY (default: 500)"
  echo "  LEARNING_RATE (default: 1e-4)"
  echo "  TRAINING_SEED (default: 7303; recorded in the training fingerprint)"
  echo "  TRACKIO_PROJECT (default: retail-bank-agent-v5)"
  echo "  TRACKIO_RUN_NAME (default derived from source commit)"
  echo "  PROJECT_LABEL (default: retail-bank-agent-v5)"
  exit 2
fi

source_commit="$1"
dataset_revision="$2"
resume_from="${3:-}"

base_model="${BASE_MODEL:-spkc83/retail-bank-servicing-agent-9b}"
base_revision="${BASE_REVISION:-1d56824995aa1adecfe20f62ca42fb1c0c443817}"
base_family="${BASE_FAMILY:-granite}"
confirmation_token="${CONFIRMATION_TOKEN:-banking-v5-grounded-dialogue-sft}"
dataset_repo="${DATASET_REPO:-spkc83/retail-bank-servicing-alignment-sft}"
hub_dest="${HF_HUB_DEST:-}"
job_timeout="${JOB_TIMEOUT:-5h}"
skip_merge_adapter="${SKIP_MERGE_ADAPTER:-}"
positive_multiplier="${POSITIVE_MULTIPLIER:-1}"
ambiguity_multiplier="${AMBIGUITY_MULTIPLIER:-1}"
policy_faq_multiplier="${POLICY_FAQ_MULTIPLIER:-1}"
tool_outcome_multiplier="${TOOL_OUTCOME_MULTIPLIER:-1}"
manifest_path="${MANIFEST_PATH:-}"
output_prefix="${OUTPUT_PREFIX:-/data/retail-bank-agent-9b-${source_commit:0:8}}"
max_steps="${MAX_STEPS:-3000}"
max_train_seconds="${MAX_TRAIN_SECONDS:-14400}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"
checkpoint_every="${CHECKPOINT_EVERY:-500}"
learning_rate="${LEARNING_RATE:-1e-4}"
training_seed="${TRAINING_SEED:-7303}"
trackio_project="${TRACKIO_PROJECT:-retail-bank-agent-v5}"
trackio_run_name="${TRACKIO_RUN_NAME:-${base_family}-tool-sft-${source_commit:0:8}}"
project_label="${PROJECT_LABEL:-retail-bank-agent-v5}"

if [[ -z "$hub_dest" ]]; then
  echo "HF_HUB_DEST must be set explicitly: the from-scratch lane publishes a NEW adapter" >&2
  echo "repository and must never default to the training base." >&2
  exit 2
fi
if [[ "$hub_dest" == "$base_model" ]]; then
  echo "HF_HUB_DEST (${hub_dest}) must differ from BASE_MODEL (${base_model}); publishing" >&2
  echo "into the base repository would overwrite the weights this run trains from." >&2
  exit 2
fi

if [[ ! "$job_timeout" =~ ^[0-9]+[smh]$ ]]; then
  echo "JOB_TIMEOUT must be a whole number followed by s, m, or h." >&2
  exit 2
fi

if [[ ! "$positive_multiplier" =~ ^[1-9][0-9]?$ ]] \
  || [[ ! "$ambiguity_multiplier" =~ ^[1-9][0-9]?$ ]] \
  || [[ ! "$policy_faq_multiplier" =~ ^[1-9][0-9]?$ ]] \
  || [[ ! "$tool_outcome_multiplier" =~ ^[1-9][0-9]?$ ]]; then
  echo "POSITIVE_MULTIPLIER, AMBIGUITY_MULTIPLIER, POLICY_FAQ_MULTIPLIER, and" >&2
  echo "TOOL_OUTCOME_MULTIPLIER must be whole numbers from 1 to 99." >&2
  exit 2
fi

if [[ -n "$skip_merge_adapter" && "$skip_merge_adapter" != "0" && "$skip_merge_adapter" != "1" ]]; then
  echo "SKIP_MERGE_ADAPTER must be unset, 0, or 1." >&2
  exit 2
fi

if [[ ! "$training_seed" =~ ^[0-9]{1,9}$ ]]; then
  echo "TRAINING_SEED must be a whole number." >&2
  exit 2
fi

if [[ ! "$base_revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "BASE_REVISION must be the exact 40-character lowercase Git revision." >&2
  exit 2
fi

script_url="https://raw.githubusercontent.com/spkc83/retail-bank-servicing/${source_commit}/scripts/retail_bank/hf_job_tool_sft.py"

if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_COMMIT must be the exact 40-character lowercase Git commit." >&2
  exit 2
fi
if [[ ! "$dataset_revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "DATASET_REVISION must be the exact 40-character lowercase Git revision." >&2
  exit 2
fi

if ! curl --fail --silent --head "$script_url" >/dev/null 2>&1; then
  echo "Could not resolve bootstrap script: ${script_url}" >&2
  exit 2
fi

job_args=(
  --flavor rtx-pro-6000
  --timeout "$job_timeout"
  --secrets HF_TOKEN
  --volume hf://buckets/spkc83/jobs-artifacts:/data
  --label project="$project_label"
  --label source="${source_commit:0:8}"
  "$script_url"
  --source-commit "$source_commit"
  --dataset-revision "$dataset_revision"
  --dataset-repo "$dataset_repo"
  --output-dir "$output_prefix"
  --base-model "$base_model"
  --base-revision "$base_revision"
  --base-family "$base_family"
  --hub-dest "$hub_dest"
  --confirmation-token "$confirmation_token"
  --max-steps "$max_steps"
  --max-train-seconds "$max_train_seconds"
  --gradient-accumulation-steps "$gradient_accumulation_steps"
  --checkpoint-every "$checkpoint_every"
  --learning-rate "$learning_rate"
  --training-seed "$training_seed"
  --trackio-project "$trackio_project"
  --trackio-run-name "$trackio_run_name"
  --positive-multiplier "$positive_multiplier"
  --ambiguity-multiplier "$ambiguity_multiplier"
  --policy-faq-multiplier "$policy_faq_multiplier"
  --tool-outcome-multiplier "$tool_outcome_multiplier"
)
if [[ "$skip_merge_adapter" == "1" ]]; then
  job_args+=(--skip-merge-adapter)
fi
if [[ -n "$manifest_path" ]]; then
  job_args+=(--manifest "$manifest_path")
fi
if [[ -n "$resume_from" ]]; then
  job_args+=(--resume-from "$resume_from")
fi

hf jobs uv run "${job_args[@]}"
