#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "usage: $0 RECOVERY_SOURCE_COMMIT [DESTINATION_REPO] [MAX_STEPS] [CHECKPOINT_STEP]" >&2
  exit 2
fi

recovery_source_commit="$1"
destination_repo="${2:-spkc83/retail-bank-servicing-agent-9b-peft-v7-grounded-generation}"
max_steps="${3:-64}"
checkpoint_step="${4:-964}"
training_source_commit="0bb83262981e16fa85ad3a411ecd1be6ddbf0b88"
dataset_revision="7b58aa748d69bf56d9b32eb65989b8c07e04f835"
failed_job="6a813914c97db76cbdf32acc"
failed_output_root="/data/retail-bank-agent-9b-peft-v6-generation-contract-0bb83262-d965816b-7b58aa74"
checkpoint_dir="${failed_output_root}/trainer/checkpoint-${checkpoint_step}"
output_dir="/data/retail-bank-agent-9b-peft-v7-checkpoint-correction-${recovery_source_commit:0:8}-${dataset_revision:0:8}-step${checkpoint_step}"
script_url="https://raw.githubusercontent.com/spkc83/retail-bank-servicing/${recovery_source_commit}/scripts/retail_bank/hf_job_correct_checkpoint_tool_sft.py"
worker_url="https://raw.githubusercontent.com/spkc83/retail-bank-servicing/${recovery_source_commit}/scripts/retail_bank/cloud_continue_tool_sft.py"

if [[ ! "$recovery_source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "RECOVERY_SOURCE_COMMIT must be an exact 40-character lowercase Git commit." >&2
  exit 2
fi

if [[ ! "$destination_repo" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
  echo "DESTINATION_REPO must be a Hugging Face repository id in owner/name form." >&2
  exit 2
fi

if [[ "$destination_repo" == "spkc83/retail-bank-servicing-agent-9b-peft-v5-remediation" ]]; then
  echo "DESTINATION_REPO must differ from the parent adapter repository." >&2
  exit 2
fi

if [[ ! "$max_steps" =~ ^[1-9][0-9]*$ || ! "$checkpoint_step" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_STEPS and CHECKPOINT_STEP must be positive integers." >&2
  exit 2
fi

if ! bootstrap_source=$(curl --fail --silent "$script_url"); then
  echo "Could not resolve checkpoint-correction bootstrap: ${script_url}" >&2
  exit 2
fi

if ! worker_source=$(curl --fail --silent "$worker_url"); then
  echo "Could not resolve checkpoint-correction worker: ${worker_url}" >&2
  exit 2
fi

protocol_marker='V7_CHECKPOINT_CORRECTION_PROTOCOL = "retail-bank-peft-v7-checkpoint-correction/v1"'
bootstrap_marker='CORRECTION_PROTOCOL = "retail-bank-peft-v7-checkpoint-correction/v1"'
if [[ "$worker_source" != *"$protocol_marker"* || "$bootstrap_source" != *"$bootstrap_marker"* ]]; then
  echo "RECOVERY_SOURCE_COMMIT does not implement the V7 checkpoint correction protocol." >&2
  exit 2
fi

job_args=(
  --flavor rtx-pro-6000
  --timeout 5h
  --secrets HF_TOKEN
  --volume hf://buckets/spkc83/jobs-artifacts:/data
  --label project=retail-bank-agent-v7-checkpoint-correction
  --label source="${recovery_source_commit:0:8}"
  --label training_source="${training_source_commit:0:8}"
  --label source_job="$failed_job"
  "$script_url"
  --recovery-source-commit "$recovery_source_commit"
  --checkpoint-dir "$checkpoint_dir"
  --checkpoint-step "$checkpoint_step"
  --destination-repo "$destination_repo"
  --output-dir "$output_dir"
  --max-steps "$max_steps"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'hf jobs uv run'
  printf ' %q' "${job_args[@]}"
  printf '\n'
  exit 0
fi

hf jobs uv run "${job_args[@]}"
