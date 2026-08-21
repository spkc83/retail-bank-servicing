#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "usage: $0 SOURCE_COMMIT DATASET_REVISION [SOURCE_ADAPTER_REVISION] [DESTINATION_REPO] [MAX_STEPS]" >&2
  echo "env: MAX_TRAIN_SECONDS caps wall-clock training time (default 3600)" >&2
  exit 2
fi

source_commit="$1"
dataset_revision="$2"
source_adapter_revision="${3:-d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2}"
destination_repo="${4:-spkc83/retail-bank-servicing-agent-9b-peft-v6-generation-contract}"
max_steps="${5:-964}"
source_adapter_repo="${SOURCE_ADAPTER_REPO:-spkc83/retail-bank-servicing-agent-9b-peft-v5-remediation}"
max_train_seconds="${MAX_TRAIN_SECONDS:-3600}"
probe_only="${PROBE_ONLY:-0}"
publish_only="${PUBLISH_ONLY:-0}"
probe_checkpoint_dir="${PROBE_CHECKPOINT_DIR:-/data/retail-bank-agent-9b-continuation-34484bb0-d965816b-715064e5/trainer/checkpoint-600}"
probe_checkpoint_step="${PROBE_CHECKPOINT_STEP:-600}"
script_url="https://raw.githubusercontent.com/spkc83/retail-bank-servicing/${source_commit}/scripts/retail_bank/hf_job_continue_tool_sft.py"
worker_url="https://raw.githubusercontent.com/spkc83/retail-bank-servicing/${source_commit}/scripts/retail_bank/cloud_continue_tool_sft.py"
output_dir="/data/retail-bank-agent-9b-peft-v6-generation-contract-${source_commit:0:8}-${source_adapter_revision:0:8}-${dataset_revision:0:8}"

if [[ "$probe_only" == "1" ]]; then
  probe_name="${probe_checkpoint_dir##*/}"
  output_dir="/data/retail-bank-agent-9b-probe-${probe_name}-${source_commit:0:8}-${dataset_revision:0:8}"
fi

if [[ ! "$max_train_seconds" =~ ^[0-9]+$ ]]; then
  echo "MAX_TRAIN_SECONDS must be a whole number of seconds." >&2
  exit 2
fi

if [[ "$probe_only" == "1" && "$publish_only" == "1" ]]; then
  echo "PROBE_ONLY and PUBLISH_ONLY cannot both be enabled." >&2
  exit 2
fi

if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_COMMIT must be the exact 40-character lowercase Git commit." >&2
  exit 2
fi

if [[ ! "$dataset_revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "DATASET_REVISION must be the exact 40-character lowercase Git commit." >&2
  exit 2
fi

if [[ "$probe_only" == "1" && "$dataset_revision" != "715064e50e7ed2f815dfd3ce19b61f345a466b9d" ]]; then
  echo "PROBE_ONLY requires candidate3 dataset revision 715064e50e7ed2f815dfd3ce19b61f345a466b9d." >&2
  exit 2
fi

if [[ ! "$source_adapter_revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_ADAPTER_REVISION must be the exact 40-character lowercase Git commit." >&2
  exit 2
fi

if [[ "$source_adapter_repo" != "spkc83/retail-bank-servicing-agent-9b-peft-v5-remediation" || "$source_adapter_revision" != "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2" ]]; then
  echo "Source adapter must be exactly spkc83/retail-bank-servicing-agent-9b-peft-v5-remediation@d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2." >&2
  exit 2
fi

if [[ "$destination_repo" == "$source_adapter_repo" ]]; then
  echo "DESTINATION_REPO must differ from the source adapter repository." >&2
  exit 2
fi

if [[ ! "$destination_repo" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
  echo "DESTINATION_REPO must be a Hugging Face repository id in owner/name form." >&2
  exit 2
fi

if [[ ! "$source_adapter_repo" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
  echo "SOURCE_ADAPTER_REPO must be a Hugging Face repository id in owner/name form." >&2
  exit 2
fi

if ! bootstrap_source=$(curl --fail --silent "$script_url"); then
  echo "Could not resolve bootstrap script: ${script_url}" >&2
  exit 2
fi

if ! worker_source=$(curl --fail --silent "$worker_url"); then
  echo "Could not resolve V6 continuation worker: ${worker_url}" >&2
  exit 2
fi

protocol_marker='V6_CONTINUATION_PROTOCOL = "retail-bank-peft-v6-generation-contract/v1"'
if [[ "$bootstrap_source" != *"$protocol_marker"* || "$worker_source" != *"$protocol_marker"* ]]; then
  echo "SOURCE_COMMIT does not implement the required V6 continuation protocol." >&2
  exit 2
fi

job_flavor="rtx-pro-6000"
job_timeout="5h"
if [[ "$publish_only" == "1" ]]; then
  job_flavor="cpu-basic"
  job_timeout="30m"
fi

job_args=(
  --flavor "$job_flavor"
  --timeout "$job_timeout"
  --secrets HF_TOKEN
  --volume hf://buckets/spkc83/jobs-artifacts:/data
  --label project=retail-bank-agent-v6-generation-contract
  --label source="${source_commit:0:8}"
  --label parent_adapter="${source_adapter_revision:0:8}"
  "$script_url"
  --source-commit "$source_commit"
  --dataset-revision "$dataset_revision"
  --source-adapter-repo "$source_adapter_repo"
  --source-adapter-revision "$source_adapter_revision"
  --destination-repo "$destination_repo"
  --output-dir "$output_dir"
  --max-steps "$max_steps"
  --max-train-seconds "$max_train_seconds"
)

if [[ "$probe_only" == "1" ]]; then
  job_args+=(
    --probe-only
    --probe-checkpoint-dir "$probe_checkpoint_dir"
    --probe-checkpoint-step "$probe_checkpoint_step"
  )
fi

if [[ "$publish_only" == "1" ]]; then
  job_args+=(--publish-only)
fi

hf jobs uv run "${job_args[@]}"
