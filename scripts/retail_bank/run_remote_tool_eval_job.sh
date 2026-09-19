#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 SOURCE_COMMIT MODEL_REVISION DATASET_REVISION [fp16|bf16]" >&2
  exit 2
fi

source_commit="$1"
model_revision="$2"
dataset_revision="$3"
dtype="${4:-bf16}"
model_repo="${MODEL_REPO:-spkc83/retail-bank-servicing-agent-9b-peft}"
dataset_repo="${DATASET_REPO:-spkc83/retail-bank-servicing-alignment-sft}"
base_model_repo="${BASE_MODEL_REPO:-spkc83/retail-bank-servicing-agent-9b}"
base_model_revision="${BASE_MODEL_REVISION:-1d56824995aa1adecfe20f62ca42fb1c0c443817}"
adapter_repo="${ADAPTER_REPO:-spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation}"
adapter_revision="${ADAPTER_REVISION:-badbc05ad1f861818ea244b462eda49bca6c6fca}"
merged_model_only="${MERGED_MODEL_ONLY:-0}"
job_timeout="${JOB_TIMEOUT:-2h}"
gpu_hourly_usd="${GPU_HOURLY_USD:-2.75}"
max_job_cost_usd="${MAX_JOB_COST_USD:-5.50}"
confirm_spend="${CONFIRM_SPEND:-}"
dry_run="${DRY_RUN:-}"
if [[ "$merged_model_only" == "1" ]]; then
  base_model_repo=""
  base_model_revision=""
  adapter_repo=""
  adapter_revision=""
elif [[ "$merged_model_only" != "0" ]]; then
  echo "MERGED_MODEL_ONLY must be 0 or 1" >&2
  exit 2
fi
script_url="https://raw.githubusercontent.com/spkc83/retail-bank-servicing/${source_commit}/scripts/retail_bank/hf_job_tool_eval.py"

for revision_name in source_commit model_revision dataset_revision; do
  revision_value="${!revision_name}"
  if [[ ! "$revision_value" =~ ^[0-9a-f]{40}$ ]]; then
    echo "${revision_name} must be an exact 40-character lowercase Git commit." >&2
    exit 2
  fi
done

peft_values=("$base_model_repo" "$base_model_revision" "$adapter_repo" "$adapter_revision")
peft_count=0
for value in "${peft_values[@]}"; do
  [[ -n "$value" ]] && peft_count=$((peft_count + 1))
done
if [[ "$peft_count" -ne 0 && "$peft_count" -ne 4 ]]; then
  echo "PEFT evaluation requires BASE_MODEL_REPO, BASE_MODEL_REVISION, ADAPTER_REPO, and ADAPTER_REVISION." >&2
  exit 2
fi
if [[ "$peft_count" -eq 4 ]]; then
  for revision_name in base_model_revision adapter_revision; do
    revision_value="${!revision_name}"
    if [[ ! "$revision_value" =~ ^[0-9a-f]{40}$ ]]; then
      echo "${revision_name} must be an exact 40-character lowercase Git commit." >&2
      exit 2
    fi
  done
fi

if [[ "$dtype" != "fp16" && "$dtype" != "bf16" ]]; then
  echo "dtype must be fp16 or bf16" >&2
  exit 2
fi

if [[ ! "$job_timeout" =~ ^[0-9]+[smh]$ ]]; then
  echo "JOB_TIMEOUT must be a whole number followed by s, m, or h." >&2
  exit 2
fi
for price_name in gpu_hourly_usd max_job_cost_usd; do
  if [[ ! "${!price_name}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "${price_name^^} must be a non-negative decimal number of US dollars." >&2
    exit 2
  fi
done

# Price the worst case before anything is billed, as the training launcher does.
timeout_hours="$(python3 -c '
import sys
raw = sys.argv[1]
value, unit = int(raw[:-1]), raw[-1]
print({"s": value / 3600, "m": value / 60, "h": float(value)}[unit])
' "$job_timeout")"
worst_case_usd="$(python3 -c '
import sys
print(f"{float(sys.argv[1]) * float(sys.argv[2]):.2f}")
' "$timeout_hours" "$gpu_hourly_usd")"

echo "Billed job: rtx-pro-6000 at \$${gpu_hourly_usd}/h, timeout ${job_timeout}" >&2
echo "Worst case if it runs to the timeout: \$${worst_case_usd} (ceiling \$${max_job_cost_usd})" >&2

if python3 -c '
import sys
sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
' "$worst_case_usd" "$max_job_cost_usd"; then
  echo "Refusing to launch: worst case \$${worst_case_usd} exceeds MAX_JOB_COST_USD (\$${max_job_cost_usd})." >&2
  echo "Lower JOB_TIMEOUT, or raise the ceiling deliberately." >&2
  exit 2
fi

if [[ "$dry_run" != "1" && "$confirm_spend" != "1" ]]; then
  echo "Refusing to launch: set CONFIRM_SPEND=1 to authorise \$${worst_case_usd} of GPU time." >&2
  exit 2
fi

if ! curl --fail --silent --head "$script_url" >/dev/null 2>&1; then
  echo "Could not resolve bootstrap script: ${script_url}" >&2
  exit 2
fi

peft_args=()
if [[ "$peft_count" -eq 4 ]]; then
  peft_args=(
    --base-model-repo "$base_model_repo"
    --base-model-revision "$base_model_revision"
    --adapter-repo "$adapter_repo"
    --adapter-revision "$adapter_revision"
  )
fi

job_args=(
  --flavor rtx-pro-6000
  --timeout "$job_timeout"
  --secrets HF_TOKEN
  --volume hf://buckets/spkc83/jobs-artifacts:/data
  --label project=retail-bank-agent-v5-eval
  --label model="${model_revision:0:8}"
  "$script_url"
  --source-commit "$source_commit"
  --model-repo "$model_repo"
  --model-revision "$model_revision"
  --dataset-repo "$dataset_repo"
  --dataset-revision "$dataset_revision"
  --evaluation-targets test granite-v7-shadow screenshot-regression
  --dtype "$dtype"
  "${peft_args[@]}"
  --output-dir "/data/retail-bank-agent-eval-${model_revision:0:8}-${dataset_revision:0:8}"
)

if [[ "$dry_run" == "1" ]]; then
  echo "DRY_RUN=1: validated and priced; not submitting. Would run:" >&2
  printf '  hf jobs uv run' >&2
  printf ' %q' "${job_args[@]}" >&2
  printf '\n' >&2
  exit 0
fi

hf jobs uv run "${job_args[@]}"
