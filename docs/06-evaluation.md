# Frozen Two-Phase Evaluation

This guide covers the active frozen evaluation for the Granite PEFT model.
Evaluation is read-only with respect to banking tools: it generates model
outputs, appends canonical replay-validated tool results from the dataset when
appropriate, scores the outputs, and publishes evaluation artifacts under the
model repository.

## Active Artifact IDs

| Artifact | Value | Owner |
| --- | --- | --- |
| Model repo | `spkc83/retail-bank-servicing-agent-9b` | [`scripts/retail_bank/hf_job_tool_eval.py`](../scripts/retail_bank/hf_job_tool_eval.py) |
| Model revision | `1d56824995aa1adecfe20f62ca42fb1c0c443817` | [`model card`](../model_cards/retail-bank-agent-9b.md) |
| Corrected dataset repo | `spkc83/retail-bank-servicing-alignment-sft` | [`data card`](../data_cards/retail-bank-servicing-alignment-sft.md) |
| Corrected dataset revision | `0ce32f9c7a3edff227005e5b89b089947b87625a` | [`data card`](../data_cards/retail-bank-servicing-alignment-sft.md) |
| Prompt-identical training revision | `fea8aa1cda716954eb7322325e2be25c9f570ea3` | [`data card`](../data_cards/retail-bank-servicing-alignment-sft.md) |
| Frozen split | `test`, 1,374 records | [`data/banking-servicing-alignment-v4/manifest.json`](../data/banking-servicing-alignment-v4/manifest.json) |
| Evaluation job | `spkc83/6a6caac1a00abefd4b289b14` | [`model card`](../model_cards/retail-bank-agent-9b.md) |
| Evaluation head | `214fc0d9e143e4fa7b658de1993113562b90958a` | [`model card`](../model_cards/retail-bank-agent-9b.md) |

## Evaluation Components

| Component | Purpose |
| --- | --- |
| [`scripts/retail_bank/evaluate_tool_model.py`](../scripts/retail_bank/evaluate_tool_model.py) | Local static evaluator CLI wrapper |
| [`src/hello_slm/banking_tool_eval.py`](../src/hello_slm/banking_tool_eval.py) | Metrics, parser adapter, dry-run fixture, report writer |
| [`scripts/retail_bank/cloud_generate_tool_eval.py`](../scripts/retail_bank/cloud_generate_tool_eval.py) | GPU prediction generator and scorer |
| [`scripts/retail_bank/rescore_tool_eval.py`](../scripts/retail_bank/rescore_tool_eval.py) | Prompt-equivalent rescore helper |
| [`scripts/retail_bank/hf_job_tool_eval.py`](../scripts/retail_bank/hf_job_tool_eval.py) | Pinned Hugging Face Jobs bootstrap |
| [`scripts/retail_bank/run_remote_tool_eval_job.sh`](../scripts/retail_bank/run_remote_tool_eval_job.sh) | Paid HF Jobs launcher |
| [`tests/test_banking_tool_eval.py`](../tests/test_banking_tool_eval.py) | Metric and parser scoring tests |
| [`tests/test_banking_tool_eval_runner.py`](../tests/test_banking_tool_eval_runner.py) | Two-phase generation, resume, and job wrapper tests |

## Two-Phase Contract

[`cloud_generate_tool_eval.py`](../scripts/retail_bank/cloud_generate_tool_eval.py)
runs deterministic generation with the Granite tagged-JSON tool adapter.

Phase 1 asks the model for the first assistant response:

- For tool records, the prompt stops before the expected assistant tool-call
  message.
- For no-tool records, the prompt stops before the expected final assistant
  response.

If the model emits a tool call that exactly matches the next expected canonical
call, the runner appends the dataset's canonical tool result and allows another
model-owned pass. It never executes live tools. It stops on no tool calls,
unmatched tool calls, missing canonical tool results, `--max-tool-passes`, or
`--max-tool-calls`.

Phase 2 asks for the grounded final response only after required tool records
have canonical tool results appended. The metadata explicitly records:

- `tool_execution: false`;
- `deterministic_output_repair: false`;
- `teacher_forced_unseen_assistant_tool_calls: false`.

The runner resumes safely from an existing predictions JSONL and skips
completed `record_id` values. Old phase-row prediction files are rejected.

### Worked two-phase example

Frozen record:

```text
User: Show my cards.
Expected call: list_cards({})
Canonical result: active card ending in 4821
Expected facts: last4=4821, status=active
```

Phase 1 asks Granite before the expected assistant call. If Granite emits
`list_cards({})`, the evaluator appends the record's canonical result.

Phase 2 asks Granite for the final response. A valid response might be “You
have an active card ending in 4821.”

The record passes only if the generated tool and arguments match, the expected
trajectory is executable, and the final response preserves required facts.

If Granite emits `list_accounts({})`, the evaluator records the mismatch. It
does not rename the tool or inject the expected call.

For a no-tool FAQ row, phase 1 is also the final answer. For a clarification row,
the scorer checks that the model asks for the missing public selector instead
of inventing one.

## Local Dry Run

Use the static dry run to verify the scorer without GPU or Hub access:

```bash
PYTHONPATH=src python scripts/retail_bank/evaluate_tool_model.py \
  --dry-run \
  --checkpoint-revision local-docs-smoke \
  --output /tmp/retail-bank-tool-eval-dry-run.json
```

This command evaluates two in-memory records. It verifies tool name and
argument scoring, executable tool success, grounded final factuality, OOD path
scoring, credential request rate, and report serialization. It does not load the
Granite model or the frozen 1,374-record split.

Run focused evaluation tests:

```bash
python -m pytest -q tests/test_banking_tool_eval.py \
  tests/test_banking_tool_eval_runner.py \
  tests/test_banking_tool_eval_rescore.py
```

## Full Local/GPU Runner

Use the generator directly only when the machine has appropriate GPU memory and
Hub access:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_generate_tool_eval.py \
  --model-repo spkc83/retail-bank-servicing-agent-9b \
  --model-revision 1d56824995aa1adecfe20f62ca42fb1c0c443817 \
  --dataset-repo spkc83/retail-bank-servicing-alignment-sft \
  --dataset-revision 0ce32f9c7a3edff227005e5b89b089947b87625a \
  --manifest data/banking-servicing-alignment-v4/manifest.json \
  --output-dir artifacts/tool-eval \
  --split test \
  --family granite \
  --device cuda \
  --dtype fp16 \
  --max-new-tokens-first 192 \
  --max-new-tokens-final 220 \
  --max-tool-passes 4 \
  --max-tool-calls 6
```

`--limit N` may be used for a local experiment, but `N` must be at least `1`.
Do not use a limited run as a release gate.

## Paid HF Jobs Evaluation

The paid launcher is
[`scripts/retail_bank/run_remote_tool_eval_job.sh`](../scripts/retail_bank/run_remote_tool_eval_job.sh).
It validates exact source, model, and dataset revisions, checks that the pinned
bootstrap URL exists, then launches:

- `hf jobs uv run`;
- `--flavor rtx-pro-6000`;
- `--timeout 2h`;
- `--secrets HF_TOKEN`;
- `--volume hf://buckets/spkc83/jobs-artifacts:/data`.

```bash
export MODEL_REPO=spkc83/retail-bank-servicing-agent-9b
export DATASET_REPO=spkc83/retail-bank-servicing-alignment-sft
bash scripts/retail_bank/run_remote_tool_eval_job.sh \
  475dc2b563ef87fa0c9aa597b0b0465d56d2ee0f \
  1d56824995aa1adecfe20f62ca42fb1c0c443817 \
  0ce32f9c7a3edff227005e5b89b089947b87625a
```

The `HF_TOKEN` secret must read the model and dataset and write evaluation
artifacts to `spkc83/retail-bank-servicing-agent-9b`. Temporary, restartable
outputs are written to the mounted durable bucket. After published files and
hashes are verified in the model repository, the bucket copy may be retired
under the policy in [`docs/04-training-and-recovery.md`](04-training-and-recovery.md).

## Rescore Correctness

The final public dataset revision is
`0ce32f9c7a3edff227005e5b89b089947b87625a`. The training/evaluation generation
used `fea8aa1cda716954eb7322325e2be25c9f570ea3`. The correction did not change
rendered prompts, target tool calls, or target final responses. Therefore
`scripts/retail_bank/rescore_tool_eval.py` can rescore the existing predictions
against the corrected dataset identity. This is prompt-equivalent rescoring, not
a second generation run.

## Metrics and Gates

[`evaluate_records`](../src/hello_slm/banking_tool_eval.py) reports numerators,
denominators, and scores for tool names, tool arguments, executable tool
success, multi-tool sequence exactness, clarification appropriateness, FAQ
quality, OOD path handling, grounded final factuality, malformed calls,
unsupported/private arguments, credential requests, in-domain false refusals,
and OOD false accepts.

The released frozen evaluation passed:

| Slice | Result |
| --- | ---: |
| Frozen test conversations | `1,374` |
| Tool names and arguments | `796/796` |
| Executable tool trajectories | `700/700` |
| Exact dependent multi-tool sequences | `96/96` |
| Appropriate clarifications | `63/63` |
| Banking FAQ answers | `258/258` |
| OOD response paths | `35/35` |
| Grounded factual responses | `1,141/1,141` |
| Malformed calls | `0` |
| Unsupported/private arguments | `0` |
| Credential requests | `0` |
| In-domain false refusals | `0` |
| OOD false accepts | `0` |

## Leakage and Interpretation Warning

`Frozen` means the test artifact and scoring inputs were fixed for the run. It
does not mean the split is contamination-free.

The current audit found:

- `147/1,374` test records contain one of three normalized user turns also seen
  in training;
- all base template families cross train, validation, and test;
- `26/27` remediation test rows reuse an exact tool-call, canonical-result, and
  final-answer combination found in remediation training;
- `894/1,374` final-answer strings occur exactly in training; and
- Stage-2 remediation and the live `alex.demo` POC share backend facts.

There are no exact full-conversation duplicates and the base split keeps state
groups disjoint. The score is valid as an in-generator protocol regression
result, but it does not prove leakage-free generalization or rule out
memorization.

See the evidence and required independent benchmark design in
[`reference/data-leakage-audit.md`](reference/data-leakage-audit.md).

The post-training
[`banking-counterfactual-eval-v1`](../data/banking-counterfactual-eval-v1)
suite addresses this limitation with independently written prompts, unseen
project-SFT/POC facts, identical-prompt result variants, and a separate exact
gate. Preparation and local 4-bit execution are documented in
[`13-counterfactual-evaluation.md`](13-counterfactual-evaluation.md). Its result
must never be merged into or substituted for the released `1,374`-row score.

## Stop Conditions

Stop evaluation and do not publish a release claim if:

- any revision is not an exact 40-character lowercase Git commit;
- the manifest is unavailable or does not declare the requested split;
- prediction JSONL is corrupt or uses the old phase-row contract;
- the model emits unmatched required tool calls and metrics fall below gate;
- `max_tool_passes` or `max_tool_calls` truncates required calls;
- the report lacks dataset fingerprint, adapter template hash, or checkpoint
  revision;
- any hard gate fails;
- the paid job cannot persist to `/data` or upload evaluation artifacts.

Evaluation does not repair model output and does not execute live tools. A
failed evaluation should produce a failed report, not a corrected prediction
file.
