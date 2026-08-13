# Model And PEFT

The active generative model is `spkc83/retail-bank-servicing-agent-9b`, a
merged FP16 LoRA adaptation of IBM Granite for a synthetic retail-bank
customer-service POC.

The source of truth for released identity and metrics is
[../model_cards/retail-bank-agent-9b.md](../model_cards/retail-bank-agent-9b.md).
The source of truth for local training defaults is
[../scripts/retail_bank/cloud_train_tool_sft.py](../scripts/retail_bank/cloud_train_tool_sft.py)
and [../configs/banking-tool-sft-granite.toml](../configs/banking-tool-sft-granite.toml).

For a detailed, example-driven explanation of instruction SFT, assistant-only
loss, LoRA matrices, the two training stages, and the merged inference model,
read [12-instruction-fine-tuning-and-peft.md](12-instruction-fine-tuning-and-peft.md).

## Base Model Identity

| Field | Value |
| --- | --- |
| Base model | `ibm-granite/granite-4.1-8b` |
| Base revision | `1504002f650e656a0a3789d99574df12e3e94ed0` |
| Architecture | Dense decoder-only causal transformer |
| Parameter count | 8,791,592,960 |
| Tool format | Granite native tagged JSON |
| Released model repo | `spkc83/retail-bank-servicing-agent-9b` |
| Immutable weights revision | `1d56824995aa1adecfe20f62ca42fb1c0c443817` |

The live POC loads that model repo and revision by default in
[../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py](../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py).

## Two-Stage PEFT Strategy

Training uses LoRA through PEFT and TRL SFTTrainer. Stage 1 adapts the pinned
IBM Granite base to the synthetic-bank tool wire. Stage 2 continues from the
tool-trained checkpoint with the v4 servicing-remediation corpus because live
POC testing exposed multi-turn conversation and tool-use failures.

Defaults in [../scripts/retail_bank/cloud_train_tool_sft.py](../scripts/retail_bank/cloud_train_tool_sft.py):

| Setting | Value |
| --- | --- |
| Precision | `bf16-lora` |
| Optional precision | `qlora` |
| LoRA rank | `32` |
| LoRA alpha | `64` |
| LoRA dropout | `0.05` |
| Learning rate | `1e-4` |
| Max sequence length | `2048` |
| Training seed | `7303` |
| Default max steps | `1000` locally, `3000` in the HF job wrapper |
| Checkpoint interval | `250` locally, `500` in the HF job wrapper |

LoRA target modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

These names are declared in `LORA_TARGET_MODULES` in
[../scripts/retail_bank/cloud_train_tool_sft.py](../scripts/retail_bank/cloud_train_tool_sft.py)
and mirrored by the local TOML configuration.

## Training Record Rendering

Training examples are rendered by `ToolWireAdapter.render_training()` in
[../src/hello_slm/banking_tool_wire.py](../src/hello_slm/banking_tool_wire.py).
That adapter is responsible for:

- accepting only the Granite family;
- rendering tokenizer chat-template messages with tool schemas;
- preserving whole user-to-final-assistant tool chains inside the sequence
  budget;
- applying assistant-only labels;
- masking context, user messages, and tool results with `-100`;
- returning `input_ids`, `attention_mask`, `labels`, a span map, and the chat
  template hash.

The training worker pre-tokenizes records through `tokenize_records()` in
[../scripts/retail_bank/cloud_train_tool_sft.py](../scripts/retail_bank/cloud_train_tool_sft.py).

## Granite Tool Wire

The active tool wire is Granite-only. `_normalize_family()` in
[../src/hello_slm/banking_tool_wire.py](../src/hello_slm/banking_tool_wire.py)
raises an error for any non-Granite family.

Tool calls use tagged JSON blocks:

```text
<tool_call>
{"name":"freeze_card","arguments":{"last4":"4821"}}
</tool_call>
```

The parser validates:

- parseable JSON;
- object payloads;
- known public tool names;
- object arguments;
- allowed argument names;
- required arguments when a schema declares them;
- JSON value types and numeric bounds;
- unique and ordered call IDs/indexes.

The adapter intentionally does not infer intent, repair malformed output,
rename tools, or fill missing arguments. Invalid model output is a model
protocol error.

## Worked Example: What LoRA Learns

Suppose a training row contains:

```text
user context: Replace card 4821.
assistant target: <tool_call>{"name":"replace_card","arguments":{"last4":"4821"}}</tool_call>
tool context: card.status=replacement_pending
assistant target: A replacement for card 4821 is pending.
```

The pretrained Granite weights already encode English and general instruction
behavior. LoRA adds trainable low-rank updates to selected attention and MLP
projections so the model can learn this repo's tool protocol and response style.

Conceptually, a frozen weight matrix `W` is used with a learned update:

```text
effective weight = W + scale * (B @ A)
```

`A` and `B` are much smaller than `W` when the LoRA rank is small. The released
rank is `32`; the base matrix is not replaced by a 32-parameter model.

During assistant-only SFT, user and tool-result tokens receive label `-100`.
Only the assistant tool call and final answer contribute to cross-entropy loss.

After training, the adapter is merged into the base weights for the published
FP16 checkpoint. The unmerged adapter is also retained for provenance and
recovery.

See [PEFT's LoRA guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
and the [original LoRA paper](https://arxiv.org/abs/2106.09685) for the general
method.

## Local Planning And Smoke Checks

The training worker is safe by default. Running it without remote execution
flags prints a dry-run plan and does not download the 8.79B base model, start a
paid job, merge weights, or push to Hugging Face:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-servicing-alignment-v4/manifest.json \
  --base-model spkc83/retail-bank-agent-9b \
  --base-revision 085df3d089cfadd77424b548542da0390a54a23e
```

The local tiny smoke path uses small offline stand-ins:

```bash
PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
  --run-tiny-smoke \
  --family granite \
  --max-steps 1 \
  --output-dir /tmp/banking-v3-tool-sft-smoke
```

Use the smoke path to prove tokenizer rendering, assistant-label masking,
checkpoint metadata, and tagged-JSON parsing without downloading the base
model.

## Remote Training Guard

Full remote execution requires all of these safeguards:

- `--execute-remote`
- `--allow-remote-execution`
- `RETAIL_BANK_ALLOW_REMOTE_TOOL_SFT=banking-v3-tool-sft`

The guarded wrapper is
[../scripts/retail_bank/run_remote_training_job.sh](../scripts/retail_bank/run_remote_training_job.sh).
It submits [../scripts/retail_bank/hf_job_tool_sft.py](../scripts/retail_bank/hf_job_tool_sft.py)
to Hugging Face Jobs with:

- exact source commit;
- exact dataset revision;
- `rtx-pro-6000` flavor;
- five-hour outer timeout;
- mounted artifact volume;
- `HF_TOKEN` as a secret;
- BF16 LoRA settings.

The job script downloads the pinned source archive, downloads the dataset
snapshot, then calls the guarded local worker with push-to-Hub enabled.

## Checkpoints And Fingerprints

`training_fingerprint()` in
[../scripts/retail_bank/cloud_train_tool_sft.py](../scripts/retail_bank/cloud_train_tool_sft.py)
captures:

- base model and revision;
- Granite family;
- tokenizer chat-template hash;
- dataset repository, revision, and manifest hash;
- training seed;
- precision;
- LoRA rank, alpha, dropout, and target modules.

`validate_resume_fingerprint()` rejects resume checkpoints whose metadata does
not match the current training inputs. This prevents accidental continuation
from a different base, dataset, template, precision, or adapter shape.

## Merge And Release Layout

The release keeps two forms:

- root checkpoint: merged FP16 weights;
- `adapter/`: retained unmerged LoRA adapter.

`merge_adapter_with_reload_parity()` in
[../scripts/retail_bank/cloud_train_tool_sft.py](../scripts/retail_bank/cloud_train_tool_sft.py)
merges the adapter, reloads the merged model, and compares adapter-vs-merged
outputs. The release helper
[../scripts/retail_bank/hf_job_finalize_tool_sft.py](../scripts/retail_bank/hf_job_finalize_tool_sft.py)
checks parity reports before publication.

The public model card reports the active servicing-remediation release metrics:

| Metric | Value |
| --- | ---: |
| Training job | `spkc83/6a6ca6276b79c09949c1d6cb` |
| Runtime | about 18 minutes 59 seconds |
| Estimated cost | about `$0.87` |
| Training loss | `0.0069123295` |
| Evaluation loss | `0.0002181597` |
| Token accuracy | `0.999976121` |

Merge parity is a release gate, not a replacement for frozen evaluation.

## Frozen Evaluation Summary

The model card records that the released checkpoint passed the frozen
1,374-record evaluation split with:

- `796/796` tool names and arguments;
- `700/700` executable tool trajectories;
- `96/96` exact dependent multi-tool sequences;
- `63/63` appropriate clarifications;
- `258/258` banking FAQ answers;
- `35/35` OOD response paths;
- `1,141/1,141` grounded factual responses;
- zero malformed calls, private arguments, credential requests, in-domain
  false refusals, or OOD false accepts.

The evaluator code is [../src/hello_slm/banking_tool_eval.py](../src/hello_slm/banking_tool_eval.py).
The prompt-equivalent rescore helper is
[../scripts/retail_bank/rescore_tool_eval.py](../scripts/retail_bank/rescore_tool_eval.py).
The remote evaluator entry points live under [../scripts/retail_bank](../scripts/retail_bank).

## Related Tests

Run the focused model/tool-wire tests from the repository root:

```bash
python -m pytest -q \
  tests/test_banking_tool_wire.py \
  tests/test_banking_tool_sft_worker.py \
  tests/test_banking_tool_sft_job.py \
  tests/test_banking_tool_sft_continuation.py \
  tests/test_banking_tool_sft_export_recovery.py \
  tests/test_banking_tool_eval.py \
  tests/test_banking_tool_eval_runner.py
```
