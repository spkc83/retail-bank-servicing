# Granite V5 Model and PEFT

The generative component is an IBM Granite 8.79B causal language model adapted
for Harborlight Bank customer service with supervised fine-tuning and LoRA.
The V5 work is incremental: it starts from the released servicing model
`spkc83/retail-bank-servicing-agent-9b` at revision
`1d56824995aa1adecfe20f62ca42fb1c0c443817` and trains on the V5 composite
dataset.

Granite V5 training job `6a7f79531f5885ae605b96cc` completed on canonical-
policy dataset revision `40a0b68...`. The published candidate is an unmerged BF16
LoRA adapter at `spkc83/retail-bank-servicing-agent-9b-peft@cc95e446...`
composed with immutable Stage-2 base revision `1d568249...`. The adapter files
were committed at `b4269445...`; the final release revision is `cc95e446...`.
Frozen evaluation job
`6a7f89edc97db76cbdf31893` failed strict gates. Five credential findings were
evaluator false positives; two genuine behavioral failures remain. A corrected
evaluator and generalized incremental SFT are underway, so this candidate is
not cleared for deployment.

## Why PEFT Instead of Training 9B From Scratch

The base checkpoint already contains general language, instruction following,
and broad reasoning. The V5 corpus teaches a narrow behavioral contract:

- Harborlight/Harbor tone and customer-facing language;
- Granite tagged-JSON banking actions;
- grounded answers from action results;
- Markdown table presentation;
- policy answers from supplied passages with `[Policy: id]` citations;
- policy detours and return to a prior servicing task;
- repair, clarification, topic-shift, and OOD behavior.

A few thousand domain records can reinforce those behaviors. They cannot teach
a 9B model language from scratch. LoRA keeps the base weights frozen and learns
small low-rank updates for selected projections, reducing optimizer memory and
training cost.

## LoRA Configuration

[`cloud_train_tool_sft.py`](../scripts/retail_bank/cloud_train_tool_sft.py)
uses these defaults:

| Setting | Value |
| --- | --- |
| Rank `r` | 32 |
| Alpha | 64 |
| Dropout | 0.05 |
| Bias | none |
| Task type | causal language modeling |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Primary precision | BF16 LoRA |
| Optional precision | NF4 QLoRA with BF16 compute |
| Maximum sequence length | 2,048 tokens |
| Gradient checkpointing | enabled |
| Packing | disabled |

Conceptually, a frozen weight matrix `W` receives a learned update:

```text
effective_weight = W + scale * (B @ A)
scale = lora_alpha / rank
```

Only `A` and `B` are trained. Although the worker can merge them into the base,
merged FP16 and BF16 candidates both failed the unchanged behavioral-parity
gates for this candidate. The only valid candidate representation therefore keeps
the base and adapter separate.

At inference, PEFT applies the learned update without rewriting base weights:

```text
load base @ 1d568249...
attach adapter @ cc95e446... with autocast_adapter_dtype=False
run the resulting PeftModel in BF16 (or quantize the base locally)
```

## What the Model Receives

The JSON `messages` array is a storage format, not a literal JSON string shown
to the model. [`ToolWireAdapter`](../src/hello_slm/banking_tool_wire.py) passes
the structured messages and action schemas through Granite's pinned chat
template. The tokenizer produces:

```text
input_ids       complete rendered conversation
attention_mask  non-padding token positions
labels          token targets; -100 outside trainable assistant spans
```

For this record:

```json
[
  {"role":"system","content":"You are Harbor ...","loss":false},
  {"role":"user","content":"Freeze my card ending 4821.","loss":false},
  {
    "role":"assistant",
    "content":null,
    "loss":true,
    "tool_calls":[{
      "function":{"name":"freeze_card","arguments":{"last4":"4821"}}
    }]
  }
]
```

the system and user tokens supply context but have label `-100`. The assistant
action tokens receive causal-language-model loss. A later action result also
supplies context, and the final assistant answer receives loss when marked
`loss: true`.

The TRL setting `assistant_only_loss=False` is intentional because the worker
pre-tokenizes records and constructs the exact assistant-only label mask
itself. TRL must not apply a second mask.

## Granite Action Wire

The runtime supports Granite tagged JSON:

```text
<tool_call>{"name":"freeze_card","arguments":{"last4":"4821"}}</tool_call>
```

The harness, not the model, assigns or validates correlated call metadata,
checks schemas, executes the action, and supplies the result. Granite then
writes the final answer. Nine public actions are available in normal servicing
turns; policy turns receive no action schemas.

## Retrieval-Grounded Policy Generation

Policy retrieval is not embedded in the model weights. The runtime retrieves
current versioned chunks and builds a policy system message such as:

```text
Authoritative Harborlight Bank policy context.
[Policy: mortgage.opening.us.v1] Customers may begin a mortgage application ...
```

Granite must answer only from those chunks and cite at least one allowed ID.
The validator rejects missing or invented citations. This separates two
concerns:

- SFT teaches the model how to use supplied policy evidence;
- the versioned knowledge base supplies the current facts at inference time.

Updating a policy therefore does not require retraining if the behavior and
schema remain stable.

## V5 Incremental Training Inputs

The active job uses:

| Input | Value |
| --- | --- |
| Base model | `spkc83/retail-bank-servicing-agent-9b` |
| Base revision | `1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| Dataset | `spkc83/retail-bank-servicing-alignment-sft` |
| Dataset revision | `9d7aed545604bb42fb02b7a0919427a0ed2b81e2` |
| Policy corpus revision | `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a` |
| Training source commit | `75b56ffff45e75ffbee11c0e0552dc35ae124d21` |
| Hardware | `rtx-pro-6000` |
| Job cap | five hours; optimizer cap four hours |
| Job | `6a7f79531f5885ae605b96cc` (completed) |
| Maximum steps | 750 |
| Learning rate | `2e-5` |
| Gradient accumulation | 2 |
| Checkpoint interval | 250 steps |
| Final training loss | `0.13014758` |
| Final evaluation loss | `0.3200804` |
| Final token accuracy | `0.96240348` |
| Adapter repository | `spkc83/retail-bank-servicing-agent-9b-peft` |
| PEFT release revision | `cc95e446af2b5e1d8d9df2751a8192613ad386e3` |
| Adapter bundle commit | `b4269445ce7b2b943d2d9531102166bf8840a074` |
| BF16 adapter SHA-256 | begins `043b22c5`; the full digest remains in the release metadata |

The final PEFT release revision `cc95e446...` was the v7 inference identity.
The bundle commit `b4269445...` proves which adapter files it contains;
neither rejected merged checkpoint is an inference substitute. The current
inference identity is the v8 adapter
`spkc83/retail-bank-servicing-agent-9b-peft-v10-longctx` at revision
`055ce38af4595b1e139a9e9baea8e0c53cba7c2e`. Its `adapter_config.json` is published under `adapter/`, so both runtimes
need `RETAIL_BANK_ADAPTER_SUBFOLDER=adapter`; PEFT reads the repo root otherwise and fails with a 404.
The previously deployed adapter was `spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation` at
`badbc05ad1f861818ea244b462eda49bca6c6fca`.

## Rebuilding on New Infrastructure

For independent reproduction, keep the same logical sequence:

1. Generate `data/banking-v5-tool-sft`.
2. Generate the composite `data/banking-servicing-alignment-v5`.
3. Choose an immutable Granite base revision.
4. Run the guarded worker with the composite manifest and record its complete
   configuration fingerprint.
5. Save the adapter, optimizer/checkpoint state, tokenizer, template hash, and
   Trackio metrics.
6. Run the unchanged behavioral-parity gates. A merged candidate is usable
   only if every gate passes.
7. If merging fails parity, validate and publish the unmerged adapter with
   [`hf_job_finalize_tool_sft_peft.py`](../scripts/retail_bank/hf_job_finalize_tool_sft_peft.py).
8. Record the immutable base and adapter revisions as one composition.
9. Evaluate that exact composition before deployment.

The current V5 job is a continuation from the released domain model because
that is the cheapest safe update. The data generators and worker remain usable
with another explicitly pinned Granite-family base; changing the base requires
a fresh full evaluation and is not equivalent to resuming the current job.

## Tiny Local Pipeline Check

This command exercises record tokenization, assistant label masking,
checkpoint metadata, and action parsing without downloading Granite, using a
GPU, or publishing:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/cloud_train_tool_sft.py \
  --manifest data/banking-servicing-alignment-v5/manifest.json \
  --output-dir /tmp/harbor-granite-v5-smoke \
  --run-tiny-smoke
```

The tiny smoke is a pipeline test, not evidence that the 8.79B model learned
the V5 behavior.
