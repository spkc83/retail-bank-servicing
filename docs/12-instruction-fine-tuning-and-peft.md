# Instruction Fine-Tuning and PEFT Design

This guide explains the model-design choices behind the retail-bank servicing
agent. It connects instruction fine-tuning, assistant-only loss, LoRA, PEFT,
the two SFT stages, merged weights, and runtime inference to the repo code.

The short version is:

```text
pretrained Granite language model
  + instruction examples in Granite chat/tool format
  + assistant-only causal-language-model loss
  + LoRA updates on selected attention and MLP projections
  = a bank-servicing instruction follower
```

The project does not pretrain a 9B model from scratch. It starts with Granite's
existing language ability and adapts how the model responds, calls tools, uses
tool results, follows conversation history, and handles supported requests.

## 1. What “Model Design” Means Here

Four layers make up the generative model design:

| Layer | Project choice | What it controls |
| --- | --- | --- |
| Foundation architecture | Granite 4.1 8B dense decoder-only transformer | Token prediction, attention, and general language capacity. |
| Behavior format | Granite chat template and tagged-JSON tools | How roles, instructions, tool calls, and tool results appear to the model. |
| Adaptation method | Instruction SFT with PEFT LoRA | Which task behavior is learned and which weights change. |
| Release form | Adapter merged into FP16 weights | How the trained model is loaded for inference. |

PEFT is not a replacement for the transformer architecture. It is the method
used to adapt selected parts of that architecture without updating every base
parameter.

The released model still has 8,791,592,960 parameters. “Parameter efficient”
means far fewer parameters are trained, not that the final transformer becomes
a tiny model.

## 2. The Pinned Foundation Model

The initial foundation checkpoint is:

| Field | Value |
| --- | --- |
| Model | `ibm-granite/granite-4.1-8b` |
| Revision | `1504002f650e656a0a3789d99574df12e3e94ed0` |
| Architecture | Dense decoder-only causal transformer |
| Parameters | `8,791,592,960` |
| Context used for SFT | At most `2,048` tokens per record |

Granite already knows tokenization, grammar, broad facts, instruction patterns,
and general dialogue. The banking corpus teaches a narrower behavioral
contract; it is not large enough to recreate those capabilities from scratch.

The exact identity is pinned in
[`retail-bank-release.toml`](../configs/retail-bank-release.toml) and the local
training defaults are in
[`banking-tool-sft-granite.toml`](../configs/banking-tool-sft-granite.toml).

## 3. Instruction Fine-Tuning

Instruction fine-tuning trains a pretrained causal language model on examples
of an instruction or conversation followed by the desired assistant behavior.
It is commonly abbreviated as instruction SFT.

In this project, an example can contain:

1. a system instruction;
2. earlier conversation turns;
3. the current user request;
4. an assistant tool call;
5. the correlated tool result; and
6. an assistant answer grounded in that result.

The model sees the context and learns to predict the assistant-owned tokens.
It does not learn by executing a database during training.

### SFT compared with nearby techniques

| Technique | Starting point | Learning signal | Used here? |
| --- | --- | --- | --- |
| Pretraining | Random or partially trained weights | Predict tokens across a very large corpus | No |
| Instruction SFT | Pretrained model | Reproduce desired assistant outputs | Yes |
| Prompt engineering | Fixed model | No weight update | Yes, at inference and data rendering |
| Preference tuning | SFT model plus preferred/rejected outputs | Rank or optimize response preferences | No |
| Retrieval | Fixed or adapted model plus external documents | Add facts to the prompt | No dedicated RAG layer |

SFT changes model weights. Prompt engineering only changes the model input.
The two are complementary because inference should use the same role and tool
format that appeared during training.

## 4. One Training Example

This simplified example mirrors records in
[`train.jsonl`](../data/banking-v3-tool-sft/train.jsonl):

```text
system [context only]
  You are the customer-service agent for a fictional retail bank.

user [context only]
  Please show my accounts and balances.

assistant [training target]
  <tool_call>{"name":"list_accounts","arguments":{}}</tool_call>

tool [context only]
  checking 1792: available USD 4,728.25
  savings 4756: available USD 7,226.08

assistant [training target]
  Main Checking ending in 1792 has USD 4,728.25 available.
  Goal Saver ending in 4756 has USD 7,226.08 available.
```

The same record also contains `expected`, `metadata`, `provenance`, and
`validation` sections. Those fields support governance and evaluation but are
not passed to Granite as conversational tokens during SFT.

Only the `messages` array becomes the chat sequence. The public tool schemas
are also supplied to the Granite chat template.

## 5. What the Causal Model Learns

A decoder-only causal model predicts each next token from all earlier tokens in
the rendered sequence.

For an assistant target with tokens `y1 ... yN`, the training objective is:

```text
loss = average of -log P(yi | all visible tokens before yi)
       over assistant-owned target tokens only
```

The model therefore learns conditional behavior such as:

```text
user request + available tool schemas
  -> correct tagged-JSON tool call

conversation + tool call + tool result
  -> grounded customer-facing answer
```

This is still next-token prediction. “Instruction following” emerges because
the input-output structure consistently places instructions in the context and
desired behavior in assistant targets.

## 6. Assistant-Only Label Masking

The token sequence contains system, user, assistant, and tool-result tokens.
The context must remain visible, but it should not be treated as text the
assistant is expected to reproduce.

The repo assigns labels as follows:

| Token owner | Visible to the model? | Contributes to loss? | Label |
| --- | --- | --- | --- |
| System | Yes | No | `-100` |
| User | Yes | No | `-100` |
| Assistant tool call | Yes | Yes | Matching token ID |
| Tool result | Yes | No | `-100` |
| Assistant final answer | Yes | Yes | Matching token ID |
| Padding | No meaningful content | No | `-100` |

PyTorch cross-entropy ignores label `-100`. The model can attend to masked
context tokens while gradients are driven only by the assistant spans.

```text
input:   [system] [user] [assistant call] [tool result] [assistant answer]
labels:  [-100  ] [-100] [token IDs     ] [-100      ] [token IDs       ]
```

[`ToolWireAdapter.render_training()`](../src/hello_slm/banking_tool_wire.py)
derives these spans from the actual Granite chat template. It rejects a
template layout when the assistant-token boundaries cannot be proven.

The worker supplies precomputed `input_ids`, `attention_mask`, and `labels` to
TRL. Its `assistant_only_loss=False` setting is intentional: the repo has
already built the assistant-only labels and TRL must not create a second mask.

## 7. Why Complete Tool Chains Matter

A tool-use example has two different assistant decisions:

1. choose a tool and generate valid public arguments;
2. use the returned facts to write the final response.

Cutting the record between those decisions would train an incomplete behavior.
The tool-wire adapter therefore keeps whole interaction chains within the
2,048-token SFT budget.

For dependent tools, each later call can use an earlier result:

```text
user asks about a transfer
  -> assistant calls list_transfers
  -> tool returns a transfer ID
  -> assistant calls get_transfer with that public ID
  -> tool returns status and amount
  -> assistant explains the result
```

Every assistant call and the final answer are targets. Each tool result remains
masked context.

## 8. PEFT and LoRA

PEFT means parameter-efficient fine-tuning. The `peft` library supports several
adapter methods; this project uses LoRA.

LoRA freezes the original weight matrix and learns a low-rank update through
two smaller matrices.

For one adapted linear layer:

```text
W_effective = W_frozen + (alpha / rank) * (B @ A)
```

For this project:

```text
rank  = 32
alpha = 64
scale = alpha / rank = 2
```

`W_frozen` preserves the pretrained layer. `A` and `B` are the trainable LoRA
matrices. Their product has the same input-output shape as `W_frozen`, but the
update is constrained to rank 32.

This constraint is useful because task adaptation often needs a structured
change in model behavior rather than an independent update to every weight.

### A shape example

Assume a projection has a `4096 x 4096` weight matrix.

```text
full matrix parameters: 4096 * 4096 = 16,777,216

LoRA A parameters:       32 * 4096  =    131,072
LoRA B parameters:       4096 * 32  =    131,072
LoRA total:                               262,144
```

That example is illustrative; Granite layer shapes differ by module. It shows
why a low-rank update can be much smaller than a full matrix update.

## 9. Which Granite Modules Are Adapted

The active `LoraConfig` targets these linear projections:

| Module | Transformer role | Adaptation intuition |
| --- | --- | --- |
| `q_proj` | Attention queries | Change what the current token looks for. |
| `k_proj` | Attention keys | Change how earlier tokens advertise relevant information. |
| `v_proj` | Attention values | Change the information retrieved from attended tokens. |
| `o_proj` | Attention output | Change how attention results return to the residual stream. |
| `gate_proj` | MLP gating | Change which feed-forward features activate. |
| `up_proj` | MLP expansion | Change the expanded feature representation. |
| `down_proj` | MLP contraction | Change how MLP features return to model width. |

Adapting attention and MLP projections gives LoRA influence over context use,
protocol generation, and response transformation without unfreezing the full
foundation model.

The exact list is `LORA_TARGET_MODULES` in
[`cloud_train_tool_sft.py`](../scripts/retail_bank/cloud_train_tool_sft.py).

## 10. What Is Trained and What Is Frozen

During a LoRA SFT step:

```text
frozen:
  pretrained Granite parameters

trainable:
  LoRA A and B matrices attached to selected projections

optimizer state:
  created for trainable adapter parameters, not every frozen parameter
```

The forward pass still uses the full 8.79B model. PEFT mainly reduces gradient
and optimizer memory, plus the amount of trainable state stored in an adapter.

Activation memory still depends on sequence length, batch size, hidden size,
layer count, and gradient checkpointing. LoRA does not make 9B inference free
and does not remove the need to load the foundation weights.

## 11. BF16 LoRA and Optional QLoRA

The released path uses BF16 LoRA on an RTX PRO 6000 job.

| Mode | Base-weight loading | Trainable state | Project role |
| --- | --- | --- | --- |
| BF16 LoRA | Base loaded in BF16 | LoRA adapters | Primary released training path |
| QLoRA | Base loaded in 4-bit NF4 | LoRA adapters with BF16 compute | Optional lower-memory path |

The optional QLoRA configuration enables NF4, double quantization, and BF16
compute through `BitsAndBytesConfig`.

QLoRA reduces training memory further, but it adds quantization behavior to the
training path. It was not the precision mode used for the active release.

## 12. The Two Sequential SFT Stages

The release uses one reproducible pipeline with two sequential SFT stages.

```text
pinned IBM Granite checkpoint
  -> Stage 1 LoRA on 9,000 general tool-use records
  -> merge Stage 1 adapter into a tool-trained checkpoint
  -> Stage 2 LoRA on 9,000 base records + 427 remediation records
  -> merge Stage 2 adapter into the final servicing checkpoint
```

### Stage 1: learn the application protocol

Stage 1 teaches broad retail-bank behavior:

- Granite tagged-JSON tool calls;
- the nine public synthetic-bank tools;
- tool-result grounding;
- clarifications;
- banking FAQs;
- OOD response examples; and
- multi-tool ordering.

The canonical release configuration uses 3,000 maximum steps and a `1e-4`
learning rate.

### Stage 2: repair observed servicing failures

Stage 2 starts from the merged Stage-1 model and attaches a new LoRA adapter.
It addresses failures found while testing the POC.

Examples include:

- card references such as “replace the active one”;
- answers to earlier clarification questions;
- service-case follow-ups;
- agent repair after a mistaken assumption;
- topic changes within one session; and
- longer customer-service responses.

The canonical release configuration uses 500 maximum steps and a `2e-5`
learning rate.

The complete Stage-1 corpus remains in the Stage-2 dataset. This replay reduces
the chance that targeted correction examples improve one case while damaging
previously learned tool behavior.

## 13. Two Meanings of “Continue Training”

The repo supports two technical continuation shapes. They should not be
confused.

| Shape | Starting object | New adapter? | Use |
| --- | --- | --- | --- |
| Sequential release stage | Merged Stage-1 model | Yes | Canonical Stage-2 release path |
| Adapter continuation | Granite base plus retained Stage-1 adapter | No; reopen existing adapter | Recovery or focused continuation helper |

[`run_release_pipeline.py`](../scripts/retail_bank/run_release_pipeline.py)
implements the canonical sequential release path.

[`cloud_continue_tool_sft.py`](../scripts/retail_bank/cloud_continue_tool_sft.py)
implements retained-adapter continuation. It is useful for recovery, but it is
not the command used by the current two-stage release definition.

## 14. From JSONL Record to Optimizer Step

The active Stage-1 and Stage-2 worker path is:

```text
JSONL messages
  -> validate record and split manifest
  -> Granite chat template with public tool schemas
  -> identify assistant token spans
  -> create input_ids, attention_mask, and labels
  -> batch with -100 label padding
  -> full Granite forward pass plus LoRA updates
  -> assistant-token cross-entropy
  -> backpropagate into LoRA A and B only
  -> optimizer updates adapter parameters
```

The relevant functions are:

| Responsibility | Code |
| --- | --- |
| Render and mask one record | `ToolWireAdapter.render_training()` |
| Convert records to tensors | `tokenize_records()` |
| Pad pre-tokenized batches | `collate_pretokenized()` |
| Build LoRA and TRL settings | `build_training_configs()` |
| Run SFT | `run_remote_training()` |

The renderer is in
[`banking_tool_wire.py`](../src/hello_slm/banking_tool_wire.py). The other
functions are in
[`cloud_train_tool_sft.py`](../scripts/retail_bank/cloud_train_tool_sft.py).

## 15. Why the Chat Template Is Part of the Model Contract

Role markers and tool syntax are tokens. Changing them can change model
behavior even when the visible English request is identical.

The tokenizer's Granite chat template controls:

- system, user, assistant, and tool role boundaries;
- tool schema placement;
- tagged tool-call serialization;
- end-of-turn tokens; and
- the generation prefix used at inference.

The training fingerprint records the chat-template hash. Resume validation
rejects a checkpoint when the template hash does not match the current run.

Training and inference both use `ToolWireAdapter`, preventing a hand-written
prompt from silently drifting away from the trained format.

## 16. Adapter, Merged Model, and Inference

After SFT, the trainable result is initially an adapter:

```text
base checkpoint + LoRA adapter = adapted model
```

For the public release, the update is merged:

```text
W_released = W_base + (alpha / rank) * (B @ A)
```

The release layout keeps both forms:

| Location | Contents | Purpose |
| --- | --- | --- |
| Model repo root | Merged FP16 model | Normal inference and ZeroGPU loading |
| `adapter/` | Unmerged LoRA adapter | Provenance, recovery, and further adaptation |

The POC loads the merged model from
`spkc83/retail-bank-servicing-agent-9b` at revision
`1d56824995aa1adecfe20f62ca42fb1c0c443817`.

It does not attach LoRA modules for each chat request. PEFT is the training
mechanism; after merging, inference uses ordinary causal-model weights.

## 17. Merge Parity Is Required

Merging should preserve the adapter model's behavior. The release process
therefore compares the adapter form with a freshly reloaded merged checkpoint.

It checks items such as:

- next-token argmax agreement;
- maximum logit difference;
- high-percentile logit difference; and
- deterministic generation behavior.

`merge_and_unload(safe_merge=True)` performs the merge. The repo then reloads
the saved FP16 model before accepting it.

Merge parity proves packaging consistency. It does not prove that the model
chooses correct tools or gives useful customer-service answers.

## 18. Evaluation After SFT

Training and validation loss answer a narrow question: how well did the model
predict held-out assistant tokens rendered by this pipeline?

The released model also undergoes frozen behavioral evaluation:

| Behavior | Evidence checked |
| --- | --- |
| Tool choice | Exact public tool name |
| Arguments | Exact keys and values |
| Multi-tool behavior | Required order and dependency |
| Execution | Canonical tool trajectory succeeds |
| Clarification | Appropriate question instead of a guessed call |
| Grounding | Final response contains required returned facts |
| OOD | Correct scope response on frozen OOD cases |
| Protocol quality | No malformed or private arguments |

The active Stage-2 job reported low token loss and passed the frozen
1,374-record behavior gate described in [`06-evaluation.md`](06-evaluation.md).

That gate is a protocol regression suite, not a leakage-free generalization
benchmark. Its template families, many targets, and the POC's synthetic facts
overlap training.

Read the dated
[`data-leakage-audit.md`](reference/data-leakage-audit.md) before interpreting
the perfect released metrics.

## 19. What SFT Can and Cannot Do

Instruction SFT is well suited to:

- teaching a stable response protocol;
- improving tool selection and argument formatting;
- conditioning behavior on conversation history;
- learning clarification patterns;
- grounding responses in supplied tool results; and
- shaping tone and scope behavior.

Instruction SFT does not guarantee:

- facts that never appear in the base model, prompt, or tool result;
- correct behavior outside the training distribution;
- reliable unseen tools without schemas and examples;
- unlimited conversation memory;
- authorization or transaction safety; or
- production banking correctness from synthetic data alone.

The model only receives the history that fits the runtime prompt budget. SFT
can teach it to use visible history, but it cannot recover turns omitted from
the prompt.

## 20. Worked End-to-End Example

Consider a follow-up conversation:

```text
User: Show my cards.
Assistant: calls list_cards
Tool: returns active card 4821 and frozen card 9134
Assistant: summarizes both cards
User: Replace the active one.
```

At inference, the selected prior interaction group is visible to Granite. The
model can resolve “the active one” to card `4821` and produce:

```text
<tool_call>{"name":"replace_card","arguments":{"last4":"4821"}}</tool_call>
```

The runtime executes the synthetic tool and returns its result. Granite then
generates the final customer-facing response from the enlarged conversation.

This behavior depends on all four design layers:

1. Granite supplies language and causal attention.
2. Instruction examples teach the desired multi-turn behavior.
3. LoRA stores task-specific updates in selected projections.
4. The runtime preserves the trained chat/tool format and relevant history.

## 21. Reproduction Map

Read or run these files in order:

| Step | Source |
| --- | --- |
| Understand data records | [`02-data-generation.md`](02-data-generation.md) |
| Inspect local LoRA defaults | [`banking-tool-sft-granite.toml`](../configs/banking-tool-sft-granite.toml) |
| Inspect message rendering | [`banking_tool_wire.py`](../src/hello_slm/banking_tool_wire.py) |
| Inspect SFT and merge code | [`cloud_train_tool_sft.py`](../scripts/retail_bank/cloud_train_tool_sft.py) |
| Inspect the two release stages | [`run_release_pipeline.py`](../scripts/retail_bank/run_release_pipeline.py) |
| Reproduce the full workflow | [`08-end-to-end-runbook.md`](08-end-to-end-runbook.md) |
| Inspect release evidence | [`retail-bank-agent-9b.md`](../model_cards/retail-bank-agent-9b.md) |
| Inspect immutable identities | [`artifacts.md`](reference/artifacts.md) |

## 22. References

- [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [Transformers chat templates](https://huggingface.co/docs/transformers/en/chat_templating)
- [PEFT documentation](https://huggingface.co/docs/peft/index)
- [PEFT LoRA guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
- [PEFT LoRA API](https://huggingface.co/docs/peft/package_reference/lora)
- [LoRA paper](https://arxiv.org/abs/2106.09685)

See [`learning-resources.md`](reference/learning-resources.md) for annotated
descriptions of these sources and related evaluation references.
