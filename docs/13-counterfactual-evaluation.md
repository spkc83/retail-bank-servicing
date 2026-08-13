# Leakage-Controlled Counterfactual Evaluation

This guide defines the post-training benchmark used to test whether the pinned
Granite servicing model follows newly returned bank facts instead of replaying
facts or answers seen during SFT.

The benchmark is deliberately small and strict. It contains `18` records,
including `5` counterfactual pairs. It is a diagnostic acceptance set, not a
new training source.

## Why This Benchmark Exists

The released `1,374`-record suite is useful as a protocol regression test, but
it shares generator families, many response targets, and some POC facts with
training. See the [data-leakage audit](reference/data-leakage-audit.md).

The counterfactual suite changes the question being tested:

> Given an independently written prompt and a newly returned synthetic fact,
> does the fixed model select the right public tool and answer from that tool
> result rather than a remembered fixture?

This is evaluated against the immutable model
`spkc83/retail-bank-servicing-agent-9b` at revision
`1d56824995aa1adecfe20f62ca42fb1c0c443817` before any benchmark row is used
for remediation.

## Files and Ownership

| File | Responsibility |
| --- | --- |
| [`banking_counterfactual_eval_data.py`](../src/hello_slm/banking_counterfactual_eval_data.py) | Records, paired variants, contamination audit, manifest validation, and benchmark gates |
| [`prepare_counterfactual_eval.py`](../scripts/retail_bank/prepare_counterfactual_eval.py) | Deterministic preparation CLI |
| [`cloud_generate_tool_eval.py`](../scripts/retail_bank/cloud_generate_tool_eval.py) | Pinned model loading, two-phase generation, 4-bit local loading, scoring, and resumable outputs |
| [`test_banking_counterfactual_eval_data.py`](../tests/test_banking_counterfactual_eval_data.py) | Evaluation-only, leakage, pair, determinism, and gate regression tests |
| [`data/banking-counterfactual-eval-v1`](../data/banking-counterfactual-eval-v1) | Tracked manifest, preparation report, README, and `test.jsonl` |

## Record Structure by Example

The JSONL uses the same `banking-tool-sft/v1` conversation shape as the trusted
two-phase evaluator, with an additional evaluation-only metadata contract. The
following example is abbreviated:

```json
{
  "record_id": "cf_accounts-returned-facts_a",
  "messages": [
    {"role": "system", "content": "...", "loss": false},
    {"role": "user", "content": "Inventory the deposit accounts ...", "loss": false},
    {
      "role": "assistant",
      "content": null,
      "loss": true,
      "tool_calls": [{
        "id": "call_cf_accounts-returned-facts_a_0",
        "type": "function",
        "function": {"name": "list_accounts", "arguments": {}}
      }]
    },
    {
      "role": "tool",
      "name": "list_accounts",
      "content": {"ok": true, "result": {"accounts": [{"last4": "1014"}]}},
      "loss": false
    },
    {"role": "assistant", "content": "... ending in 1014 ...", "loss": true}
  ],
  "expected": {
    "requires_tool": true,
    "tool_calls": [{"name": "list_accounts", "arguments": {}}],
    "grounding_facts": ["account.last4=1014"],
    "forbidden_facts": ["1119"]
  },
  "metadata": {
    "split": "test",
    "trainable": false,
    "counterfactual_pair_id": "accounts-returned-facts",
    "counterfactual_variant": "a",
    "varied_facts": ["1014"]
  }
}
```

The real row includes all returned account fields and all facts required for
scoring.

## How a Counterfactual Pair Works

Each pair has variants `a` and `b`.

1. Both variants provide byte-equivalent messages before the first target tool
   call. The model cannot infer the variant from the prompt.
2. Both expect the same public tool name and arguments.
3. The canonical tool result changes names, last-four digits, amounts, dates,
   or statuses.
4. Each final answer must contain its own returned facts.
5. Each row explicitly forbids the other variant's facts.

For example, the accounts pair returns `1014` in variant `a` and `1119` in
variant `b`. A model that answers `1014` for both variants fails grounding on
variant `b`, even if its grammar and tool call are otherwise correct.

The record validator also rejects a pair if a changed fact is visible before
the canonical tool result. This prevents the user prompt from accidentally
giving away the expected answer.

## Contamination Controls

Preparation reads both SFT training files and the POC source/state as
read-only audit inputs. The build fails on:

- an exact normalized benchmark user turn in either SFT training stage;
- an exact benchmark final target in training;
- a reused training `template_id`;
- an exact POC prompt literal;
- a varied benchmark fact found in the POC state;
- a varied benchmark fact found in training, including the underlying integer
  cents representation of a formatted amount; or
- a nearest training-prompt sequence similarity of `0.90` or greater after a
  trigram shortlist.

The current preparation report records:

- `0` exact training user overlaps;
- `0` exact training final overlaps;
- `0` training template overlaps;
- `0` POC prompt overlaps;
- `0` POC fact overlaps;
- `0` training varied-fact overlaps; and
- maximum observed prompt similarity `0.561404`.

These controls substantially strengthen the test, but they do not prove that
the base Granite pretraining corpus never contained a phrase or generic banking
concept. That corpus is not locally enumerable. The claim is restricted to the
two project SFT stages and tracked POC fixtures.

## Evaluation-Only Enforcement

The benchmark has only a `test` split. Its manifest declares:

```json
{
  "contract": "banking-counterfactual-eval-manifest/v1",
  "training_allowed": false,
  "allowed_use": ["counterfactual-evaluation"]
}
```

The normal SFT manifest loader rejects this manifest contract and rejects any
individual record whose `metadata.trainable` value is `false`. Both gates must
be bypassed to train on this data accidentally.

Once a row has been inspected and used to design remediation data, that row is
no longer a clean final test for the remediated model. Preserve it only as a
regression case and author a new hidden acceptance set.

## Prepare and Validate

From the repository root:

```bash
uv run python scripts/retail_bank/prepare_counterfactual_eval.py
```

The command deterministically rewrites and validates:

- `data/banking-counterfactual-eval-v1/test.jsonl`;
- `data/banking-counterfactual-eval-v1/manifest.json`;
- `data/banking-counterfactual-eval-v1/preparation-report.json`; and
- `data/banking-counterfactual-eval-v1/README.md`.

Run the contract tests:

```bash
uv run pytest -q \
  tests/test_banking_counterfactual_eval_data.py \
  tests/test_banking_tool_eval_runner.py \
  tests/test_banking_tool_sft_worker.py
```

## Run the Pinned 9B Model on a 12GB NVIDIA GPU

The merged checkpoint is about `17.9 GB` in its published precision and does
not fit wholly in 12GB VRAM. `--load-in-4bit` loads linear weights through
bitsandbytes NF4 double quantization. Quantization happens at load time; this
command does not alter or save new model weights.

The TITAN V has compute capability `7.0`. Use a CUDA 12.6 PyTorch wheel that
contains `sm_70` kernels; CUDA 13 wheels may omit them. The script's inline UV
metadata pins `torch 2.12.1` to the official `cu126` index. On a preconfigured
environment, verify the stack before starting a 17.9GB download:

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.get_arch_list())'
python -m pip install bitsandbytes==0.50.0
```

First derive the exact local benchmark identity:

```bash
export COUNTERFACTUAL_DATASET_REVISION="sha256:$(sha256sum \
  data/banking-counterfactual-eval-v1/manifest.json | cut -d' ' -f1)"
```

Run two records as a hardware and tool-wire smoke test:

```bash
HF_XET_HIGH_PERFORMANCE=1 HF_XET_NUM_CONCURRENT_RANGE_GETS=32 \
uv run scripts/retail_bank/cloud_generate_tool_eval.py \
  --model-repo spkc83/retail-bank-servicing-agent-9b \
  --model-revision 1d56824995aa1adecfe20f62ca42fb1c0c443817 \
  --dataset-repo local/retail-bank-counterfactual-eval-v1 \
  --dataset-revision "$COUNTERFACTUAL_DATASET_REVISION" \
  --manifest data/banking-counterfactual-eval-v1/manifest.json \
  --output-dir artifacts/banking-counterfactual-eval-v1 \
  --split test \
  --family granite \
  --device cuda \
  --dtype fp16 \
  --load-in-4bit \
  --limit 2
```

If the smoke test succeeds, rerun the same command without `--limit`. The
runner resumes the prediction JSONL by `record_id`, so the first two rows are
not generated twice.

Hugging Face documents that NF4/FP4 quantization supports NVIDIA Pascal or
newer GPUs and that `BitsAndBytesConfig` is passed to `from_pretrained` to load
4-bit weights. See the official
[Transformers bitsandbytes guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

## What Proves the 9B Model Generated the Output

The evidence is the combination of immutable identity and raw generation
artifacts, not a UI label:

- `metadata-*.json` records the exact model repo and 40-character revision;
- `decode.weight_quantization` records `bitsandbytes-nf4-double` for the local
  run;
- `predictions-*.jsonl` stores every raw model pass, parsed assistant message,
  emitted tool call, appended canonical result, stop reason, and timestamp;
- prediction and report SHA-256 values are stored in metadata; and
- the read-only contract states that deterministic output repair is disabled
  and unseen assistant tool calls are never teacher-forced.

The evaluator appends a canonical result only after the model emits the exact
next tool call. It does not invent, rename, or repair a failed call. The final
natural-language response is another model generation pass after the result is
appended.

## Benchmark Gate

Because `18` rows are too few for a stable percentage tolerance, the v1 gate is
exact. Every applicable positive metric must score `1.0`, every error metric
must score `0.0`, and every paired row must pass both exact tool arguments and
grounded final factuality.

A failure is diagnostic rather than an instruction to copy the failing row
into training. Group failures into:

- tool selection or argument serialization;
- iterative orchestration or multi-tool order;
- result grounding or counterfactual fact substitution;
- clarification, FAQ, or OOD response path; and
- quantized-runtime-only differences.

If a 4-bit local failure is close to the gate, confirm it in the published
precision on larger hardware before attributing the failure to the SFT model.
If both fail, create new training examples from the behavior family while
keeping the acceptance prompts and facts out of training, then evaluate on a
new clean set.
