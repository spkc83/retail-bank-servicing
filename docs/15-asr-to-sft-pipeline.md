# ASR Output to Granite Fine-Tuning Data

This document explains how reviewed speech-to-text output becomes additional
Granite tool-use SFT data without turning transcription mistakes into banking
actions. The implementation is
[`banking_asr_sft_data.py`](../src/hello_slm/banking_asr_sft_data.py); the CLI
entry point is
[`prepare_asr_sft_data.py`](../scripts/retail_bank/prepare_asr_sft_data.py).

## The core rule

An ASR transcript is evidence of how a customer phrase may sound after speech
recognition. It is not evidence of the correct tool call or assistant answer.

The pipeline therefore uses an **overlay**:

```text
validated semantic SFT record
  + reviewed ASR customer utterance with the same meaning
  -> replace only the latest user-message text
  -> preserve tool calls, tool results, expected facts, and answer content
  -> inherit the source split and split group
  -> write trainer-compatible messages JSONL
```

This separates two questions:

1. What should the assistant do? The validated source SFT record answers this.
2. What noisy text might the model receive from speech recognition? The ASR
   overlay answers this.

Raw call transcripts are not automatically treated as labels. Agent speech,
unreviewed tool guesses, and low-quality summaries do not become assistant
targets.

## Input contract

The CLI accepts JSON or JSONL. Each object describes one reviewed customer
utterance and references one existing source record.

```json
{
  "record_id": "asr-transfer-cancel-0042",
  "source_record_id": "transfer_cancel_state_0042_realization_1",
  "utterance_id": "utt-call-2026-0042-07",
  "recording_id": "call-2026-0042",
  "speaker": "customer",
  "transcript": "can you cancel the river consulting transfer uh please",
  "start_ms": 8120,
  "end_ms": 12140,
  "language": "en-US",
  "confidence": 0.84,
  "alternatives": [
    {
      "text": "can you cancel the river consulting transfer please",
      "confidence": 0.12
    }
  ],
  "asr_model_id": "organization/asr-model",
  "asr_model_revision": "immutable-model-revision",
  "audio_sha256": "64-lowercase-hexadecimal-characters",
  "review": {
    "semantic_match": true,
    "pii_reviewed": true,
    "consent_for_training": true,
    "reviewer": "review-batch-2026-08",
    "license": "approved-training-license"
  }
}
```

### Why each field exists

| Field | Purpose |
| --- | --- |
| `source_record_id` | Binds the noisy utterance to an already-validated semantic behavior. |
| `recording_id` / `utterance_id` | Trace the row back to its source segment and prevent duplicates. |
| `speaker=customer` | Prevents agent speech from becoming a user-message input. |
| `start_ms` / `end_ms` | Preserves the ASR segment boundary and catches malformed segments. |
| `confidence` / `alternatives` | Preserves recognition uncertainty for later analysis; alternatives are metadata, not extra training messages. |
| model ID and revision | Makes ASR behavior reproducible instead of depending on a moving model branch. |
| `audio_sha256` | Detects source-audio drift without storing audio in the SFT record. |
| review flags | Make semantic equivalence, PII review, consent, and licensing explicit gates. |

The repository example is
[`examples/asr/asr-utterances.jsonl`](../examples/asr/asr-utterances.jsonl).
Its recordings and hashes are synthetic.

## What the output looks like

Suppose the source record contains:

```json
{
  "messages": [
    {"role": "user", "content": "Cancel the River Consulting transfer."},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "function": {
            "name": "cancel_transfer",
            "arguments": {"recipient": "River Consulting"}
          }
        }
      ]
    },
    {"role": "tool", "content": {"ok": true, "result": {"transfer": {"status": "cancelled"}}}},
    {"role": "assistant", "content": "The River Consulting transfer is now cancelled."}
  ]
}
```

The generated overlay changes the latest user message to the reviewed ASR
text. It leaves the assistant tool call, tool result, final-answer content,
expected facts, and split keys unchanged. During training, the chat template
consumes the `messages` array exactly like the existing SFT dataset.

For a read-only list trajectory, loss remains on the assistant tool call but is
disabled on the final prose answer because serving renders the exact list as a
table. This makes the ASR overlay a speech-robustness/tool-selection target for
that path. Direct answers and action answers remain model-authored trainable
targets.

The generated record additionally contains:

- `schema_version=banking-asr-tool-sft/v1`;
- provenance linking the source record, recording, utterance, model revision,
  audio hash, license, and reviewer;
- a digest of the immutable source semantics;
- ASR confidence, alternatives, language, speaker, and timestamps in metadata;
- validation flags proving the row passed the required review gates.

## Failure policy

Preparation stops with an error when any row is unsafe or ambiguous. It does
not silently skip rejected records. The current gates reject:

- unknown or non-trainable source records;
- source records that were not previously accepted by their validator;
- duplicate record IDs, utterance IDs, or normalized transcripts;
- empty text, malformed JSON, invalid confidence, or invalid timestamps;
- non-customer speakers;
- malformed audio hashes or unpinned ASR identity;
- missing semantic, PII, consent, reviewer, or license approval;
- email addresses, Social Security number patterns, or long payment/account
  number patterns in the transcript or alternatives;
- source split files whose declared SHA-256 does not match their content.

`semantic_match=true` means a reviewer verified that the ASR text still
supports the source behavior. For example, if ASR changes “River Consulting”
to “Jamie Lee,” the row must not inherit the River Consulting cancellation
target. Correct the transcript from the audio, link it to a different semantic
record, or author a clarification scenario.

## Split and leakage behavior

An ASR row always inherits the source record's `train`, `validation`, or `test`
assignment and its `split_group`. The pipeline never hashes the utterance into
a new split. That prevents an ASR paraphrase of a validation or test scenario
from leaking into training.

Multiple ASR realizations may reference the same source record, but all remain
in the same split. Keep speaker/session families together in the source data
when building a larger corpus.

## Run the pipeline

From the repository root:

```bash
python scripts/retail_bank/prepare_asr_sft_data.py \
  --asr-input examples/asr/asr-utterances.jsonl \
  --base-manifest data/banking-servicing-alignment-v4/manifest.json \
  --output-dir /tmp/retail-bank-asr-sft
```

The output directory contains:

```text
train.jsonl
validation.jsonl
test.jsonl
manifest.json
preparation-report.json
```

All three splits must contain at least one reviewed ASR overlay. Preparation
fails before writing output when train, validation, or test coverage is empty.

The manifest uses `tool_sft` entries, so the existing Granite training worker
can read it with its real `--manifest /path/to/manifest.json` argument. The row
schema is new, but the trainer-facing `messages` field is unchanged.

Before any training run, inspect the preparation report, sample every scenario
family and confidence band, and run the tests below. The command itself starts
no GPU job and publishes nothing.

```bash
pytest -q tests/test_banking_asr_sft_data.py
```

## How to scale beyond the example

For a production-sized speech robustness corpus:

1. Capture or synthesize consented audio against an approved scenario set.
2. Run the chosen ASR model and pin its exact revision.
3. Normalize model-specific output into the input contract above.
4. Review PII and semantic equivalence against the referenced source record.
5. Include realistic disfluencies, accents, background noise, truncation, and
   recognition alternatives without changing the intended banking entity.
6. Stratify reports by ASR model, language, confidence band, noise condition,
   capability, and conversation relation.
7. Keep clean text and ASR variants in the same split group.
8. Fine-tune incrementally, then compare clean-text, ASR-text, tool-execution,
   grounding, and counterfactual evaluations before release.

Do not optimize only for average word-error rate. A single corrupted recipient,
amount, negation, or card selector can be more important than several harmless
filler-word errors.

## Relationship to the inference harness

The harness improvements and ASR data solve different failure classes:

- ASR SFT teaches Granite to select the correct existing tool behavior from
  speech-like text.
- deterministic read rendering prevents the model from dropping or changing
  table facts after a successful list tool.
- grounded action validation checks essential selectors and outcomes, then
  permits one text-only repair pass with tools disabled.
- complete prior interaction groups preserve tool calls and correlated results
  while they fit the model context budget; browser transcript data is never
  promoted into a trusted system-memory block.

The harness does not repair malformed tool calls or change arguments. ASR
training data must therefore preserve entity and action semantics before it is
accepted.
