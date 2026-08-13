# Granite SFT Data-Leakage Audit

Audit date: `2026-07-31`

Status: **the strict leakage-free claim is not supported**.

The released data has useful split protections, but the frozen evaluation and
live POC are not independent of the SFT data. Current results cannot rule out
memorization as one contributor to the 9B model's performance.

This finding does not prove that the model memorized every answer. It means the
current evidence cannot distinguish memorization from generalization strongly
enough to claim a clean, independent test.

## Claim Tested

The falsifiable claim was:

> No Stage-1 or Stage-2 training record contains POC presets, POC test data, or
> frozen evaluation examples in a way that could let the model pass by
> memorizing prompts, facts, tool targets, or final answers.

The claim was tested against the local release-shaped JSONL, data generators,
POC source, POC synthetic state, and two-phase evaluator.

The audit did not redownload the immutable Hub revisions. The local manifests,
locks, model card, and release config identify the corresponding release
artifacts.

## What Is Correctly Isolated

The audit found these protections:

- no `record_id` overlap between train, validation, and test;
- no `split_group` overlap between Stage-1 train and test;
- no Stage-1 `state_seed` or `customer_id` overlap between train and test;
- no exact full-conversation signature overlap between train and test; and
- exact screenshot current-turn strings are excluded from Stage-2 training.

The last rule is implemented by `SCREENSHOT_HELDOUT_CURRENTS` in
[`banking_servicing_alignment_data.py`](../../src/hello_slm/banking_servicing_alignment_data.py).

These checks prevent copying an entire record or state group into two splits.
They do not prevent shared templates, target text, backend facts, or POC assets.

## Findings That Refute the Strict Claim

### 1. A POC preset appears in training

The POC exposes this preset:

```text
yo, sup?
```

Stage-1 and Stage-2 training contain:

```text
yo sup
```

After lowercase and punctuation normalization, these strings are identical.
The training record is `small_talk_greeting`.

This is a small-talk case rather than a banking tool case, but it is still a
direct preset-to-training overlap.

### 2. POC prompts have close generated matches

Several other presets are near duplicates of training prompts:

| POC preset | Closest training current | Sequence similarity |
| --- | --- | ---: |
| `My card was stolen. Freeze it.` | `Please my card was stolen freeze it` | `0.889` |
| `Please replace my debit card.` | `Please replace my card` | `0.880` |
| `Can you help me open a mortgage account?` | `Would you can you help me open a mortgage account` | `0.886` |
| `Show my five most recent transactions.` | `Please show my three most recent transactions` | `0.829` |

Similarity is from Python `difflib.SequenceMatcher` after lowercase and
punctuation normalization. It is a diagnostic, not a semantic-leakage proof.

The examples show that the UI presets exercise the same deterministic language
patterns used to generate SFT records. They are not an independently authored
human acceptance set.

### 3. User turns repeat across train and frozen test

Three unique normalized user strings occur in both Stage-2 train and test:

- `My wallet was stolen. Find my active debit card and freeze it.`
- `Replace my card.`
- `I need to dispute a debit card charge.`

Those shared turns affect `147` of the `1,374` test records because the same
turn is reused with multiple state and realization variants.

The complete conversations are not identical. The repeated visible user turns
still reduce the strength of a claim that test language was unseen.

### 4. Every base template family crosses splits

Stage-1 split assignment keeps a specific combination of scenario family,
state, customer, and template together.

However, all `27` base `template_id` values and all `26` scenario families
appear across train, validation, and test. Realization patterns also cross
splits.

One audited train/test pair from the same realization pattern had normalized
sequence similarity `0.965`; only the leading request phrase and synthetic
state differed.

This is an interpolation test over a shared generator, not a template-family
holdout.

### 5. Stage-2 remediation uses the POC backend facts

The Stage-2 alignment generator uses customer `alex.demo` and hard-coded POC
facts such as:

- active card `4821`;
- `Everyday Visa Debit`;
- service case `Confirm mailing address update`;
- creation time `2026-06-18T14:00:00Z`; and
- merchant `North Harbor Market`.

The same facts appear in
[`synthetic_bank.json`](../../poc/retail-bank-customer-service-poc/synthetic_bank.json).
The Stage-2 manifest also records that file's SHA-256 identity.

Therefore, a live conversation against `alex.demo` does not test whether the
model can ground an answer in previously unseen backend values.

### 6. Stage-2 train and test targets are mostly identical

The remediation-only portion has:

| Split | Records |
| --- | ---: |
| Train | `320` |
| Validation | `80` |
| Test | `27` |

For `26` of the `27` remediation test rows, an exact combination of expected
tool call, canonical tool result, and final answer already occurs in remediation
training.

Across the complete `1,374`-record test split, `894` final-answer strings occur
exactly in training. Many are intentionally fixed FAQ, clarification, error,
or stock responses.

Exact target reuse makes the gate useful for protocol regression, but weaker as
evidence of response generalization.

### 7. Evaluation replays dataset tool results

The two-phase evaluator does not execute a newly randomized backend.

After an exact expected tool call, it appends the record's canonical tool result
and asks the model for the final response. This is deterministic and reproducible,
but the Stage-2 remediation results contain the same facts used in training.

The behavior is implemented in
[`cloud_generate_tool_eval.py`](../../scripts/retail_bank/cloud_generate_tool_eval.py).

## What the Existing Score Does Prove

The `1,374`-record score shows that the released checkpoint can reproduce the
expected protocol on the released synthetic generator distribution:

- valid Granite tagged-JSON tool calls;
- correct public argument shapes;
- required multi-tool order;
- responses containing expected replayed facts;
- expected FAQ, clarification, and OOD behavior; and
- no scored malformed or private arguments.

The base `1,347` test records use disjoint state groups and no exact full
conversation duplicates. That is meaningful evidence beyond a byte-for-byte
record copy.

The score does **not** establish performance on independently authored prompts,
unseen template families, unseen POC customer state, or a live randomized
backend.

In this project, `frozen` means the test artifact was fixed during a given
evaluation. It does not mean the test is contamination-free.

## Impact on Live Human Testing

The live POC uses the same synthetic customer facts that appear in Stage-2 SFT.
Human rewording helps test conversational flexibility, but unchanged backend
entities and outcomes leave a memorization path open.

A convincing live test must change both language and state. For example, a new
account could expose an active card ending in `7365`. A correct answer must use
`7365` from the tool result and never fall back to memorized `4821`.

## Required Clean Evaluation

Before claiming leakage-free generalization, create an evaluation set after the
training corpus is frozen with these properties:

1. Independently author prompts; do not reuse generator templates or POC presets.
2. Hold out complete template and scenario subfamilies, not only state groups.
3. Generate new customer states with unseen names, last-four values, amounts,
   dates, merchants, cases, and tool outcomes.
4. Execute or replay results from the new state, never the training fixture.
5. Include counterfactual pairs where only a returned fact changes.
6. Fail if the response repeats a training fact instead of the new tool fact.
7. Keep the acceptance set unavailable to data-generation code and remediation
   authors until the checkpoint is fixed.
8. Report clean metrics separately from in-generator regression metrics.

The existing model can be evaluated on this new set before retraining. A pass
would supply stronger evidence of generalization.

This recommendation is now implemented as the evaluation-only
[`banking-counterfactual-eval-v1`](../../data/banking-counterfactual-eval-v1)
suite and documented in
[`13-counterfactual-evaluation.md`](../13-counterfactual-evaluation.md). Its
preparation audit passes against both local SFT training stages and the POC
source/state. The benchmark result must remain separate from the released
`1,374`-record regression score; preparation success alone is not a model pass.

If it fails, revise the data pipeline and retrain. Do not add the new acceptance
rows to the next test set after using them as remediation training data.

## Recommended Automated Gates

A future data build should fail on:

- normalized POC preset or test-prompt overlap;
- exact train/test current-turn overlap;
- train/test final-target overlap above an explicit per-slice allowance;
- cross-split `template_id` or realization-pattern overlap for clean slices;
- POC synthetic entity values inside clean evaluation data; and
- high n-gram or embedding similarity above a reviewed threshold.

Fixed stock responses may need an allowlist. Any allowlisted overlap should be
reported separately instead of silently counted as clean generalization.

## Audit Conclusion

Do not describe the current model or `1,374`-record gate as leakage-free.

The accurate claim is:

> Exact record and state groups are isolated, and the model passed a frozen
> in-generator protocol regression suite. POC facts, template families, and
> many targets are shared with training, so independent generalization remains
> unproven.

This audit covers the Granite SFT model. The conversation router has separate
data and evaluation contracts and needs its own contamination audit before a
similar claim is made for classifier performance.
