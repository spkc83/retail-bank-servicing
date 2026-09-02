---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- banking
- tool-calling
- synthetic
- conversational
- alignment
pretty_name: Retail Bank Servicing Alignment SFT
---

# Retail Bank Servicing Alignment SFT

The training corpus for the Granite retail-bank servicing agent. It is the
released tool-use SFT corpus merged with a servicing-alignment continuation
curriculum that teaches multi-turn behaviours the base corpus does not: what to
do when the customer says "that one", when a policy question interrupts a
transfer, when the agent's own previous turn was wrong, and when the honest
answer is that the agent cannot see what it was asked about.

Every conversation is self-authored synthetic data about a fictional bank. It
contains no real customers, credentials, accounts, or financial events.

## Splits

| Split | Rows | Alignment rows | Base rows | Allowed use |
| --- | ---: | ---: | ---: | --- |
| `train` | 3,959 | 3,118 | 841 | `granite-continuation-sft` |
| `validation` | 447 | 268 | 179 | `granite-continuation-evaluation` |
| `test` | 215 | 35 | 180 | `granite-continuation-evaluation` |

Three further files are **not trainable** and are declared as such in
`manifest.json`:

| File | Rows | Purpose |
| --- | ---: | --- |
| `coreference-shadow.jsonl` | 32 (16 pairs) | Post-selection behavioural gate; never used for checkpoint selection |
| `granite-v7-shadow.jsonl` | 13 | Checkpoint selection and generalization evaluation |
| `screenshot-regression.jsonl` | 9 | Regression fixture for the demo surface |

Loading the dataset with `datasets` gives you the three splits. The gate files
are plain JSONL beside them and are deliberately not exposed as splits, so a
naive `load_dataset` cannot train on them.

## What the alignment curriculum teaches

Counts are train rows. The largest families are the deictic-coreference pairs,
where the same history is followed by a turn that either resolves to one card or
is genuinely ambiguous, and the model must act in the first case and ask in the
second.

| Family | Rows | Behaviour |
| --- | ---: | --- |
| `deictic_replace_action` / `_ambiguity` | 784 / 784 | Act on a resolved referent; ask when two candidates fit |
| `long_context_tool_fidelity` | 200 | Keep the tool contract when a long history ends in a misleading turn |
| `history_entity_action` / `_ambiguity` | 128 / 32 | Resolve an entity named earlier in the conversation |
| `no_evidence_honesty` | 112 | Do not claim a fact the tools did not return |
| `capability_boundary`, `credential_hygiene`, `scope_refusal` | 84 each | Decline what the agent cannot or must not do |
| `deictic_*_clarification` | 72 each | Clarify when the referent is ineligible or missing |
| `agent_repair` | 64 | Recover when the agent's own previous turn was wrong |
| `card_anaphora_action`, `clarification_answer`, `service_case_context` | 64 each | Carry a referent across turns |
| `deictic_replace_reinforcement_*` | 64 each | The same mapping over a wider entity pool |
| `banking_topic_shift`, `external_topic_shift`, `policy_detour`, `policy_resume` | 32 each | Leave and return to a servicing task |
| `natural_social_style` | 12 | Answer a greeting without starting a transaction |

## Provenance

Rows are generated deterministically, then two teacher passes rewrite wording
only:

| Pass | Rows moved | Field |
| --- | ---: | --- |
| Finals realization | 651 train + 168 validation | `final_response` |
| Prompt realization | 366 train + validation | `user_content` |

Both are safe by construction. An immutable record hash covers the tool calls,
the tool results, the grounding facts, and the split keys, and deliberately
excludes the two wording fields — so a teacher can rephrase a question or an
answer but can never edit what makes a row's supervision correct. A pass that
tried would fail the hash check at build time. Each pass records the model it
credits in every touched row's `provenance`.

The prompt pass exists because the evaluation splits had become the training
questions with a different trailing phrase. Rows whose prompt moved are held to
an isolation invariant: a rewritten prompt may not share a 4-gram with any test,
shadow, or regression prompt.

## Build-time gates

Generation fails rather than reports. The build asserts, among others:

- **PII**: zero matches for emails, SSNs, and long digit runs across *every*
  message, not only the trainable pair (`pii_matches: 0`).
- **Held-out isolation**: no test-split current turn appears in train, and no
  long n-gram from a held-out row leaks into train (both empty).
- **Trainable-text word ban**: train and validation may not say `app`, `demo`,
  `synthetic`, `sandbox`, `test`, and similar — a model should not learn to talk
  about a product surface it cannot see. The frozen evaluation splits are exempt
  by split so their fixtures stay byte-identical.
- **Coreference pairing**: a normalized current turn may group multiple pairs
  only across distinct history forms, so a pair cannot be won by memorising the
  question.

## Contamination

Measured two ways, because the obvious one misleads on a domain corpus:

| Split | Median nearest-train similarity | ≥0.95 | ≥0.90 |
| --- | ---: | ---: | ---: |
| `test` | 0.843 | 0 | 8 |
| `validation` | 0.789 | 0 | 34 |

Raw 4-gram overlap over-reports badly here: a banking corpus shares its own
vocabulary ("my card ending in", "freeze the active card"), and counting that as
contamination would argue for training a model to avoid the nouns of its own
domain. Nearest-neighbour similarity separates the cases — a test question that
differs from a training question only in a trailing phrase scores above 0.95,
and no row now does.

## Reproducing it

The corpus is generated, not collected. See
[the generator](https://github.com/spkc83/retail-bank-servicing) —
`scripts/retail_bank/prepare_servicing_alignment_data.py` with the two teacher
files reproduces this revision byte for byte, and `manifest.json` carries the
SHA-256 of every file plus the digests of the base corpus, the synthetic bank,
and the policy corpus it was built from.

## Intended use and limits

This is a research demonstration of governed synthetic SFT data, not a banking
product. The behaviours it teaches are enforced at the data layer and gated at
training time; nothing here is a substitute for the safety architecture of the
serving harness, which carries the guarantees this corpus can only bias toward.
