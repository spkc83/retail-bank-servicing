# Use-Case Coverage

A corpus can pass every mechanism gate — PII, leakage, digests, provenance —
and still be missing what customers actually say. Those are different
questions. The mechanism gates ask *does the generator do what it claims?*;
this page asks *is each thing a customer brings represented, in the shape they
bring it?*

The shape matters as much as the intent. A router trained on "Show my account
balances." and "Could you pull up my accounts?" has not learned "What is my
balance?" — the first two are commands, the third is a question about an
amount, and a lane head that has never seen the third routes it wherever
questions usually go.

## The matrix

[`banking_corpus_coverage.py`](../src/hello_slm/banking_corpus_coverage.py)
classifies every row along two axes.

**Categories**, multi-label, read from the row's own metadata wherever it
exists and from wording only where nothing better does:

| category | router corpus reads | alignment corpus reads |
| --- | --- | --- |
| `in_domain` / `social` / `out_of_domain` | `domain_name` | `metadata.path`, family |
| `first_turn` / `multi_turn` / `long_running` | `history` length (≥6 is long) | user-turn count |
| `counterfactual` | `counterfactual_pair_id` | `deictic_*_action` / `_ambiguity` families |
| `policy_question` | intent `policy_knowledge` | policy path, `faq_*` families |
| `intent_drift` / `loop_back` / `agent_repair` / `clarification_answer` | relation labels | family name |
| `adversarial` | wording: instruction override, credential extraction | `credential_hygiene`, `hard_negative_private_id` |
| `multi_intent` | wording: two verb-plus-object clauses joined | same |

**Phrasing form**, one per row: `imperative`, `wh_question` ("What is my
balance?"), `modal_request` ("Could you pull up my accounts?"), `elliptical`
("Balance?"), `deictic` ("Freeze that one."). `wh_question` and
`modal_request` are kept apart deliberately: the router handles modal requests
as the polite imperatives they are, so the real question is the form a corpus
built from commands is most likely to under-supply.

A **cell** is `(intent, form, first_turn | multi_turn)`. Every detector has a
test that plants a row and proves it fires, and a negative that proves it stays
quiet — a coverage gate whose detectors cannot fire certifies nothing.

## The declared expectation

[`configs/corpus-coverage.toml`](../configs/corpus-coverage.toml) names the
cells that matter and gives each a `minimum` and a `target`. `minimum` is a
ratchet: `make verify` fails if the corpus drops below it. `target` is the
goal; a cell below target is reported, worst first, and that ranking is the
authoring order. A cell that is empty today is declared with minimum 0 and a
real target — declaring it is the point.

```bash
make coverage          # both corpora, full matrix and the ranked shortfalls
PYTHONPATH=src uv run python scripts/retail_bank/measure_corpus_coverage.py \
  --corpus router --gate     # what make verify runs
```

Raise a minimum only after the rows exist *and* the router has been retrained
on them; a minimum describes what the shipped artifact learned from.

## What the router corpus contains today

21,686 training rows in `data/banking-conversation-router-v9-surface-form`. The
corpus is built around **transitions** — 45% of rows carry `topic_shift`, 74%
are multi-turn — so the plain first ask stays the thinner side of the corpus
even where it is covered.

| category | rows | share |
| --- | ---: | ---: |
| in_domain / social / out_of_domain | 11,440 / 1,978 / 8,268 | 53% / 9% / 38% |
| first_turn / multi_turn | 5,646 / 16,040 | 26% / 74% |
| long_running (≥6 turns) | 32 | 0.1% |
| counterfactual | 1,568 | 7.2% |
| policy_question | 1,202 | 5.5% |
| intent_drift / loop_back | 9,746 / 360 | 45% / 1.7% |
| agent_repair / clarification_answer | 1,103 / 571 | 5.1% / 2.6% |
| **adversarial** | **0** | — |
| **multi_intent** | **0** | — |

First-turn rows per servicing intent, by form:

| intent | imperative | wh_question | modal_request | deictic |
| --- | ---: | ---: | ---: | ---: |
| view_accounts | 10 | 15 | 51 | 0 |
| view_cards | 24 | 21 | 51 | 0 |
| view_transactions | 35 | 21 | 54 | 0 |
| view_transfers | 2 | 13 | 55 | 10 |
| view_service_cases | 22 | 50 | 49 | 0 |
| freeze_card | 49 | 22 | 47 | 0 |
| replace_card | 110 | 19 | 58 | 8 |
| dispute_transaction | 35 | 5 | 72 | 5 |
| cancel_transfer | 90 | 5 | 65 | 17 |

Every servicing intent now has first-turn questions and modal requests, which
is what the phrasing family in the router derivation supplies. The `wh_question`
column is still the thinnest across the reads, and `dispute_transaction` and
`cancel_transfer` sit at five rows each, so the class is covered rather than
saturated. The imperative column is uneven for the same reason it always was:
the demo presets are all imperative and the reads inherited whatever the
transition curricula happened to produce.

The two cells with the largest absolute shortfall are the deictic follow-ups
for `cancel_transfer` (19) and `dispute_transaction` (31), against 2,005 for
`replace_card`: the counterfactual pairs are concentrated on cards.

## What the alignment corpus contains today

3,959 training rows. It is better balanced on categories the router corpus
lacks — 116 adversarial rows (`credential_hygiene`, `hard_negative_private_id`),
200 long-running, 43% counterfactual — and shares the router corpus's two
absences: **multi-intent is 0**, and question-form first asks are thin.

## Authoring order

The report ranks every declared cell below its target, and that ranking is the
authoring order:

1. deictic follow-ups for `cancel_transfer` and `dispute_transaction`, the two
   largest shortfalls;
2. first-turn rows for the reads, imperative and `wh_question` alike, all of
   which sit well under a 60-row target;
3. `wh_question × first_turn` for `dispute_transaction` and `cancel_transfer`,
   at five rows each;
4. adversarial turns in the **router** corpus — the router has no notion of an
   instruction-override or credential request and routes one as whatever it
   superficially resembles;
5. multi-intent turns, absent from both corpora;
6. long-running conversations beyond six turns.

New rows for the router corpus derive from the alignment corpus through
`prepare_conversation_router_data.py`, so the first three items are authored
there. The first-turn cells are filled by a hand-written phrasing family in the
router derivation, described in
[Data generation](02-data-generation.md#derivation-guards). Coverage is
measured against the release corpus, and the minimums for those cells are set
high enough that they cannot silently empty again.
