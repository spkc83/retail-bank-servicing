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
as the polite imperatives they are, and it is the real question that it had
never seen.

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

20,439 training rows. The corpus is built around **transitions** — 45% of rows
carry `topic_shift`, 75% are multi-turn — and the plain first ask is the
neglected case across every intent.

| category | rows | share |
| --- | ---: | ---: |
| in_domain / social / out_of_domain | 10,408 / 1,986 / 8,045 | 51% / 10% / 39% |
| first_turn / multi_turn | 5,092 / 15,347 | 25% / 75% |
| long_running (≥6 turns) | 32 | 0.2% |
| counterfactual | 1,344 | 6.6% |
| policy_question | 1,152 | 5.6% |
| intent_drift / loop_back | 9,276 / 360 | 45% / 1.8% |
| agent_repair / clarification_answer | 1,100 / 546 | 5.4% / 2.7% |
| **adversarial** | **0** | — |
| **multi_intent** | **0** | — |

First-turn rows per servicing intent, by form:

| intent | imperative | wh_question | modal_request | deictic |
| --- | ---: | ---: | ---: | ---: |
| view_accounts | 14 | **0** | 10 | 0 |
| view_cards | 33 | **0** | 11 | 0 |
| view_transactions | 41 | **0** | 18 | 0 |
| view_transfers | 16 | **0** | 12 | 0 |
| view_service_cases | 28 | 32 | 9 | 0 |
| freeze_card | 58 | **0** | 14 | 0 |
| replace_card | 114 | **0** | 26 | 8 |
| dispute_transaction | 41 | 4 | 22 | 3 |
| cancel_transfer | 93 | 2 | 28 | 8 |

Read the `wh_question` column. Six servicing intents have **no** first-turn
question at all; only `view_service_cases` has a meaningful number. The
question a human types most — "What is my balance?" — is one instance of a
class that is absent for the whole servicing lane. The demo presets never
showed it because every preset is imperative.

The two cells with the largest absolute shortfall are the deictic follow-ups
for `cancel_transfer` (18) and `dispute_transaction` (29), against 1,685 and
162 for `replace_card` and `freeze_card`: the counterfactual pairs are
concentrated on cards.

## What the alignment corpus contains today

3,959 training rows. It is better balanced on categories the router corpus
lacks — 116 adversarial rows (`credential_hygiene`, `hard_negative_private_id`),
200 long-running, 43% counterfactual — and shares the router corpus's two
absences: **multi-intent is 0**, and question-form first asks are thin.

## Authoring order

The report ranks every declared cell below its target. As of this
measurement, in order:

1. `wh_question × first_turn` for all six empty servicing intents — the class
   that shipped;
2. deictic follow-ups for `cancel_transfer` and `dispute_transaction`;
3. `modal_request × first_turn` for the reads;
4. adversarial turns in the **router** corpus — today the router has no notion
   of an instruction-override or credential request and routes it as whatever
   it superficially resembles;
5. multi-intent turns, absent from both corpora;
6. long-running conversations beyond six turns.

New rows for the router corpus derive from the alignment corpus through
`prepare_conversation_router_data.py`, so the first three items are authored
there. The router cannot simply be rebuilt on the result: at HEAD a rebuild
fails five release gates for unrelated reasons recorded in the
[runbook](08-end-to-end-runbook.md), and that has to be resolved first.
