# Who Should Decide the Turn?

Two components could classify a customer turn: the V6 seven-head DistilBERT
cross-encoder, or the fine-tuned 9B itself. This page is the experiment that
compares them, and the reason it is shaped the way it is.

## The comparison has to hold the guards fixed

The tempting version of this experiment — "run the harness without a router" —
is confounded and unsafe. Removing the router also removes single-schema
exposure, entity gating and turn guidance, so it changes *who decides* and
*what is enforced* at the same time. If the un-routed configuration looks worse,
nothing tells you whether that was the classifier or the missing guards. It is
also how a public demo ends up exposing all four mutations
(see [P0 #2](superpowers/plans/2026-08-29-improvement-register.md)).

So [`model_router.py`](../poc/retail-bank-customer-service-poc/model_router.py)
produces the **same decision shape** the learned router produces, and
everything downstream runs unchanged. A test pins the property the experiment
rests on: for an identical tuple, both classifiers reach an identical tool
surface.

Two rules keep it fair rather than flattering:

- **Unknown means rejected.** A label outside the taxonomy is a parse failure,
  never snapped to the nearest legal value. Coercion would repair the model's
  mistakes and count them as successes.
- **Illegal tuples are corrected and *reported*.** The learned router cannot
  emit one — its joint decoder enumerates the legal tuples — so the same
  legality is applied to the model's proposal, with every correction recorded
  in `constraint_diagnostics`. "How often did the classifier propose something
  illegal" stays a measurable difference instead of a hidden repair.

## Three configurations, not two

| mode | who decides | tools exposed |
| --- | --- | --- |
| `router` (default) | seven-head cross-encoder | one intent-compatible schema |
| `model` | the SLM, structured pass | one intent-compatible schema, same guards |
| unrouted | nobody | all nine, including four mutations |

The third is not a deployment option; it is the V3 legacy surface and the
baseline that measures what the routing layer contributes. It is opt-in
(`tool_authority: "unrouted"`) precisely so it cannot be reached by accident.

`RETAIL_BANK_ROUTING_MODE=router|model` sets the deployment default, and the
Gradio surface carries a radio — **Who decides the turn?** — that overrides it
per request, so the same session can ask the same question both ways. An
unrecognised value, from either source, resolves to the router and never to
anything unrouted. Technical details then report `Classifier` and, in model
mode, the tuple the model proposed *before* the legality check beside the one
the harness acted on.

The radio costs one extra generation per turn in model mode. That is
autoregressive work — the model writes its decision one token at a time, each
token a full 9B forward pass — against a single classification pass through a
66M-parameter encoder, which is where the two differ in cost. On the Space
that pass runs inside the same `@spaces.GPU(duration=90)` wall as the rest of
the turn.

## Scoring

[`compare_routing_classifiers.py`](../scripts/retail_bank/compare_routing_classifiers.py)
scores both arms against the router's own held-out split — 4,921 test rows
carrying gold `domain`, `action`, `entity_resolution` and `intent` beside the
turn and its history.

Each head is graded only over rows that carry it, and each is read the way
`_generation_plan` reads it. Both details matter: 2,785 rows carry no gold
intent, and the learned router reports intent under `intent` rather than
`fine_intent`. Getting either wrong produces a plausible, wrong number — the
first draft of this script scored the shipped router at 0.558 and then at 0.0
on intent before both were fixed.

```bash
PYTHONPATH=src uv run python scripts/retail_bank/compare_routing_classifiers.py \
  --arm router --limit 800
```

## Results

**Learned router**, 800 sampled rows, seed 711:

| metric | value |
| --- | ---: |
| exact tuple accuracy | 0.9187 |
| domain | 0.9487 |
| action | 0.9313 |
| entity resolution | 0.9725 |
| intent (347 labelled rows) | 0.8588 |
| seconds per turn | 0.084 (CPU) |

**SLM as classifier:** not yet measured at a publishable sample size. A five-row
smoke on the local TITAN V returned 8.2 s/turn with four of five turns
producing no usable routing object. That latency is a local, NF4-quantized,
Volta-era number and should not be read across to the Space, which serves
bf16 on newer hardware; the ratio to the cross-encoder is real, the absolute
figure is not portable.
That direction is unsurprising: v14 was fine-tuned to *be* a servicing agent,
not to emit routing labels, and nothing in its curriculum teaches the JSON
contract this asks for. **Five rows is an anecdote, not a result**, and the raw
outputs have not yet been read closely enough to rule out the prompt or the
parser as the cause. Treat the number as a reason to run the real sample, not
as the finding.

## What this does not measure

Downstream answer quality. Both arms feed the same harness, so this compares
routing decisions, not the responses that follow them. A classifier that routes
correctly and a model that then answers badly would look identical here.
