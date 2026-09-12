# Rotating the frozen test splits

**Goal:** remove 31 template-mangled prompts from the two frozen evaluation
splits without destroying the evidence for any number already published.

**Status:** design, revised after a failed execution attempt on 2026-09-12.
The mechanical version does not work and the reasons are recorded below. What
remains is a bounded authoring task plus one priced decision, both stated below.

## What is wrong with the fixtures

`data/banking-v5-tool-sft/test.jsonl` (180 rows) and
`data/banking-servicing-alignment-v5/test.jsonl` (215 rows) each carry 31 user
turns of a shape no customer types: a request opener stacked on a sentence that
was already a question.

> Can you what information is needed for a card dispute
>
> Help me how do I safely report card fraud so I can finish this banking task
>
> Please what do banks usually require for a new account

They are an artefact of a realizer template that stacked
`REALIZER_OPENERS` onto a stem without checking whether the stem was a
question. The template was retired on 2026-08-20 and the training splits were
regenerated without it. The test splits were not, because they are frozen.

Counting them precisely, because the number has been quoted three ways. The
strict filter in `banking_conversation_router_data.is_retired_realizer_prompt`
matches **28** rows in each split. Three more rows match the same defect
through a looser shape that begins "Please can you ..." and were deliberately
excluded from the filter, since "Please can you freeze my card" is ordinary
English. **31** is the total, and it is 14% of the alignment test split. Only 28 of them
are defective; see the rotation scope below.

Two consequences, both already paid for:

- The router's in-domain false-refusal gate was scoring a retired template
  rather than the router. That is the 2026-09-03 rebuild failure, fixed on the
  derivation side by excluding these rows from the router corpus.
- Any model scored on these splits is scored partly on prompts it will never
  see in service.

## What must not break

| Invariant | Why | Where it is enforced |
| --- | --- | --- |
| Every number already published stays reproducible | Model cards, the register and the release ledger quote scores measured on these exact bytes | `tests/test_banking_servicing_alignment_data.py` pins the 215-row composite digest |
| The shadow and screenshot fixtures do not move | They gate checkpoint selection and a live regression | `granite-v7-shadow.jsonl`, `coreference-shadow.jsonl`, `screenshot-regression.jsonl` |
| Regenerated prompts stay clear of training wording | Otherwise the rotation reintroduces contamination | `assert_realized_prompts_stay_clear_of_eval` |
| The router corpus keeps excluding the mangled shape | The retired-realizer filter is what makes the router rebuildable | `is_retired_realizer_prompt` |

## Three ways to do it

**A. Regenerate `test.jsonl` in place, archive the old bytes.**
The current filename always means the current fixture; the superseded one is
kept beside it as `test-v1-2026-08-20.jsonl` with its digest recorded. Every
consumer that reads `test.jsonl` picks up the rotation with no change.
The cost is that the digest test has to pin two files instead of one, and any
number quoted elsewhere has to name which fixture it came from. That naming is
work, but it is honest work: those numbers already differ from what a rescore
would produce and nothing says so today.

**B. Add `test-v2.jsonl` beside the untouched `test.jsonl`.**
Nothing existing moves, so no published number is disturbed and no digest test
changes. The cost is that the default filename keeps pointing at the defective
fixture, so every consumer has to be edited to opt in, and the one that is
forgotten silently keeps scoring the old bytes. That is the same failure mode
as the stale v4 lock: an artifact whose name promises currency and does not
deliver it.

**C. New corpus directories at v6.**
Cleanest naming, but it duplicates 12 MB of train data that is not changing,
and it forces a v6 identity onto the tool-SFT and alignment corpora for a
change that touches 31 rows of one split each. The identity churn would ripple
through the release config, both data cards and the docs for no gain.

**Chosen: A.** The current name should mean the current fixture, and the
archived file plus its recorded digest preserves everything B protects. C buys
nothing that A does not, at a much higher cost.

## How the rows get regenerated

The teacher pass stays available for voice, through the existing
`--prompt-responses` / `--prompt-teacher-model` route, but it is not needed to
fix the defect and adding it would move these rows twice.

### What the first attempt assumed, and what it cost

This section originally read: "no new authoring is required ... the correct
realization is the stem itself, with the opener dropped ... a deterministic
transformation of rows the generator already produces". That was tried on
2026-09-12 and is wrong. Three things came out of the attempt, each caught by a
gate rather than by review, and all three are why the mechanical version cannot
work.

**Dropping the opener collapses rows into duplicates.** `_realize_user` cycles
the stem every occurrence, the opener every `len(stems)` occurrences and the
closer every `len(stems) * len(openers)`. Occurrences `n` and `n + len(stems)`
therefore share a stem and a closer and differ **only** by opener. Remove it and
they normalize to the same text, which
`validate_records` rejects. The opener is load-bearing for uniqueness, so the
variety it provides is real even where the wording it produces is not.

**An inverted question cannot be embedded.** Bridging the opener into the
question instead of dropping it preserves uniqueness one-to-one, and it works
for stems like "what information is needed for a card dispute": "Can you tell me
what information is needed for a card dispute" is correct. It fails for every
stem that keeps subject-auxiliary inversion. "Can you tell me how does a card
purchase dispute work" is as wrong as the shape being replaced, and roughly half
the affected stems are of that form.

**The product-wording scrub is case-sensitive.** `TRAINABLE_TEXT_SUBSTITUTIONS`
rewrites literal lowercase substrings after split assignment. A stem promoted to
the start of a prompt capitalises, the substring stops matching, and "demo"
survives into trainable text. The banned-wording gate caught it on the first
run. Any change that moves a stem to the front of a prompt has to account for
this.

A measurement worth keeping from the attempt: with the generator changed,
**train and validation did not move at all** — zero rows in either. The teacher
realization file overwrites those prompts, so the generator's wording reaches
only the test split. The rotation really is confined to the fixture, which is
the one thing the original design got right.

### What it actually takes

Each question stem needs an authored **embedded form** alongside its direct
form, so the bridge has something grammatical to embed:

| direct stem | embedded form |
| --- | --- |
| how does a card purchase dispute work | how a card purchase dispute works |
| how do I safely report card fraud | how to safely report card fraud |
| what should I do if I see card fraud | what to do about card fraud |
| when is savings interest usually credited | when savings interest is usually credited |

That is roughly 30 strings across the six FAQ families, hand-written. It is
bounded and it is the honest fix: these rows are questions, and a question needs
either to stand on its own or to be de-inverted before something embeds it.
There is no rule that derives one from the other reliably.

The count also drops. Of the 31 prompts, the three of the form "Please can you
open another checking account" are ordinary English and are left alone, which is
exactly why the router's retired-realizer filter excludes that shape. **28 rows
move, not 31.**

## Steps

1. Author an embedded form for every interrogative stem in the six FAQ
   families, and teach `_realize_user` to bridge the opener into that form
   rather than stacking it on the direct one. Unconditionally: an earlier draft
   put the change behind a rotation flag so the old bytes stayed reproducible
   from HEAD, but that is complexity this repository does not need. It already
   has a place for an artifact HEAD no longer derives —
   `FROZEN_RELEASE_ARTIFACTS` in `check_corpora_reproduce.py`, which is how the
   v8 router corpus is handled — and the archived fixture belongs there for the
   same reason. Extend `TRAINABLE_TEXT_SUBSTITUTIONS` for any stem the change
   promotes to the front of a prompt.
2. Regenerate both test splits. Assert exactly 28 rows moved in each, that
   train and validation do not move at all, and that no other row's bytes
   changed. The second and third assertions are the ones that matter: they
   prove the change is surgical rather than a re-voicing. The first attempt
   confirmed train and validation hold still, because the teacher realization
   file overwrites their prompts.
3. Archive the superseded files as `test-v1-2026-08-20.jsonl` and record their
   digests in the manifest under a `superseded` key.
4. Update the pinned digests. The composite 215-row test keeps its old value
   under the archived name and gains the new one.
5. Re-run `assert_realized_prompts_stay_clear_of_eval` across the rotated
   splits against the current train split.
6. Rescore the deployed v14 adapter on the rotated splits, once, and record
   both numbers side by side with the fixture each belongs to.
7. Note in `docs/06-evaluation.md` which scores belong to which fixture.

Steps 1 to 5 are local, free and reversible. Step 6 is not.

## The decision this is blocked on

Step 6 needs a GPU. The deployed adapter is Granite 9B, which does not fit the
local TITAN V alongside the user's running services, so the rescore is an HF
job on `rtx-pro-6000` at $2.75/hour.

Scoring 395 rows is far shorter than a training run. Measured history puts
per-run overhead at roughly ten minutes, and the v14 gate suite scored in about
thirteen. **Estimate: 20 to 25 minutes, $0.92 to $1.15**, launched with
`JOB_TIMEOUT=30m` for a hard ceiling of **$1.38**. Price it with `DRY_RUN=1`
first, which is free.

The alternative is to rotate the fixtures and not rescore, which leaves the
published v14 numbers attached to the archived fixture and clearly labelled as
such. That is free and honest, but it means no model has a score on the fixture
that will be used from now on, so the next model has nothing to be compared
against.

**Recommendation: spend the ~$1.15.** A benchmark with no baseline on it is not
yet a benchmark, and this is the cheapest job in the project's history.
