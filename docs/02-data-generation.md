# V6 Hierarchical Router Data Generation

The active router dataset trains one shared encoder to predict seven related
views of a customer turn. It combines leakage-controlled banking examples from
the servicing-alignment corpus with checksum-pinned CLINC150 external OOD
examples.

## Published Identity

| Item | Value |
| --- | --- |
| Dataset | `spkc83/retail-bank-conversation-router-data` |
| Immutable revision | `b33c27170e27cdb11783704ede14f7d25f70625e` |
| Local directory | `data/banking-conversation-router-v8-first-turn-mutation` |
| Source lock | `data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json` |
| Train rows | 20,439 |
| Validation rows | 4,158 |
| Test rows | 4,921 |
| Manifest SHA-256 | `caae2209063beb9370d0f3a6fc166e4c35658fafdd2420b21e5920c6c9e90de5` |

The split SHA-256 values are:

| Split | SHA-256 |
| --- | --- |
| train | `c838134cdecc22723fda887c1dd561329ab5cac2c72eabc2de484c54a4d4f733` |
| validation | `5491dcbe64ef5c4d7a15d440076ef9964a3767a0adb94d0c4edbb33ecc3c2168` |
| test | `135e2c16962a19c2752b85ca626e83e067eaa9222ff7e1b9029bbdbe681584e8` |

## Inputs and Generator

[`prepare_conversation_router_data.py`](../scripts/retail_bank/prepare_conversation_router_data.py)
loads:

1. the governed `data/banking-servicing-alignment-v5` splits for banking,
   social, policy, action, clarification, and multi-turn examples;
2. `UCI/clinc150` from a pinned archive/member checksum for external OOD;
3. deterministic V6 templates for state switches, policy detours, social
   detours, repairs, coreference, entity ambiguity, and counterfactual pairs.

Generate the exact release data with:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/prepare_conversation_router_data.py \
  --sft-dir data/banking-servicing-alignment-v5 \
  --output-dir data/banking-conversation-router-v8-first-turn-mutation \
  --source-lock data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json \
  --expected-release-lock data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json
```

The expected lock makes regeneration fail if split digests drift. Use
`--skip-release-digest-check` only while deliberately creating a new candidate,
never when reproducing this release.

## Record Structure

Each JSONL record contains the source conversation plus supervised hierarchy:

| Field | Meaning |
| --- | --- |
| `current_text` | Current customer utterance. |
| `history` | Recent visible user/assistant messages. |
| `prior_dialogue_state` | Trusted state before the current turn. |
| `text` | Canonical cross-encoder rendering of those inputs. |
| `domain_name` / `domain_index` | `out_of_domain`, `banking`, or `social`. |
| `lane_name` / `lane_index` | Orchestration lane. |
| `family_name` / `family_index` | Product/topic family. |
| `intent` / `intent_label` | Fine intent; absent for OOD rows. |
| `relation_labels` | Five-element multi-hot relation vector. |
| `action_name` / `action_index` | Intended generation disposition. |
| `entity_resolution_name` / `entity_resolution_index` | Whether a servicing target is usable. |
| `group_id` / `trajectory_id` | Split-isolation keys. |
| `counterfactual_pair_id` | Link for controlled state-pair tests. |
| `example_kind` / `source` | Audit and metric grouping. |

### Direct servicing example

```json
{
  "current_text": "Please show the accounts available to me and their balances",
  "history": [],
  "prior_dialogue_state": {},
  "domain_name": "banking",
  "lane_name": "servicing",
  "family_name": "accounts",
  "intent": "view_accounts",
  "relation_labels": [0, 0, 0, 0, 0],
  "action_name": "execute_tool",
  "entity_resolution_name": "not_required",
  "text": "[CURRENT_USER]\nPlease show the accounts available to me and their balances"
}
```

The label `execute_tool` means the harness may expose the one schema mapped
from `view_accounts`. It is not a tool call and contains no arguments.

### Intent change with stale state

```json
{
  "current_text": "Change of plan: list my debit cards.",
  "history": [
    {"role": "user", "content": "What are the rules for interest on these accounts?"},
    {"role": "assistant", "content": "I can explain the applicable policy."}
  ],
  "prior_dialogue_state": {
    "knowledge_detour_active": true,
    "pending_servicing": {
      "intent": "view_accounts",
      "anchor_user_message": "Show my account balances.",
      "anchor_assistant_message": "Which record should I use?",
      "phase": "awaiting_user"
    },
    "version": 1
  },
  "domain_name": "banking",
  "lane_name": "servicing",
  "family_name": "cards",
  "intent": "view_cards",
  "relation_labels": [0, 0, 1, 0, 0],
  "action_name": "execute_tool",
  "entity_resolution_name": "not_required"
}
```

This row teaches the current explicit intent to override a stale pending task.

### Ineligible entity example

```json
{
  "current_text": "Freeze that closed card anyway.",
  "history": [
    {"role": "user", "content": "Show my cards."},
    {"role": "assistant", "content": "The card ending in 1846 is closed and cannot be used."}
  ],
  "domain_name": "banking",
  "lane_name": "servicing",
  "family_name": "cards",
  "intent": "freeze_card",
  "relation_labels": [1, 0, 0, 0, 0],
  "action_name": "converse",
  "entity_resolution_name": "ineligible"
}
```

The customer intent remains `freeze_card`, but the action/entity heads prevent
the harness from exposing `freeze_card` for an unusable target.

## Label Ontology

### Hierarchy

```text
domain
  -> lane
     -> family
        -> intent
```

Examples:

```text
banking -> servicing -> cards -> replace_card
banking -> policy -> policy -> policy_knowledge
social -> conversation -> social -> conversation
out_of_domain -> out_of_domain -> external -> no intent
```

### Relations

The multi-hot vector order is fixed:

```text
context_dependent, agent_repair, topic_shift,
clarification_answer, resume_previous_service
```

### Action and entity resolution

Actions are `refuse_ood`, `execute_tool`, `clarify`, `retrieve_policy`, and
`converse`. Entity states are `not_required`, `resolved`, `missing`,
`ambiguous`, and `ineligible`.

These labels supervise orchestration, not customer data access. The dataset
does not include current-turn action results, expected final answers, or tool
arguments in router input.

## Counterfactual and Leakage Controls

The generator enforces:

- zero group, trajectory, state-current-text, state-paraphrase-family, and
  counterfactual-pair leakage across splits;
- split-isolated paraphrases for state-dependent tests;
- held-out screenshot regressions only in test;
- no email, SSN-like, or long card-number matches;
- exact source/archive and prepared-split digests;
- paired examples in which history/state changes while the current wording is
  controlled.

The release manifest reports zero leakage findings and zero PII-like matches.
These controls detect direct contamination; they do not prove generalization
to production traffic.

## Verify

```bash
python -m json.tool data/banking-conversation-router-v8-first-turn-mutation/manifest.json >/dev/null
python -m json.tool data/sources/banking-conversation-router-v8-first-turn-mutation.lock.json >/dev/null

PYTHONPATH=src uv run pytest -q \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py
```

Do not mix the `expected`, grounding, or assistant-answer fields from Granite
SFT records into the router input. Those fields are evaluation truth for the
generative model, not classifier features.

## Granite SFT Corpora

The alignment corpus consumed above is one of two governed Granite SFT datasets.
Both were regenerated in the conversational-voice pass so that every
loss-bearing customer-visible final reads like a human bank-chat agent rather
than a filler-wrapped template. Record counts, split membership, tool calls, and
gate multipliers are unchanged; only customer-visible text moved.

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| `data/banking-v5-tool-sft` | 841 | 179 | 180 |
| `data/banking-servicing-alignment-v5` | 3,043 | 397 | 215 |

### Voice contract

[`2026-08-21-conversational-voice-spec.md`](superpowers/plans/2026-08-21-conversational-voice-spec.md)
is the authoring contract for every rewritten final: persona, per-mode sentence
shapes, the phrases each mode must keep for its corpus and runtime validators,
the opening-trigram diversity rule, and the substrings banned from finals.

### Teacher realizations

Rewritten finals are supplied as teacher-realization files and applied through
the existing export/import hook in
[`banking_tool_sft_data.py`](../src/hello_slm/banking_tool_sft_data.py).

| Source file | Rows | Rewrites |
| --- | ---: | --- |
| `data/sources/banking-v5-tool-sft-teacher-realizations-v2.jsonl` | 1,020 | `final_response` and `user_content` for every base train (841) and validation (179) row |
| `data/sources/banking-servicing-alignment-v5-teacher-realizations.jsonl` | 819 | `final_response` only, for the 651 train and 168 validation alignment rows that `_varied_final` had wrapped in opener/closer filler |

The alignment preparer exposes the same three teacher flags as the base
preparer: `--teacher-responses`, `--teacher-model`, and `--teacher-prompt-hash`.
Alignment rows edit `final_response` only, because alignment user text feeds the
leakage and held-out gates; the preparer enforces this. Neither teacher file may
contain a test-split record id of either dataset, which is also enforced in
code.

Regenerate both corpora in place:

```bash
PYTHONPATH=src uv run python scripts/retail_bank/prepare_tool_sft_data.py \
  --output-dir data/banking-v5-tool-sft \
  --teacher-responses data/sources/banking-v5-tool-sft-teacher-realizations-v2.jsonl \
  --teacher-model claude-opus-5 \
  --teacher-prompt-hash 95dd4bc84e7b5a3a89a89c27e888b5505e6c72ebd25add9be58c143cee7f7ace

PYTHONPATH=src uv run python scripts/retail_bank/prepare_servicing_alignment_data.py \
  --output-dir data/banking-servicing-alignment-v5 \
  --teacher-responses data/sources/banking-servicing-alignment-v5-teacher-realizations.jsonl \
  --teacher-model claude-opus-5 \
  --teacher-prompt-hash 21d264fe83490f5ef3666041490a493828847c368a0aca619a034a3c3f72ed6f
```

The prompt hash is a digest over the voice spec plus the request file sent to
the teacher. The base preparer stamps `teacher_model` and `teacher_prompt_hash`
onto every prepared row's `provenance`; the alignment preparer additionally
records `report.alignment_teacher_realization.realized_counts` in its
`manifest.json` (`{"train": 651, "validation": 168}`).

Regeneration must leave these files byte-identical:
`data/banking-v5-tool-sft/test.jsonl` and the alignment `test.jsonl`,
`coreference-shadow.jsonl`, `granite-v7-shadow.jsonl`, and
`screenshot-regression.jsonl`.

### Trainable-text word ban

Training must not teach the model to talk about a product surface it cannot see
or to describe itself as a demo. `validate_records` therefore rejects any `user`
or `assistant` message whose text matches `TRAINABLE_TEXT_BANNED_WORDS`
(`app`/`apps`, `mobile app`, `demo`/`demos`, `synthetic`, `mock`, `sandbox`,
`fictional`, `prototype`, `poc`, `placeholder`, `dummy`, `sample`,
`test`/`testing`, word-boundary matched, case-insensitive) when the record's
`metadata.split` is `train` or `validation`. The frozen evaluation splits
(`test`, the two shadow gates, and the held-out screenshot rows) are exempt by
split so their fixtures stay byte-identical; the pre-existing
`FINAL_RESPONSE_FORBIDDEN` and system-prompt checks are unchanged.

Two consequences for the generators:

- The alignment templates were rewritten for `train`/`validation` only. The
  clarification finals now end `Please share the last four digits.` and the
  `deictic_replace_ambiguity` final ends `Please share its last four digits.`,
  while the shadow branches keep the legacy `... shown in the app.` string that
  `coreference-shadow.jsonl` and `granite-v7-shadow.jsonl` pin.
- Base tool-SFT scenario templates are shared by every split and the base test
  split is byte-frozen, so they cannot be edited in place. Instead
  `_scrub_trainable_product_wording` rewrites `TRAINABLE_TEXT_SUBSTITUTIONS`
  (`while I am checking the mobile app`, ` shown in the app`, and the two
  `can this demo ...` stems) out of the dialogue text *after* split assignment,
  on trainable records only. Teacher-realization files must be app-free on their
  own: the teacher pass overwrites `user_content` and `final_response` after the
  scrub, and its `validate_records` call then enforces the ban.

### Ambiguity clarification template

`_deictic_replace_curriculum` emits one clarification template for every
`deictic_replace_ambiguity` row in all splits. A 32-phrasing conversational pool
was tried for train/validation in the v9 iteration and regressed the coreference
dev gate (ambiguity accuracy 0.44 after 964 continuation steps: the parent
adapter's prior was not overwritten at the continuation learning rate), so the
single template was restored. It names both candidate cards and keeps `which`
and `card` early, which is what the gate matches on. Train and validation close
with `Please share its last four digits.`; the shadow split keeps the frozen
`Please share the last four digits shown in the app.`

### Documented limitations

- The `_suffix` concatenation defect is retained. The split suffix is appended
  without a separating space, so alignment user turns read `get createdon my
  account?` in train and `get createdfrom my profile?` in test. The same defect is
  present in the frozen `test.jsonl`, so correcting it would change a
  byte-identical split; it is deferred to the next frozen-split rotation.
- Alignment user turns remain scaffolded ("... I am going through my accounts.
  Please keep the answer concise."). Only assistant finals were rewritten in
  this pass, for the leakage-gate reason above; the scaffolding itself is now
  app-free but still reads as a template.
