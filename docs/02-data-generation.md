# V6 Hierarchical Router Data Generation

The active router dataset trains one shared encoder to predict seven related
views of a customer turn. It combines leakage-controlled banking examples from
the servicing-alignment corpus with checksum-pinned CLINC150 external OOD
examples.

## Published Identity

| Item | Value |
| --- | --- |
| Dataset | `spkc83/retail-bank-conversation-router-data` |
| Immutable revision | `80c0edfea84b341d2ee4092f5c4a4bbb05405e40` |
| Local directory | `data/banking-conversation-router-v6-hierarchical` |
| Source lock | `data/sources/banking-conversation-router-v6-hierarchical.lock.json` |
| Train rows | 16,693 |
| Validation rows | 4,061 |
| Test rows | 4,895 |
| Manifest SHA-256 | `0886dd8037e59d73b41c4ee60cde57dc865c85494309accd821d8a423681da11` |

The split SHA-256 values are:

| Split | SHA-256 |
| --- | --- |
| train | `10b6d4316719f4cd9f162d9faef36a3c5f264bf7a3a0ae759a2c5562993032f3` |
| validation | `cb9660b696d1b9d4ee81922c0fc042861c8cae9a55a9d0760b02ad98b94646f9` |
| test | `58bc7c10dd11797988e987afd17bdd9f5e0f79b2dbc2a17ed2ea33a8cb176b68` |

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
  --output-dir data/banking-conversation-router-v6-hierarchical \
  --source-lock data/sources/banking-conversation-router-v6-hierarchical.lock.json \
  --expected-release-lock data/sources/banking-conversation-router-v6-hierarchical.lock.json
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
python -m json.tool data/banking-conversation-router-v6-hierarchical/manifest.json >/dev/null
python -m json.tool data/sources/banking-conversation-router-v6-hierarchical.lock.json >/dev/null

PYTHONPATH=src uv run pytest -q \
  tests/test_banking_conversation_router_data.py \
  tests/test_banking_conversation_router_preparation.py
```

Do not mix the `expected`, grounding, or assistant-answer fields from Granite
SFT records into the router input. Those fields are evaluation truth for the
generative model, not classifier features.
