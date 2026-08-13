---
license: cc-by-4.0
task_categories:
  - text-classification
language:
  - en
---

# Retail Bank Conversation Router Data

Governed classifier-only data for the released history-aware conversation
router. It is not included in generative SFT.

Preparation and audit code:
https://github.com/spkc83/retail-bank-servicing

Trained router:
https://huggingface.co/spkc83/retail-bank-conversation-router

- Dataset repository: `spkc83/retail-bank-conversation-router-data`
- Released revision: `e9a64a2e7f2b622d5412c15eac4618ceca2150da`
- Train rows: 61,759
- Validation rows: 13,173
- Test rows: 15,466
- Domain labels: OOD=0, supported retail banking=1
- Capability labels: accounts, cards, card actions, transactions, transfers,
  service cases, FAQ, conversation
- Relation labels: `context_dependent`, `agent_repair`, `topic_shift`,
  `clarification_answer`

The data is built from governed synthetic retail-bank SFT conversations,
checksum-pinned UCI CLINC150 OOD language, and deterministic synthetic
conversation variants. Exact captured POC failures are reserved for held-out
regression rows and are not copied into training.

## Example Row

```text
[CURRENT_USER]
When was that created?
[PREVIOUS_ASSISTANT]
You have a closed mailing-address update case.
[PREVIOUS_USER]
Show my service cases.
```

Targets mark the row as supported banking, `service_cases`, and
`context_dependent`. The row excludes the current assistant answer, target tool
call, tool result, and grounding facts to prevent label leakage.

At serving time, a middle-domain score is `uncertain` and continues to the 9B
agent. Only high-confidence OOD stops generation.

See `manifest.json` in the dataset repository for source revisions, hashes,
mapping policy, and audit counts.
