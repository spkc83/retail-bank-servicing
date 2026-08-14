---
license: cc-by-4.0
task_categories:
  - text-classification
language:
  - en
---

# Retail Bank Conversation Router Data V5

Governed classifier-only data for the state-aware retail-bank conversation
router. It is not part of generative SFT.

- Dataset repository: `spkc83/retail-bank-conversation-router-data`
- Released revision: `8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc`
- Manifest SHA-256: `9cb527bdc337ce4da06e391f1d1e341da80092ab1ac46bf619bd33947f7a3608`
- Train rows: `19,363`; SHA-256 `1e67741213b2ee48a61b6aa20be485f9f634850434637f533f928a858e1572f5`
- Validation rows: `5,056`; SHA-256 `4df22958f9519355204bcc2910a2874ead44425644056165133126042abcdafa`
- Test rows: `6,171`; SHA-256 `6af19f8079ff07c087d692ae4c331c55ef33adcdbcd316aa425e866452bd5d97`
- Domain labels: external OOD `0`, supported banking/conversation `1`
- Fine intents: nine servicing intents plus policy, conversation, and other banking
- Relations: context dependence, repair, topic shift, clarification, and resume

The data combines governed synthetic retail-bank SFT conversations,
checksum-pinned UCI CLINC150 external-OOD language, and deterministic V5
trajectory families. Whole groups and trajectory IDs remain in one split.
Audits report zero group or trajectory split leakage and zero PII-pattern
matches.

V5 adds pre-turn state, policy-detour, service-resume, explicit intent-switch,
state-conditioned OOD, social-detour, and orphan-resume examples. Negative
rows prevent the stored service intent from overriding the meaning of the
current user turn.

## Example Shape

```text
[PRIOR_DIALOGUE_STATE]
{"knowledge_detour_active":true,"pending_servicing":{"intent":"dispute_transaction",...}}
[CURRENT_USER]
Actually freeze my card instead.
[PREVIOUS_ASSISTANT]
I can explain the applicable policy.
[PREVIOUS_USER]
What is the policy for reviewing a purchase dispute?
```

Targets mark this as in-domain, `freeze_card`, and `topic_shift`, with no
`resume_previous_service` relation. The row excludes target tool calls, tool
results, expected assistant answers, and post-turn state.

See the published `manifest.json` and
[`docs/02-data-generation.md`](../docs/02-data-generation.md) for exact source
hashes, split hashes, schema, audit rules, and reproduction commands.
