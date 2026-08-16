# Retail Bank Conversation Router V6 Hierarchical Data

Governed cross-encoder data for a history-aware OOD, hierarchical intent, action, entity-resolution, and relation classifier.

Rows include only prior visible user/assistant messages and the current user message.
They exclude current-turn tool plans, tool results, expected outputs, and final assistant responses.

- Train rows: 16887
- Validation rows: 4140
- Test rows: 4921
- Intent labels: view_accounts, view_cards, freeze_card, replace_card, view_transactions, dispute_transaction, view_transfers, cancel_transfer, view_service_cases, policy_knowledge, conversation, other_banking
- Domain labels: out_of_domain, banking, social
- Lane labels: out_of_domain, servicing, policy, conversation, other_banking
- Family labels: external, accounts, cards, transactions, transfers, service_cases, policy, social, other_banking
- Action labels: refuse_ood, execute_tool, clarify, retrieve_policy, converse
- Entity-resolution labels: not_required, resolved, missing, ambiguous, ineligible
- Relation labels: context_dependent, agent_repair, topic_shift, clarification_answer, resume_previous_service
