# Retail Bank Conversation Router V5 Data

Governed cross-encoder data for a history-aware OOD, fine-intent, and relation classifier.

Rows include only prior visible user/assistant messages and the current user message.
They exclude current-turn tool plans, tool results, expected outputs, and final assistant responses.

- Train rows: 19363
- Validation rows: 5056
- Test rows: 6171
- Intent labels: view_accounts, view_cards, freeze_card, replace_card, view_transactions, dispute_transaction, view_transfers, cancel_transfer, view_service_cases, policy_knowledge, conversation, other_banking
- Relation labels: context_dependent, agent_repair, topic_shift, clarification_answer, resume_previous_service
