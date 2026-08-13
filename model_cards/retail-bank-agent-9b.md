---
license: apache-2.0
base_model: ibm-granite/granite-4.1-8b
datasets:
- spkc83/retail-bank-agent-sft
- spkc83/retail-bank-servicing-alignment-sft
pipeline_tag: text-generation
tags:
- retail-banking
- tool-calling
- conversational
- peft
---

# Retail Bank Servicing Agent 9B

Retail Bank Servicing Agent 9B is an experimental customer-service and tool-use
model for the linked synthetic retail-bank demonstration. It is a merged FP16
LoRA adaptation of `ibm-granite/granite-4.1-8b`.

- Source: https://github.com/spkc83/retail-bank-servicing
- Initial tool-use dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-agent-sft
- Servicing-remediation dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-servicing-alignment-sft
- Public ZeroGPU POC:
  https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc

## Artifact Identity

- Model repository: `spkc83/retail-bank-servicing-agent-9b`
- Immutable weights revision:
  `1d56824995aa1adecfe20f62ca42fb1c0c443817`
- Published evaluation head:
  `214fc0d9e143e4fa7b658de1993113562b90958a`
- Base model revision:
  `1504002f650e656a0a3789d99574df12e3e94ed0`
- Source revision:
  `475dc2b563ef87fa0c9aa597b0b0465d56d2ee0f`
- Initial tool-use dataset revision:
  `183e7e1ed1aba9c3d7155e7b83b64dc854935055`
- Corrected servicing-remediation dataset revision:
  `0ce32f9c7a3edff227005e5b89b089947b87625a`
- Prompt-identical training dataset revision:
  `fea8aa1cda716954eb7322325e2be25c9f570ea3`
- Servicing-remediation training job:
  `spkc83/6a6ca6276b79c09949c1d6cb`
- Exact frozen evaluation job:
  `spkc83/6a6caac1a00abefd4b289b14`
- Parameters: 8,791,592,960
- Architecture: dense decoder-only causal transformer
- Tool format: Granite native tagged JSON

## Training Stages

Stage 1 fine-tuned IBM Granite on the initial 9,000-record synthetic tool-use
SFT corpus. That stage taught the tagged-JSON tool wire, public synthetic-bank
tools, tool-result grounding, clarification, FAQ, OOD refusal, and multi-tool
ordering.

Stage 2 continued from the released tool-trained checkpoint with the composite
v4 servicing-remediation corpus. The second stage exists because POC testing
exposed conversation and tool-use failures around service-case follow-ups, card
anaphora, clarification answers, agent repair, and topic shifts. The composite
corpus keeps the full initial SFT corpus and appends 427 targeted remediation
records in split.

## Inference Example

```text
User: Show my cards.
Model pass 1: <tool_call>{"name":"list_cards","arguments":{}}</tool_call>
Runtime: executes the synthetic list_cards tool
Model pass 2: You have an active card ending in 4821.
```

For “Replace the active one,” the model also needs retained visible history.
The router may label the turn `in_domain` or `uncertain`; either route invokes
this model. Router capability candidates do not enter the model prompt.

## Servicing-Remediation Training Result

- Training job: `spkc83/6a6ca6276b79c09949c1d6cb`
- Runtime: about 18 minutes 59 seconds
- Estimated cost: about `$0.87`
- Training loss: `0.0069123295`
- Evaluation loss: `0.0002181597`
- Token accuracy: `0.999976121`
- Adaptation: BF16 LoRA over attention and MLP projection modules
- Maximum training sequence: 2,048 tokens
- Output: merged FP16 weights in `spkc83/retail-bank-servicing-agent-9b`

## Frozen Evaluation

Evaluation job `spkc83/6a6caac1a00abefd4b289b14` evaluated 1,374 frozen
records with deterministic FP16 generation and the exact tool/final-response
scorer.

- tool names and arguments: `796/796`
- executable tool trajectories: `700/700`
- exact dependent multi-tool sequences: `96/96`
- appropriate clarifications: `63/63`
- banking FAQ answers: `258/258`
- OOD response paths: `35/35`
- grounded factual responses: `1,141/1,141`
- malformed calls, unsupported/private arguments, credential requests,
  in-domain false refusals, and OOD false accepts: `0`

The public corrected dataset revision is
`0ce32f9c7a3edff227005e5b89b089947b87625a`. Training used revision
`fea8aa1cda716954eb7322325e2be25c9f570ea3`. The final score is a rescore
because the corrected rows are prompt-identical to the training/evaluation
rows: the rendered prompts, target tool calls, and target final responses are
equivalent for generation and scoring. This card does not claim that a second
generation run was performed.

### Evaluation-integrity limitation

The frozen score is an in-generator protocol regression result, not a
leakage-free generalization benchmark. A repository audit found shared POC
facts, shared template families, repeated user turns, and repeated targets
between training and test.

In particular, `26/27` remediation test rows reuse an exact expected tool call,
canonical tool result, and final answer found in remediation training. Across
the complete test split, `894/1,374` final-answer strings occur in training.

No exact full-conversation duplicates were found, and base state groups remain
disjoint. These protections do not rule out memorization as one contributor to
the reported score or live `alex.demo` behavior.

See
[`docs/reference/data-leakage-audit.md`](../docs/reference/data-leakage-audit.md)
for scope, evidence, and the clean counterfactual benchmark required before
claiming independent generalization.

That benchmark is now implemented as the evaluation-only local
[`banking-counterfactual-eval-v1`](../data/banking-counterfactual-eval-v1)
suite. Its preparation audit and any pinned-model result are reported
separately from the released `1,374`-record regression score; a preparation
pass is not itself evidence that this checkpoint passed the benchmark.

## Intended Use

The model is intended only for research evaluation in the linked synthetic
banking POC. It must receive the published tool schemas, conversation history,
and correlated tool results. The model does not connect to real banking
systems.

## Limitations

The dataset is synthetic and deliberately narrow. The model may choose the
wrong tool, produce invalid arguments, mishandle conversation context, or make
unsupported claims. It is not financial advice and must not receive
credentials, full account numbers, payment-card details, or real customer data.

Release evaluation covers tool-call syntax and names, public arguments,
multi-tool ordering, clarification, tool errors, grounded final responses,
multi-turn follow-ups, OOD behavior, and malformed-call handling.
