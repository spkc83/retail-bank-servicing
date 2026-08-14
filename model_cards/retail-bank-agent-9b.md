---
license: apache-2.0
base_model: spkc83/retail-bank-servicing-agent-9b
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

# Retail Bank Servicing Agent 9B — PEFT Candidate

Retail Bank Servicing Agent 9B is an experimental customer-service and tool-use
model for the linked synthetic retail-bank demonstration. The active candidate
is an unmerged BF16 LoRA adapter over immutable Stage-2 base revision
`1d56824995aa1adecfe20f62ca42fb1c0c443817`.

- Source: https://github.com/spkc83/retail-bank-servicing
- Initial tool-use dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-agent-sft
- Servicing-remediation dataset:
  https://huggingface.co/datasets/spkc83/retail-bank-servicing-alignment-sft
- Public ZeroGPU POC:
  https://huggingface.co/spaces/spkc83/retail-bank-servicing-poc

## Artifact Identity

- Adapter repository: `spkc83/retail-bank-servicing-agent-9b-peft`
- PEFT release revision: `cc95e446af2b5e1d8d9df2751a8192613ad386e3`
- Adapter bundle commit: `b4269445ce7b2b943d2d9531102166bf8840a074`
- Training job: `spkc83/6a7f79531f5885ae605b96cc` (`COMPLETED`)
- Incremental SFT parent revision:
  `1d56824995aa1adecfe20f62ca42fb1c0c443817`
- Base model revision:
  `1504002f650e656a0a3789d99574df12e3e94ed0`
- V5 training source revision:
  `75b56ffff45e75ffbee11c0e0552dc35ae124d21`
- Initial tool-use dataset revision:
  `183e7e1ed1aba9c3d7155e7b83b64dc854935055`
- V5 composite training dataset revision:
  `40a0b68b9f746131ffff32a83e077fd7e4a344d1`
- Canonical policy corpus revision:
  `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a`
- Strict evaluation job: `spkc83/6a7f89edc97db76cbdf31893` (`FAILED`)
- Evaluation source: `42c89ae6d6b6792268b36e2162c4b19688e4e617`
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

Stage 3 continued incrementally from the Stage-2 revision with the governed V5
composite corpus. It adds Harborlight Bank presentation, retrieval-grounded
policy citations, policy-detour and servicing-resume trajectories, broader
tool-use phrasing, response empathy, and Markdown table targets. Static policy
facts are supplied by the runtime knowledge base rather than treated as model
memory.

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

## Corrected V5 Grounded-Dialogue Training

- Training job: `spkc83/6a7f79531f5885ae605b96cc`
- Status: `COMPLETED`
- Adaptation: PEFT/LoRA continued from the immutable Stage-2 parent
- Source: `75b56ffff45e75ffbee11c0e0552dc35ae124d21`
- Dataset: `40a0b68b9f746131ffff32a83e077fd7e4a344d1`
- Optimizer steps: `750`
- Training loss: `0.13014758`
- Evaluation loss: `0.3200804`
- Token accuracy: `0.96240348`
- Output: BF16 LoRA PEFT release `cc95e446...`, with adapter bundle committed
  at `b4269445...`

Merged FP16 and BF16 candidates were rejected by unchanged behavioral-parity
gates. Inference and evaluation therefore load the exact base revision and
attach the immutable adapter with PEFT; they do not use merged weights.

## Current Strict-Evaluation Status

Evaluation job `spkc83/6a7f89edc97db76cbdf31893` failed the strict gates.
Five credential-request findings are evaluator false positives caused by the
customer-safe phrase “do not share a password.” Two genuine behaviors remain:

- after an action error, the final answer incorrectly claims success;
- a history-resolved card-replacement request asks for information again.

A corrected evaluator and generalized incremental SFT are underway. No new
artifact identity or passing metric exists yet, so this candidate is not
cleared for deployment.

## Historical V4 Servicing-Remediation Result

- Training job: `spkc83/6a6ca6276b79c09949c1d6cb`
- Runtime: about 18 minutes 59 seconds
- Estimated cost: about `$0.87`
- Training loss: `0.0069123295`
- Evaluation loss: `0.0002181597`
- Token accuracy: `0.999976121`
- Adaptation: BF16 LoRA over attention and MLP projection modules
- Maximum training sequence: 2,048 tokens
- Output: merged FP16 weights in `spkc83/retail-bank-servicing-agent-9b`

## Historical V4 Frozen Evaluation

The prior evaluation job `spkc83/6a6caac1a00abefd4b289b14` evaluated 1,374 frozen
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

## Superseded Pre-Canonical-Policy V5 Evidence

Job `spkc83/6a7f6d01c97db76cbdf3170b` completed against the immutable V5 model and
model revision `1799d068906c0da2a8739668857b096d20fed549` and dataset revision
`f7784b34b41094b1e771323b2df046ed4664b9a4`. The enforced gate reported
`eligible: true` with no failures, but this evidence predates canonical policy
corpus revision `sha256:ec6e7500...` and is not the current release result.

- tool names: `125/125`
- tool arguments: `125/125`
- executable tool trajectories: `113/113`
- exact dependent multi-tool sequences: `12/12`
- grounded factual responses: `175/175`
- grounded policy responses: `44/44`
- appropriate clarifications: `6/6`
- OOD/small-talk response paths: `11/11`
- malformed calls, unsupported private arguments, credential requests,
  in-domain false refusals, and OOD false accepts: `0`

The 216-record first-pass evaluation includes 113 grounded-final generations.
Its immutable report and predictions are stored under
`evaluation/1799d068906c-f7784b34b410/` at model-repository revision
`9806174bacbe7bd268d0d72b2eaff6f98b668386`. The report SHA-256 is
`4a90ea779a20de0c72c293d49cc69a8c44d9067c3e70408ac806988060651dac`.

### Evaluation-integrity limitation

The historical V4 score is an in-generator protocol regression result, not a
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
