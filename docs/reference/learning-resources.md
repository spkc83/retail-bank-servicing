# Learning Resources

These references explain the general techniques used by this repository. The
repository code and pinned artifact revisions remain the source of truth for
the released implementation.

## Conversation and Tool-Use Data

- [Transformers chat templates](https://huggingface.co/docs/transformers/en/chat_templating)
  explains why role messages must be rendered with the selected model's chat
  template rather than concatenated by hand.
- [Transformers tool use](https://huggingface.co/docs/transformers/en/chat_extras)
  explains tool schemas, model-generated tool calls, and tool-result messages.
- [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) documents
  conversational dataset formats, assistant-only loss, packing, PEFT
  integration, and evaluation hooks.
- [Hugging Face Datasets processing](https://huggingface.co/docs/datasets/en/process)
  documents mapping, shuffling, selecting, splitting, and sharding datasets.

Assistant-only loss depends on a reliable assistant-token mask. In this repo,
[`banking_tool_wire.py`](../../src/hello_slm/banking_tool_wire.py) proves token
spans from the pinned Granite chat template and rejects an unprovable layout.

## PEFT and LoRA

- [PEFT documentation](https://huggingface.co/docs/peft/index) introduces
  parameter-efficient fine-tuning and adapter workflows.
- [PEFT LoRA guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
  explains frozen base weights and trainable low-rank update matrices.
- [PEFT LoRA API reference](https://huggingface.co/docs/peft/package_reference/lora)
  documents rank, alpha, dropout, target modules, initialization, and related
  configuration fields.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
  is the original paper behind the adaptation method.

## Calibration and the Uncertain Band

- [Scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
  explains why classifier probabilities may need calibration before they are
  used as confidence values.
- [Scikit-learn decision-threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)
  separates probability estimation from the later decision policy.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
  is a primary reference on neural-network confidence calibration.
- [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500)
  explains classification with an abstention or reject option.
- [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) describes a
  network that learns prediction and rejection together.

This project's `uncertain` route is an operational abstention band. It does not
claim to implement SelectiveNet. It sends a low-confidence decision to Granite
instead of pretending the classifier knows the correct capability.

## Evaluation

- [Scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
  describes classification metrics and scoring concepts.
- [Scikit-learn F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
  is relevant to the router capability and relation heads.
- [Scikit-learn ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
  is useful when studying OOD score separation before threshold selection.

For the generative agent, generic token metrics are insufficient. This repo
also measures exact tools, public arguments, call order, executable
trajectories, clarifications, OOD behavior, and grounded response facts.
