# Retail Bank Counterfactual Evaluation v1

This dataset is an evaluation-only counterfactual benchmark for the pinned
Granite servicing model. It contains no train or validation split and must not
be used for SFT, remediation, prompt development, or model selection.

- Records: `18`
- Counterfactual pairs: `5`
- Training allowed: `false`
- Manifest contract: `banking-counterfactual-eval-manifest/v1`
- Contamination audit: `pass`

Each pair presents an identical first-phase prompt but changes the canonical
tool result. The changed values are absent before the result and the other
variant's values are forbidden in the final response.

See `docs/13-counterfactual-evaluation.md` for preparation, scoring, and
interpretation instructions.
