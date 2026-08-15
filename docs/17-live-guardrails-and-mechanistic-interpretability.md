# Live Guardrails and Mechanistic-Interpretability Shadow Mode

This phase adds privacy-bounded activation observations to the exact Granite
model used by both POC runtimes. It does **not** allow activation scores to
approve, deny, rewrite, or stop a customer request. The observer is off by
default until local NF4 and hosted BF16 measurements are calibrated
independently.

## What Enforces Behavior Today

The live guardrails remain the existing deterministic system controls:

- the V6 hierarchical router and its joint compatibility decoder;
- action-guided tool allow-listing;
- tool-call schema and argument validation;
- the synthetic bank execution boundary;
- policy retrieval, citation, and factual grounding checks;
- response-policy validation and governed OOD behavior.

The activation observer is an additional evidence surface. Calling this
phase `MI shadow instrumentation` avoids implying that uncalibrated activation
statistics are already a safety control.

## Framework Decision

The live implementation uses native PyTorch module hooks on the already-loaded
Granite PEFT model.

| Option | Decision | Reason |
| --- | --- | --- |
| Native PyTorch hooks | Use in the POC | No model conversion or new runtime dependency; works on the local NF4 and hosted BF16 compositions. |
| NNsight | Later, offline causal research | Useful for tracing and interventions, but its remote mode targets NDIF rather than Hugging Face ZeroGPU. Customer traffic must not be sent to an unrelated remote tracing service. |
| TransformerLens | Defer to an isolated research environment | Strong circuit-analysis abstractions, but a larger dependency surface and exact PEFT/quantization parity must be proven first. |

References:

- [PyTorch module hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [NNsight remote execution](https://nnsight.net/features/13_remote_execution/)
- [TransformerLens model bridge](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.model_bridge.bridge.html)
- [Hugging Face Granite model documentation](https://huggingface.co/docs/transformers/model_doc/granite)

## Runtime Contract

`activation_guardrails.py` implements schema
`retail-bank-activation-observation/v1`.

```text
Granite generation pass
  -> hooks on selected Granite decoder layers
  -> select the final sequence position
  -> reduce on GPU
  -> aggregate prefill and decode separately
  -> copy one small aggregate per layer to CPU
  -> attach immutable observation to that exact model pass
  -> diagnostics render only allow-listed aggregate fields
```

The default layers are the zero-based end of the first half and final decoder
blocks. For the current 40-layer Granite configuration, those are layers 19 and 39. The
capture is capped at 256 forward samples per selected layer and 16 MiB of
admitted buffer space.

Collected values are aggregate RMS, mean absolute value, maximum absolute
value, finite-value status, sample count, and sequence-width bounds. The
system does not retain prompts, token IDs, token text, or activation vectors.

## Configuration

```bash
# Default and recommended until calibration is complete.
export RETAIL_BANK_MI_MODE=off

# Shadow observation only; does not alter routing or generation.
export RETAIL_BANK_MI_MODE=observe

# Optional zero-based Granite decoder layers.
export RETAIL_BANK_MI_LAYERS=19,39
```

Invalid modes, duplicate layers, out-of-range layers, and oversized capture
plans fail during configuration. A runtime module mismatch or hook/reduction
problem yields a sanitized `unavailable` observation and does not replace the
model output. Hook handles are removed in a `finally` path. Local and hosted
generation are serialized while request-scoped hooks are installed.

## Local and Hosted Composition

Thresholds must not be shared across these deployments without evidence:

| Runtime | Weight path | Compute path |
| --- | --- | --- |
| Local Streamlit | PEFT over 4-bit NF4 double-quantized base | FP16 on Titan V |
| Hugging Face Space | PEFT over unquantized base | BF16 by default on ZeroGPU |

Quantization changes activation distributions. Every observation therefore
records the relevant model/base/adapter revision and runtime composition.

## Required Calibration Before Enforcement

Run the following independently for local NF4 and hosted BF16:

1. Use two fixed, non-sensitive prompts and `max_new_tokens=32`.
2. Run three warmups, then twenty measured generations with observation off.
3. Repeat with observation enabled.
4. Record p50/p95 latency and peak allocated-memory delta.
5. Require less than 5% p95 latency overhead and less than 16 MiB incremental
   observer memory.
6. Require identical generated-output hashes with observation off and on.
7. Build a labeled calibration corpus and report AUROC, AUPRC, calibration
   error, and false-positive rate at the required recall separately for each
   runtime composition.

Only a later reviewed release may convert a versioned, prospectively validated
score into an enforcing guardrail. Until then, `observe` remains diagnostic
shadow mode and `off` remains the deployment default.

## Code Map

| File | Responsibility |
| --- | --- |
| `poc/retail-bank-customer-service-poc/activation_guardrails.py` | Bounded hook lifecycle, aggregation, failure sanitization, and observation schema. |
| `poc/retail-bank-customer-service-poc/model_service.py` | Pass-scoped result propagation and diagnostics allow-list. |
| `poc/retail-bank-customer-service-poc/local_gpu_runtime.py` | Local NF4 capture around exact Granite generation. |
| `poc/retail-bank-customer-service-poc/zero_gpu_runtime.py` | Hosted BF16 capture around exact Granite generation. |
| `poc/retail-bank-customer-service-poc/local_app_service.py` | Local technical-details rendering. |
| `poc/retail-bank-customer-service-poc/app.py` | Hosted technical-details rendering. |
