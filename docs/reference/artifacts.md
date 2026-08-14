# Artifact Ledger

This page records the active immutable artifacts for the retail-bank servicing
agent, history-aware router, datasets, local manifests, paid job outputs, and
runtime defaults.

## Published Repositories

| Artifact | Repository | Immutable revision | Source |
| --- | --- | --- | --- |
| Granite V5 PEFT candidate | `spkc83/retail-bank-servicing-agent-9b-peft` | release revision `cc95e446af2b5e1d8d9df2751a8192613ad386e3`; bundle commit `b4269445ce7b2b943d2d9531102166bf8840a074`; failed strict evaluation | [`04-training-and-recovery.md`](../04-training-and-recovery.md) |
| Agent base model | `ibm-granite/granite-4.1-8b` | `1504002f650e656a0a3789d99574df12e3e94ed0` | [`configs/banking-tool-sft-granite.toml`](../../configs/banking-tool-sft-granite.toml), [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Stage-1 Granite tool-use checkpoint | `spkc83/retail-bank-agent-9b` | `085df3d089cfadd77424b548542da0390a54a23e` | [`release config`](../../configs/retail-bank-release.toml) |
| Initial tool-use SFT dataset | `spkc83/retail-bank-agent-sft` | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` | [`data card`](../../data_cards/retail-bank-agent-sft.md) |
| Canonical-policy V5 servicing-alignment dataset | `spkc83/retail-bank-servicing-alignment-sft` | `40a0b68b9f746131ffff32a83e077fd7e4a344d1` | [`02-data-generation.md`](../02-data-generation.md) |
| Generalized state-aware V5 router | `spkc83/retail-bank-conversation-router` | `c8f154266612e79afe20af8abef25761fa56d589` | [`router.py`](../../poc/retail-bank-customer-service-poc/router.py) |
| Generalized state-aware V5 router dataset | `spkc83/retail-bank-conversation-router-data` | `8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc` | [`05-dual-head-router.md`](../05-dual-head-router.md) |
| Public Space | `spkc83/retail-bank-servicing-poc` | Space commit is exposed at runtime as `SPACE_COMMIT_SHA` | [`app.py`](../../poc/retail-bank-customer-service-poc/app.py), [`README.md`](../../poc/retail-bank-customer-service-poc/README.md) |

## Agent Model Details

| Field | Value |
| --- | --- |
| Adapter repository | `spkc83/retail-bank-servicing-agent-9b-peft` |
| PEFT release revision | `cc95e446af2b5e1d8d9df2751a8192613ad386e3` |
| Adapter bundle commit | `b4269445ce7b2b943d2d9531102166bf8840a074` |
| Incremental SFT parent revision | `1d56824995aa1adecfe20f62ca42fb1c0c443817` |
| Base model | `ibm-granite/granite-4.1-8b` |
| Base revision | `1504002f650e656a0a3789d99574df12e3e94ed0` |
| V5 training source revision | `75b56ffff45e75ffbee11c0e0552dc35ae124d21` |
| Initial dataset revision | `183e7e1ed1aba9c3d7155e7b83b64dc854935055` |
| V5 composite training dataset revision | `40a0b68b9f746131ffff32a83e077fd7e4a344d1` |
| Canonical policy corpus revision | `sha256:ec6e75000209f34a1c84d5904d203b275842e441401e6db82ac883301fabe10a` |
| Parameters | 8,791,592,960 |
| Tool format | Granite native tagged JSON |
| Evaluation job/source | `6a7f89edc97db76cbdf31893` / `42c89ae6d6b6792268b36e2162c4b19688e4e617` |
| Evaluation result | failed strict gates; replacement evaluator and incremental SFT are underway |

### How to follow one released artifact chain

Read the ledger as a dependency chain:

```text
IBM base revision
  -> stage-1 dataset revision
  -> stage-1 checkpoint revision
  -> V5 composite dataset revision
  -> immutable Stage-2 base + V5 adapter-bundle revisions
  -> frozen evaluation artifacts
  -> Space runtime model variable
```

The PEFT candidate loads immutable base `1d568249...` plus adapter bundle
`cc95e446...`, never mutable `main`. It is not cleared for deployment because
strict evaluation failed. Its immutable training dataset is `40a0b68...`.

Changing any upstream revision creates a new experiment. Reusing a downstream
metric after such a change would break provenance.

## Paid Job Records

These are the job records for the released artifacts. Do not start new paid jobs
unless explicitly authorized.

| Purpose | Job ID | Evidence |
| --- | --- | --- |
| Canonical-policy V5 SFT | `spkc83/6a7f79531f5885ae605b96cc` | completed; PEFT release `cc95e446...`, bundle `b4269445...` |
| PEFT strict evaluation | `spkc83/6a7f89edc97db76cbdf31893` | failed; five credential flags are evaluator false positives, two behavioral failures remain |
| Pre-canonical-policy V5 SFT | `spkc83/6a7f60401f5885ae605b94bf` | superseded; produced `1799d068...` |
| Pre-canonical-policy V5 evaluation | `spkc83/6a7f6d01c97db76cbdf3170b` | superseded evidence; not the corrected release gate |
| Servicing-remediation SFT training | `spkc83/6a6ca6276b79c09949c1d6cb` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Exact frozen tool/final-response evaluation | `spkc83/6a6caac1a00abefd4b289b14` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |

The superseded V5 training job ran for 1,162 seconds. It completed 750 optimizer steps
with training loss `0.1271855585`; the merge/reload parity check reported
generation equality and `1.0` token-argmax agreement.

### Current PEFT evaluation failure

Job `6a7f89edc97db76cbdf31893` evaluated base `1d568249...` plus adapter bundle
`cc95e446...` from source `42c89ae...`. It did not pass the release gate:

- five credential-request detections are evaluator false positives triggered
  by “do not share a password” language;
- one action-error trajectory incorrectly claims the action succeeded;
- one history-resolved card-replacement turn asks for the card information
  again.

A corrected evaluator and generalized incremental SFT are underway. No
replacement model identity or passing evaluation metric has been published.

## Private Job-Bucket Retention

Bucket: `spkc83/jobs-artifacts`

The bucket is a private, restartable job workspace. It is not read by the
public Space and it is not the authoritative location for published model,
router, dataset, or evaluation artifacts.

| State | Files | Logical size |
| --- | ---: | ---: |
| Before 2026-07-31 cleanup | 290 | 449,461,595,301 bytes |
| After cleanup | 58 | 1,252,559,272 bytes |
| Removed | 232 | about 448.2 GB |

The retained set preserves the selected recovery checkpoint adapter, trainer
state, and provenance JSON. Failed runs, superseded checkpoints, duplicate
merged weights, temporary merge files, and bucket copies of already-published
evaluation outputs were removed. New runs must write a new prefix and apply the
same publish-verify-retain policy.

## Historical V4 Evaluation

The prior V4 released score is from evaluation job
`spkc83/6a6caac1a00abefd4b289b14`.

| Metric | Result | Source |
| --- | ---: | --- |
| Frozen test conversations | 1,374 | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Tool names and arguments | `796/796` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Executable trajectories | `700/700` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Dependent multi-tool sequences | `96/96` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Clarifications | `63/63` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Banking FAQ answers | `258/258` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| OOD paths | `35/35` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Grounded factual responses | `1,141/1,141` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |
| Malformed calls, unsupported/private arguments, credential requests, in-domain false refusals, OOD false accepts | `0` | [`model card`](../../model_cards/retail-bank-agent-9b.md) |

The V4 corrected dataset revision
`0ce32f9c7a3edff227005e5b89b089947b87625a` is prompt-identical to the training
revision `fea8aa1cda716954eb7322325e2be25c9f570ea3` for generation and scoring.
The final report is therefore a rescore of equivalent prompts, not a second
generation run. The rescore helper is `scripts/retail_bank/rescore_tool_eval.py`.

This historical metric is an in-generator protocol regression result. It is not a
leakage-free generalization claim. The local audit found shared POC facts,
template families, and targets between training and test. See
[`data-leakage-audit.md`](data-leakage-audit.md).

## Superseded Pre-Canonical-Policy V5 Evaluation

Job `spkc83/6a7f6d01c97db76cbdf3170b` completed against model revision
`1799d068906c0da2a8739668857b096d20fed549`, dataset revision
`f7784b34b41094b1e771323b2df046ed4664b9a4`, and source revision
`19b4c11aa7abaa175a7153acd7e880ab7ebd22bf`. The immutable evaluation files are
published in the model repository at revision
`9806174bacbe7bd268d0d72b2eaff6f98b668386` under
`evaluation/1799d068906c-f7784b34b410/`.

This chain predates the canonical policy corpus and is retained only as
superseded evidence. Its perfect gate result cannot release the current PEFT
candidate or any later incremental adapter.

| Metric | Result |
| --- | ---: |
| First-pass records | `216` |
| Grounded-final records | `113` |
| Tool-name accuracy | `125/125` |
| Tool-argument accuracy | `125/125` |
| Executable tool success | `113/113` |
| Exact multi-tool sequences | `12/12` |
| Grounded final factuality | `175/175` |
| Grounded policy quality | `44/44` |
| Appropriate clarifications | `6/6` |
| OOD/small-talk response path | `11/11` |
| Malformed calls, unsupported private arguments, credential requests, in-domain false refusals, OOD false accepts | `0` |

The enforced `banking-tool-release-gate/v1` result is eligible with no
failures. The predictions SHA-256 is
`459a4abd65390232a434e19479af65877314cf0a7752fa71bf12c1d7897b7a00`; the
report SHA-256 is
`4a90ea779a20de0c72c293d49cc69a8c44d9067c3e70408ac806988060651dac`.

## Local Counterfactual Evaluation

The clean project-SFT/POC acceptance set is tracked locally and is not a
published training dataset.

| Field | Value |
| --- | --- |
| Manifest | [`data/banking-counterfactual-eval-v1/manifest.json`](../../data/banking-counterfactual-eval-v1/manifest.json) |
| Manifest SHA-256 | `3737655087d79136f719ecf7f251d6a79f89f28392684bb3059cdab243a97b5b` |
| Test rows | `18` |
| Counterfactual pairs | `5` |
| Test JSONL SHA-256 | `d0b60766a659496fa1809c6a8124027172ebce22081db087fedfca4e2b71c6f6` |
| Training allowed | `false` |
| Preparation audit | pass |

The preparation pass proves only that the tracked rows satisfy the local
contamination and pair contracts. A model result exists only when a run emits
predictions, a report, and metadata under
`artifacts/banking-counterfactual-eval-v1`. See
[`13-counterfactual-evaluation.md`](../13-counterfactual-evaluation.md).

## Initial Tool-Use SFT Dataset

Published repository: `spkc83/retail-bank-agent-sft`

Published revision:
`183e7e1ed1aba9c3d7155e7b83b64dc854935055`

Corpus fingerprint:
`2bb7a400ed2556b15c7e5eb6147668041b5deef8ae4f037f9e2e52295ff29ab5`

Local manifest:
[`data/banking-v3-tool-sft/manifest.json`](../../data/banking-v3-tool-sft/manifest.json)

| Split | Records | Local SHA-256 |
| --- | ---: | --- |
| train | 6,304 | `8d92fa0ab1d39875f0c4d918bc5aeaf670f71bf660a01a0a376cb4edc1cced53` |
| validation | 1,349 | `a8c7871b33689fce026ea570ad0a8a90a609cde232a89486e5437b028279e6d3` |
| test | 1,347 | `76b485fa507d56002f12b556f100fd842c77146804cf49be3426be031cc692c0` |

## Servicing-Remediation SFT Dataset

Published repository: `spkc83/retail-bank-servicing-alignment-sft`

Corrected published revision:
`0ce32f9c7a3edff227005e5b89b089947b87625a`

Prompt-identical training revision:
`fea8aa1cda716954eb7322325e2be25c9f570ea3`

Local manifest:
[`data/banking-servicing-alignment-v4/manifest.json`](../../data/banking-servicing-alignment-v4/manifest.json)

| Split | Initial base | Remediation additions | Composite total |
| --- | ---: | ---: | ---: |
| train | 6,304 | 320 | 6,624 |
| validation | 1,349 | 80 | 1,429 |
| test | 1,347 | 27 | 1,374 |

## Router Artifact

Published repository: `spkc83/retail-bank-conversation-router`

Published revision:
`c8f154266612e79afe20af8abef25761fa56d589`

Training-data revision:
`8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc`

Router code:
[`poc/retail-bank-customer-service-poc/router.py`](../../poc/retail-bank-customer-service-poc/router.py)

| Field | Value |
| --- | ---: |
| Fine-intent macro F1 | `0.990312` |
| Relation macro F1 | `0.996474` |
| In-domain false-refusal rate | `0.000000` |
| OOD false-accept rate | `0.007899` |
| Contextual false-refusal rate | `0.000000` |
| Repair false-refusal rate | `0.000000` |
| External topic-shift false-accept rate | `0.003876` |
| Resume intent / relation error rate | `0.000000 / 0.000000` |
| State-conditioned route / intent error rate | `0.000000 / 0.000000` |
| Runtime transition error rate | `0.000000` |
| State-conditioned false-resume rate | `0.000000` |
| Captured-regression route/intent/relation errors | `0 / 0 / 0` |
| Held-out social/policy generalization errors | `0 / 0` |
| OOD banking boundary | `0.45` |
| Serving in-domain boundary | `0.50` |
| Relation rescue boundary | `0.40` |

## Router Dataset

Published repository: `spkc83/retail-bank-conversation-router-data`

Published revision:
`8efa57dc335d8cfa8e6f2c51446c3d1aa83215dc`

Release lock:
[`data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json`](../../data/sources/banking-conversation-router-v5-social-policy-generalization-candidate5.lock.json)

Manifest SHA-256:
`9cb527bdc337ce4da06e391f1d1e341da80092ab1ac46bf619bd33947f7a3608`

| Split | Rows | Local SHA-256 |
| --- | ---: | --- |
| train | 19,363 | `1e67741213b2ee48a61b6aa20be485f9f634850434637f533f928a858e1572f5` |
| validation | 5,056 | `4df22958f9519355204bcc2910a2874ead44425644056165133126042abcdafa` |
| test | 6,171 | `6af19f8079ff07c087d692ae4c331c55ef33adcdbcd316aa425e866452bd5d97` |

## Runtime Artifact Defaults

| Runtime field | Default | Source |
| --- | --- | --- |
| `RETAIL_BANK_MODEL_ID` | `spkc83/retail-bank-servicing-agent-9b-peft` | [`zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| `RETAIL_BANK_MODEL_REVISION` | `cc95e446af2b5e1d8d9df2751a8192613ad386e3` | [`zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| `RETAIL_BANK_BASE_MODEL_ID` | `spkc83/retail-bank-servicing-agent-9b` | [`zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| `RETAIL_BANK_BASE_MODEL_REVISION` | `1d56824995aa1adecfe20f62ca42fb1c0c443817` | [`zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| `RETAIL_BANK_ADAPTER_ID` | `spkc83/retail-bank-servicing-agent-9b-peft` | [`zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| `RETAIL_BANK_ADAPTER_REVISION` | `cc95e446af2b5e1d8d9df2751a8192613ad386e3` | [`zero_gpu_runtime.py`](../../poc/retail-bank-customer-service-poc/zero_gpu_runtime.py) |
| `RETAIL_BANK_ROUTER_ID` | `spkc83/retail-bank-conversation-router` | [`router.py`](../../poc/retail-bank-customer-service-poc/router.py) |
| `RETAIL_BANK_ROUTER_REVISION` | `c8f154266612e79afe20af8abef25761fa56d589` | [`router.py`](../../poc/retail-bank-customer-service-poc/router.py) |
| `INPUT_TOKEN_BUDGET` | `8192` | [`model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) |
| `MAX_NEW_TOKENS` | `512` | [`model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) |
| `MAX_TOOL_CALLS` | `8` | [`model_service.py`](../../poc/retail-bank-customer-service-poc/model_service.py) |
| Demo usernames | `alex.demo`, `maya.demo` | [`auth.py`](../../poc/retail-bank-customer-service-poc/auth.py) |
| Session database directory | `/tmp/retail-bank-servicing-poc` unless `POC_SESSION_DB_DIR` is set | [`state.py`](../../poc/retail-bank-customer-service-poc/state.py) |

## Source Commit For A New Run

The corrected V5 training source revision is
`75b56ffff45e75ffbee11c0e0552dc35ae124d21`.
For a new paid run, commit and push the intended source state, then obtain its
immutable revision with:

```bash
git rev-parse HEAD
```

Paid launchers require that value to be an exact 40-character commit and verify
that the remote job script exists at the same revision before starting work.
