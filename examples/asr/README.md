# Reviewed ASR overlay example

`asr-utterances.jsonl` demonstrates the normalized input accepted by the ASR
to SFT preparation pipeline. The rows and audio hashes are synthetic; no audio
files or real customer data are included.

Each row points to one already-validated semantic SFT record. The pipeline
replaces only that record's latest customer utterance with the reviewed ASR
transcript. It does not copy an agent transcript into the target and it does
not infer a tool call from raw ASR text.

Run the example from the repository root:

```bash
python scripts/retail_bank/prepare_asr_sft_data.py \
  --asr-input examples/asr/asr-utterances.jsonl \
  --base-manifest data/banking-servicing-alignment-v4/manifest.json \
  --output-dir /tmp/retail-bank-asr-sft-example
```

The output contains trainer-compatible `train.jsonl`, `validation.jsonl`, and
`test.jsonl` files plus a manifest and preparation report. The command writes
only to the explicitly supplied output directory.
