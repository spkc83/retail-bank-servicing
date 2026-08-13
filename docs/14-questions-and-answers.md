# Questions and Answers

## Does the model receive the `messages` JSON array?

No. The `messages` array is the dataset and application representation of a
conversation. The transformer ultimately receives integer token tensors, not
the JSON array itself.

The common path is:

```text
messages array
  -> validate and normalize message objects
  -> apply the Granite chat template and public tool schemas
  -> tokenize the rendered conversation
  -> input_ids and attention_mask
  -> Granite causal language model
```

For example, a stored SFT record can contain:

```json
[
  {
    "role": "system",
    "content": "You are a retail banking assistant.",
    "loss": false
  },
  {
    "role": "user",
    "content": "Show my cards.",
    "loss": false
  },
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "list_cards",
          "arguments": {}
        }
      }
    ],
    "loss": true
  },
  {
    "role": "tool",
    "tool_call_id": "call_1",
    "name": "list_cards",
    "content": "{\"cards\":[{\"last_four\":\"4821\"}]}",
    "loss": false
  },
  {
    "role": "assistant",
    "content": "Your active debit card ends in 4821.",
    "loss": true
  }
]
```

`ToolWireAdapter` removes pipeline-only fields such as `loss`, normalizes tool
calls and tool results, and invokes the tokenizer's Granite chat template. The
template determines the exact system, user, assistant, and tool role markers,
tagged tool-call syntax, end-of-turn tokens, and generation prefix. The model
therefore sees the tokenized Granite wire format rather than literal JSON keys
such as `"role"` and `"content"`.

The tool definitions are also not embedded in the dataset row. The public tool
manifest is passed separately as the `tools` argument to the chat template, so
the same tool schemas can be used consistently during training and inference.

### During training

The complete conversation, including the correct assistant outputs, is
rendered into `input_ids`. The training labels decide which tokens contribute
to the loss:

```text
system tokens                 -> label -100, ignored
user tokens                   -> label -100, ignored
assistant tool-call tokens    -> trained when loss is true
tool-result tokens            -> label -100, ignored
assistant answer tokens       -> trained when loss is true
padding                       -> label -100, ignored
```

Including the assistant target in `input_ids` is normal causal-language-model
teacher forcing. The model's loss is shifted by one token, so it must predict
each assistant token from only the tokens that precede it. Future target tokens
are not available as context for the current prediction.

The repository implements this assistant-span masking in
[`ToolWireAdapter.render_training()`](../src/hello_slm/banking_tool_wire.py).
[`tokenize_records()`](../scripts/retail_bank/cloud_train_tool_sft.py) converts
the rendered records into `input_ids`, `attention_mask`, and `labels`, and the
collator pads them into training batches.

Fields outside `messages`, including `expected`, `grounding_facts`, metadata,
provenance, and split keys, are used by dataset validation, governance, and
evaluation. They are not included in the training prompt unless code
explicitly copies their values into a message.

### During inference

The expected assistant response is not present. The first generation pass
receives:

```text
system message
+ selected conversation history
+ current user message
+ public tool schemas
+ assistant generation prefix
```

If Granite emits a tool call, the application:

1. Parses and validates the generated tool call.
2. Executes the corresponding operation against the synthetic banking backend.
3. Appends the generated assistant tool-call message and actual tool-result
   message to the conversation.
4. Applies the same Granite chat template again.
5. Invokes Granite again to generate the grounded customer-facing response.

This multi-pass loop is implemented in
[`model_service.py`](../poc/retail-bank-customer-service-poc/model_service.py).
The evaluation runner deliberately removes the expected assistant target before
generation in
[`first_phase_messages()`](../scripts/retail_bank/cloud_generate_tool_eval.py).

### Why the application sometimes serializes messages as JSON

The servicing application serializes `{messages, tools}` to canonical JSON to
calculate a diagnostic prompt hash. That serialized string is evidence for
tracing which structured prompt was used; it is not the prompt sent to Granite.
The generation backend separately passes the message objects and tools through
the tokenizer's chat template and sends the resulting token tensors to the
model.

### Short answer

The JSON array is a structured interchange format for the data pipeline and
application. Granite consumes the model-specific, chat-template-rendered token
sequence derived from that array.
