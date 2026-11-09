# Structured outputs: XGrammar

warp-34 uses **`xgrammar-rs = 0.2.8`**, the pure-Rust port of
[mlc-ai/XGrammar](https://github.com/mlc-ai/xgrammar), with the structured-output
architecture of vLLM 0.31.0. There is no llguidance backend or fallback.

## Ownership

- `request_constraints.rs`: normalize API controls, reject conflicting constraints,
  and enforce request-size limits (64 KiB output specifications / 512 KiB tools).
- `constraints.rs`: one tokenizer-aware compiler per server, an explicitly bounded
  512 MiB compiler cache, request-local matcher, reasoning activation, and packed
  32-bit masks. Compilation runs through the HTTP CPU-work boundary, not on the
  GPU engine thread.
- `tool_constraints.rs`: native XGrammar structural-tag descriptors, not a custom
  schema-to-grammar converter. Tools use `json_schema` with `style: qwen_xml`.
  Qwen sets the native
  `max_whitespace_cnt: 1` option to prevent empty-argument calls from exhausting
  their output budget in a whitespace loop.
- `parser.rs`: sampler integration and output extraction. Native XML parameter
  strings are raw text; declared tool schemas distinguish strings such as `true`
  and `123` from booleans/numbers. JSON arrays and objects remain structured.

XGrammar owns grammar compilation, matching, completion, and EOS acceptance.
The engine owns GPU mask application and committed/speculative output state.
A private matcher fork advances along hypothetical draft prefixes; rejected or
uncommitted tokens never advance the persistent matcher. Deferred mask generation
continues to overlap with GPU forward execution. Masks exclude vocabulary holes
and model-padding IDs.

## Request behavior

Supported forms include:

- `response_format`: `text`, `json_object`, `json_schema`, `structural_tag`.
- `structured_outputs`: exactly one of `json`, `json_object`, `regex`, `grammar`,
  `choice`, `structural_tag`. JSON schemas and structural tags may be objects or,
  where supported by the field, JSON-encoded strings.
- Grammar strings accept native EBNF (`root ::= ...`) or Lark (`start: ...`).
- Structural tags accept native XGrammar `format` descriptions or the older
  `structures` / `triggers` request shape, lowered to native descriptors.
- `strict: true` is not required to request JSON-schema enforcement. Omitted or
  false does not turn a supplied output schema into an unconstrained request.
- Unsupported known schema constraints and compilation failures are request
  errors, including auto-tool compilation failures. No silent fallback.

Tools use a single explicit policy: declared tools are constrained by native
structural tags when active, including `auto`; there is no separate configurable
vLLM backend/strict-tool feature-flag matrix. `none` installs no tool grammar,
removes tool definitions from the generation prompt, and disables output call
extraction. `auto` allows ordinary text and triggered calls. `required` requires
at least one call; named choice restricts output to the selected function.
`parallel_tool_calls: false` stops tool generation after its first call.
Tools and a separate output grammar cannot currently be combined.

### Deliberate compatibility changes

The generated tool envelopes now follow the upstream builders, **not the old
whitespace-tolerant custom grammar**:

```text
<tool_call>
<function=weather>
<parameter=city>Paris</parameter>
</function>
</tool_call>
```

Qwen XML is the only supported tool-call envelope. Tagged JSON objects are not
extracted as tool calls; JSON object and array values inside XML parameters
remain supported. Schema-aware server extraction requires the canonical envelope
covered by the native grammar; compact or whitespace-altered envelopes in `auto`
mode's ordinary text are not dispatched. Only the generic compatibility parser
(`Qwen3Parser::new`) retains XML whitespace tolerance.

Parameter boundaries take precedence over outer function/tool delimiters. Raw
string values retain their whitespace and may contain outer delimiter text;
object/array alternatives shield delimiters inside JSON strings. Ambiguous raw
strings that look like incomplete JSON remain buffered until `finish()`, whose
returned delta must also be consumed. Numeric conversion preserves JSON integer
precision, and native non-string padding is not retained as value data.

Ambiguous XML values are checked against request-owned, optimized native JSON
value grammars, compiled once during parser preparation without a tokenizer.
References retain the original tool-schema root; literal `const`/`enum` data is
not rewritten. Non-string candidates are preferred only if accepted; exact raw
string bytes are tried before grammar-owned padding is removed. This resolves
mixed enum/const/bounded alternatives without a second handwritten schema
validator. Candidate object key order is retained through `serde_json`'s
`preserve_order` feature, so validation does not need relaxed any-order grammar
rules. Whole-call requirements remain enforced by the generation grammar.

Reasoning is not a grammar production. Before `</think>`, real vocabulary tokens
are unconstrained by the answer grammar, including special tokens. Qwen tools also
support the parser's implicit reasoning-to-tool transition at `<tool_call>`.
Split delimiters and tokens containing both a delimiter and answer bytes are
handled at the byte boundary. Gate-prefix masks are cached across tokens and
speculative forks. Rollback restores both gate and matcher state.

Native XGrammar syntax and support replace llguidance extensions. In particular,
there is no compatibility layer for `x-guidance` options or the old custom Lark
emitter. Known unsupported schema features are rejected before compilation;
this is not a general-purpose JSON Schema validator or a promise to support all
JSON Schema vocabularies.

## Verification

`tests/xgrammar_oracle.rs` checks every vocabulary decision against next-token
masks generated by the official Python XGrammar package used by the local vLLM
installation, then checks acceptance, EOS, and rollback. The checked-in fixture
records its oracle version. Regenerate with:

```sh
python tools/oracle/xgrammar_masks.py
```

`tests/structured_constraints.rs` covers schemas, recursive grammars, native tool
formats, structural tags, reasoning boundaries, deferred masks, and rollback.
`tests/http_constraints.rs` covers real-model plain/vision, MTP, DFlash2, and
TP=2 MoE serving. The full GPU suite remains `mise run check`.

An opt-in CPU benchmark uses a local Qwen tokenizer without downloads:

```sh
W34_QWEN_TOKENIZER=/path/to/tokenizer.json \
  cargo test --release --features cuda --test structured_constraints \
  benchmark_cached_qwen_constraint -- --ignored --nocapture
```

It reports tokenizer setup, cold/cache-hit compilation, and per-token costs. On
this host with the pinned Qwen3.5-4B GPTQ tokenizer, one release-mode run measured
**609.9 ms** tokenizer metadata setup, **5.85 ms** cold schema compilation,
**10.88 µs** cache-hit construction, **0.585 µs/token** cached mask copy, and
**3.482 µs/token** acceptance plus successor-mask generation (five output tokens).
These small measurements are not an end-to-end throughput comparison with vLLM.

## Review decisions

The initial review's requested restoration of whitespace-tolerant Qwen
prefixes was declined: those prefixes intentionally follow XGrammar's Qwen
builder. Reintroducing the
custom language would defeat this migration. The parser metadata reference-budget
finding was fixed: parsing metadata is constructed once before dispatch, limits
and unresolved references return errors, and nested object/array data does not
require recursive schema expansion. No permissive `{}` fallback remains.
