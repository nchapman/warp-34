"""Regenerate CPU token-mask fixtures using the official XGrammar Python package.

Run in the pinned vLLM environment (xgrammar==0.2.7):
    python tools/oracle/xgrammar_masks.py
No model, GPU, network, or Rust implementation is used by this oracle.
"""

import importlib.metadata
import json
from pathlib import Path

import tokenizers
import transformers
import xgrammar as xgr


def main():
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocab = {char: index for index, char in enumerate(alphabet)}
    vocab["<|im_end|>"] = 256
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, []))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    tokenizer.add_special_tokens(["<|im_end|>"])
    hf = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="<|im_end|>"
    )
    info = xgr.TokenizerInfo.from_huggingface(hf, vocab_size=260)
    compiler = xgr.GrammarCompiler(info)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    cases = [
        ("regex", "[A-Z]{2}[0-9]{2}", "AB12"),
        ("schema", schema, '{"ok":true}'),
        ("grammar", 'root ::= "(" root ")" | "x"', "((x))"),
        ("qwen_tools", [{"type": "function", "function": {
            "name": "echo", "parameters": {"type": "object", "properties": {
                "value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}
        }}], "<tool_call>\n<function=echo>\n<parameter=value>true</parameter>\n</function>\n</tool_call>"),
    ]
    results = []
    for kind, source, text in cases:
        if kind == "regex":
            compiled = compiler.compile_regex(source)
        elif kind == "schema":
            compiled = compiler.compile_json_schema(json.dumps(source))
        elif kind == "qwen_tools":
            descriptor = xgr.get_model_structural_tag(
                "qwen_3_5", tools=source, tool_choice="required", reasoning=False,
                max_whitespace_cnt=1,
            )
            compiled = compiler.compile_structural_tag(descriptor.model_dump_json())
        else:
            compiled = compiler.compile_grammar(source)
        matcher = xgr.GrammarMatcher(compiled)
        mask = xgr.allocate_token_bitmask(1, 260)
        ids = tokenizer.encode(text).ids + [256]
        masks = []
        for token in ids:
            matcher.fill_next_token_bitmask(mask)
            masks.append([int(word) & 0xFFFFFFFF for word in mask[0].tolist()])
            assert matcher.accept_token(token), (kind, token)
        assert matcher.is_terminated()
        results.append({"kind": kind, "source": source, "tokens": ids, "masks": masks})
    result = {
        "oracle": "mlc-ai/xgrammar",
        "version": importlib.metadata.version("xgrammar"),
        "tokenizer": json.loads(tokenizer.to_str()),
        "vocab_size": 260,
        "eos": 256,
        "cases": results,
    }
    output = Path(__file__).resolve().parents[2] / "crates/w34-host/tests/fixtures/xgrammar_masks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
