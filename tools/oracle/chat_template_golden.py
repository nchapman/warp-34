#!/usr/bin/env python3
"""Generate text ChatML fixtures with the native Qwen checkpoint Jinja oracle.

Pass the unmodified Qwen3.5/3.6 chat_template.jinja explicitly. Some quantized
checkpoints ship custom templates; the source hash is recorded to prevent
silently treating those as the native template. No model/GPU loading required.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from transformers.utils.chat_template_utils import _compile_jinja_template


def cases():
    tools = [{"type": "function", "function": {
        "name": "weather", "description": "Weather in <city> & région",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "options": {"type": "object"}},
            "required": ["city"], "additionalProperties": False}}}]
    calls = [{"id": "call_a", "type": "function", "function": {
        "name": "weather", "arguments": '{"city":"Paris","options":{"units":"°C","days":[1,2],"rain":true}}'}},
        {"id": "call_b", "type": "function", "function": {
            "name": "weather", "arguments": {"city": "Rome"}}}]
    histories = [
        ("plain", [{"role": "user", "content": "  hello  "}], None),
        ("system", [{"role": "system", "content": "  Be precise. "},
                    {"role": "user", "content": "weather?"}], tools),
        ("tools_without_system", [{"role": "user", "content": "weather?"}], tools),
        ("multi_tool", [{"role": "user", "content": "Compare weather."},
                        {"role": "assistant", "content": None,
                         "reasoning_content": "  Need both cities.  ", "tool_calls": calls},
                        {"role": "tool", "content": " sunny ", "tool_call_id": "call_a"},
                        {"role": "tool", "content": "rain", "tool_call_id": "call_b"}], tools),
        ("old_reasoning", [{"role": "user", "content": "first"},
                           {"role": "assistant", "content": "<think>\nold\n</think>\n\nanswer"},
                           {"role": "user", "content": "next"},
                           {"role": "assistant", "content": "more", "reasoning_content": "new"}], None),
        ("text_before_call", [{"role": "user", "content": "weather?"},
                              {"role": "assistant", "content": "Checking.", "tool_calls": calls[:1]},
                              {"role": "tool", "content": "sunny"},
                              {"role": "assistant", "content": "Sunny."},
                              {"role": "user", "content": "thanks"}], tools),
        ("wrapped_response", [{"role": "user", "content": "weather?"},
                              {"role": "assistant", "content": "", "tool_calls": calls[:1]},
                              {"role": "user", "content": "<tool_response>sunny</tool_response>"}], tools),
    ]
    for name, messages, tools in histories:
        for thinking in (False, True):
            for preserve in (False, True):
                yield {"name": f"{name}_thinking={thinking}_preserve={preserve}",
                       "messages": messages, "tools": tools,
                       "enable_thinking": thinking, "preserve_thinking": preserve}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.template.read_text()
    template = _compile_jinja_template(source)
    output = []
    for case in cases():
        # OpenAI encodes arguments as a string; Transformers expects a mapping.
        messages = copy.deepcopy(case["messages"])
        for message in messages:
            for call in message.get("tool_calls", []):
                arguments = call["function"].get("arguments", {})
                if isinstance(arguments, str):
                    call["function"]["arguments"] = json.loads(arguments)
        case["expected"] = template.render(
            messages=messages, tools=case["tools"], add_generation_prompt=True,
            enable_thinking=case["enable_thinking"], preserve_thinking=case["preserve_thinking"])
        output.append(case)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"template_sha256": hashlib.sha256(source.encode()).hexdigest(),
                                    "cases": output}, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(output)} native Jinja fixtures to {args.out}")


if __name__ == "__main__":
    main()
