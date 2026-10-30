"""Engine-independent payloads; synthetic sizes are bytes, never claimed as tokens."""

import copy
import json


def messages(case, index, namespace):
    data = case["fixture_data"]
    cache = case.get("cache", "unique")
    # Put the nonce FIRST, before the shared system text, to defeat prefix reuse.
    tag = f"Benchmark session: {namespace}:{index if cache == 'unique' else 'shared'}.\n"
    if case.get("kind", "text") == "agent_replay":
        return [{"role": "system", "content": tag + data["system"]}]
    prompt = data["prompts"][index % len(data["prompts"])]
    repeats = case.get("context_repetitions", 1)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("context_repetitions must be a positive integer")
    context = data.get("context", "") * repeats
    return [{"role": "system", "content": tag + data.get("system", "You are a helpful assistant.")},
            {"role": "user", "content": context + prompt}]


def body(engine, case, history):
    payload = {"model": engine["model"], "messages": copy.deepcopy(history),
               "max_tokens": case["max_tokens"], "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}}
    payload.update(copy.deepcopy(engine.get("request_overrides", {})))
    if case.get("kind", "text") == "agent_replay":
        payload["tool_choice"] = "none"
    return payload


def append_replay(history, turn, session, step):
    """Replay recorded observations, not an autonomous tool execution score.

    Tools never execute commands or read the local filesystem. Actual generated
    assistant prose is kept in history by the runner before this is called.
    """
    history.append({"role": "user", "content": turn["user"]})
    tools = turn.get("tools", [])
    if tools:
        calls = [{"id": f"replay_{session}_{step}_{i}", "type": "function",
                  "function": {"name": tool["name"],
                               "arguments": json.dumps(tool["arguments"], sort_keys=True)}}
                 for i, tool in enumerate(tools)]
        history.append({"role": "assistant", "content": None, "tool_calls": calls})
        history.extend({"role": "tool", "tool_call_id": call["id"], "content": tool["result"]}
                       for call, tool in zip(calls, tools))
