"""Engine-independent payloads; synthetic sizes are bytes, never claimed as tokens."""

import copy
import json


def agent_trajectory(data, index):
    if not isinstance(data, dict):
        raise ValueError("agent fixture must be a JSON object")
    if "trajectories" in data:
        if "turns" in data:
            raise ValueError("agent fixture cannot specify both turns and trajectories")
        trajectories = data["trajectories"]
        if not isinstance(trajectories, list) or not trajectories:
            raise ValueError("agent replay requires nonempty trajectories")
        traj = trajectories[index % len(trajectories)]
    else:
        traj = data
    if (not isinstance(traj, dict) or not isinstance(traj.get("system"), str)
            or not traj.get("turns") or not isinstance(traj["turns"], list)):
        raise ValueError("agent replay requires system and turns")
    return traj


def messages(case, index, namespace):
    data = case["fixture_data"]
    cache = case.get("cache", "unique")
    # Put the nonce FIRST, before the shared system text, to defeat prefix reuse.
    tag = f"Benchmark session: {namespace}:{index if cache == 'unique' else 'shared'}.\n"
    if case.get("kind", "text") == "agent_replay":
        traj = agent_trajectory(data, index)
        return [{"role": "system", "content": tag + traj["system"]}]
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
    Follow-up turns in Open-SWE-Traces-style tool loops may omit `user` (`None`)
    so the single initial user query remains `last_query` and prior `<think>`
    blocks stay prefix-cached verbatim across engines.
    """
    if turn.get("user") is not None:
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


def decision_request(case, index, namespace):
    data = case["fixture_data"]
    requests = data.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("decision fixture requires nonempty requests")
    item = copy.deepcopy(requests[index % len(requests)])
    cache = case.get("cache", "unique")
    tag = f"Benchmark session: {namespace}:{index if cache == 'unique' else 'shared'}.\n"
    repeats = case.get("context_repetitions", 1)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or not 1 <= repeats <= 256:
        raise ValueError("context_repetitions must be an integer in 1..256")
    state = item["state"]
    if isinstance(state, str):
        item["state"] = tag + (state * repeats)
    elif isinstance(state, dict):
        merged = {"_benchmark_session": tag.strip(), **state}
        if repeats > 1:
            merged["_context_padding"] = [copy.deepcopy(state) for _ in range(repeats - 1)]
        item["state"] = merged
    elif isinstance(state, list):
        item["state"] = [tag.strip(), *(state * repeats)]
    return item


def decision_body(engine, case, item):
    payload = {
        "model": engine["model"],
        "state": copy.deepcopy(item["state"]),
        "questions": copy.deepcopy(item["questions"]),
    }
    if "order_passes" in case:
        payload["order_passes"] = case["order_passes"]
    payload.update(copy.deepcopy(engine.get("request_overrides", {})))
    return payload

