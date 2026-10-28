#!/usr/bin/env python3
"""vLLM 0.31.0 greedy oracle generator for warp-34.

Generates greedy token IDs (temp=0, ignore_eos=True) and per-step top-5 logprobs
on a single prompt or across the canonical 12-prompt suite (tools/oracle/prompts.json).
"""
import argparse
import json
import os
from pathlib import Path

ORACLE_DIR = Path(__file__).resolve().parent


def write_atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        os.unlink(path)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model ID or snapshot path")
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES string")
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    p.add_argument("--util", type=float, default=0.85, help="gpu_memory_utilization")
    p.add_argument("--max-len", type=int, default=8192, help="max_model_len")
    p.add_argument("--tokens", type=int, default=128, help="max generated tokens")
    p.add_argument("--prompt", default=None, help="Single literal prompt string")
    p.add_argument("--prompt-file", default=None, help="Single prompt file")
    p.add_argument("--suite", action="store_true", help="Run the 12-prompt canonical suite")
    p.add_argument("--out-dir", default=None, help="Output directory for --suite (.ids + .top5.json)")
    p.add_argument("--out", default=None, help="Output .ids file for single prompt")
    p.add_argument("--topk-out", default=None, help="Output .top5.json file for single prompt")
    a = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=a.model,
        dtype="float16",
        gpu_memory_utilization=a.util,
        max_model_len=a.max_len,
        enforce_eager=False,
        tensor_parallel_size=a.tp,
    )

    if a.suite:
        prompts_data = json.loads((ORACLE_DIR / "prompts.json").read_text())
        out_dir = Path(a.out_dir) if a.out_dir else (ORACLE_DIR / "goldens" / "custom")
        out_dir.mkdir(parents=True, exist_ok=True)
        for group in ("chat", "agentic"):
            for idx, prompt_text in enumerate(prompts_data.get(group, [])):
                tag = f"{group}{idx:02d}"
                sp = SamplingParams(
                    temperature=0.0,
                    max_tokens=a.tokens,
                    ignore_eos=True,
                    logprobs=5,
                )
                out = llm.generate([prompt_text], sp)[0].outputs[0]
                ids = list(out.token_ids)
                top5 = [
                    sorted(((tid, float(v.logprob)) for tid, v in lp.items()), key=lambda kv: -kv[1])[:5]
                    for lp in (out.logprobs or [])
                ]
                if len(top5) != len(ids) or any(len(step) == 0 for step in top5):
                    raise SystemExit(f"vllm_greedy: incomplete logprobs on {tag}")
                (out_dir / f"{tag}.ids").write_text(" ".join(map(str, ids)) + "\n")
                write_atomic_json(out_dir / f"{tag}.top5.json", top5)
                print(f"[{tag}] {len(ids)} tokens -> {out_dir / tag}.ids")
        return

    if a.prompt is not None:
        prompt_text = a.prompt
    elif a.prompt_file is not None:
        prompt_text = Path(a.prompt_file).read_text().strip()
    else:
        raise SystemExit("Must pass --prompt, --prompt-file, or --suite")

    want_topk = a.topk_out is not None
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=a.tokens,
        ignore_eos=True,
        logprobs=5 if want_topk else None,
    )
    out = llm.generate([prompt_text], sp)[0].outputs[0]
    ids = list(out.token_ids)
    line = " ".join(map(str, ids))
    print(line)
    if want_topk:
        top5 = [
            sorted(((tid, float(v.logprob)) for tid, v in lp.items()), key=lambda kv: -kv[1])[:5]
            for lp in (out.logprobs or [])
        ]
        if len(top5) != len(ids) or any(len(step) == 0 for step in top5):
            raise SystemExit("vllm_greedy: missing/short logprobs")
        write_atomic_json(Path(a.topk_out), top5)
    if a.out:
        Path(a.out).write_text(line + "\n")


if __name__ == "__main__":
    main()
