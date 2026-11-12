#!/usr/bin/env python3
"""Capture actual native Qwen3_5MultiTokenPredictor inputs/outputs in vLLM.

Uses eager inference and worker RPC to attach hooks after model initialization.
No substitute attention or hand-written layer reference is used. CUDA_VISIBLE_DEVICES
selects the GPU. Captures the first four MTP calls during one greedy request.
Worker RPC requires VLLM_ALLOW_INSECURE_SERIALIZATION=1 for this trusted local
function; do not enable that setting on an untrusted service. Use HF_HUB_OFFLINE=1.
"""
import argparse
import json
from pathlib import Path


def install_hook(worker, out_dir):
    import hashlib
    import inspect
    import torch
    import vllm
    from safetensors.torch import save_file
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
    runner = worker.model_runner
    speculator = getattr(runner, "speculator", None)
    if speculator is None:
        speculator = runner.drafter
    draft = speculator.model
    modules = [m for m in draft.modules() if isinstance(m, Qwen3_5MultiTokenPredictor)]
    if len(modules) != 1:
        raise RuntimeError(f"expected one native MTP predictor, got {len(modules)}")
    module = modules[0]
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    source = Path(inspect.getfile(Qwen3_5MultiTokenPredictor))
    state = {"count": 0, "inputs": None}
    # Clone module boundaries immediately: fused norms mutate residual inputs.
    for name, child in module.named_modules():
        if not name:
            continue
        original = child.forward
        def traced(*args, _name=name, _forward=original, **kwargs):
            active = state["inputs"] is not None and state["count"] < 4
            def capture(label, value):
                if isinstance(value, torch.Tensor):
                    state["inputs"][f"trace.{_name}.{label}"] = value.detach().cpu().contiguous().clone()
                elif isinstance(value, (tuple, list)):
                    for i, item in enumerate(value):
                        capture(f"{label}.{i}", item)
            if active:
                capture("in", args)
            result = _forward(*args, **kwargs)
            if active:
                capture("out", result)
            return result
        child.forward = traced
    sig = inspect.signature(module.forward)
    def before(mod, args, kwargs):
        if state["count"] >= 4:
            return
        bound = sig.bind(*args, **kwargs)
        # Clone before the donor can overwrite any workspace-backed input.
        state["inputs"] = {name: value.detach().cpu().contiguous().clone()
                           for name, value in bound.arguments.items()
                           if isinstance(value, torch.Tensor)}
    def after(mod, args, kwargs, output):
        if state["count"] >= 4:
            return
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("expected last-PP-stage hidden output")
        tensors = state["inputs"]
        tensors["output"] = output.detach().cpu().contiguous().clone()
        meta = {"vllm": vllm.__version__, "class": type(mod).__name__,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "scope": "actual eager native MTP forward, including donor attention"}
        dest = path / f"call_{state['count']}.safetensors"
        save_file(tensors, dest, metadata={"provenance": json.dumps(meta)})
        dest.chmod(0o644)
        print(f"MTP_CAPTURE {dest} {[(k,tuple(v.shape)) for k,v in tensors.items()]}", flush=True)
        state["count"] += 1
        state["inputs"] = None
    # support_torch_compile overrides __call__ and bypasses nn.Module hooks,
    # even in eager mode. Wrap forward itself while retaining the real donor.
    original_forward = module.forward
    def captured_forward(*args, **kwargs):
        before(module, args, kwargs)
        output = original_forward(*args, **kwargs)
        after(module, args, kwargs, output)
        return output
    module.forward = captured_forward
    return type(module).__name__


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    if not Path(args.model).is_dir():
        raise ValueError("--model must be a pinned local snapshot directory, not a mutable Hub ref")
    llm = LLM(model=args.model, dtype="float16", tensor_parallel_size=1,
              speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 3},
              enforce_eager=True, gpu_memory_utilization=0.65, max_model_len=256,
              max_num_seqs=1, max_num_batched_tokens=256, enable_prefix_caching=False)
    print(llm.collective_rpc(install_hook, args=(str(Path(args.out).resolve()),)), flush=True)
    result = llm.generate(["The capital of France is"],
                          SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
                          use_tqdm=False)[0]
    if len(list(Path(args.out).glob("call_*.safetensors"))) != 4:
        raise RuntimeError("expected four actual MTP captures; hooks did not execute")
    Path(args.out, "request.json").write_text(json.dumps({
        "model": args.model, "prompt_ids": result.prompt_token_ids,
        "output_ids": result.outputs[0].token_ids,
        "dtype": "float16", "k": 3,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
