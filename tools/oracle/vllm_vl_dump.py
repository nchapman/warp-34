#!/usr/bin/env python3
"""Native vLLM 0.31.0 Qwen3.5 single-image oracle (no parity assertion).

Trusted local workers only: VLLM_ALLOW_INSECURE_SERIALIZATION=1 is required
for collective_rpc callables. Output must be a new/empty directory on a filesystem
shared by all TP workers. No chat template is applied. Imports are host-only until
main/install_hook. Each rank must capture exactly one real image invocation.
"""
import argparse
import hashlib
import json
from pathlib import Path

MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
PROCESSOR = dict(min_pixels=32 * 32, max_pixels=1024 * 1024,
                 patch_size=16, merge_size=2, temporal_patch_size=2)


def validate_prompt(prompt):
    if prompt.count(MARKER) != 1 or prompt.count("<|image_pad|>") != 1:
        raise ValueError("one image requires exactly one native " + MARKER)
    if "<|video_pad|>" in prompt:
        raise ValueError("video is not supported")


def validate_shape(shape, grid, hidden):
    if len(grid) != 1 or len(grid[0]) != 3:
        raise ValueError("expected one image_grid_thw row")
    t, h, w = grid[0]
    if t != 1 or h <= 0 or w <= 0 or h % 2 or w % 2:
        raise ValueError("invalid still-image patch16/merge2/temporal2 grid")
    if list(shape) != [t * h * w // 4, hidden]:
        raise ValueError("visual_embeddings must be [merged_rows, hidden]")


def generation_json(output, tokens):
    ids = list(output.token_ids)
    if len(ids) != tokens or output.logprobs is None or len(output.logprobs) != tokens:
        raise RuntimeError("incomplete greedy token/logprob capture")
    steps = []
    for token, entries in zip(ids, output.logprobs):
        best = sorted(entries.items(), key=lambda kv: (-kv[1].logprob, kv[0]))[:5]
        if len(best) != 5 or token not in entries:
            raise RuntimeError("missing top5 or sampled token logprob")
        steps.append({"token_id": token, "logprob": entries[token].logprob,
                      "top5": [{"token_id": k, "logprob": v.logprob,
                                "rank": v.rank} for k, v in best]})
    return {"token_ids": ids, "steps": steps, "score_kind": "raw_logprobs"}


def trace_vision_boundary(module, sample, output_name, input_name=None):
    # support_torch_compile replaces __call__ and bypasses nn.Module hooks,
    # even in eager mode. Wrap forward only for these decorated boundaries.
    original = module.forward

    def traced(*args, **kwargs):
        if input_name is not None:
            sample(input_name, args[0] if args else kwargs["x"])
        result = original(*args, **kwargs)
        sample(output_name, result)
        return result

    module.forward = traced


def install_hook(worker, out_dir, override_visual=None, capture_layers=False, capture_vision_layers=False):
    import inspect
    import torch
    from safetensors.torch import load_file, save_file
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration)
    model = worker.model_runner.model
    if not isinstance(model, (Qwen3_5ForConditionalGeneration,
                              Qwen3_5MoeForConditionalGeneration)):
        raise RuntimeError(f"not native Qwen3.5: {type(model)}")
    if model.deepstack_num_level or model.is_multimodal_pruning_enabled or model.use_data_parallel:
        raise RuntimeError("requires unpruned, non-deepstack, weight-TP vision")
    vc = model.config.vision_config
    if (vc.patch_size, vc.spatial_merge_size, vc.temporal_patch_size) != (16, 2, 2):
        raise RuntimeError("vision config does not match w34 geometry")
    state = {"count": 0, "text_count": 0, "rank": worker.rank}
    worker._w34_vl_capture = state
    original = model._process_image_input
    source = Path(inspect.getfile(original))
    state["source"] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}

    vision_tensors = {}
    if capture_vision_layers:
        # Audited qwen3_vl: patch_embed returns [L,H]; blocks use [L,1,H].
        # Qwen2_5 attention proj input is [L,1,H], QKV is pre-RoPE.
        if model.config.text_config is None or worker.rank != 0:
            raise RuntimeError("vision layer diagnostics require TP=1")
        visual_source = Path(inspect.getfile(type(model.visual)))
        state["vision_source"] = {"path": str(visual_source), "sha256": hashlib.sha256(visual_source.read_bytes()).hexdigest()}
        state["vision_trace_layout"] = "[3,channels], rows [0,L//2,L-1]; singleton axes removed; duplicates retained"

        def sample(name, value):
            key = "vision_trace." + name
            if key in vision_tensors:
                return
            if isinstance(value, tuple):
                value = value[0]
            flat = value.detach().reshape(-1, value.shape[-1])
            rows = flat.shape[0]
            vision_tensors[key] = flat[[0, rows // 2, rows - 1]].float().cpu().contiguous().clone()

        def hook(module, name, pre=False):
            if pre:
                module.register_forward_pre_hook(lambda m, a: sample(name, a[0]))
            else:
                module.register_forward_hook(lambda m, a, o: sample(name, o))

        trace_vision_boundary(model.visual.patch_embed, sample, "patch_embed")
        for i, block in enumerate(model.visual.blocks):
            prefix = f"blocks.{i}."
            trace_vision_boundary(block, sample, prefix + "output", prefix + "input")
            for name, module in [("norm1", block.norm1), ("qkv", block.attn.qkv),
                                 ("proj", block.attn.proj), ("norm2", block.norm2),
                                 ("fc1", block.mlp.linear_fc1), ("fc2", block.mlp.linear_fc2)]:
                hook(module, prefix + name)
            hook(block.attn.proj, prefix + "attn_output", pre=True)
        hook(model.visual.merger, "merger.input", pre=True)
        hook(model.visual.merger, "merger.output")

    def captured(image_input):
        if state["count"] or image_input["type"] != "pixel_values":
            raise RuntimeError("expected exactly one real pixel_values image call")
        # Clone before native execution; never substitute the vision computation.
        pixels = image_input["pixel_values"].detach().cpu().contiguous().clone()
        grid = image_input["image_grid_thw"].detach().cpu().contiguous().clone()
        result = original(image_input)
        if override_visual is not None:
            values = json.loads(Path(override_visual).read_text())["visual_embeddings"]
            replacement = torch.tensor(values, dtype=result[0].dtype, device=result[0].device).reshape(result[0].shape)
            result = (replacement,)
        if len(result) != 1:
            raise RuntimeError("expected one native image embedding")
        visual = result[0].detach().cpu().contiguous().clone()
        validate_shape(visual.shape, grid.tolist(), model.config.text_config.hidden_size)
        if visual.dtype != torch.float16 or not torch.isfinite(visual).all():
            raise RuntimeError("expected finite FP16 visual output")
        # w34 dump stores FP16 values widened to F32; preserve exact native FP16 too.
        tensors = {"visual_embeddings": visual.float(), "visual_embeddings_fp16": visual,
                   "pixel_values": pixels, "image_grid_thw": grid}
        tensors.update(vision_tensors)
        name = f"rank_{worker.rank}.safetensors"
        save_file(tensors, str(Path(out_dir) / name))
        state.update(count=1, file=name, grid=grid.tolist(), tensors={
            k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()})
        return result

    model._process_image_input = captured
    original_text = model.language_model.model.forward
    layer_tensors = {}
    if capture_layers:
        def layer_hook(index):
            def capture(module, args, output):
                if state["text_count"] == 0:
                    hidden, residual = output
                    layer_tensors[f"layer_{index}_hidden_after"] = (hidden + residual).detach().float().cpu().contiguous().clone()
            return capture
        for index, layer in enumerate(model.language_model.model.layers):
            layer.register_forward_hook(layer_hook(index))

    def captured_text(*args, **kwargs):
        if state["text_count"]:
            return original_text(*args, **kwargs)
        if state["count"] != 1:
            raise RuntimeError("text prefill preceded real image capture")
        embeds = kwargs["inputs_embeds"].detach().cpu().contiguous().clone()
        positions = kwargs["positions"].detach().cpu().contiguous().clone()
        result = original_text(*args, **kwargs)
        final = result.detach().cpu().contiguous().clone()
        if final.shape != embeds.shape or final.dtype != torch.float16 or not torch.isfinite(final).all():
            raise RuntimeError("invalid native text prefill capture")
        tensors = load_file(str(Path(out_dir) / state["file"]))
        if not torch.equal(positions, positions.to(torch.int32).to(positions.dtype)):
            raise RuntimeError("mrope positions exceed INT32 capture range")
        tensors.update(embed=embeds.float(), final_norm=final.float(), mrope_positions=positions.to(torch.int32))
        tensors.update(layer_tensors)
        save_file(tensors, str(Path(out_dir) / state["file"]))
        state.update(text_count=1, tensors={
            k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()})
        return result

    model.language_model.model.forward = captured_text
    return {"rank": worker.rank, "model_class": type(model).__name__}


def capture_status(worker):
    state = worker._w34_vl_capture
    if state["count"] != 1 or state.get("text_count") != 1:
        raise RuntimeError(f"incomplete vision/text hook capture on rank {worker.rank}")
    return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("model", "image", "prompt", "out-dir"):
        ap.add_argument("--" + flag, required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--diagnostic-override-visual", help="Diagnostic only: replace native visual outputs with explicit JSON values; never a canonical oracle")
    ap.add_argument("--capture-layers", action="store_true", help="Also capture every decoder layer's post-MLP residual sum")
    ap.add_argument("--capture-vision-layers", action="store_true", help="Diagnostic TP=1 first/middle/last vision rows")
    args = ap.parse_args()
    if args.capture_vision_layers and args.tp != 1:
        ap.error("--capture-vision-layers requires --tp 1")
    validate_prompt(args.prompt)
    if min(args.tp, args.tokens, args.max_len) <= 0 or args.tokens >= args.max_len:
        ap.error("positive tp/tokens/max-len required, tokens < max-len")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("--out-dir must be empty (no stale captures)")
    from PIL import Image
    image_path = Path(args.image).resolve(strict=True)
    image_bytes = image_path.read_bytes()
    import io
    with Image.open(io.BytesIO(image_bytes)) as src:
        if src.format not in ("PNG", "JPEG"):
            raise ValueError("only real PNG/JPEG inputs are supported")
        image_info = {"path": str(image_path), "sha256": hashlib.sha256(image_bytes).hexdigest(),
                      "format": src.format, "size": list(src.size), "source_mode": src.mode}
        image = src.convert("RGB")
    from huggingface_hub import snapshot_download
    model = Path(args.model).expanduser()
    if not model.is_dir():
        model = Path(snapshot_download(repo_id=args.model))
    model = model.resolve(strict=True)
    import vllm
    if vllm.__version__ != "0.31.0":
        raise RuntimeError(f"requires audited vLLM 0.31.0, got {vllm.__version__}")
    # Qwen3_5ProcessingInfo inherits Qwen3VLProcessingInfo.get_hf_processor,
    # explicitly requesting Qwen3VLProcessor, whose image_processor is Qwen2VL.
    from transformers import Qwen3VLProcessor
    processor = Qwen3VLProcessor.from_pretrained(str(model), **PROCESSOR)
    ip = processor.image_processor
    attrs = {k: getattr(ip, k) for k in ("patch_size", "merge_size", "temporal_patch_size")}
    attrs["size"] = dict(ip.size)
    if any(attrs[k] != PROCESSOR[k] for k in ("patch_size", "merge_size", "temporal_patch_size")) or attrs["size"] != {"shortest_edge": 1024, "longest_edge": 1048576}:
        raise RuntimeError(f"unexpected HF processor geometry: {attrs}")
    from vllm import LLM, SamplingParams
    settings = dict(temperature=0, max_tokens=args.tokens, ignore_eos=True, logprobs=5, seed=0)
    engine = dict(dtype="float16", tensor_parallel_size=args.tp, enforce_eager=True,
                  max_model_len=args.max_len, max_num_seqs=1,
                  enable_prefix_caching=False, enable_chunked_prefill=False,
                  max_num_batched_tokens=args.max_len, mm_processor_cache_gb=0,
                  mm_encoder_tp_mode="weights", limit_mm_per_prompt={"image": 1, "video": 0},
                  mm_processor_kwargs=PROCESSOR, logprobs_mode="raw_logprobs", seed=0)
    llm = LLM(model=str(model), **engine)
    override = str(Path(args.diagnostic_override_visual).resolve(strict=True)) if args.diagnostic_override_visual else None
    installed = llm.collective_rpc(install_hook, args=(str(out), override, args.capture_layers, args.capture_vision_layers))
    result = llm.generate([{"prompt": args.prompt, "multi_modal_data": {"image": image}}],
                          SamplingParams(**settings), use_tqdm=False)[0]
    captures = llm.collective_rpc(capture_status)
    if len(captures) != args.tp or {c["rank"] for c in captures} != set(range(args.tp)):
        raise RuntimeError("missing TP capture ranks")
    if not result.prompt_token_ids or len(result.outputs) != 1:
        raise RuntimeError("missing processed prompt or greedy output")
    generation = generation_json(result.outputs[0], args.tokens)
    (out / "generation.json").write_text(json.dumps(generation, indent=2, allow_nan=False) + "\n")
    request = {"schema_version": 1, "diagnostic_override_visual": override,
               "diagnostic_override_sha256": hashlib.sha256(Path(override).read_bytes()).hexdigest() if override else None,
               "model_requested": args.model, "model_resolved_snapshot": str(model),
               "config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
               "image": image_info, "raw_prompt": args.prompt,
               "prompt_token_ids": list(result.prompt_token_ids), "generation_settings": settings,
               "engine_settings": engine, "vllm_version": vllm.__version__,
               "processor_class": type(processor).__name__, "image_processor_class": type(ip).__name__,
               "image_processor_attrs": attrs, "hooks": installed, "tensor_schema": captures,
               "canonical_tensor_file": "rank_0.safetensors", "generation_file": "generation.json",
               "scope": "native visual embeddings, spliced embed, final_norm, and mrope_positions; no parity claim",
               "mrope": "captured from native text model positions input"}
    (out / "request.json").write_text(json.dumps(request, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
