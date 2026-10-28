#!/usr/bin/env python3
"""Actual SGLang DFlash2 model capture, with an explicit attention-boundary adapter.

Run ONLY with physical GPU1 isolated by Docker, e.g.:
 docker run --rm --gpus device=1 --ipc=host -e CUDA_VISIBLE_DEVICES=0 \
   -v "$PWD":/work -v "$HOME/.cache/huggingface":/hf:ro \
   --entrypoint python lmsysorg/sglang:v0.5.19 \
   /work/tools/oracle/sglang_dflash_model_dump.py --inputs /work/.pi/dflash-model-inputs \
   --model /hf/hub/models--nchapman--Qwen3.5-4B-DFlash2/snapshots/8d8bb62873a4fe6165e644163e9ff76e06ab78ba \
   --out /work/tools/oracle/goldens/sglang_dflash_model_4b.safetensors

Not an end-to-end serving oracle: no scheduler, target model, target head, or
selector walk. Genuine donor forward/load_weights/context projection, norms,
convolutions, QK/RoPE and MLP; only KV storage/attention backend is adapted.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

IMAGE = "lmsysorg/sglang:v0.5.19"
IMAGE_ID = "sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--diagnostics", action="store_true", help="Capture real-shape operator operands into a NEW fixture")
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite oracle capture: {args.out}")
    import torch
    import sglang
    import flashinfer
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import Qwen3Config
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.models import dflash
    from sglang.srt.speculative import dflash_utils
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    assert sglang.__version__ == "0.5.19"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Use documented GPU1 Docker isolation"
    assert torch.cuda.device_count() == 1, "Expose physical GPU1 only"
    torch.set_default_dtype(torch.bfloat16)
    raw = json.loads((args.model / "config.json").read_text())
    info = json.loads((args.inputs / "input.json").read_text())
    n, m, h = int(info["rows"]), int(info["block_size"]), int(info["hidden_size"])
    assert (n, m, h) == (5, 16, 2560)
    config = Qwen3Config(**raw["transformer_layer_config"])
    config.dflash_config = {k: raw[k] for k in (
        "block_size", "conv_group_size", "conv_kernel_size", "selector_rank",
        "selector_top_k", "mask_token_id", "sample_from_anchor")}
    config.dflash_config["target_layer_ids"] = [x - 1 for x in raw["aux_hidden_state_layer_ids"]]
    # ServerArgs resolves HF architecture metadata; vendor outer config is not
    # AutoConfig-compatible. Publish from the same normalized nested Qwen3 body.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="dflash-normalized-") as tmp:
        config.save_pretrained(tmp)
        set_global_server_args_for_scheduler(ServerArgs(model_path=tmp, tp_size=1, dtype="bfloat16", attention_backend="flashinfer"))
    from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method="tcp://127.0.0.1:29591", backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1, backend="nccl")
    tensors = {}

    def capture(name, x):
        # Clone immediately: donor fused residual/norm kernels may mutate inputs.
        tensors[name] = x.detach().cpu().contiguous().clone()

    def read_raw(name, dtype, shape):
        data = (args.inputs / name).read_bytes()
        x = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape).clone()
        capture("input." + name, x)
        return x.cuda()

    class AttentionAdapter:
        def __init__(self):
            self.context = {}
            self.workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
            assert not kwargs, kwargs
            assert layer.sliding_window_size == -1
            assert layer.attn_type == dflash.AttentionType.ENCODER_ONLY
            ck, cv = self.context[layer.layer_id]
            k = torch.cat((ck, k), dim=0).contiguous()
            v = torch.cat((cv, v), dim=0).contiguous()
            q = q.reshape(m, layer.tp_q_head_num, layer.head_dim).contiguous()
            # Page size one represents exactly the concatenated committed+draft KV.
            kv = torch.stack((k, v), dim=1).unsqueeze(2).contiguous()
            i32 = lambda a: torch.tensor(a, dtype=torch.int32, device="cuda")
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, kv_layout="NHD", backend="fa2")
            wrapper.plan(i32([0, m]), i32([0, n + m]), i32(list(range(n + m))),
                         i32([1]), layer.tp_q_head_num, layer.tp_k_head_num,
                         layer.head_dim, 1, causal=False, window_left=-1,
                         q_data_type=q.dtype, kv_data_type=k.dtype)
            out = wrapper.forward(q, kv, causal=False, window_left=-1, sm_scale=layer.scaling)
            for key, value in (("q", q), ("k_all", k), ("v_all", v), ("attention", out)):
                capture(f"layers.{layer.layer_id}.{key}", value)
            return out.reshape(m, -1)

    # TP1 topology injection only; no model arithmetic is replaced.
    with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0), torch.device("cuda"), torch.no_grad():
        model = dflash.DFlash2DraftModel(config).eval()
        # Poison all parameters so an ignored/missing checkpoint weight fails closed.
        for p in model.parameters():
            p.fill_(float("nan"))
        weight_files = sorted(args.model.glob("*.safetensors"))
        assert weight_files
        def weights():
            for path in weight_files:
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        yield key, f.get_tensor(key)
        model.load_weights(weights())
        for name, p in model.named_parameters():
            if not torch.isfinite(p).all().item():
                raise RuntimeError(f"Unloaded/nonfinite donor parameter: {name}")
        model.set_block_size(m)
        model.fc.register_forward_hook(lambda mod, inp, out: capture("context.fc", out))
        for i, layer in enumerate(model.layers):
            def hook(mod, inp, out, i=i):
                capture(f"layers.{i}.hidden", out[0])
                capture(f"layers.{i}.residual", out[1])
            layer.register_forward_hook(hook)
        if args.diagnostics:
            # RMSNorm may mutate both operands. Capture BEFORE forward and use
            # invocation IDs because context preparation reuses layer norms.
            norm_calls = {}
            for name, module in model.named_modules():
                if name.startswith("layers.0.") and name.endswith(("qkv_proj", "o_proj", "gate_up_proj", "down_proj")):
                    def linear_pre(mod, inp, name=name):
                        capture(f"gemms.{name}.input", inp[0])
                        capture(f"gemms.{name}.weight", mod.weight)
                    def linear_post(mod, inp, out, name=name):
                        capture(f"gemms.{name}.output", out[0] if isinstance(out, tuple) else out)
                    module.register_forward_pre_hook(linear_pre)
                    module.register_forward_hook(linear_post)
                if isinstance(module, dflash.RMSNorm):
                    def norm_pre(mod, inp, name=name):
                        index = norm_calls.get(name, 0)
                        norm_calls[name] = index + 1
                        prefix = f"norms.{name}.call{index}"
                        mod._oracle_norm_prefix = prefix
                        for j, value in enumerate(inp):
                            if isinstance(value, torch.Tensor):
                                capture(f"{prefix}.input{j}", value)
                        capture(f"{prefix}.weight", mod.weight)
                        capture(f"{prefix}.eps", torch.tensor([mod.variance_epsilon], dtype=torch.float32))
                    def norm_post(mod, inp, out):
                        prefix = mod._oracle_norm_prefix
                        for j, value in enumerate(out if isinstance(out, tuple) else (out,)):
                            if isinstance(value, torch.Tensor):
                                capture(f"{prefix}.output{j}", value)
                    module.register_forward_pre_hook(norm_pre)
                    module.register_forward_hook(norm_post)
                if isinstance(module, (dflash.RMSNorm, dflash.DFlashGroupedConv)) or name.endswith(("qkv_proj", "gate_up_proj", "down_proj", "act_fn")):
                    def diagnostic_hook(mod, inp, out, name=name):
                        for j, value in enumerate(inp):
                            if isinstance(value, torch.Tensor):
                                capture(f"ops.{name}.input{j}", value)
                        for j, value in enumerate(out if isinstance(out, tuple) else (out,)):
                            if isinstance(value, torch.Tensor):
                                capture(f"ops.{name}.output{j}", value)
                    module.register_forward_hook(diagnostic_hook)
            original_conv = dflash._grouped_conv
            def diagnostic_conv(x, delta, base, block, groups, group_size, taps):
                prefix = f"ops.conv{diagnostic_conv.index}"
                diagnostic_conv.index += 1
                for key, value in (("input", x), ("delta", delta), ("base", base)):
                    capture(prefix + "." + key, value)
                out = original_conv(x, delta, base, block, groups, group_size, taps)
                capture(prefix + ".output", out)
                return out
            diagnostic_conv.index = 0
            dflash._grouped_conv = diagnostic_conv
            original_rope = dflash_utils.table_qk_norm_rope_
            def diagnostic_rope(qkv, positions, qw, kw, table, nq, nk, dim, eps):
                index = diagnostic_rope.index
                diagnostic_rope.index += 1
                prefix = f"ops.rope{index}"
                for key, value in (("input", qkv), ("positions", positions), ("qw", qw), ("kw", kw), ("table", table[:int(positions.max().item()) + 1])):
                    capture(prefix + "." + key, value)
                original_rope(qkv, positions, qw, kw, table, nq, nk, dim, eps)
                capture(prefix + ".output", qkv)
            diagnostic_rope.index = 0
            dflash_utils.table_qk_norm_rope_ = diagnostic_rope
        taps = read_raw("taps.f32", torch.float32, (n, 5 * h)).to(torch.bfloat16)
        embeds = read_raw("embeddings.bf16", torch.bfloat16, (m, h))
        capture("input.taps_bf16", taps)
        ctx = model.project_target_hidden(taps)
        capture("context.hidden", ctx)
        adapter = AttentionAdapter()
        context_positions = torch.arange(n, device="cuda", dtype=torch.int64)
        for i, layer in enumerate(model.layers):
            prepared = model.prepare_context_hidden_for_kv(layer, ctx)
            capture(f"layers.{i}.context_prepared", prepared)
            att = layer.self_attn
            k, v = att.kv_proj_only(prepared)
            k = att.apply_k_rope(context_positions, att.apply_k_norm(k))
            k = k.reshape(n, att.num_kv_heads, att.head_dim)
            v = v.reshape(n, att.num_kv_heads, att.head_dim)
            adapter.context[i] = (k, v)
            capture(f"layers.{i}.context_k", k)
            capture(f"layers.{i}.context_v", v)
        positions = torch.arange(n, n + m, device="cuda", dtype=torch.int64)
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, attn_backend=adapter)
        from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
        with forward_context(ForwardContext(attn_backend=adapter)):
            output = model(input_ids=None, positions=positions, forward_batch=batch, input_embeds=embeds)
        capture("final_hidden", output.hidden_states)
        capture("selector.projected_hidden", model.candidate_selector.hidden_projection(output.hidden_states))
        torch.cuda.synchronize()
    for key, value in tensors.items():
        assert torch.isfinite(value).all(), key
    sources = {}
    for module in (dflash, dflash_utils):
        p = Path(inspect.getfile(module))
        sources[str(p)] = sha(p)
    metadata = dict(image=IMAGE, image_id=IMAGE_ID, sglang=sglang.__version__,
                    torch=torch.__version__, flashinfer=flashinfer.__version__,
                    gpu=torch.cuda.get_device_name(), physical_gpu=1,
                    scope="actual DFlash2DraftModel forward; attention-boundary adapter; NOT end-to-end serving",
                    adapter="FlashInfer fa2 paged prefill; concatenated committed+query KV; page_size=1; causal=false; window_left=-1",
                    config=raw, effective_config=config.to_dict(), input=info,
                    input_sha256={p: sha(args.inputs / p) for p in ("input.json", "taps.f32", "embeddings.bf16")},
                    checkpoint_sha256={p.name: sha(p) for p in weight_files},
                    source_sha256=sources, script_sha256=sha(__file__),
                    context_positions=list(range(n)), draft_positions=list(range(n, n+m)),
                    contract="taps supplied as F32 target FP16 residual captures, cast BF16 before donor FC; BF16 draft ops; no target head or selector walk")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata={"provenance": json.dumps(metadata)})
    args.out.chmod(0o644)
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {args.out} ({len(tensors)} tensors)", flush=True)


if __name__ == "__main__":
    main()
