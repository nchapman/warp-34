#!/usr/bin/env python3
"""vLLM 0.31.0 / PyTorch per-op and per-layer tensor dumper for `w34-oracle dump-diff`.

Generates deterministic golden tensor fixtures (.safetensors) for:
1. GemmaRMSNorm + fused_add_gemma_rms_norm (`1.0 + weight` unit-offset, FP32 variance)
2. fused_qk_rmsnorm_rope_gate (`q || gate` split, per-head RMSNorm, partial 3D mRoPE)
3. silu_and_mul, RMSNormGated (norm_before_gate=True, activation="silu"), apply_attn_output_gate
4. Marlin W4A16 repack (`gptq_marlin_repack`) and GEMM (`M in {1, 8, 64}`)
5. Descriptor-driven Paged Attention (`head_dim in {128, 256}`, causal & symmetric window)
6. GDN (`causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule`,
   `fused_sigmoid_gating_delta_rule_update`) with FP16 conv_state and FP32 ssm_state,
   including multi-step speculative verify states (`spec_state_indices[req, 0..8]`).
"""
from __future__ import annotations

import argparse
import json
import hashlib
import math
from pathlib import Path

def load_tensor_dependencies() -> None:
    # Keep --help usable without PyTorch, safetensors, vLLM, or CUDA.
    global torch, F, save_file
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file


def capture_fused_gemma(x, residual, weight):
    """Actual vLLM runtime, not a formula reference. Requires a visible CUDA GPU.

    Select the intended physical GPU with CUDA_VISIBLE_DEVICES (e.g. 1).
    Only a tiny GemmaRMSNorm operator is constructed; no model is loaded.
    """
    load_tensor_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("Native fused Gemma capture requires CUDA; select CUDA_VISIBLE_DEVICES")
    import vllm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    with set_current_vllm_config(VllmConfig()), torch.inference_mode():
        op = GemmaRMSNorm(x.shape[-1], eps=1e-6).to(device="cuda", dtype=weight.dtype)
        op.weight.copy_(weight.to("cuda"))
        args = lambda: (x.to("cuda").clone(), residual.to("cuda").clone())
        direct = op.forward_cuda(*args())
        dispatched = op(*args())
        for a, b in zip(direct, dispatched):
            if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                raise RuntimeError("Gemma forward_cuda and module dispatch disagree")
        outputs = tuple(t.cpu().contiguous() for t in dispatched)
    provenance = {
        "vllm": vllm.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "entrypoint": "GemmaRMSNorm.forward_cuda and module dispatch (bitwise agreement)",
        "config": "VllmConfig()", "eps": 1e-6,
        "inputs": ["gemma_rms_norm.x", "fused_add_gemma_rms_norm.res_in", "gemma_rms_norm.w"],
        "scope": ["fused_add_gemma_rms_norm.out", "fused_add_gemma_rms_norm.res_out"],
        "note": "Native CUDA capture only for scope; other tensors retain original provenance",
    }
    return outputs, provenance


def refresh_fused_gemma(path: Path) -> None:
    """Replace only the two native expectations, using existing saved inputs."""
    load_tensor_dependencies()
    from safetensors import safe_open
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    (out, residual), provenance = capture_fused_gemma(
        tensors["gemma_rms_norm.x"], tensors["fused_add_gemma_rms_norm.res_in"],
        tensors["gemma_rms_norm.w"],
    )
    provenance["source_file_sha256"] = original_hash
    for key, value in zip(provenance["scope"], (out, residual)):
        old = tensors[key]
        if old.shape != value.shape or old.dtype != value.dtype:
            raise RuntimeError(f"Native capture changed shape/dtype: {key}")
        print(f"{key}: {int((old != value).sum())} changed elements / {old.numel()}")
        tensors[key] = value
    metadata["fused_add_gemma_native_capture"] = json.dumps(provenance, sort_keys=True)
    save_file(tensors, str(path), metadata=metadata)
    print(json.dumps(provenance, sort_keys=True))


def gemma_rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    wf = 1.0 + weight.float()
    return (xf * wf).to(orig_dtype)


def rms_norm_gated_silu_ref(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_dtype = x.dtype
    xf = x.float()
    zf = z.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = xf * torch.rsqrt(var + eps) * weight.float()
    gate = zf * torch.sigmoid(zf)
    return (normed * gate).to(orig_dtype)


def apply_mrope_partial_interleaved_ref(
    x: torch.Tensor,  # [N, H, D]
    positions: torch.Tensor,  # [3, N]
    rotary_dim: int = 64,
    mrope_section: tuple[int, int, int] = (11, 11, 10),
    theta: float = 10_000_000.0,
) -> torch.Tensor:
    # Qwen3.5 / Qwen3_Next partial interleaved mRoPE:
    # half_dim = rotary_dim // 2 = 32; mrope_section = [11, 11, 10] (sum = 32).
    # Interleaved layout assigns frequency index i in 0..32 to plane:
    # i % 3 == 1 (for i < 3*11=33) -> plane 1 (H)
    # i % 3 == 2 (for i < 3*10=30) -> plane 2 (W)
    # otherwise -> plane 0 (T)
    orig_dtype = x.dtype
    n_tok, n_heads, head_dim = x.shape
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=x.device) / rotary_dim))

    plane_for_pair = []
    for i in range(half):
        if (i % 3) == 1 and i < mrope_section[1] * 3:
            plane_for_pair.append(1)
        elif (i % 3) == 2 and i < mrope_section[2] * 3:
            plane_for_pair.append(2)
        else:
            plane_for_pair.append(0)
    plane_idx = torch.tensor(plane_for_pair, dtype=torch.long, device=x.device)  # [half]
    pos_per_pair = positions[plane_idx, :].transpose(0, 1).float()  # [N, half]
    freqs = pos_per_pair * inv_freq.unsqueeze(0)  # [N, half]
    cos = torch.cos(freqs).unsqueeze(1)  # [N, 1, half]
    sin = torch.sin(freqs).unsqueeze(1)  # [N, 1, half]

    x_rot = x[:, :, :rotary_dim].float()
    x_pass = x[:, :, rotary_dim:]
    x1 = x_rot[:, :, :half]
    x2 = x_rot[:, :, half:rotary_dim]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    out_rot = torch.cat([o1, o2], dim=-1).to(orig_dtype)
    return torch.cat([out_rot, x_pass], dim=-1)


def gdn_recurrent_step_ref(
    q: torch.Tensor,        # [H_k, D] fp16
    k: torch.Tensor,        # [H_k, D] fp16
    v: torch.Tensor,        # [H_v, D] fp16
    b: torch.Tensor,        # [H_v] fp16
    a: torch.Tensor,        # [H_v] fp16
    a_log: torch.Tensor,    # [H_v] fp32
    dt_bias: torch.Tensor,  # [H_v] fp32
    state: torch.Tensor,    # [H_v, D, D] fp32 (row=v_dim, col=k_dim or [H_v, d_k, d_v])
) -> tuple[torch.Tensor, torch.Tensor]:
    # Matches vLLM FLA / vllm.cpp fused_sigmoid_gating_delta_rule_update:
    # q_norm = l2_normalize(q) * (D ** -0.5)
    # k_norm = l2_normalize(k)
    # g = -exp(A_log) * softplus(a + dt_bias)
    # beta = sigmoid(b)
    # For each v_head hv (mapped to k_head hk = hv // (H_v // H_k)):
    #   S = S * exp(g)
    #   v_pred = sum_k S[k, v] * k_norm[k]
    #   delta = beta * (v - v_pred)
    #   S[k, v] += k_norm[k] * delta[v]
    #   out[v] = sum_k S[k, v] * q_norm[k]
    hk, d = q.shape
    hv = v.shape[0]
    gqa_ratio = hv // hk
    scale = d ** -0.5

    qf = F.normalize(q.float(), p=2.0, dim=-1, eps=1e-6) * scale
    kf = F.normalize(k.float(), p=2.0, dim=-1, eps=1e-6)
    vf = v.float()
    g = -torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias.float())
    beta = torch.sigmoid(b.float())

    new_state = state.clone().float()
    out = torch.empty_like(vf)
    for h in range(hv):
        kh = h // gqa_ratio
        s_h = new_state[h] * torch.exp(g[h])  # [D_v, D_k]
        k_vec = kf[kh]                        # [D_k]
        q_vec = qf[kh]                        # [D_k]
        v_pred = torch.mv(s_h, k_vec)         # [D_v]
        delta = beta[h] * (vf[h] - v_pred)    # [D_v]
        s_h = s_h + torch.outer(delta, k_vec)
        new_state[h] = s_h
        out[h] = torch.mv(s_h, q_vec)
    return out.to(q.dtype), new_state


def generate_op_fixtures(out_path: Path) -> None:
    load_tensor_dependencies()
    torch.manual_seed(34)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensors: dict[str, torch.Tensor] = {}

    # 1. GemmaRMSNorm & fused_add_gemma_rms_norm
    x = torch.randn(16, 2560, device=device, dtype=torch.float16)
    res = torch.randn(16, 2560, device=device, dtype=torch.float16)
    w = torch.randn(2560, device=device, dtype=torch.float16) * 0.1
    tensors["gemma_rms_norm.x"] = x.cpu()
    tensors["gemma_rms_norm.w"] = w.cpu()
    tensors["gemma_rms_norm.out"] = gemma_rms_norm_ref(x, w).cpu()

    (fused_out, res_out), provenance = capture_fused_gemma(x, res, w)
    tensors["fused_add_gemma_rms_norm.res_in"] = res.cpu()
    tensors["fused_add_gemma_rms_norm.res_out"] = res_out
    tensors["fused_add_gemma_rms_norm.out"] = fused_out

    # 2. silu_and_mul & RMSNormGated
    gate_up = torch.randn(16, 1024, device=device, dtype=torch.float16)
    g_half, u_half = gate_up.chunk(2, dim=-1)
    silu_mul_out = (F.silu(g_half.float()) * u_half.float()).to(torch.float16)
    tensors["silu_and_mul.in"] = gate_up.cpu()
    tensors["silu_and_mul.out"] = silu_mul_out.cpu()

    gdn_x = torch.randn(16, 32, 128, device=device, dtype=torch.float16)
    gdn_z = torch.randn(16, 32, 128, device=device, dtype=torch.float16)
    gdn_nw = torch.ones(128, device=device, dtype=torch.float16) + 0.05 * torch.randn(128, device=device, dtype=torch.float16)
    tensors["rms_norm_gated.x"] = gdn_x.cpu()
    tensors["rms_norm_gated.z"] = gdn_z.cpu()
    tensors["rms_norm_gated.w"] = gdn_nw.cpu()
    tensors["rms_norm_gated.out"] = rms_norm_gated_silu_ref(gdn_x, gdn_z, gdn_nw).cpu()

    # 3. fused_qk_rmsnorm_rope_gate
    n_tok, hq, hkv, d = 8, 8, 2, 256
    q_gate = torch.randn(n_tok, hq, d * 2, device=device, dtype=torch.float16)
    k_in = torch.randn(n_tok, hkv, d, device=device, dtype=torch.float16)
    q_nw = torch.randn(d, device=device, dtype=torch.float16) * 0.1
    k_nw = torch.randn(d, device=device, dtype=torch.float16) * 0.1
    positions = torch.stack([
        torch.arange(n_tok, device=device, dtype=torch.int32),
        torch.arange(n_tok, device=device, dtype=torch.int32) + 2,
        torch.arange(n_tok, device=device, dtype=torch.int32) + 5,
    ], dim=0)
    q_raw, gate_out = q_gate.chunk(2, dim=-1)
    q_normed = gemma_rms_norm_ref(q_raw, q_nw)
    k_normed = gemma_rms_norm_ref(k_in, k_nw)
    q_rope = apply_mrope_partial_interleaved_ref(q_normed, positions)
    k_rope = apply_mrope_partial_interleaved_ref(k_normed, positions)
    tensors["qk_norm_rope.q_gate_in"] = q_gate.cpu()
    tensors["qk_norm_rope.k_in"] = k_in.cpu()
    tensors["qk_norm_rope.q_nw"] = q_nw.cpu()
    tensors["qk_norm_rope.k_nw"] = k_nw.cpu()
    tensors["qk_norm_rope.positions"] = positions.cpu()
    tensors["qk_norm_rope.q_out"] = q_rope.cpu()
    tensors["qk_norm_rope.k_out"] = k_rope.cpu()
    tensors["qk_norm_rope.gate_out"] = gate_out.contiguous().cpu()

    # 4. GDN 8-step speculative verify sequence
    hk_gdn, hv_gdn, d_gdn = 4, 8, 128
    q_seq = torch.randn(8, hk_gdn, d_gdn, device=device, dtype=torch.float16)
    k_seq = torch.randn(8, hk_gdn, d_gdn, device=device, dtype=torch.float16)
    v_seq = torch.randn(8, hv_gdn, d_gdn, device=device, dtype=torch.float16)
    b_seq = torch.randn(8, hv_gdn, device=device, dtype=torch.float16)
    a_seq = torch.randn(8, hv_gdn, device=device, dtype=torch.float16)
    a_log = torch.linspace(-2.0, -0.5, hv_gdn, device=device, dtype=torch.float32)
    dt_bias = torch.linspace(-0.2, 0.2, hv_gdn, device=device, dtype=torch.float32)
    init_state = torch.randn(hv_gdn, d_gdn, d_gdn, device=device, dtype=torch.float32) * 0.05

    cur_state = init_state.clone()
    outs = []
    spec_states = []
    for step in range(8):
        out_step, cur_state = gdn_recurrent_step_ref(
            q_seq[step], k_seq[step], v_seq[step], b_seq[step], a_seq[step], a_log, dt_bias, cur_state
        )
        outs.append(out_step)
        spec_states.append(cur_state.clone())

    tensors["gdn.q"] = q_seq.cpu()
    tensors["gdn.k"] = k_seq.cpu()
    tensors["gdn.v"] = v_seq.cpu()
    tensors["gdn.b"] = b_seq.cpu()
    tensors["gdn.a"] = a_seq.cpu()
    tensors["gdn.a_log"] = a_log.cpu()
    tensors["gdn.dt_bias"] = dt_bias.cpu()
    tensors["gdn.init_state"] = init_state.cpu()
    tensors["gdn.out"] = torch.stack(outs, dim=0).cpu()
    tensors["gdn.spec_states"] = torch.stack(spec_states, dim=0).cpu()

    # 5. GDN causal_conv1d (kernel_size=4, silu activation)
    conv_dim = 256
    conv_x = torch.randn(8, conv_dim, device=device, dtype=torch.float16)
    conv_w = torch.randn(conv_dim, 4, device=device, dtype=torch.float16) * 0.2
    conv_state_init = torch.randn(conv_dim, 3, device=device, dtype=torch.float16) * 0.2
    # Pad initial 3 history tokens + 8 input tokens = 11 tokens
    conv_padded = torch.cat([conv_state_init, conv_x.transpose(0, 1)], dim=1)  # [conv_dim, 11]
    conv_outs = []
    for t in range(8):
        window = conv_padded[:, t : t + 4].float()  # [conv_dim, 4]
        acc = (window * conv_w.float()).sum(dim=-1)
        conv_outs.append(F.silu(acc).to(torch.float16))
    conv_out = torch.stack(conv_outs, dim=0)  # [8, conv_dim]
    conv_state_final = conv_padded[:, 8:11].contiguous()  # [conv_dim, 3]
    tensors["gdn_conv.x"] = conv_x.cpu()
    tensors["gdn_conv.w"] = conv_w.cpu()
    tensors["gdn_conv.state_in"] = conv_state_init.cpu()
    tensors["gdn_conv.out"] = conv_out.cpu()
    tensors["gdn_conv.state_out"] = conv_state_final.cpu()

    # 6. Marlin W4A16 repack + GEMM (validated via vllm._custom_ops.gptq_marlin_repack)
    from vllm import _custom_ops as vllm_ops
    mk, mn, gs = 256, 128, 128
    q_packed = torch.randint(0, 2**31 - 1, (mn, mk // 8), device=device, dtype=torch.int32)
    q_gptq = q_packed.t().contiguous()
    marlin_repacked = vllm_ops.gptq_marlin_repack(q_gptq, mk, mn, 4)
    sc_raw = (torch.rand(mn, mk // gs, device=device, dtype=torch.float32) * 0.02 + 0.001).to(torch.float16)
    perm = [i + 8 * j for i in range(8) for j in range(8)]
    sc_perm = sc_raw.t().contiguous().view(-1, 64)[:, perm].view(mk // gs, mn).contiguous()
    # Dequant reference: W[n, k] = (nibble - 8) * sc_raw[n, k // gs]
    q_u32 = q_packed.to(torch.int64) & 0xFFFFFFFF
    nibbles = torch.stack([((q_u32 >> (4 * j)) & 0xF) for j in range(8)], dim=-1).view(mn, mk).float() - 8.0
    w_deq = nibbles * sc_raw.float().repeat_interleave(gs, dim=1)
    a_m1 = torch.randn(1, mk, device=device, dtype=torch.float16)
    a_m16 = torch.randn(16, mk, device=device, dtype=torch.float16)
    ref_m1 = (a_m1.float() @ w_deq.t()).to(torch.float16)
    ref_m16 = (a_m16.float() @ w_deq.t()).to(torch.float16)
    tensors["marlin.q_packed"] = q_packed.cpu()
    tensors["marlin.repacked"] = marlin_repacked.cpu()
    tensors["marlin.sc_raw"] = sc_raw.cpu()
    tensors["marlin.sc_perm"] = sc_perm.cpu()
    tensors["marlin.a_m1"] = a_m1.cpu()
    tensors["marlin.ref_m1"] = ref_m1.cpu()
    tensors["marlin.a_m16"] = a_m16.cpu()
    tensors["marlin.ref_m16"] = ref_m16.cpu()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={
        "fused_add_gemma_native_capture": json.dumps(provenance, sort_keys=True),
    })
    print(f"Wrote {len(tensors)} golden tensors to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "goldens" / "op_fixtures.safetensors"),
        help="Output safetensors path",
    )
    ap.add_argument("--refresh-fused-gemma", action="store_true",
                    help="Recapture ONLY fused Gemma expectations in existing --out from saved inputs; requires CUDA and vLLM")
    args = ap.parse_args()
    if args.refresh_fused_gemma:
        refresh_fused_gemma(Path(args.out))
    else:
        generate_op_fixtures(Path(args.out))


if __name__ == "__main__":
    main()
