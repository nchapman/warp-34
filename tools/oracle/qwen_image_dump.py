#!/usr/bin/env python3
"""Diffusers `QwenImage21Pipeline` (`huggingface/diffusers#14804`) oracle tensor dumper.

Generates deterministic golden tensor fixtures (`.safetensors`) directly from the
upstream `diffusers` `QwenImage21Transformer2DModel` and `AutoencoderKLQwenImage21`
modules, plus `Qwen3VL` conditioning encoder reference ops:
1. `QwenImage21Rope` (`theta=10000`, `axes_dim=[16, 56, 56]`) + `apply_rotary_emb_qwen(use_real=False)`.
2. `QwenImage21TimestepProjEmbeddings` + `QwenImage21TextProjection` + `QwenImage21ZeroCenterRMSNorm`.
3. `QwenImage21Transformer2DModel` (`extract` step-0 block-causal prefill + prefix KV cache,
   and `cached` step-1 target-only decode against `[cached_k, target_k]`).
4. `AutoencoderKLQwenImage21` 64-channel RGBA VAE (`encode` + `decode` at `T=1`,
   `QwenImage21RMS_norm`, `AvgDown3D`, `DupUp3D`, `latents_mean` / `latents_std`).
5. `Qwen3VL` full-head interleaved `mRoPE` (`[24, 20, 20]`, `theta=5_000_000.0`) and
   `deepstack` post-shuffle `PatchMerger`.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def load_deps():
    global torch, F, save_file
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file


def apply_qwen3_vl_mrope_interleaved(
    x, positions, mrope_section=(24, 20, 20), theta: float = 5_000_000.0
):
    # x: [N, H, D=128], positions: [3, N]
    _, _, head_dim = x.shape
    half_dim = head_dim // 2
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device)
            / float(head_dim)
        )
    )
    freqs_3d = torch.einsum("cn,d->cnd", positions.float(), inv_freq)  # [3, N, 64]
    freqs = freqs_3d[0].clone()
    h_indices = slice(1, mrope_section[1] * 3, 3)
    w_indices = slice(2, mrope_section[2] * 3, 3)
    freqs[:, h_indices] = freqs_3d[1, :, h_indices]
    freqs[:, w_indices] = freqs_3d[2, :, w_indices]

    cos = freqs.cos().unsqueeze(1)  # [N, 1, 64]
    sin = freqs.sin().unsqueeze(1)  # [N, 1, 64]
    xf = x.float()
    x1 = xf[..., :half_dim]
    x2 = xf[..., half_dim:]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1).to(x.dtype)


def generate_synthetic_fixture(out_path: Path):
    load_deps()
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import (
        AutoencoderKLQwenImage21,
        QwenImage21RMS_norm,
    )
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21KVCache,
        QwenImage21Rope,
        QwenImage21Transformer2DModel,
        apply_rotary_emb_qwen,
    )

    torch.manual_seed(20260419)
    tensors = {}

    # 1. QwenImage21Rope + apply_rotary_emb_qwen(use_real=False)
    # 2 text tokens, 1 condition image (1x2x2 = 4 tokens), 1 text token, 1 target image (1x2x4 = 8 tokens)
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    image_pad_mask = torch.tensor(
        [False, False] + [True] * 4 + [False] + [True] * 8, dtype=torch.bool
    )
    img_shapes = [(1, 2, 2), (1, 2, 4)]
    freqs_cis = rope(img_shapes, image_pad_mask, device=torch.device("cpu"))
    tensors["rope.cos"] = freqs_cis.real.to(torch.float32).contiguous()
    tensors["rope.sin"] = freqs_cis.imag.to(torch.float32).contiguous()
    q_in = torch.randn(1, 15, 2, 128, dtype=torch.float16)
    q_out = apply_rotary_emb_qwen(q_in, freqs_cis, use_real=False)
    tensors["rope.q_in"] = q_in.contiguous()
    tensors["rope.q_out"] = q_out.contiguous()

    # 2. Miniature QwenImage21Transformer2DModel (1 layer, 2 heads x 128 = 256 inner_dim, in/out=8)
    # Verify Step 0 ("extract" with prefix KV cache) and Step 1 ("cached" target-only decode).
    torch.manual_seed(20260420)
    dit = QwenImage21Transformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=1,
        attention_head_dim=128,
        num_attention_heads=2,
        context_in_dim=64,
        mlp_ratio=2,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
        causal_condition=True,
    ).eval()
    # Initialize zero-centered text_norm weight to non-zero values so (1 + w) is nontrivial
    with torch.no_grad():
        dit.txt_in.text_norm.weight.copy_(torch.randn_like(dit.txt_in.text_norm.weight) * 0.1)
    dit = dit.to(torch.float16)

    for k, v in dit.state_dict().items():
        tensors[f"dit.weight.{k}"] = v.contiguous()

    # VLM sequence before 4x image slot expansion:
    # [txt0, txt1, img_slot_cond, txt2] -> len = 4, plus 2 target slots appended -> len = 6
    enc_hidden = torch.randn(1, 4, 64, dtype=torch.float16)
    img_mask = torch.tensor([[False, False, True, False, True, True]], dtype=torch.bool)
    cond_latents = torch.randn(1, 4, 8, dtype=torch.float16)
    target_latents_0 = torch.randn(1, 8, 8, dtype=torch.float16)
    target_latents_1 = torch.randn(1, 8, 8, dtype=torch.float16)
    hidden_0 = torch.cat([cond_latents, target_latents_0], dim=1)
    hidden_1 = torch.cat([cond_latents, target_latents_1], dim=1)
    t0 = torch.tensor([0.978453], dtype=torch.float16)
    t1 = torch.tensor([0.845148], dtype=torch.float16)

    kv_cache = QwenImage21KVCache(num_layers=1)
    with torch.no_grad():
        out_extract_full = dit(
            hidden_states=hidden_0,
            encoder_hidden_states=enc_hidden,
            timestep=t0,
            img_shapes=[img_shapes],
            img_mask=img_mask,
            kv_cache=kv_cache,
            kv_cache_mode="extract",
            return_dict=False,
        )[0]
        out_extract_target = out_extract_full[:, -8:].clone()

        out_cached_target = dit(
            hidden_states=hidden_1,
            encoder_hidden_states=enc_hidden,
            timestep=t1,
            img_shapes=[img_shapes],
            img_mask=img_mask,
            kv_cache=kv_cache,
            kv_cache_mode="cached",
            return_dict=False,
        )[0][:, -8:]

    tensors["dit.enc_hidden"] = enc_hidden.contiguous()
    tensors["dit.cond_latents"] = cond_latents.contiguous()
    tensors["dit.target_latents_0"] = target_latents_0.contiguous()
    tensors["dit.target_latents_1"] = target_latents_1.contiguous()
    tensors["dit.t0"] = t0.contiguous()
    tensors["dit.t1"] = t1.contiguous()
    tensors["dit.out_extract_full"] = out_extract_full.contiguous()
    tensors["dit.out_extract_target"] = out_extract_target.contiguous()
    tensors["dit.out_cached_target"] = out_cached_target.contiguous()
    tensors["dit.cache_l0_k"] = kv_cache.get_layer(0).k.contiguous()
    tensors["dit.cache_l0_v"] = kv_cache.get_layer(0).v.contiguous()

    # 3. Miniature AutoencoderKLQwenImage21 (base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 2, 2])
    torch.manual_seed(20260421)
    vae = AutoencoderKLQwenImage21(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=4,
        dim_mult=[1, 2, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, False],
        in_channels=4,
        out_channels=4,
        is_residual=True,
        latents_mean=[0.1 * i for i in range(4)],
        latents_std=[1.0 + 0.1 * i for i in range(4)],
    ).eval()
    for k, v in vae.state_dict().items():
        tensors[f"vae.weight.{k}"] = v.contiguous()

    vae_img_in = torch.randn(1, 4, 1, 16, 16, dtype=torch.float32)
    with torch.no_grad():
        vae_enc_dist = vae.encode(vae_img_in).latent_dist
        vae_z = vae_enc_dist.mode()
        vae_dec_out = vae.decode(vae_z).sample
    tensors["vae.img_in"] = vae_img_in.contiguous()
    tensors["vae.z_mode"] = vae_z.contiguous()
    tensors["vae.dec_out"] = vae_dec_out.contiguous()

    # Standalone QwenImage21RMS_norm (channel-wise RMSNorm)
    rms = QwenImage21RMS_norm(16, images=True)
    with torch.no_grad():
        rms.gamma.copy_(torch.randn_like(rms.gamma))
    rms_in = torch.randn(1, 16, 8, 8, dtype=torch.float32)
    with torch.no_grad():
        rms_out = rms(rms_in)
    tensors["vae.rms_gamma"] = rms.gamma.view(-1).contiguous()
    tensors["vae.rms_in"] = rms_in.contiguous()
    tensors["vae.rms_out"] = rms_out.contiguous()

    # 4. Qwen3-VL full-head mRoPE [24, 20, 20] at theta=5_000_000.0
    pos_3d = torch.tensor(
        [
            [0, 1, 2, 2, 2, 2, 4, 5],
            [0, 1, 2, 2, 3, 3, 4, 5],
            [0, 1, 2, 3, 2, 3, 4, 5],
        ],
        dtype=torch.int32,
    )
    vl_q = torch.randn(8, 4, 128, dtype=torch.float16)
    tensors["vl_mrope.positions"] = pos_3d.contiguous()
    tensors["vl_mrope.q_in"] = vl_q.contiguous()
    tensors["vl_mrope.q_out"] = apply_qwen3_vl_mrope_interleaved(vl_q, pos_3d).contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    print(f"Wrote {len(tensors)} oracle tensors to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/oracle/fixtures/qwen_image_ops.safetensors"),
        help="Output .safetensors fixture path",
    )
    args = parser.parse_args()
    generate_synthetic_fixture(args.output)


if __name__ == "__main__":
    main()
