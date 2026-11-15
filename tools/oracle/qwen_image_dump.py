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

    # 2b. Multi-image conditioning (2 condition images [(1,2,2), (1,2,3)] + target (1,2,4))
    # + 3-step Euler + CFG (true_cfg_scale=4.0) with asymmetric positive vs negative prompt lengths.
    multi_img_shapes = [(1, 2, 2), (1, 2, 3), (1, 2, 4)]
    # Positive VLM sequence: [txt, img1_slot(1->4), txt, img2_slot(2->8, truncated to 6 by img_shapes? No: in QwenImage21,
    # vlm_img_mask has 1 bool per VLM token, and each True expands by 4x! Wait: in QwenImage21TransformerEngine,
    # vlm_img_mask is already the expanded image_pad_mask of length text_tokens + sum(img_shapes)!)
    # Let's check how dit() uses img_mask: in diffusers, encoder_hidden_states has 1 token per VLM vision token
    # (so 1 VLM token -> 4 DiT image tokens after 2x2 repeat_interleave).
    # So for img_shapes=[(1,2,2), (1,4,2), (1,2,4)] (4, 8, 8 tokens = 1, 2, 2 VLM image slots):
    multi_img_shapes = [(1, 2, 2), (1, 4, 2), (1, 2, 4)]
    pos_enc_hidden = torch.randn(1, 6, 64, dtype=torch.float16)  # 3 text + 1 img1 + 2 img2
    pos_img_mask = torch.tensor(
        [[False, True, False, True, True, False, True, True]], dtype=torch.bool
    )
    neg_enc_hidden = torch.randn(1, 4, 64, dtype=torch.float16)  # 1 text + 1 img1 + 2 img2
    neg_img_mask = torch.tensor(
        [[False, True, True, True, True, True]], dtype=torch.bool
    )
    multi_cond_latents = torch.randn(1, 12, 8, dtype=torch.float16)  # 4 + 8 = 12 cond tokens
    multi_init_latents = torch.randn(1, 8, 8, dtype=torch.float16)
    cfg_timesteps = [0.978453, 0.845148, 0.422574]
    cfg_sigmas = [1.0, 0.7, 0.3, 0.0]
    cfg_scale = 4.0

    pos_kv = QwenImage21KVCache(num_layers=1)
    neg_kv = QwenImage21KVCache(num_layers=1)
    sample = multi_init_latents.clone()
    with torch.no_grad():
        for step_idx, t_val in enumerate(cfg_timesteps):
            t_ten = torch.tensor([t_val], dtype=torch.float16)
            mode = "extract" if step_idx == 0 else "cached"
            h_in = torch.cat([multi_cond_latents, sample], dim=1)
            cond_pred = dit(
                hidden_states=h_in,
                encoder_hidden_states=pos_enc_hidden,
                timestep=t_ten,
                img_shapes=[multi_img_shapes],
                img_mask=pos_img_mask,
                kv_cache=pos_kv,
                kv_cache_mode=mode,
                return_dict=False,
            )[0][:, -8:]
            if step_idx == 0:
                multi_extract_target = cond_pred.clone()
            uncond_pred = dit(
                hidden_states=h_in,
                encoder_hidden_states=neg_enc_hidden,
                timestep=t_ten,
                img_shapes=[multi_img_shapes],
                img_mask=neg_img_mask,
                kv_cache=neg_kv,
                kv_cache_mode=mode,
                return_dict=False,
            )[0][:, -8:]
            # Match QwenImage21Pipeline normed CFG + Euler step
            comb = uncond_pred.float() + cfg_scale * (cond_pred.float() - uncond_pred.float())
            cond_norm = torch.linalg.vector_norm(cond_pred.float(), dim=-1, keepdim=True)
            comb_norm = torch.linalg.vector_norm(comb, dim=-1, keepdim=True)
            pred = (comb * (cond_norm / (comb_norm + 1e-12))).to(torch.float16)
            dt = cfg_sigmas[step_idx + 1] - cfg_sigmas[step_idx]
            sample = (sample.float() + dt * pred.float()).to(torch.float16)

    tensors["dit.multi.pos_enc_hidden"] = pos_enc_hidden.contiguous()
    tensors["dit.multi.neg_enc_hidden"] = neg_enc_hidden.contiguous()
    tensors["dit.multi.cond_latents"] = multi_cond_latents.contiguous()
    tensors["dit.multi.init_latents"] = multi_init_latents.contiguous()
    tensors["dit.multi.extract_target"] = multi_extract_target.contiguous()
    tensors["dit.multi.final_latents"] = sample.contiguous()

    # 3. Miniature AutoencoderKLQwenImage21 (base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 2, 2])
    torch.manual_seed(20260421)
    vae = (
        AutoencoderKLQwenImage21(
            base_dim=8,
            decoder_base_dim=8,
            z_dim=4,
            dim_mult=[1, 2, 2],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, True],
            in_channels=4,
            out_channels=4,
            is_residual=True,
            scale_factor_spatial=4,
            latents_mean=[0.1 * i for i in range(4)],
            latents_std=[1.0 + 0.1 * i for i in range(4)],
        )
        .eval()
        .to(torch.float16)
    )
    for k, v in vae.state_dict().items():
        tensors[f"vae.weight.{k}"] = v.contiguous()

    vae_img_in = torch.randn(1, 4, 1, 16, 16, dtype=torch.float16)
    with torch.no_grad():
        vae_enc_dist = vae.encode(vae_img_in).latent_dist
        vae_z = vae_enc_dist.mode()
        vae_dec_out = vae.decode(vae_z).sample
        vae.enable_tiling(
            tile_sample_min_height=8,
            tile_sample_min_width=8,
            tile_sample_stride_height=4,
            tile_sample_stride_width=4,
        )
        vae_tiled_z = vae.encode(vae_img_in).latent_dist.mode()
        vae_tiled_dec_out = vae.decode(vae_z).sample
    tensors["vae.img_in"] = vae_img_in.contiguous()
    tensors["vae.z_mode"] = vae_z.contiguous()
    tensors["vae.dec_out"] = vae_dec_out.contiguous()
    tensors["vae.tiled_z_mode"] = vae_tiled_z.contiguous()
    tensors["vae.tiled_dec_out"] = vae_tiled_dec_out.contiguous()

    # Standalone QwenImage21RMS_norm (channel-wise RMSNorm)
    rms = QwenImage21RMS_norm(16, images=True).to(torch.float16)
    with torch.no_grad():
        rms.gamma.copy_(torch.randn_like(rms.gamma))
    rms_in = torch.randn(1, 16, 8, 8, dtype=torch.float16)
    with torch.no_grad():
        rms_out = rms(rms_in)
    tensors["vae.rms_gamma"] = rms.gamma.view(-1).contiguous()
    tensors["vae.rms_in"] = rms_in.contiguous()
    tensors["vae.rms_out"] = rms_out.contiguous()

    # 3b. 5-stage AutoencoderKLQwenImage21 (dim_mult=[1, 2, 4, 4, 4]) high-dynamic-range overflow test:
    # Simulates the exact `alpha = 0` transparent RGBA decoder regime where `up_blocks.2` and `up_blocks.3`
    # amplify the residual highway past `65,504.0` (`~1.5e6` in BF16/FP32) before `decoder.norm_out` (`RMSNorm`)
    # normalizes it back to `O(1)`. Unscaled FP16 overflows to `Inf -> NaN`; rescaled FP16 matches BF16/FP32.
    torch.manual_seed(20260423)
    vae5 = (
        AutoencoderKLQwenImage21(
            base_dim=4,
            decoder_base_dim=4,
            z_dim=4,
            dim_mult=[1, 2, 4, 4, 4],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, False, True, True],
            in_channels=4,
            out_channels=4,
            is_residual=True,
            scale_factor_spatial=16,
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )
        .eval()
        .to(torch.float32)
    )
    with torch.no_grad():
        vae5.decoder.up_blocks[2].resnets[1].conv2.weight.normal_(mean=0.0, std=25.0)
        vae5.decoder.up_blocks[2].upsampler.resample[1].weight.normal_(mean=0.0, std=4.0)
        vae5.decoder.up_blocks[3].upsampler.resample[1].weight.normal_(mean=0.0, std=4.0)
    vae5_z = torch.randn(1, 4, 1, 2, 2, dtype=torch.float16) * 4.0
    with torch.no_grad():
        # Reference output in FP32 (from FP16 weights):
        vae5_f16_state = {k: v.half().contiguous() for k, v in vae5.state_dict().items()}
        vae5.load_state_dict({k: v.float() for k, v in vae5_f16_state.items()})
        vae5_dec_oracle = vae5.decode(vae5_z.float()).sample.to(torch.float16)
        assert torch.isfinite(vae5_dec_oracle).all()
        # Verify unscaled FP16 indeed overflows to NaN on this fixture:
        unscaled_out = vae5.half().decode(vae5_z).sample
        assert not torch.isfinite(unscaled_out).all(), "expected unscaled FP16 5-stage VAE to overflow"
    for k, v in vae5_f16_state.items():
        tensors[f"vae5.weight.{k}"] = v.contiguous()
    tensors["vae5.z_in"] = vae5_z.contiguous()
    tensors["vae5.dec_out"] = vae5_dec_oracle.contiguous()

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

    # 5. Miniature 2-layer Qwen3VLModel (vision + deepstack_visual_indexes=[0, 1] + language_model pre-norm tap)
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    torch.manual_seed(20260422)
    vcfg = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        out_hidden_size=256,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0, 1],
    )
    tcfg = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 5000000.0,
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        },
    )
    vl_cfg = Qwen3VLConfig(
        vision_config=vcfg.to_dict(),
        text_config=tcfg.to_dict(),
        image_token_id=10,
    )
    vl_model = Qwen3VLModel(vl_cfg).eval().to(torch.float16)
    with torch.no_grad():
        for p in vl_model.parameters():
            p.normal_(mean=0.0, std=0.05)

    for k, v in vl_model.state_dict().items():
        tensors[f"model.{k}"] = v.contiguous()

    captured = {}

    def _capture_pre_norm(_mod, args):
        captured["pre_norm"] = args[0].detach().clone()

    hook_handle = vl_model.language_model.norm.register_forward_pre_hook(_capture_pre_norm)

    # 5a. Text-only conditioning forward pass
    txt_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    with torch.no_grad():
        vl_model(input_ids=txt_ids)
    tensors["vl.txt_input_ids"] = txt_ids.view(-1).to(torch.int32).contiguous()
    tensors["vl.txt_pre_norm"] = captured["pre_norm"].squeeze(0).contiguous()

    # 5b. Multimodal conditioning forward pass (16 patches = 1x4x4 -> 4 merged tokens at id=10)
    mm_ids = torch.tensor([[1, 2, 10, 10, 10, 10, 3, 4]], dtype=torch.long)
    mm_token_type_ids = (mm_ids == 10).long()
    pixel_values = torch.randn(16, 1536, dtype=torch.float16)
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    with torch.no_grad():
        vis_out = vl_model.visual(pixel_values, grid_thw=image_grid_thw)
        vl_model(
            input_ids=mm_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
    hook_handle.remove()

    tensors["vl.mm_input_ids"] = mm_ids.view(-1).to(torch.int32).contiguous()
    tensors["vl.mm_pixel_values"] = pixel_values.to(torch.float32).contiguous()
    tensors["vl.mm_grid_thw"] = image_grid_thw.view(-1).to(torch.int32).contiguous()
    tensors["vl.mm_vis_pooler"] = vis_out.pooler_output.contiguous()
    tensors["vl.mm_vis_deepstack_0"] = vis_out.deepstack_features[0].contiguous()
    tensors["vl.mm_vis_deepstack_1"] = vis_out.deepstack_features[1].contiguous()
    tensors["vl.mm_pre_norm"] = captured["pre_norm"].squeeze(0).contiguous()

    # 5c. Multi-image conditioning forward pass (2 images: 1x4x4=16 patches -> 4 tokens, 1x2x4=8 patches -> 2 tokens)
    hook_handle2 = vl_model.language_model.norm.register_forward_pre_hook(_capture_pre_norm)
    mm2_ids = torch.tensor([[1, 10, 10, 10, 10, 2, 10, 10, 3]], dtype=torch.long)
    mm2_token_type_ids = (mm2_ids == 10).long()
    mm2_pixel_values = torch.randn(24, 1536, dtype=torch.float16)
    mm2_grid_thw = torch.tensor([[1, 4, 4], [1, 2, 4]], dtype=torch.long)
    with torch.no_grad():
        vl_model(
            input_ids=mm2_ids,
            pixel_values=mm2_pixel_values,
            image_grid_thw=mm2_grid_thw,
            mm_token_type_ids=mm2_token_type_ids,
        )
    hook_handle2.remove()
    tensors["vl.mm2_input_ids"] = mm2_ids.view(-1).to(torch.int32).contiguous()
    tensors["vl.mm2_pixel_values_1"] = mm2_pixel_values[:16].to(torch.float32).contiguous()
    tensors["vl.mm2_pixel_values_2"] = mm2_pixel_values[16:].to(torch.float32).contiguous()
    tensors["vl.mm2_pre_norm"] = captured["pre_norm"].squeeze(0).contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    print(f"Wrote {len(tensors)} oracle tensors to {out_path}")


def generate_real_golden_suite(out_dir: Path):
    load_deps()
    from diffusers import QwenImage21Pipeline
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = "A serene alpine lake at golden hour reflecting snow-capped peaks, crisp foreground wildflowers, photorealistic 8k"
    coco_raw = Image.open("tools/oracle/fixtures/coco_sample.png")
    coco_576 = coco_raw.convert("RGBA").resize((576, 448), Image.Resampling.LANCZOS)
    coco_576.save(out_dir / "coco_576x448.png")
    coco_512 = coco_raw.convert("RGBA").resize((512, 512), Image.Resampling.LANCZOS)
    coco_512.save(out_dir / "coco_512x512.png")

    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1-Turbo", dtype=torch.float16, local_files_only=True
    )
    pipe.enable_model_cpu_offload()

    # 1. T2I 512x512 RGB
    g = torch.Generator("cpu").manual_seed(42)
    img_512 = pipe(prompt=prompt, width=512, height=512, num_inference_steps=8, generator=g).images[0]
    img_512.convert("RGB").save(out_dir / "turbo_512.png")

    # 2. T2I 768x512 Non-Square
    g = torch.Generator("cpu").manual_seed(42)
    img_768x512 = pipe(prompt=prompt, width=768, height=512, num_inference_steps=8, generator=g).images[0]
    img_768x512.convert("RGB").save(out_dir / "turbo_768x512.png")

    # 3. Single-image edit on raw 640x480 COCO -> 576x448 @ res=512
    g = torch.Generator("cpu").manual_seed(42)
    img_raw_edit = pipe(
        prompt="Turn this scene into a watercolor painting",
        image=[coco_raw],
        width=576,
        height=448,
        output_resolution=512,
        num_inference_steps=8,
        generator=g,
    ).images[0]
    img_raw_edit.convert("RGB").save(out_dir / "diffusers_raw_640x480_edit.png")

    # 4. Multi-image edit on [512x512, 576x448] -> 512x512
    g = torch.Generator("cpu").manual_seed(123)
    img_multi_edit = pipe(
        prompt="Combine lighting from image 2 onto image 1",
        image=[coco_512, coco_576],
        width=512,
        height=512,
        output_resolution=512,
        num_inference_steps=8,
        generator=g,
    ).images[0]
    img_multi_edit.convert("RGB").save(out_dir / "diffusers_512_576_seed123.png")
    del pipe
    torch.cuda.empty_cache()

    # 5. RGBA 1024x1024 transparent T2I + RGBA single-image edit (BF16 VAE reference)
    pipe_bf16 = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1-Turbo", dtype=torch.bfloat16, local_files_only=True
    )
    pipe_bf16.enable_model_cpu_offload()
    g = torch.Generator("cpu").manual_seed(42)
    out_rgba = pipe_bf16(
        prompt="This is an RGBA image with transparency. A cute isometric 3D icon of a glowing crystal potion bottle with a cork stopper. The image has alpha channel and the background is transparent.",
        width=1024,
        height=1024,
        output_resolution=1024,
        num_inference_steps=8,
        true_cfg_scale=1.0,
        generator=g,
    ).images[0]
    out_rgba.save(out_dir / "diffusers_rgba_transparent_1024.png")

    g = torch.Generator("cpu").manual_seed(42)
    out_edit_rgba = pipe_bf16(
        image=[out_rgba],
        prompt="Change the glowing potion inside the crystal bottle from purple to bright emerald green and add golden runes etched on the glass",
        width=1024,
        height=1024,
        output_resolution=1024,
        num_inference_steps=8,
        true_cfg_scale=1.0,
        generator=g,
    ).images[0]
    out_edit_rgba.save(out_dir / "diffusers_rgba_edit_1024.png")
    del pipe_bf16
    torch.cuda.empty_cache()

    # 6. Standard Qwen-Image-2.1 CFG=4.0 + empty negative_prompt
    pipe_std = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1", dtype=torch.float16, local_files_only=True
    )
    pipe_std.enable_model_cpu_offload()
    g = torch.Generator("cpu").manual_seed(42)
    img_cfg = pipe_std(
        prompt=prompt,
        negative_prompt="",
        true_cfg_scale=4.0,
        width=512,
        height=512,
        num_inference_steps=8,
        generator=g,
    ).images[0]
    img_cfg.convert("RGB").save(out_dir / "std_cfg4_emptyneg_512.png")
    print(f"Wrote real-model diffusers golden suite to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/oracle/fixtures/qwen_image_ops.safetensors"),
        help="Output .safetensors fixture path",
    )
    parser.add_argument(
        "--real-golden-dir",
        type=Path,
        default=None,
        help="Optional directory to generate full real-model diffusers golden PNG suite",
    )
    args = parser.parse_args()
    if args.real_golden_dir is not None:
        generate_real_golden_suite(args.real_golden_dir)
    else:
        generate_synthetic_fixture(args.output)


if __name__ == "__main__":
    main()
