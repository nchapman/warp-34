use core::ffi::c_void;
use w34_kernels::ffi::{self, W34AttentionBatch, W34Exec, W34GdnBatch, W34KvLayerLayout};

use crate::cuda::{CudaError, Result};

#[inline]
fn check(rc: i32, op: &'static str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(CudaError::new(rc, op))
    }
}

pub fn hc_init_streams_f16(
    ex: &W34Exec,
    embed_in: u64,
    residual_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    hc_count: usize,
    hidden_size: usize,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller and sized for `num_tokens`.
    let rc = unsafe {
        ffi::w34_hc_init_streams_f16(
            ex,
            embed_in as usize as *const c_void,
            residual_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            hc_count as i32,
            hidden_size as i32,
        )
    };
    check(rc, "w34_hc_init_streams_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn hc_rmsnorm_f16(
    ex: &W34Exec,
    residual_in: u64,
    norm_weight: u64,
    normed_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    hc_count: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller and sized for `num_tokens * hc_count * hidden_size`.
    let rc = unsafe {
        ffi::w34_hc_rmsnorm_f16(
            ex,
            residual_in as usize as *const c_void,
            norm_weight as usize as *const c_void,
            normed_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            hc_count as i32,
            hidden_size as i32,
            eps,
        )
    };
    check(rc, "w34_hc_rmsnorm_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn hc_down_silu_inject_f16(
    ex: &W34Exec,
    down_in: u64,
    down_out: u64,
    inj_in: u64,
    inject_weights_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    hc_rank_dim: usize,
    hc_count: usize,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_hc_down_silu_inject_f16(
            ex,
            down_in as usize as *const c_void,
            down_out as usize as *mut c_void,
            inj_in as usize as *const c_void,
            inject_weights_out as usize as *mut f32,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            hc_rank_dim as i32,
            hc_count as i32,
        )
    };
    check(rc, "w34_hc_down_silu_inject_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn hc_mix_streams_f16(
    ex: &W34Exec,
    residual_in: u64,
    up_raw_in: u64,
    mixed_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    hc_count: usize,
    hidden_size: usize,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_hc_mix_streams_f16(
            ex,
            residual_in as usize as *const c_void,
            up_raw_in as usize as *const c_void,
            mixed_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            hc_count as i32,
            hidden_size as i32,
        )
    };
    check(rc, "w34_hc_mix_streams_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn hc_mix_streams_hadamard_f16(
    ex: &W34Exec,
    residual_in: u64,
    up_raw_in: u64,
    mixed_out: u64,
    had64_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    hc_count: usize,
    hidden_size: usize,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_hc_mix_streams_hadamard_f16(
            ex,
            residual_in as usize as *const c_void,
            up_raw_in as usize as *const c_void,
            mixed_out as usize as *mut c_void,
            had64_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            hc_count as i32,
            hidden_size as i32,
        )
    };
    check(rc, "w34_hc_mix_streams_hadamard_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn hc_combine_f16(
    ex: &W34Exec,
    residual_in: u64,
    block_out: u64,
    inject_weights: u64,
    ple_delta: u64,
    residual_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    hc_count: usize,
    hidden_size: usize,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_hc_combine_f16(
            ex,
            residual_in as usize as *const c_void,
            block_out as usize as *const c_void,
            inject_weights as usize as *const f32,
            ple_delta as usize as *const c_void,
            residual_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            hc_count as i32,
            hidden_size as i32,
        )
    };
    check(rc, "w34_hc_combine_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn hc_combine_tap_f16(
    ex: &W34Exec,
    residual_in: u64,
    block_out: u64,
    inject_weights: u64,
    ple_delta: u64,
    residual_out: u64,
    tap_bf16_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    hc_count: usize,
    hidden_size: usize,
    tap_stride_elems: i32,
    tap_offset_elems: i32,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_hc_combine_tap_f16(
            ex,
            residual_in as usize as *const c_void,
            block_out as usize as *const c_void,
            inject_weights as usize as *const f32,
            ple_delta as usize as *const c_void,
            residual_out as usize as *mut c_void,
            tap_bf16_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            hc_count as i32,
            hidden_size as i32,
            tap_stride_elems,
            tap_offset_elems,
        )
    };
    check(rc, "w34_hc_combine_tap_f16")
}

pub fn ple_dequant_f8_to_f16(
    ex: &W34Exec,
    f8_in: u64,
    f16_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    embed_dim: usize,
    weight_scale: f32,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_ple_dequant_f8_to_f16(
            ex,
            f8_in as usize as *const u8,
            f16_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            embed_dim as i32,
            weight_scale,
        )
    };
    check(rc, "w34_ple_dequant_f8_to_f16")
}

pub fn ple_dequant_f8_hadamard_to_f16(
    ex: &W34Exec,
    f8_in: u64,
    f16_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    embed_dim: usize,
    weight_scale: f32,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_ple_dequant_f8_hadamard_to_f16(
            ex,
            f8_in as usize as *const u8,
            f16_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            embed_dim as i32,
            weight_scale,
        )
    };
    check(rc, "w34_ple_dequant_f8_hadamard_to_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn ple_dequant_lut4_to_f16(
    ex: &W34Exec,
    raw_in: u64,
    lut16: u64,
    f16_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    group_size: usize,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_ple_dequant_lut4_to_f16(
            ex,
            raw_in as usize as *const u8,
            lut16 as usize as *const f32,
            f16_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            num_heads as i32,
            head_dim as i32,
            group_size as i32,
        )
    };
    check(rc, "w34_ple_dequant_lut4_to_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn ple_dequant_lut4_hadamard_to_f16(
    ex: &W34Exec,
    raw_in: u64,
    lut16: u64,
    f16_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    group_size: usize,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_ple_dequant_lut4_hadamard_to_f16(
            ex,
            raw_in as usize as *const u8,
            lut16 as usize as *const f32,
            f16_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            num_heads as i32,
            head_dim as i32,
            group_size as i32,
        )
    };
    check(rc, "w34_ple_dequant_lut4_hadamard_to_f16")
}

pub fn ple_dilated_conv1d_f16(
    ex: &W34Exec,
    key_in: u64,
    conv_weight: u64,
    conv_state_pool: u64,
    key_out: u64,
    batch: &W34GdnBatch,
    conv_dim: usize,
) -> Result<()> {
    // SAFETY: Device pointers and `batch` metadata are validated by the caller.
    let rc = unsafe {
        ffi::w34_ple_dilated_conv1d_f16(
            ex,
            key_in as usize as *const c_void,
            conv_weight as usize as *const c_void,
            conv_state_pool as usize as *mut c_void,
            key_out as usize as *mut c_void,
            batch,
            conv_dim as i32,
        )
    };
    check(rc, "w34_ple_dilated_conv1d_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn ple_dilated_conv1d_add_f16(
    ex: &W34Exec,
    key_in: u64,
    conv_weight: u64,
    conv_state_pool: u64,
    key_out: u64,
    residual_inout: u64,
    batch: &W34GdnBatch,
    conv_dim: usize,
) -> Result<()> {
    // SAFETY: Device pointers and `batch` metadata are validated by the caller.
    let rc = unsafe {
        ffi::w34_ple_dilated_conv1d_add_f16(
            ex,
            key_in as usize as *const c_void,
            conv_weight as usize as *const c_void,
            conv_state_pool as usize as *mut c_void,
            key_out as usize as *mut c_void,
            residual_inout as usize as *mut c_void,
            batch,
            conv_dim as i32,
        )
    };
    check(rc, "w34_ple_dilated_conv1d_add_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn ple_signed_sqrt_gate_f16(
    ex: &W34Exec,
    norm_query: u64,
    norm_key: u64,
    value_in: u64,
    ple_delta_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    hc_count: usize,
    embed_dim: usize,
    eps: f32,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_ple_signed_sqrt_gate_f16(
            ex,
            norm_query as usize as *const c_void,
            norm_key as usize as *const c_void,
            value_in as usize as *const c_void,
            ple_delta_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            hc_count as i32,
            embed_dim as i32,
            eps,
        )
    };
    check(rc, "w34_ple_signed_sqrt_gate_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn ple_signed_sqrt_gate_norm_add_f16(
    ex: &W34Exec,
    norm_query: u64,
    norm_key: u64,
    value_in: u64,
    ple_delta_out: u64,
    residual_inout: u64,
    norm_conv_weight: u64,
    norm_conv_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    hc_count: usize,
    embed_dim: usize,
    eps: f32,
    norm_eps: f32,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_ple_signed_sqrt_gate_norm_add_f16(
            ex,
            norm_query as usize as *const c_void,
            norm_key as usize as *const c_void,
            value_in as usize as *const c_void,
            ple_delta_out as usize as *mut c_void,
            residual_inout as usize as *mut c_void,
            norm_conv_weight as usize as *const c_void,
            norm_conv_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            hc_count as i32,
            embed_dim as i32,
            eps,
            norm_eps,
        )
    };
    check(rc, "w34_ple_signed_sqrt_gate_norm_add_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_indexer_norm_rope_f16(
    ex: &W34Exec,
    qk_in: u64,
    q_norm_weight: u64,
    k_norm_weight: u64,
    positions: u64,
    q_out: u64,
    k_out: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    eps: f32,
    mrope_s1: usize,
    mrope_s2: usize,
    pos_plane_stride: usize,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_qsa_indexer_norm_rope_f16(
            ex,
            qk_in as usize as *const c_void,
            q_norm_weight as usize as *const c_void,
            k_norm_weight as usize as *const c_void,
            positions as usize as *const i32,
            q_out as usize as *mut c_void,
            k_out as usize as *mut c_void,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            rope_theta,
            eps,
            mrope_s1 as i32,
            mrope_s2 as i32,
            pos_plane_stride as i32,
        )
    };
    check(rc, "w34_qsa_indexer_norm_rope_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_indexer_cache_update_f16(
    ex: &W34Exec,
    k_in: u64,
    raw_k_cache: u64,
    comp_k_cache: u64,
    k_norm_weight: u64,
    positions: u64,
    write_slot_mapping: u64,
    valid_token_mask: u64,
    num_tokens: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    eps: f32,
    mrope_s1: usize,
    mrope_s2: usize,
    pos_plane_stride: usize,
) -> Result<()> {
    // SAFETY: Device pointers are validated by the caller.
    let rc = unsafe {
        ffi::w34_qsa_indexer_cache_update_f16(
            ex,
            k_in as usize as *const c_void,
            raw_k_cache as usize as *mut c_void,
            comp_k_cache as usize as *mut c_void,
            k_norm_weight as usize as *const c_void,
            positions as usize as *const i32,
            write_slot_mapping as usize as *const i32,
            valid_token_mask as usize as *const u8,
            num_tokens as i32,
            head_dim as i32,
            rotary_dim as i32,
            rope_theta,
            eps,
            mrope_s1 as i32,
            mrope_s2 as i32,
            pos_plane_stride as i32,
        )
    };
    check(rc, "w34_qsa_indexer_cache_update_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_select_blocks_f16(
    ex: &W34Exec,
    q_indexer: u64,
    comp_k_cache: u64,
    batch: &W34AttentionBatch,
    out_selected_phys_blocks: u64,
    out_selected_logical_blocks: u64,
    out_num_selected_blocks: u64,
    out_token_seq_pos: u64,
    q_heads: usize,
    head_dim: usize,
    block_size: usize,
    select_k_blocks: usize,
    sink_blocks: usize,
    recent_blocks: usize,
) -> Result<()> {
    // SAFETY: Device pointers and `batch` metadata are validated by the caller.
    let rc = unsafe {
        ffi::w34_qsa_select_blocks_f16(
            ex,
            q_indexer as usize as *const c_void,
            comp_k_cache as usize as *const c_void,
            batch,
            out_selected_phys_blocks as usize as *mut i32,
            out_selected_logical_blocks as usize as *mut i32,
            out_num_selected_blocks as usize as *mut i32,
            out_token_seq_pos as usize as *mut i32,
            q_heads as i32,
            head_dim as i32,
            block_size as i32,
            select_k_blocks as i32,
            sink_blocks as i32,
            recent_blocks as i32,
        )
    };
    check(rc, "w34_qsa_select_blocks_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_sparse_paged_attn_f16(
    ex: &W34Exec,
    q: u64,
    gate: u64,
    kv_layout: &W34KvLayerLayout,
    selected_phys_blocks: u64,
    selected_logical_blocks: u64,
    num_selected_blocks: u64,
    token_seq_pos: u64,
    valid_token_mask: u64,
    out: u64,
    num_tokens: usize,
    local_q_heads: usize,
    select_k_blocks: usize,
    sm_scale: f32,
) -> Result<()> {
    // SAFETY: Device pointers and `kv_layout` are validated by the caller.
    let rc = unsafe {
        ffi::w34_qsa_sparse_paged_attn_f16(
            ex,
            q as usize as *const c_void,
            gate as usize as *const c_void,
            kv_layout,
            selected_phys_blocks as usize as *const i32,
            selected_logical_blocks as usize as *const i32,
            num_selected_blocks as usize as *const i32,
            token_seq_pos as usize as *const i32,
            valid_token_mask as usize as *const u8,
            out as usize as *mut c_void,
            num_tokens as i32,
            local_q_heads as i32,
            select_k_blocks as i32,
            sm_scale,
        )
    };
    check(rc, "w34_qsa_sparse_paged_attn_f16")
}

#[allow(clippy::too_many_arguments)]
pub fn qsa_sparse_paged_attn_hadamard_f16(
    ex: &W34Exec,
    q: u64,
    gate: u64,
    kv_layout: &W34KvLayerLayout,
    selected_phys_blocks: u64,
    selected_logical_blocks: u64,
    num_selected_blocks: u64,
    token_seq_pos: u64,
    valid_token_mask: u64,
    out: u64,
    num_tokens: usize,
    local_q_heads: usize,
    select_k_blocks: usize,
    sm_scale: f32,
) -> Result<()> {
    // SAFETY: Device pointers and `kv_layout` are validated by the caller.
    let rc = unsafe {
        ffi::w34_qsa_sparse_paged_attn_hadamard_f16(
            ex,
            q as usize as *const c_void,
            gate as usize as *const c_void,
            kv_layout,
            selected_phys_blocks as usize as *const i32,
            selected_logical_blocks as usize as *const i32,
            num_selected_blocks as usize as *const i32,
            token_seq_pos as usize as *const i32,
            valid_token_mask as usize as *const u8,
            out as usize as *mut c_void,
            num_tokens as i32,
            local_q_heads as i32,
            select_k_blocks as i32,
            sm_scale,
        )
    };
    check(rc, "w34_qsa_sparse_paged_attn_hadamard_f16")
}
