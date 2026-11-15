//! Extended `warp34` FFI declarations (Vision, MoE, CLEF, HyperConnection, PLE, QSA, and Hadamard-fused ops).

#[cfg(feature = "cuda")]
use core::ffi::c_void;

#[cfg(feature = "cuda")]
use super::{
    Warp34AttentionBatch, Warp34Exec, Warp34GdnBatch, Warp34KvLayerLayout, Warp34SamplerRowCfg,
    Warp34SamplerSparseEntry,
};

#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn warp34_vision_bias_add_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        out: *mut c_void,
        rows: i32,
        cols: i32,
    ) -> i32;

    pub fn warp34_vision_layernorm_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        out: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
    ) -> i32;

    pub fn warp34_vision_bias_gelu_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        bias: *const c_void,
        out: *mut c_void,
        rows: i32,
        cols: i32,
        use_tanh: i32,
    ) -> i32;

    pub fn warp34_vision_qkv_rope_attn_f16(
        ex: *const Warp34Exec,
        qkv: *mut c_void,
        qkv_bias: *const c_void,
        pos_hw: *const i32,
        cu_seqlens: *const i32,
        out: *mut c_void,
        total_patches: i32,
        num_images: i32,
        num_heads: i32,
        head_dim: i32,
        rope_theta: f32,
    ) -> i32;

    pub fn warp34_embedding_splice_f16(
        ex: *const Warp34Exec,
        dst_embeds: *mut c_void,
        src_visual_embeds: *const c_void,
        token_indices: *const i32,
        num_visual_tokens: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn warp34_moe_topk_router_f16(
        ex: *const Warp34Exec,
        router_logits: *const c_void,
        topk_weights: *mut f32,
        topk_ids: *mut i32,
        num_tokens: i32,
        num_experts: i32,
        top_k: i32,
        norm_topk_prob: i32,
        scoring_func: i32,
    ) -> i32;

    pub fn warp34_moe_w4a16_gate_up_silu_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        gate_up_packed: *const u32,
        gate_up_scales: *const c_void,
        topk_ids: *const i32,
        interm_out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
    ) -> i32;

    pub fn warp34_moe_w4a16_down_accum_f16(
        ex: *const Warp34Exec,
        interm_in: *const c_void,
        down_packed: *const u32,
        down_scales: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        k_hidden: i32,
        n_interm: i32,
        group_size: i32,
    ) -> i32;

    pub fn warp34_moe_w4a16_mlp_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        gate_up_packed: *const u32,
        gate_up_scales: *const c_void,
        down_packed: *const u32,
        down_scales: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        interm_buf: *mut c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
    ) -> i32;

    pub fn warp34_moe_w4a16_mlp_ex_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        gate_up_packed: *const u32,
        gate_up_scales: *const c_void,
        down_packed: *const u32,
        down_scales: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        interm_buf: *mut c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
        down_hadamard_block_size: i32,
    ) -> i32;

    pub fn warp34_moe_w2a16_gate_up_silu_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        gate_up_packed: *const u32,
        gate_up_scales: *const c_void,
        topk_ids: *const i32,
        interm_out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
    ) -> i32;

    pub fn warp34_moe_w2a16_down_accum_f16(
        ex: *const Warp34Exec,
        interm_in: *const c_void,
        down_packed: *const u32,
        down_scales: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        k_hidden: i32,
        n_interm: i32,
        group_size: i32,
    ) -> i32;

    pub fn warp34_moe_w2a16_mlp_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        gate_up_packed: *const u32,
        gate_up_scales: *const c_void,
        down_packed: *const u32,
        down_scales: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        interm_buf: *mut c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
    ) -> i32;

    pub fn warp34_moe_w2a16_mlp_ex_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        gate_up_packed: *const u32,
        gate_up_scales: *const c_void,
        down_packed: *const u32,
        down_scales: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        interm_buf: *mut c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
        down_hadamard_block_size: i32,
    ) -> i32;

    pub fn warp34_moe_w4a16_mlp_zp_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        gate_up_packed: *const u32,
        gate_up_scales: *const c_void,
        gate_up_qzeros: *const u32,
        down_packed: *const u32,
        down_scales: *const c_void,
        down_qzeros: *const u32,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        interm_buf: *mut c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
    ) -> i32;

    pub fn warp34_moe_w4a16_mlp_zp_ex_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        gate_up_packed: *const u32,
        gate_up_scales: *const c_void,
        gate_up_qzeros: *const u32,
        down_packed: *const u32,
        down_scales: *const c_void,
        down_qzeros: *const u32,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        interm_buf: *mut c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
        down_hadamard_block_size: i32,
    ) -> i32;

    pub fn warp34_moe_f16_gate_up_silu(
        ex: *const Warp34Exec,
        x: *const c_void,
        gate_up_w: *const c_void,
        topk_ids: *const i32,
        interm_out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        n_interm: i32,
        k_hidden: i32,
    ) -> i32;

    pub fn warp34_moe_f16_down_accum(
        ex: *const Warp34Exec,
        interm_in: *const c_void,
        down_w: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        shared_expert_out: *const c_void,
        shared_expert_gate_logits: *const c_void,
        out: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        num_experts: i32,
        k_hidden: i32,
        n_interm: i32,
    ) -> i32;

    pub fn warp34_hc_init_streams_f16(
        ex: *const Warp34Exec,
        embed_in: *const c_void,
        residual_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn warp34_hc_rmsnorm_f16(
        ex: *const Warp34Exec,
        residual_in: *const c_void,
        norm_weight: *const c_void,
        normed_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
        eps: f32,
    ) -> i32;

    pub fn warp34_hc_down_silu_inject_f16(
        ex: *const Warp34Exec,
        down_in: *const c_void,
        down_out: *mut c_void,
        inj_in: *const c_void,
        inject_weights_out: *mut f32,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_rank_dim: i32,
        hc_count: i32,
    ) -> i32;

    pub fn warp34_hc_mix_streams_f16(
        ex: *const Warp34Exec,
        residual_in: *const c_void,
        up_raw_in: *const c_void,
        mixed_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn warp34_hc_mix_streams_hadamard_f16(
        ex: *const Warp34Exec,
        residual_in: *const c_void,
        up_raw_in: *const c_void,
        mixed_out: *mut c_void,
        had64_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn warp34_hc_combine_f16(
        ex: *const Warp34Exec,
        residual_in: *const c_void,
        block_out: *const c_void,
        inject_weights: *const f32,
        ple_delta: *const c_void,
        residual_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn warp34_hc_combine_tap_f16(
        ex: *const Warp34Exec,
        residual_in: *const c_void,
        block_out: *const c_void,
        inject_weights: *const f32,
        ple_delta: *const c_void,
        residual_out: *mut c_void,
        tap_bf16_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
        tap_stride_elems: i32,
        tap_offset_elems: i32,
    ) -> i32;

    pub fn warp34_qwen4_mtp_broadcast_add_f16(
        ex: *const Warp34Exec,
        fc_embed: *const c_void,
        fc_hidden: *const c_void,
        out_multi_hidden: *mut c_void,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn warp34_ple_dequant_f8_to_f16(
        ex: *const Warp34Exec,
        f8_in: *const u8,
        f16_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        embed_dim: i32,
        weight_scale: f32,
    ) -> i32;

    pub fn warp34_ple_dequant_f8_hadamard_to_f16(
        ex: *const Warp34Exec,
        f8_in: *const u8,
        f16_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        embed_dim: i32,
        weight_scale: f32,
    ) -> i32;

    pub fn warp34_ple_dequant_lut4_to_f16(
        ex: *const Warp34Exec,
        raw_in: *const u8,
        lut16: *const f32,
        f16_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        group_size: i32,
    ) -> i32;

    pub fn warp34_ple_dequant_lut4_hadamard_to_f16(
        ex: *const Warp34Exec,
        raw_in: *const u8,
        lut16: *const f32,
        f16_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        group_size: i32,
    ) -> i32;

    pub fn warp34_ple_dilated_conv1d_f16(
        ex: *const Warp34Exec,
        key_in: *const c_void,
        conv_weight: *const c_void,
        conv_state_pool: *mut c_void,
        key_out: *mut c_void,
        batch: *const Warp34GdnBatch,
        conv_dim: i32,
    ) -> i32;

    pub fn warp34_ple_dilated_conv1d_add_f16(
        ex: *const Warp34Exec,
        key_in: *const c_void,
        conv_weight: *const c_void,
        conv_state_pool: *mut c_void,
        key_out: *mut c_void,
        residual_inout: *mut c_void,
        batch: *const Warp34GdnBatch,
        conv_dim: i32,
    ) -> i32;

    pub fn warp34_ple_signed_sqrt_gate_f16(
        ex: *const Warp34Exec,
        norm_query: *const c_void,
        norm_key: *const c_void,
        value_in: *const c_void,
        ple_delta_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        embed_dim: i32,
        eps: f32,
    ) -> i32;

    pub fn warp34_ple_signed_sqrt_gate_norm_add_f16(
        ex: *const Warp34Exec,
        norm_query: *const c_void,
        norm_key: *const c_void,
        value_in: *const c_void,
        ple_delta_out: *mut c_void,
        residual_inout: *mut c_void,
        norm_conv_weight: *const c_void,
        norm_conv_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hc_count: i32,
        embed_dim: i32,
        eps: f32,
        norm_eps: f32,
    ) -> i32;

    pub fn warp34_qsa_indexer_norm_rope_f16(
        ex: *const Warp34Exec,
        qk_in: *const c_void,
        q_norm_weight: *const c_void,
        k_norm_weight: *const c_void,
        positions: *const i32,
        q_out: *mut c_void,
        k_out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        rope_theta: f32,
        eps: f32,
        mrope_s1: i32,
        mrope_s2: i32,
        pos_plane_stride: i32,
    ) -> i32;

    pub fn warp34_qsa_indexer_cache_update_f16(
        ex: *const Warp34Exec,
        k_in: *const c_void,
        raw_k_cache: *mut c_void,
        comp_k_cache: *mut c_void,
        k_norm_weight: *const c_void,
        positions: *const i32,
        write_slot_mapping: *const i32,
        valid_token_mask: *const u8,
        num_tokens: i32,
        head_dim: i32,
        rotary_dim: i32,
        rope_theta: f32,
        eps: f32,
        mrope_s1: i32,
        mrope_s2: i32,
        pos_plane_stride: i32,
    ) -> i32;

    pub fn warp34_qsa_select_blocks_f16(
        ex: *const Warp34Exec,
        q_indexer: *const c_void,
        comp_k_cache: *const c_void,
        batch: *const Warp34AttentionBatch,
        out_selected_phys_blocks: *mut i32,
        out_selected_logical_blocks: *mut i32,
        out_num_selected_blocks: *mut i32,
        out_token_seq_pos: *mut i32,
        q_heads: i32,
        head_dim: i32,
        block_size: i32,
        select_k_blocks: i32,
        sink_blocks: i32,
        recent_blocks: i32,
    ) -> i32;

    pub fn warp34_qsa_sparse_paged_attn_f16(
        ex: *const Warp34Exec,
        q: *const c_void,
        gate: *const c_void,
        kv_layout: *const Warp34KvLayerLayout,
        selected_phys_blocks: *const i32,
        selected_logical_blocks: *const i32,
        num_selected_blocks: *const i32,
        token_seq_pos: *const i32,
        valid_token_mask: *const u8,
        out: *mut c_void,
        num_tokens: i32,
        local_q_heads: i32,
        select_k_blocks: i32,
        sm_scale: f32,
    ) -> i32;

    pub fn warp34_qsa_sparse_paged_attn_hadamard_f16(
        ex: *const Warp34Exec,
        q: *const c_void,
        gate: *const c_void,
        kv_layout: *const Warp34KvLayerLayout,
        selected_phys_blocks: *const i32,
        selected_logical_blocks: *const i32,
        num_selected_blocks: *const i32,
        token_seq_pos: *const i32,
        valid_token_mask: *const u8,
        out: *mut c_void,
        num_tokens: i32,
        local_q_heads: i32,
        select_k_blocks: i32,
        sm_scale: f32,
    ) -> i32;

    pub fn warp34_moe_unpack_staged_experts(
        ex: *const Warp34Exec,
        staged_packs: *const c_void,
        dst_slots: *const i32,
        gate_up_packed_base: *mut u32,
        gate_up_scales_base: *mut c_void,
        down_packed_base: *mut u32,
        down_scales_base: *mut c_void,
        num_staged: i32,
        dst_capacity: i32,
        pack_stride_bytes: usize,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
        bits: i32,
    ) -> i32;

    pub fn warp34_moe_pull_mapped_experts(
        ex: *const Warp34Exec,
        mapped_host_packs: *const c_void,
        src_local_experts: *const i32,
        dst_slots: *const i32,
        gate_up_packed_base: *mut u32,
        gate_up_scales_base: *mut c_void,
        down_packed_base: *mut u32,
        down_scales_base: *mut c_void,
        num_staged: i32,
        local_num_experts: i32,
        dst_capacity: i32,
        pack_stride_bytes: usize,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
        bits: i32,
    ) -> i32;

    pub fn warp34_moe_remap_topk_ids_i32(
        ex: *const Warp34Exec,
        global_topk_ids: *const i32,
        expert_to_slot: *const i32,
        remapped_topk_ids: *mut i32,
        total_pairs: i32,
        global_expert_offset: i32,
        local_num_experts: i32,
    ) -> i32;

    pub fn warp34_add_inplace_f16(
        ex: *const Warp34Exec,
        dst_f16: *mut c_void,
        src_f16: *const c_void,
        num_elems: i64,
    ) -> i32;

    pub fn warp34_midloop_damped_euler_f16(
        ex: *const Warp34Exec,
        dst_f16: *mut c_void,
        x_prev_f16: *const c_void,
        g_x_f16: *const c_void,
        num_elems: i64,
        alpha: f32,
    ) -> i32;

    pub fn warp34_moe_gpu_lru_update_and_pull(
        ex: *const Warp34Exec,
        global_topk_ids: *const i32,
        expert_to_slot: *mut i32,
        slot_to_expert: *mut i32,
        slot_last_used: *mut u64,
        step_counter: *mut u64,
        remapped_topk_ids: *mut i32,
        miss_local_experts: *mut i32,
        miss_victim_slots: *mut i32,
        num_misses_buf: *mut i32,
        mapped_host_packs: *const c_void,
        gate_up_packed_base: *mut u32,
        gate_up_scales_base: *mut c_void,
        down_packed_base: *mut u32,
        down_scales_base: *mut c_void,
        total_pairs: i32,
        global_expert_offset: i32,
        local_num_experts: i32,
        k_hot: i32,
        max_misses: i32,
        pack_stride_bytes: usize,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
        bits: i32,
    ) -> i32;

    pub fn warp34_moe_hybrid_submit_and_pull(
        ex: *const Warp34Exec,
        d_req_slot: *mut c_void,
        x_f16: *const c_void,
        global_topk_ids: *const i32,
        topk_weights: *const f32,
        expert_to_slot: *mut i32,
        slot_to_expert: *mut i32,
        slot_last_used: *mut u64,
        step_counter: *mut u64,
        remapped_topk_ids: *mut i32,
        miss_local_experts: *mut i32,
        miss_victim_slots: *mut i32,
        num_misses_buf: *mut i32,
        mapped_host_packs: *const c_void,
        gate_up_packed_base: *mut u32,
        gate_up_scales_base: *mut c_void,
        down_packed_base: *mut u32,
        down_scales_base: *mut c_void,
        num_tokens: i32,
        top_k: i32,
        global_expert_offset: i32,
        local_num_experts: i32,
        k_static: i32,
        k_hot: i32,
        pack_stride_bytes: usize,
        n_interm: i32,
        k_hidden: i32,
        group_size: i32,
        bits: i32,
        gpu_share_frac: f32,
    ) -> i32;

    pub fn warp34_moe_hybrid_wait_and_add(
        ex: *const Warp34Exec,
        d_req_slot: *const c_void,
        d_resp_slot: *const c_void,
        out_f16: *mut c_void,
        num_misses_buf: *const i32,
        num_tokens: i32,
        k_hidden: i32,
    ) -> i32;

    pub fn warp34_gemma_rms_norm_hadamard_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        out_had64: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
    ) -> i32;

    pub fn warp34_fused_add_gemma_rms_norm_tap_hadamard_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        residual: *mut c_void,
        weight: *const c_void,
        out: *mut c_void,
        out_had64: *mut c_void,
        tap_dst_bf16: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
        tap_row_stride: i32,
        tap_col_offset: i32,
    ) -> i32;

    pub fn warp34_tp2_all_reduce_residual_gemma_rms_norm_hadamard_f16(
        ex: *const Warp34Exec,
        local_in: *const c_void,
        ar_out: *mut c_void,
        residual: *mut c_void,
        weight: *const c_void,
        out: *mut c_void,
        out_had64: *mut c_void,
        tap_dst_bf16: *mut c_void,
        stage0: *mut c_void,
        stage1: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        rank: i32,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
        tap_row_stride: i32,
        tap_col_offset: i32,
    ) -> i32;

    pub fn warp34_tp4_all_reduce_residual_gemma_rms_norm_hadamard_f16(
        ex: *const Warp34Exec,
        local_in: *const c_void,
        ar_out: *mut c_void,
        residual: *mut c_void,
        weight: *const c_void,
        out: *mut c_void,
        out_had64: *mut c_void,
        tap_dst_bf16: *mut c_void,
        stage0: *mut c_void,
        stage1: *mut c_void,
        stage2: *mut c_void,
        stage3: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        sig2: *mut c_void,
        sig3: *mut c_void,
        rank: i32,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
        tap_row_stride: i32,
        tap_col_offset: i32,
    ) -> i32;

    pub fn warp34_rms_norm_gated_silu_strided_hadamard_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        z: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        num_v_heads: i32,
        head_dim: i32,
        z_token_stride: i32,
        eps: f32,
    ) -> i32;

    pub fn warp34_rms_norm_gated_sigmoid_strided_hadamard_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        z: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        num_v_heads: i32,
        head_dim: i32,
        z_token_stride: i32,
        eps: f32,
    ) -> i32;

    pub fn warp34_silu_and_mul_hadamard_f16(
        ex: *const Warp34Exec,
        gate_up: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        intermediate_size: i32,
    ) -> i32;

    pub fn warp34_apply_attn_output_gate_hadamard_f16(
        ex: *const Warp34Exec,
        attn_out: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        total_dim: i32,
    ) -> i32;

    pub fn warp34_tp2_all_gather_topk_f32_i32(
        ex: *const Warp34Exec,
        local_vals: *const f32,
        local_ids: *const i32,
        global_vals: *mut f32,
        global_ids: *mut i32,
        stage0: *mut c_void,
        stage1: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        rank: i32,
        num_rows: i32,
        top_k: i32,
    ) -> i32;

    pub fn warp34_tp4_all_gather_topk_f32_i32(
        ex: *const Warp34Exec,
        local_vals: *const f32,
        local_ids: *const i32,
        global_vals: *mut f32,
        global_ids: *mut i32,
        stage0: *mut c_void,
        stage1: *mut c_void,
        stage2: *mut c_void,
        stage3: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        sig2: *mut c_void,
        sig3: *mut c_void,
        rank: i32,
        num_rows: i32,
        top_k: i32,
    ) -> i32;

    pub fn warp34_tp2_all_reduce_hc_combine_tap_f16(
        ex: *const Warp34Exec,
        residual_in: *const c_void,
        local_block_out: *const c_void,
        ar_out: *mut c_void,
        inject_weights: *const f32,
        ple_delta: *const c_void,
        residual_out: *mut c_void,
        tap_bf16_out: *mut c_void,
        valid_token_mask: *const u8,
        stage0: *mut c_void,
        stage1: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        rank: i32,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
        tap_stride_elems: i32,
        tap_offset_elems: i32,
    ) -> i32;

    pub fn warp34_tp4_all_reduce_hc_combine_tap_f16(
        ex: *const Warp34Exec,
        residual_in: *const c_void,
        local_block_out: *const c_void,
        ar_out: *mut c_void,
        inject_weights: *const f32,
        ple_delta: *const c_void,
        residual_out: *mut c_void,
        tap_bf16_out: *mut c_void,
        valid_token_mask: *const u8,
        stage0: *mut c_void,
        stage1: *mut c_void,
        stage2: *mut c_void,
        stage3: *mut c_void,
        sig0: *mut c_void,
        sig1: *mut c_void,
        sig2: *mut c_void,
        sig3: *mut c_void,
        rank: i32,
        num_tokens: i32,
        hc_count: i32,
        hidden_size: i32,
        tap_stride_elems: i32,
        tap_offset_elems: i32,
    ) -> i32;

    pub fn warp34_lora_shrink_expand_f16(
        ex: *const Warp34Exec,
        x_f16: *const c_void,
        lora_a_stacked: *const c_void,
        lora_b_stacked: *const c_void,
        token_lora_ids: *const i32,
        lowrank_f32_ws: *mut f32,
        y_inout_f16: *mut c_void,
        m: i32,
        size_n: i32,
        size_k: i32,
        rank_pad: i32,
        num_adapters: i32,
        default_lora_id: i32,
        adapter_mask: u64,
    ) -> i32;

    pub fn warp34_sampler_shard_fast_greedy_f16(
        ex: *const Warp34Exec,
        logits_f16: *const c_void,
        row_cfgs: *const Warp34SamplerRowCfg,
        bitmask_u32: *const u32,
        sparse_entries: *const Warp34SamplerSparseEntry,
        out_local_vals: *mut f32,
        out_local_ids: *mut i32,
        num_rows: i32,
        local_vocab_size: i32,
        vocab_offset: i32,
    ) -> i32;

    pub fn warp34_rms_norm_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
    ) -> i32;

    pub fn warp34_fused_add_rms_norm_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        residual: *mut c_void,
        weight: *const c_void,
        out: *mut c_void,
        valid_token_mask: *const u8,
        num_tokens: i32,
        hidden_size: i32,
        eps: f32,
    ) -> i32;

    pub fn warp34_fused_qk_rmsnorm_rope_f16(
        ex: *const Warp34Exec,
        q_in: *const c_void,
        k_in: *const c_void,
        q_norm_w: *const c_void,
        k_norm_w: *const c_void,
        positions: *const i32,
        valid_token_mask: *const u8,
        q_out: *mut c_void,
        k_out: *mut c_void,
        num_tokens: i32,
        pos_plane_stride: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        mrope_s0: i32,
        mrope_s1: i32,
        mrope_s2: i32,
        rope_theta: f32,
        rms_eps: f32,
        weight_bias: f32,
    ) -> i32;

    pub fn warp34_embedding_add_f16(
        ex: *const Warp34Exec,
        dst_embeds: *mut c_void,
        src_visual_embeds: *const c_void,
        token_indices: *const i32,
        num_visual_tokens: i32,
        dst_rows: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn warp34_mha_fa2_d128_f16(
        ex: *const Warp34Exec,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        b: i32,
        seqlen_q: i32,
        seqlen_k: i32,
        h: i32,
        h_k: i32,
        is_causal: i32,
    ) -> i32;

    pub fn warp34_qwen_image_timestep_sinusoidal_f16(
        ex: *const Warp34Exec,
        timesteps: *const f32,
        out: *mut c_void,
        num_rows: i32,
        timestep_dim: i32,
        max_period: f32,
        time_factor: f32,
    ) -> i32;

    pub fn warp34_qwen_image_silu_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        out: *mut c_void,
        n: i64,
    ) -> i32;

    pub fn warp34_qwen_image_build_joint_hidden_f16(
        ex: *const Warp34Exec,
        txt_hidden: *const c_void,
        img_hidden: *const c_void,
        row_map: *const i32,
        out_joint: *mut c_void,
        total_rows: i32,
        txt_rows: i32,
        img_rows: i32,
        hidden_size: i32,
    ) -> i32;

    pub fn warp34_qwen_image_adaln_modulate_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        modulation: *const c_void,
        target_token_mask: *const u8,
        out: *mut c_void,
        num_tokens: i32,
        hidden_size: i32,
        mod_stride: i32,
        scale_col_offset: i32,
        eps: f32,
    ) -> i32;

    pub fn warp34_qwen_image_gated_residual_f16(
        ex: *const Warp34Exec,
        residual: *mut c_void,
        branch_out: *const c_void,
        modulation: *const c_void,
        target_token_mask: *const u8,
        num_tokens: i32,
        hidden_size: i32,
        mod_stride: i32,
        gate_col_offset: i32,
    ) -> i32;

    pub fn warp34_qwen_image_qk_rmsnorm_complex_rope_d128_f16(
        ex: *const Warp34Exec,
        q_in: *const c_void,
        k_in: *const c_void,
        norm_q_w: *const c_void,
        norm_k_w: *const c_void,
        cos_table: *const f32,
        sin_table: *const f32,
        q_out: *mut c_void,
        k_out: *mut c_void,
        cache_k_out: *mut c_void,
        num_tokens: i32,
        num_heads: i32,
        cache_write_len: i32,
        eps: f32,
    ) -> i32;

    pub fn warp34_cublas_conv_gemm_f16(
        handle: usize,
        col: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        spatial: i32,
        out_c: i32,
        k_dim: i32,
        ld_out: i32,
    ) -> i32;

    pub fn warp34_cublas_vae_qk_f16_f32(
        handle: usize,
        q_cs: *const c_void,
        k_cs: *const c_void,
        scores_f32: *mut f32,
        spatial: i32,
        channels: i32,
        scale: f32,
    ) -> i32;

    pub fn warp34_cublas_vae_sv_f16(
        handle: usize,
        probs_f16: *const c_void,
        v_cs: *const c_void,
        out_cs: *mut c_void,
        spatial: i32,
        channels: i32,
    ) -> i32;

    pub fn warp34_qwen_image_vae_rms_norm_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        gamma: *const c_void,
        out: *mut c_void,
        channels: i32,
        spatial: i32,
        apply_silu: i32,
    ) -> i32;

    pub fn warp34_qwen_image_vae_im2col_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        col: *mut c_void,
        in_c: i32,
        h_in: i32,
        w_in: i32,
        k_h: i32,
        k_w: i32,
        pad_top: i32,
        pad_left: i32,
        stride_h: i32,
        stride_w: i32,
        oh_start: i32,
        chunk_h: i32,
        w_out: i32,
    ) -> i32;

    pub fn warp34_qwen_image_vae_bias_residual_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        out: *mut c_void,
        channels: i32,
        spatial: i32,
        clamp_pm1: i32,
    ) -> i32;

    pub fn warp34_qwen_image_vae_upsample2d_nearest2x_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        out: *mut c_void,
        channels: i32,
        h_in: i32,
        w_in: i32,
    ) -> i32;

    pub fn warp34_qwen_image_vae_avg_down_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        residual: *const c_void,
        out: *mut c_void,
        in_c: i32,
        out_c: i32,
        h_in: i32,
        w_in: i32,
        factor_t: i32,
        factor_s: i32,
    ) -> i32;

    pub fn warp34_qwen_image_vae_dup_up_f16(
        ex: *const Warp34Exec,
        x: *const c_void,
        residual: *const c_void,
        out: *mut c_void,
        in_c: i32,
        out_c: i32,
        h_in: i32,
        w_in: i32,
        factor_t: i32,
        factor_s: i32,
    ) -> i32;

    pub fn warp34_qwen_image_vae_softmax_f32_to_f16(
        ex: *const Warp34Exec,
        scores: *const f32,
        probs: *mut c_void,
        seq_len: i32,
    ) -> i32;
}
