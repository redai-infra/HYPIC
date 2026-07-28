import os
from typing import Optional, Tuple, Union

import torch
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.pic import diag_layer_dump as _diag_dump
from sglang.srt.pic.conv_tails import (
    build_prev_tail_slots,
    capture_conv_tails,
    load_conv_history,
)
from sglang.srt.pic.policy import PICCompose
from sglang.srt.pic.state_composition import build_addition_prefix_states
from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h as _fla_fwd_h,
    )
    from sglang.srt.layers.attention.fla.chunk_fwd import (
        chunk_gated_delta_rule_fwd_intra as _fla_fwd_intra,
    )
    from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o as _fla_fwd_o
    from sglang.srt.layers.attention.fla.cumsum import (
        chunk_local_cumsum as _fla_cumsum,
    )
    from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd as _fla_l2norm

if is_cuda() or is_hip():
    from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkv_split_gdn_prefill

MAX_FUSED_QKV_SPLIT_DIM = 8192

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

        cutedsl_kernel = None
        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            cutedsl_kernel = CuteDSLGDNKernel()
            self.decode_kernel = cutedsl_kernel
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            # Reuse the CuteDSL kernel if already created for decode
            if cutedsl_kernel is None:
                from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                    CuteDSLGDNKernel,
                )

                cutedsl_kernel = CuteDSLGDNKernel()
            # The CuteDSL prefill kernel only exists on SM100+ (Blackwell).
            # On SM90 (Hopper) fall back to Triton so users can pick
            # `cutedsl` uniformly across hardware.
            if cutedsl_kernel.supports_prefill:
                self.extend_kernel = cutedsl_kernel
            else:
                rank0_log(
                    "CuTe DSL GDN prefill is not supported on this GPU "
                    "(requires SM100+). Falling back to Triton for prefill."
                )
                self.extend_kernel = triton_kernel
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel: use FlashInfer when the selected FlashInfer kernel
        # supports MTP verify. SM90 uses the fp32-state path; SM100 uses the
        # bf16-state adapter in FlashInferGDNKernel.
        if (
            decode_backend.is_flashinfer() or prefill_backend.is_flashinfer()
        ) and flashinfer_kernel.supports_target_verify:
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)
        self.verify_intermediate_state_indices = torch.arange(
            self.req_to_token_pool.size, dtype=torch.int32, device=model_runner.device
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            self.forward_metadata.mamba_track_mask_indices = (
                forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_indices = self.verify_intermediate_state_indices
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            _diag_dump.dump_conv_slots(layer.layer_id, "base_conv_pre", cache_indices, has_initial_states, conv_states)
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        actual_seq_len = mixed_qkv.shape[0]
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        if (is_cuda() or is_hip()) and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM:
            query, key, value = fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )
            query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            core_attn_out = self.kernel_dispatcher.target_verify(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_state_cache,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
            )
        else:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            _diag_dump.dump_qkvgb(layer.layer_id, "base", query[0], key[0], value[0], g[0], beta[0])
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )
            _diag_dump.dump_ssm_state(layer.layer_id, "base_ssm_final", ssm_states[cache_indices[0].item()])
            _diag_dump.dump_o(layer.layer_id, "base", core_attn_out)

            if (is_npu() or is_cpu()) and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out

    def init_pic_metadata(self, forward_batch: ForwardBatch):
        """Pre-compute PIC segment metadata (called once per batch from HybridLinearAttnBackend)."""
        device = forward_batch.input_ids.device
        pic_miss_segments = forward_batch.pic_miss_segments
        batch_size = forward_batch.batch_size
        req_cache_indices = self.forward_metadata.mamba_cache_indices
        p = forward_batch.pic_policy
        is_recompute = p.recompute
        is_addition = p.compose is PICCompose.ADDITION

        # --- Shared GDN workspace shape prelude (all modes). Batch-level shapes
        # from layer 0's cache, identical across layers in current hybrid configs.
        layer0_cache = self.req_to_token_pool.mamba2_layer_cache(0)
        ssm0 = layer0_cache.temporal           # (N_slots, H_v, V, K)
        conv0 = layer0_cache.conv[0]           # used as qkv dtype proxy
        _, H_v, V, K = ssm0.shape
        seq_len = forward_batch.input_ids.shape[0]
        ssm_dtype = ssm0.dtype
        qkv_dtype = conv0.dtype
        # transition pool present == transition-family mode (pool has it iff
        # a transition pic_mode was passed at MambaPool init time).
        has_transition = layer0_cache.transition is not None

        def _alloc(buf, shape, dtype):
            if (buf is None or buf.shape != shape
                    or buf.dtype != dtype or buf.device != device):
                return torch.empty(shape, dtype=dtype, device=device)
            return buf

        # Final per-req dst mamba slot — same across all modes and exactly
        # req_cache_indices in batch order, so build it vectorized (no per-req
        # .item() host-device sync).
        self._pic_req_dst_indices_long = req_cache_indices[:batch_size].to(torch.long)

        # h_accum staging + read-only u_zero: used by transition AND recompute
        # compose (not addition). transition pool present == transition-family.
        if has_transition:
            # Per-req composed final-state staging (fp32, matches compose
            # h_accum dtype) → one batched scatter instead of per-req writes.
            self._pic_h_accum_buf = _alloc(
                getattr(self, "_pic_h_accum_buf", None),
                (batch_size, H_v, V, K), torch.float32,
            )
            # u shape from _fla_fwd_intra is (1, seq_len, H_v, V). Zero once at
            # alloc, then never reset (read-only across all layers).
            u_shape = (1, seq_len, H_v, V)
            u_buf = getattr(self, "_pic_u_zero_buf", None)
            if (u_buf is None or u_buf.shape != u_shape
                    or u_buf.dtype != qkv_dtype or u_buf.device != device):
                self._pic_u_zero_buf = torch.zeros(u_shape, dtype=qkv_dtype, device=device)

        # recompute owns its own phys_seg/compute-level metadata + S/T buffers
        # (fully self-contained in _build_pic_recompute_metadata) and reads NONE
        # of the segment-level metadata below — skip building it entirely.
        if is_recompute:
            self._build_pic_recompute_metadata(
                forward_batch=forward_batch,
                H_v=H_v, V=V, K=K, seq_len=seq_len,
                ssm_dtype=ssm_dtype, qkv_dtype=qkv_dtype, device=device,
                _alloc=_alloc,
            )
            return

        # ===== segment-level metadata (addition + transition only) =====
        seg_lengths = []
        for req_idx in range(batch_size):
            # Invariant: pic_miss_segments[req] is start-monotonic (segmenter
            # appends segs in left→right text order; schedule_batch zip+filter
            # preserves order). Compose-loop payload index (seg_cursor+local_i)
            # aligns with _pic_seg_cu_seqlens / _pic_seg_indices only under
            # this invariant — a future scheduler that reorders miss_segs
            # would silently mis-seed Pass3 _fwd_h. Assert to fail loud.
            prev_start = -1
            for (start, end) in pic_miss_segments[req_idx]:
                assert start > prev_start, (
                    f"pic_miss_segments[{req_idx}] not start-monotonic: "
                    f"{pic_miss_segments[req_idx]}"
                )
                prev_start = start
                seg_lengths.append(end - start)
        num_segments = len(seg_lengths)

        pic_hit_mamba_slots = forward_batch.pic_hit_mamba_slots
        pic_miss_mamba_slots = forward_batch.pic_miss_mamba_slots
        seg_offsets: list = [0]
        persist_src_list: list = []
        persist_dst_list: list = []
        # Transition-mode persist plan: covers ALL miss segs (incl. the last
        # question seg per req) — its slot is freed at request end without
        # being indexed, so the extra write is harmless and lets us scatter
        # in a single fancy-index op instead of a per-seg Python loop.
        trans_persist_src_list: list = []
        trans_persist_dst_list: list = []
        # Per-req start-sorted [(slot, compute_payload_or_None), ...] for the
        # transition-mode compose loop (transition only; addition never reads
        # it). `compute_payload is None` ↔ REUSE cached (T,S) at `slot`;
        # non-None ↔ COMPUTE — payload row in this batch's Pass1/Pass2 output.
        trans_compose_order: list = []
        seg_cursor = 0
        seg_conv_slot_list: list = []  # Fix: per-segment unique conv slot
        pic_hit_segments = forward_batch.pic_hit_segments
        for req_idx in range(batch_size):
            hit_slots = pic_hit_mamba_slots[req_idx] if pic_hit_mamba_slots else {}
            miss_segs = pic_miss_segments[req_idx]
            req_seg_count = len(miss_segs)
            seg_offsets.append(seg_cursor + req_seg_count)

            miss_slots = pic_miss_mamba_slots[req_idx] if pic_miss_mamba_slots else {}
            # addition mode has no reuse/hit segments; default keeps the later
            # is_single_seg (scatter) check well-defined without an UnboundLocalError.
            reuse_segs = []
            if not is_addition:
                reuse_segs = pic_hit_segments[req_idx] if pic_hit_segments else []
                req_order = [(s, hit_slots[h], None) for (s, e, h) in reuse_segs]
                req_order.extend(
                    (s, miss_slots[(s, e)], seg_cursor + local_i)
                    for local_i, (s, e) in enumerate(miss_segs)
                )
                req_order.sort(key=lambda x: x[0])
                trans_compose_order.append(
                    [(slot, compute_payload) for (_, slot, compute_payload) in req_order]
                )
            req_dst_slot = req_cache_indices[req_idx].item()
            for local_i, (start, end) in enumerate(miss_segs):
                persist_slot_any = miss_slots.get((start, end))
                if persist_slot_any is not None:
                    trans_persist_src_list.append(seg_cursor + local_i)
                    trans_persist_dst_list.append(persist_slot_any)
                # ponytail: capture conv tail for non-last segs, AND for the lone
                # segment of a scatter single_seg request (total 1 segment). That
                # segment IS cached + later reused as a non-last (prefix) segment
                # on the combine worker, so its tail must be persisted now — else
                # the combine's next-seg conv1d seeds from zeros and multi-seg
                # compose diverges. (Normal reqs always have >=2 segs, unchanged.)
                is_single_seg = (len(reuse_segs) + req_seg_count) == 1
                if local_i < req_seg_count - 1 or is_single_seg:
                    persist_slot = persist_slot_any
                    if persist_slot is not None:
                        persist_src_list.append(seg_cursor + local_i)
                        persist_dst_list.append(persist_slot)
                    # Non-last miss seg uses its own persist slot for conv;
                    # if no persist slot (shouldn't normally happen), fall
                    # back to req slot — but then this seg's tail won't be
                    # cached for the next seg, so flag.
                    seg_conv_slot_list.append(
                        persist_slot if persist_slot is not None else req_dst_slot
                    )
                else:
                    # Last miss seg of a request uses the request's mamba slot.
                    seg_conv_slot_list.append(req_dst_slot)
            seg_cursor += req_seg_count

        self._pic_fused_persist_src = torch.tensor(persist_src_list, dtype=torch.int32, device=device)
        self._pic_fused_persist_dst = torch.tensor(persist_dst_list, dtype=torch.int32, device=device)
        self._pic_trans_persist_src = torch.tensor(trans_persist_src_list, dtype=torch.long, device=device)
        self._pic_trans_persist_dst = torch.tensor(trans_persist_dst_list, dtype=torch.long, device=device)
        if is_addition:
            # Per-req "last miss payload index" (long, fancy-index): kernel
            # writes composed end-state to temp_states[last_miss_payload[req]],
            # which then lands at ssm_states[req_dst_slot]. Addition-only.
            self._pic_req_last_miss_payload = (
                torch.tensor(seg_offsets[1:], dtype=torch.long, device=device) - 1
            )
            self._pic_seg_offsets = seg_offsets
            self._pic_has_multiple_misses = any(
                end - start > 1
                for start, end in zip(seg_offsets, seg_offsets[1:])
            )
        else:
            self._pic_trans_compose_order = trans_compose_order

        seg_lens_tensor = torch.tensor(seg_lengths, dtype=torch.int32, device=device)
        seg_cu_seqlens = torch.zeros(num_segments + 1, dtype=torch.int32, device=device)
        seg_cu_seqlens[1:] = torch.cumsum(seg_lens_tensor, dim=0)

        self._pic_seg_cu_seqlens = seg_cu_seqlens
        self._pic_seg_indices = torch.arange(num_segments, dtype=torch.int32, device=device)
        # Per-segment UNIQUE conv slot (was: all = req's mamba slot, which
        # caused fancy-index race in load_conv_history and broke fused
        # multi-seg causal_conv1d_fn's per-seg initial state isolation).
        self._pic_seg_conv_indices = torch.tensor(
            seg_conv_slot_list, dtype=torch.int32, device=device,
        )
        self._pic_seg_lengths = seg_lengths

        # PIC conv cross-segment state passing: for each miss segment, find
        # the mamba slot of its position-preceding segment (hit or miss). If
        # the preceding slot exists, the miss segment must load its conv tail
        # and tell causal_conv1d_fn `has_initial_state=True`.
        prev_tail_slots = build_prev_tail_slots(
            batch_size=batch_size,
            pic_hit_segments=forward_batch.pic_hit_segments,
            pic_hit_mamba_slots=forward_batch.pic_hit_mamba_slots,
            pic_miss_segments=forward_batch.pic_miss_segments,
            pic_miss_mamba_slots=forward_batch.pic_miss_mamba_slots,
            req_cache_indices=req_cache_indices,
        )
        self._pic_prev_tail_slot = torch.tensor(
            prev_tail_slots, dtype=torch.int32, device=device,
        )
        self._pic_has_initial_states = torch.tensor(
            [s >= 0 for s in prev_tail_slots], dtype=torch.bool, device=device,
        )

        # --- Segment-level GDN workspaces (addition + transition).
        # ponytail: fp32 (was ssm_dtype/bf16). S in bf16 → compose loses ~1e-3
        # per segment; ssm_final accumulated to ~3e-3. fp32 keeps full kernel
        # precision through the compose chain.
        state_shape = (num_segments, H_v, V, K)
        self._pic_temp_states_buf = _alloc(
            getattr(self, "_pic_temp_states_buf", None), state_shape, torch.float32,
        )
        if is_addition and self._pic_has_multiple_misses:
            self._pic_addition_h0_buf = _alloc(
                getattr(self, "_pic_addition_h0_buf", None),
                state_shape,
                torch.float32,
            )
        # transition-only S/T + seeded-h0 buffers.
        if has_transition:
            # ponytail: fp32 (was qkv_dtype/bf16). T in bf16 → ssm_final ~4e-3.
            self._pic_temp_transitions_buf = _alloc(
                getattr(self, "_pic_temp_transitions_buf", None),
                (num_segments, H_v, K, K), torch.float32,
            )
            # ponytail: fp32 so Pass3's seeded h0 keeps compose precision.
            self._pic_kernel_h0_buf = _alloc(
                getattr(self, "_pic_kernel_h0_buf", None), state_shape, torch.float32,
            )

    # ------------------------------------------------------------------
    # transition_rope_recompute helpers
    # ------------------------------------------------------------------
    def _build_pic_recompute_metadata(
        self, *, forward_batch, H_v, V, K, seq_len,
        ssm_dtype, qkv_dtype, device, _alloc,
    ):
        """Build metadata for the mini-seg 3-pass recompute scheme.

        Phys_seg (conv1d) breakpoints sit at each pic's interior_end (cold
        prefill) or batch-interior_end (hit replay, where interior_end =
        sink_end because interior tokens are absent from batch). Each phys_seg
        carries this pic's sink + interior (if in batch). Conv tail capture is at offset=0 from the
        phys_seg end (= interior_tail for pic / seg_end_tail for sys), so the
        next phys_seg loads the right K-1 lookback for cross-seg continuity.

        Compute rows (Pass1/2/3 cu_seqlens) partition by LOGICAL mini-seg:
          - sys (always): 1 row over full sys; persist (S_full, T_full) to pool.
          - mid pic miss (cold): sink + interior rows; persist ONLY
            interior row's (S, T) to pool.
          - mid pic hit (replay): sink row over batch tokens; no
            persist (pool was filled at cold prefill of an earlier request).
          - query (always last, always miss): 1 row over full query.

        Compose chain per req walks seg_order as sink → interior. Interior
        stage uses pool for hit, batch row for miss. Pass3 h0 per row =
        h_accum entering that mini-seg.
        """
        # Sub-seg lengths in batch order (for fla cumsum/fwd_intra cu_seqlens).
        # Each sub-seg corresponds to either an interior subset segment or a seam
        # subset segment, contiguous in mixed_qkv. fla kernels reset state at
        # sub-seg boundaries so that downstream fwd_h (on interior/seam subsets)
        # sees correct per-segment-relative g_cumsum.
        fla_seg_lengths_in_batch_order: list = []

        rope_meta = forward_batch.pic_rope_meta or []
        pic_hit_segments = forward_batch.pic_hit_segments or []
        pic_miss_segments = forward_batch.pic_miss_segments or []
        pic_hit_mamba_slots = forward_batch.pic_hit_mamba_slots or []
        pic_miss_mamba_slots = forward_batch.pic_miss_mamba_slots or []
        req_cache_indices = self.forward_metadata.mamba_cache_indices
        batch_size = forward_batch.batch_size

        # ----- per-req: build sorted abs-pos list (matches batch input_ids order)
        # plus pic_segments (hit+miss merged, sorted by start).
        per_req_pic_segments: list = []
        per_req_abs_pos_sorted: list = []
        per_req_offset: list = []
        running_offset = 0
        for req_idx in range(batch_size):
            hit_segs = pic_hit_segments[req_idx] if pic_hit_segments else []
            miss_segs = pic_miss_segments[req_idx] if pic_miss_segments else []
            # Build merged segs (kind tag).
            segs: list = []
            for (s, e, h) in hit_segs:
                segs.append((s, e, "hit", h))
            for (s, e) in miss_segs:
                segs.append((s, e, "miss", None))
            segs.sort(key=lambda x: x[0])
            per_req_pic_segments.append(segs)

            # Reconstruct abs-pos list for this req in batch order:
            # miss segs contribute every position; hit segs contribute only seam.
            seam_meta = (rope_meta[req_idx] or {}).get("seam") if rope_meta else None
            hit_seam_dict = (seam_meta or {}).get("hit_seam") or {}
            abs_pos = []
            for (s, e) in miss_segs:
                abs_pos.extend(range(s, e))
            for sink_pos in hit_seam_dict.values():
                abs_pos.extend(sink_pos)
            abs_pos.sort()
            per_req_abs_pos_sorted.append(abs_pos)
            per_req_offset.append(running_offset)
            running_offset += len(abs_pos)

        # Flat abs_pos for all batch tokens — used by diag comparator to
        # align PIC's per-token dumps against BASE's full-sequence dump.
        self._pic_abs_pos_flat = [p for req in per_req_abs_pos_sorted for p in req]

        # ----- iterate per req per pic_segment to build phys_seg / mini-seg
        # compute rows / compose plan.
        #
        # Mini-seg model:
        #   - sys:       1 mini-seg = full sys (no sink).
        #   - mid pic:   sink + interior. Interior is in
        #                batch only for cold-prefill misses; absent for hits.
        #   - query:     1 mini-seg = full query.
        #
        # Phys_seg model (conv1d cu_seqlens):
        #   - Boundaries at end of each pic's interior_end (in BATCH order),
        #     end of sys, end of query.
        #   - Each phys_seg's conv slot = THIS seg's mamba slot (write-through
        #     irrelevant except for query which holds the final state).
        #   - Each phys_seg's prev_tail_slot = previous seg's mamba slot, so
        #     load_conv_history copies conv_tails[prev_slot] → conv_states.
        #   - Capture at offset=0 from phys_seg end (= interior_tail for pic
        #     / seg_end_tail for sys / nothing for query).
        phys_seg_lengths: list = []
        phys_seg_conv_slots: list = []
        phys_seg_prev_slots: list = []
        phys_persist_src: list = []
        phys_persist_dst: list = []
        phys_persist_end_offset: list = []  # always 0 in new scheme

        compute_seg_lengths: list = []     # per compute row, batch-order
        compute_seg_indices: list = []     # row idx 0..C-1 (== position)
        compute_persist_src: list = []     # rows whose S/T persist to pool
        compute_persist_dst: list = []     # mamba slot for each

        compose_plan: list = []  # per-req entry

        for req_idx in range(batch_size):
            segs = per_req_pic_segments[req_idx]
            n_segs = len(segs)

            hit_slots_dict = pic_hit_mamba_slots[req_idx] if pic_hit_mamba_slots else {}
            miss_slots_dict = pic_miss_mamba_slots[req_idx] if pic_miss_mamba_slots else {}
            req_dst_slot = int(req_cache_indices[req_idx].item())
            seam_meta = (rope_meta[req_idx] or {}).get("seam") if rope_meta else None
            hit_seam_dict = (seam_meta or {}).get("hit_seam") or {}

            seg_order_entries: list = []  # for compose_plan

            # State across segs in this req for phys_seg assembly:
            cur_phys_start = per_req_offset[req_idx]
            cur_phys_prev_slot = -1     # mamba slot whose conv_tail next phys loads
            tok_cursor = per_req_offset[req_idx]

            for i, (s, e, kind, hkey) in enumerate(segs):
                seg_len = e - s
                is_first = (i == 0)
                is_last = (i == n_segs - 1)
                if is_first or is_last:
                    x_sink_eff = 0
                elif kind == "hit" and (s, e) in hit_seam_dict:
                    sink_pos = hit_seam_dict[(s, e)]
                    x_sink_eff = len(sink_pos)
                else:
                    x_sink_eff = 0
                interior_len = seg_len - x_sink_eff

                # mamba slot for this seg
                if kind == "hit":
                    seg_mamba_slot = int(hit_slots_dict[hkey])
                elif is_last:
                    seg_mamba_slot = req_dst_slot
                else:
                    seg_mamba_slot = int(miss_slots_dict.get((s, e), req_dst_slot))

                # Tokens-in-batch for this seg:
                if kind == "hit":
                    in_batch_interior_len = 0
                    in_batch_len = x_sink_eff
                else:
                    in_batch_interior_len = interior_len
                    in_batch_len = seg_len

                # ===== Mini-seg compute rows + token positions =====
                sink_start = tok_cursor
                sink_end = sink_start + x_sink_eff
                interior_start_b = sink_end
                interior_end_b = interior_start_b + in_batch_interior_len
                seg_end_b = interior_end_b
                tok_cursor = seg_end_b

                sink_row = None
                interior_row = None

                if is_first:
                    # sys: 1 row over full sys (cold-miss only — sys hit has 0 batch tokens)
                    if in_batch_len > 0:
                        interior_row = len(compute_seg_lengths)
                        compute_seg_lengths.append(in_batch_len)
                        compute_seg_indices.append(interior_row)
                        fla_seg_lengths_in_batch_order.append(in_batch_len)
                        # Persist full sys (S, T) to pool[sys_slot] for future hits.
                        if kind == "miss" and not is_last:
                            persist_slot = miss_slots_dict.get((s, e), None)
                            if persist_slot is not None:
                                compute_persist_src.append(interior_row)
                                compute_persist_dst.append(int(persist_slot))
                    entry_kind = "sys"
                elif is_last:
                    # query: 1 row over full query
                    if seg_len > 0:
                        interior_row = len(compute_seg_lengths)
                        compute_seg_lengths.append(seg_len)
                        compute_seg_indices.append(interior_row)
                        fla_seg_lengths_in_batch_order.append(seg_len)
                    entry_kind = "query"
                else:
                    # mid pic: emit sink / (interior if in batch)
                    if x_sink_eff > 0:
                        sink_row = len(compute_seg_lengths)
                        compute_seg_lengths.append(x_sink_eff)
                        compute_seg_indices.append(sink_row)
                        fla_seg_lengths_in_batch_order.append(x_sink_eff)
                    if in_batch_interior_len > 0:
                        interior_row = len(compute_seg_lengths)
                        compute_seg_lengths.append(in_batch_interior_len)
                        compute_seg_indices.append(interior_row)
                        fla_seg_lengths_in_batch_order.append(in_batch_interior_len)
                        # Persist interior (S, T) to pool[pic_slot] — interior ONLY.
                        if kind == "miss":
                            persist_slot = miss_slots_dict.get((s, e), None)
                            if persist_slot is not None:
                                compute_persist_src.append(interior_row)
                                compute_persist_dst.append(int(persist_slot))
                    entry_kind = "mid_hit" if kind == "hit" else "mid_miss"

                seg_order_entries.append({
                    "kind": entry_kind,
                    "mamba_slot": seg_mamba_slot,
                    "sink_row": sink_row,
                    "interior_row": interior_row,
                    "use_pool_interior": (kind == "hit" and interior_len > 0),
                })

                # ===== Phys_seg close: at this seg's "interior_end" boundary =====
                # close_at = end of sink + in-batch-interior (= sink_end + interior_len_in_batch).
                # For sys (no sink): close_at = sys_end.
                # For mid pic: close_at = interior_end_b (sink_end for hit since no interior in batch).
                # For query (last, miss): close_at = query_end.
                if is_first or is_last:
                    close_at = seg_end_b
                elif kind == "hit" and interior_len == 0:
                    close_at = seg_end_b
                else:
                    close_at = interior_end_b

                phys_seg_len = close_at - cur_phys_start
                if phys_seg_len > 0:
                    phys_idx = len(phys_seg_lengths)
                    phys_seg_lengths.append(phys_seg_len)
                    phys_seg_conv_slots.append(seg_mamba_slot)
                    phys_seg_prev_slots.append(cur_phys_prev_slot)
                    # Capture conv tail for non-last segs whose persist slot exists.
                    if not is_last:
                        persist_slot = None
                        if kind == "miss":
                            persist_slot = miss_slots_dict.get((s, e), None)
                        # mid-hit: no recapture (interior_tail already in conv_tails from cold).
                        if persist_slot is not None:
                            phys_persist_src.append(phys_idx)
                            phys_persist_dst.append(int(persist_slot))
                            phys_persist_end_offset.append(0)
                    cur_phys_prev_slot = seg_mamba_slot
                    cur_phys_start = close_at
                else:
                    # Empty phys_seg (e.g. sys hit with no batch tokens):
                    # still update prev_slot so next phys_seg loads from this seg.
                    cur_phys_prev_slot = seg_mamba_slot
                    # cur_phys_start unchanged (still at the same batch position)

            compose_plan.append({
                "seg_order": seg_order_entries,
                "req_dst_slot": req_dst_slot,
            })

        # ----- materialize tensors -----
        C = len(compute_seg_lengths)
        P = len(phys_seg_lengths)

        self._pic_recompute_C = C
        self._pic_compose_plan = compose_plan

        # phys_seg cu_seqlens
        if P > 0:
            phys_lens = torch.tensor(phys_seg_lengths, dtype=torch.int32, device=device)
            phys_cu = torch.zeros(P + 1, dtype=torch.int32, device=device)
            phys_cu[1:] = torch.cumsum(phys_lens, dim=0)
        else:
            phys_cu = torch.zeros(1, dtype=torch.int32, device=device)
        self._pic_phys_seg_cu_seqlens = phys_cu
        self._pic_phys_seg_lengths = phys_seg_lengths
        self._pic_phys_seg_conv_indices = torch.tensor(
            phys_seg_conv_slots, dtype=torch.int32, device=device,
        )
        self._pic_phys_seg_prev_tail_slot = torch.tensor(
            phys_seg_prev_slots, dtype=torch.int32, device=device,
        )
        self._pic_phys_seg_has_initial_states = torch.tensor(
            [s >= 0 for s in phys_seg_prev_slots], dtype=torch.bool, device=device,
        )
        self._pic_phys_persist_src = torch.tensor(
            phys_persist_src, dtype=torch.int32, device=device,
        )
        self._pic_phys_persist_dst = torch.tensor(
            phys_persist_dst, dtype=torch.int32, device=device,
        )
        self._pic_phys_persist_end_offset = torch.tensor(
            phys_persist_end_offset, dtype=torch.int32, device=device,
        )

        # Compute-row cu_seqlens. Used for cumsum / fwd_intra / Pass1 / Pass2 /
        # Pass3 — single scheme. compute_seg_lengths is already in batch order
        # (token positions per row are contiguous in mixed_qkv), so we don't
        # need an explicit compute_token_idx; kernel inputs are passed directly.
        if C > 0:
            compute_lens = torch.tensor(
                compute_seg_lengths, dtype=torch.int32, device=device,
            )
            compute_cu = torch.zeros(C + 1, dtype=torch.int32, device=device)
            compute_cu[1:] = torch.cumsum(compute_lens, dim=0)
        else:
            compute_cu = torch.zeros(1, dtype=torch.int32, device=device)
        self._pic_compute_cu_seqlens = compute_cu
        self._pic_compute_seg_indices = torch.tensor(
            compute_seg_indices, dtype=torch.int32, device=device,
        )
        self._pic_compute_persist_src = torch.tensor(
            compute_persist_src, dtype=torch.long, device=device,
        )
        self._pic_compute_persist_dst = torch.tensor(
            compute_persist_dst, dtype=torch.long, device=device,
        )

        # buffers
        C_safe = max(C, 1)
        self._pic_recompute_temp_states = _alloc(
            getattr(self, "_pic_recompute_temp_states", None),
            (C_safe, H_v, V, K), torch.float32,
        )
        # ponytail: fp32 (was qkv_dtype/bf16). bf16 T → compose loses ~e-3
        # per stage; ssm_final accumulated to 4e-3. fp32 brings ssm_final to
        # kernel-internal floor (~e-5).
        self._pic_recompute_temp_transitions = _alloc(
            getattr(self, "_pic_recompute_temp_transitions", None),
            (C_safe, H_v, K, K), torch.float32,
        )
        # ponytail: fp32 so Pass3's seeded h0 doesn't lose compose precision.
        self._pic_recompute_kernel_h0 = _alloc(
            getattr(self, "_pic_recompute_kernel_h0", None),
            (C_safe, H_v, V, K), torch.float32,
        )
        # u_zero is shared with the transition path's `_pic_u_zero_buf`
        # (shape (1, seq_len, H_v, V), allocated above in init_pic_metadata
        # under the transition_pool branch which recompute mode also takes).

    def forward_extend_pic_addition(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        """PIC lapic_addition for GDN linear-attn layers (optimized: metadata pre-computed)."""
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]
        device = mixed_qkv.device

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal

        pic_hit_mamba_slots = forward_batch.pic_hit_mamba_slots
        pic_miss_segments = forward_batch.pic_miss_segments
        batch_size = forward_batch.batch_size

        temp_states = self._pic_temp_states_buf
        temp_states.zero_()
        # The common single-miss case stays on the original one-pass path.
        # Multi-miss requests first compute each segment from zero so those
        # position-independent states can seed the additive prefix pass below.
        if not self._pic_has_multiple_misses:
            seg_cursor_a = 0
            for req_idx in range(batch_size):
                hit_slots = pic_hit_mamba_slots[req_idx] if pic_hit_mamba_slots else {}
                n_miss = len(pic_miss_segments[req_idx])
                if n_miss > 0 and hit_slots:
                    hit_slot_list = list(hit_slots.values())
                    idx_t = torch.tensor(
                        hit_slot_list, dtype=torch.long, device=device
                    )
                    temp_states[seg_cursor_a] = (
                        ssm_states.index_select(0, idx_t)
                        .sum(dim=0)
                        .to(temp_states.dtype)
                    )
                seg_cursor_a += n_miss

        # --- Conv1d (uses pre-computed metadata) ---
        conv_tails_per_layer = self.req_to_token_pool.mamba2_conv_tails_cache(
            layer.layer_id
        )
        conv_tails_layer = conv_tails_per_layer[0] if conv_tails_per_layer else None
        # Capture each persisted miss segment's last K-1 raw mixed_qkv tokens
        # BEFORE conv1d (the kernel mutates conv_states but the raw input is
        # what we need to cache).
        capture_conv_tails(
            mixed_qkv=mixed_qkv,
            seg_cu_seqlens=self._pic_seg_cu_seqlens,
            persist_src_idx=self._pic_fused_persist_src,
            persist_dst_slot=self._pic_fused_persist_dst,
            conv_tails_layer=conv_tails_layer,
        )
        # Load conv history into conv_states for any seg whose preceding seg
        # is known. has_initial_state[seg] was already set in init_pic_metadata
        # for exactly those segments.
        load_conv_history(
            conv_states=conv_states,
            conv_tails_layer=conv_tails_layer,
            seg_conv_indices=self._pic_seg_conv_indices,
            prev_tail_slots=self._pic_prev_tail_slot,
            has_initial_state=self._pic_has_initial_states,
        )
        mixed_qkv_t = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=self._pic_has_initial_states,
            cache_indices=self._pic_seg_conv_indices,
            query_start_loc=self._pic_seg_cu_seqlens,
            seq_lens_cpu=self._pic_seg_lengths,
        ).transpose(0, 1)[:seq_len]

        # --- Split + reshape ---
        query, key, value = torch.split(
            mixed_qkv_t, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1,
        )
        query = query.view(1, seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, seq_len, layer.num_v_heads, layer.head_v_dim)

        # --- Gating ---
        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

        # --- GDN kernel ---
        def _run_pic_segments(initial_states):
            return self.kernel_dispatcher.extend(
                q=query, k=key, v=value, g=g, beta=beta,
                ssm_states=initial_states,
                cache_indices=self._pic_seg_indices,
                query_start_loc=self._pic_seg_cu_seqlens,
            )

        core_attn_out, _last_recurrent_state, _h = _run_pic_segments(temp_states)

        # Persist zero-start miss states before the optional output pass mutates
        # its independent prefix-state workspace.
        if self._pic_trans_persist_src.numel() > 0:
            ssm_states[self._pic_trans_persist_dst] = temp_states[self._pic_trans_persist_src]

        if self._pic_has_multiple_misses:
            addition_h0 = build_addition_prefix_states(
                segment_states=temp_states,
                state_pool=ssm_states,
                hit_segments_per_request=forward_batch.pic_hit_segments,
                hit_slots_per_request=pic_hit_mamba_slots,
                miss_segments_per_request=pic_miss_segments,
                segment_offsets=self._pic_seg_offsets,
                out=self._pic_addition_h0_buf,
            )
            core_attn_out, _last_recurrent_state, _h = _run_pic_segments(
                addition_h0
            )
            final_states = addition_h0
        else:
            final_states = temp_states
        ssm_states[self._pic_req_dst_indices_long] = final_states[
            self._pic_req_last_miss_payload
        ]

        return core_attn_out

    def forward_extend_pic_transition(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        """PIC transition mode for GDN linear-attn layers.

        Key differences from addition:
        1. Inline FLA steps to get intermediate 'w' (needed for T computation).
        2. Compute per-segment transition matrices T_segment from (w, k, g_cumsum).
        3. Use transition_blend (right-multiply scan) instead of fused_state_gather_sum.
        4. Store T_i on req for later persistence to transition buffer.

        Env-gated diag dumps (PIC_DIAG_GDN) are emitted inline via the
        prefix-parameterized helpers in diag_layer_dump — each dump site is
        one line and the primitives short-circuit when the env is unset.
        """
        # Recompute mode dispatch: 3-pass interior/seam split (M4).
        if forward_batch.pic_policy.recompute:
            return self.forward_extend_pic_transition_recompute(
                layer, forward_batch, mixed_qkv, a, b, **kwargs,
            )

        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        transitions = mamba_cache_params.transition
        assert transitions is not None, (
            "PIC transition mode requires transition buffer in MambaPool. "
            "Ensure pic_mode is passed to HybridReqToTokenPool/MambaPool at init time."
        )

        # --- Conv1d (uses pre-computed metadata, same as addition) ---
        conv_tails_per_layer = self.req_to_token_pool.mamba2_conv_tails_cache(
            layer.layer_id
        )
        conv_tails_layer = conv_tails_per_layer[0] if conv_tails_per_layer else None
        capture_conv_tails(
            mixed_qkv=mixed_qkv,
            seg_cu_seqlens=self._pic_seg_cu_seqlens,
            persist_src_idx=self._pic_fused_persist_src,
            persist_dst_slot=self._pic_fused_persist_dst,
            conv_tails_layer=conv_tails_layer,
        )
        load_conv_history(
            conv_states=conv_states,
            conv_tails_layer=conv_tails_layer,
            seg_conv_indices=self._pic_seg_conv_indices,
            prev_tail_slots=self._pic_prev_tail_slot,
            has_initial_state=self._pic_has_initial_states,
        )
        _diag_dump.dump_conv_slots(layer.layer_id, "pic_trans_conv_pre", self._pic_seg_conv_indices, self._pic_has_initial_states, conv_states)
        mixed_qkv_t = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=self._pic_has_initial_states,
            cache_indices=self._pic_seg_conv_indices,
            query_start_loc=self._pic_seg_cu_seqlens,
            seq_lens_cpu=self._pic_seg_lengths,
        ).transpose(0, 1)[:seq_len]

        # --- Split + reshape ---
        query, key, value = torch.split(
            mixed_qkv_t, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1,
        )
        query = query.view(1, seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, seq_len, layer.num_v_heads, layer.head_v_dim)

        # --- Gating ---
        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        _diag_dump.dump_qkvgb(layer.layer_id, "pic", query[0], key[0], value[0], g[0], beta[0])

        # --- Inline Triton FLA (single pass: o + state + w) ---
        # PIC transition forces Triton backend. We inline the 3 FLA steps to get w
        # in the same pass as o and state (no extra fwd_intra call needed).
        q_normed = _fla_l2norm(query.contiguous())
        k_normed = _fla_l2norm(key.contiguous())
        v_contig = value.contiguous()
        g_contig = g.contiguous()
        beta_contig = beta.contiguous()
        scale = layer.head_k_dim ** -0.5

        g_cumsum = _fla_cumsum(g_contig, chunk_size=64,
                               cu_seqlens=self._pic_seg_cu_seqlens)
        w, u, _A = _fla_fwd_intra(
            k=k_normed, v=v_contig, g=g_cumsum, beta=beta_contig,
            cu_seqlens=self._pic_seg_cu_seqlens,
        )
        # A1: dropped the first _fla_fwd_o call here — its output was unconditionally
        # overwritten by the seeded re-run below.
        temp_states = self._pic_temp_states_buf
        temp_states.zero_()
        # ponytail: bf16 w (NOT w.float()). On triton 3.6.0 (v0.5.14) a full
        # fp32 tl.dot in chunk_delta_h's v-reconstruction produces garbage,
        # degenerating the composed state to token 0 ("!!!"). fp32 w was only a
        # ~2e-3 precision nicety on the old base's older triton; bf16 w keeps
        # S/T/seeded mutually consistent and coherent. See gdn_backend fp32-dot
        # note. Cost: ~2e-3 abs (same order as core_attn_out's ~5e-3 floor).
        _fla_fwd_h(
            k=k_normed, w=w, u=u, g=g_cumsum,
            initial_state=temp_states,
            initial_state_indices=self._pic_seg_indices,
            cu_seqlens=self._pic_seg_cu_seqlens,
        )
        # temp_states updated in-place by fwd_h (INPLACE_UPDATE=True)

        # --- Compute per-segment T = ∏_t α_t (I − β_t k_t k_t^T) ---
        # Reuse the (g_cumsum, w) we already computed above; only the extra
        # fwd_h pass with u=0 and initial_state=I is new work.
        # sglang convention is right-multiply on K axis (h_new = h_old @ T + S).
        # The kernel runs h0 @ ∏(I − β k kᵀ) when initial_state=I and u=0, so
        # the written-back buffer IS T_seg directly — NO transpose needed.
        temp_transitions = self._pic_temp_transitions_buf
        temp_transitions.zero_()
        temp_transitions.diagonal(dim1=-2, dim2=-1).fill_(1)
        # u_zero is allocated + zeroed once per batch and never written; safe to reuse.
        u_zero = self._pic_u_zero_buf
        # ponytail: bf16 w (NOT w.float()) — triton 3.6.0 fp32 tl.dot is broken.
        # S (Pass1) and T (Pass2) both use bf16 w so the compose formula
        # h@T+S stays mutually consistent with the seeded pass.
        _fla_fwd_h(
            k=k_normed, w=w, u=u_zero, g=g_cumsum,
            initial_state=temp_transitions,
            initial_state_indices=self._pic_seg_indices,
            cu_seqlens=self._pic_seg_cu_seqlens,
        )
        # temp_transitions now holds T_seg in-place (INPLACE_UPDATE=True).

        # --- Pre-persist miss segs' (S, T) to pool in one fancy-index scatter
        # per buffer (src/dst tensors are precomputed in init_pic_metadata).
        # After this, compose treats hit and miss uniformly (both read from
        # pool). Persist plan covers ALL miss segs incl. the last (question)
        # one — its slot is freed at request end without being indexed, so the
        # extra write is harmless.
        if self._pic_trans_persist_src.numel() > 0:
            ssm_states[self._pic_trans_persist_dst] = temp_states[self._pic_trans_persist_src]
            transitions[self._pic_trans_persist_dst] = (
                temp_transitions[self._pic_trans_persist_src].to(transitions.dtype)
            )

        # --- Compose: unified hit/miss read-from-pool.
        # PIC Pass3 invariant: every miss payload sees Pass3 h0 = the prefix
        # h_accum at the moment compose enters this segment. This matches the
        # `transition_rope_recompute` path (forward_extend_pic_transition_recompute)
        # and keeps `o` for every miss seg query consistent with the BASE forward.
        # `ssm_states[slot]` / `transitions[slot]` are unchanged (still
        # Pass1 h0=0 + Pass2 h0=I,u=0 outputs); only Pass3's per-payload h0
        # injection broadens.
        kernel_h0_buf = self._pic_kernel_h0_buf
        kernel_h0_buf.zero_()
        h_accum_buf = self._pic_h_accum_buf
        h_accum_buf.zero_()
        # transition_pool / ssm_states are fp32 (mamba_ssm_dtype=float32), so
        # compose reads them directly — no fp32 mirror needed.
        for req_idx, req_order in enumerate(self._pic_trans_compose_order):
            h_accum = h_accum_buf[req_idx]  # fp32 view, zeroed above
            for slot, compute_payload in req_order:
                if compute_payload is not None:
                    kernel_h0_buf[compute_payload] = h_accum.to(kernel_h0_buf.dtype)
                h_accum = torch.bmm(h_accum, transitions[slot]) + ssm_states[slot]
            h_accum_buf[req_idx] = h_accum

        # Batched final state write from compose chain.
        ssm_states[self._pic_req_dst_indices_long] = h_accum_buf.to(ssm_states.dtype)

        # Pass3: re-run prefill kernel seeded with composed kernel_h0_buf so
        # core_attn_out for miss tokens reflects the correct prefix history.
        # bf16 w (NOT w.float()) — triton 3.6.0 fp32 tl.dot is broken.
        _h2, v_new2 = _fla_fwd_h(
            k=k_normed, w=w, u=u, g=g_cumsum,
            initial_state=kernel_h0_buf,
            initial_state_indices=self._pic_seg_indices,
            cu_seqlens=self._pic_seg_cu_seqlens,
        )
        core_attn_out = _fla_fwd_o(
            q=q_normed, k=k_normed, v=v_new2, h=_h2, g=g_cumsum,
            scale=scale,
            cu_seqlens=self._pic_seg_cu_seqlens,
        )
        _diag_dump.dump_o(layer.layer_id, "pic", core_attn_out)
        _diag_dump.dump_ssm_finals(layer.layer_id, h_accum_buf, forward_batch.batch_size)
        return core_attn_out

    def forward_extend_pic_transition_recompute(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        """PIC transition_rope_recompute: unified per-compute-row 3-pass GDN.

        Per pic_segment in batch:
          - mid-miss / sys / query: ONE compute row covering the full seg
            tokens. Pass1 produces per-row full-seg
            S; Pass2 produces full-seg T; both go into compose.
          - hit: up to one compute row (sink). State source for compose
            comes from pool (cold-prefill full-seg S/T); the sink row
            are Pass3-only — re-run with seeded h0 to recover correct o at
            rerotated abs_pos. (Pass1 still runs on them; output is unused.)

        Pass3 is a SINGLE fwd_h + fwd_o over all compute tokens, seeded with
        per-row composed h0 in kernel_h0_buf. Tokens within each row see their
        true prefix via the kernel's internal state advance. Final ssm_state
        is h_accum_buf after composing query (Pass1 included query tokens, so
        h_accum naturally advances past query).
        """
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]
        device = mixed_qkv.device

        mamba = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba.conv[0]
        ssm_states = mamba.temporal
        transition_pool = mamba.transition
        assert transition_pool is not None, (
            "PIC transition_rope_recompute mode requires transition buffer in MambaPool."
        )

        C = self._pic_recompute_C

        # ----- Conv1d (phys_seg layout: one phys_seg per pic_segment) -----
        conv_tails_per_layer = self.req_to_token_pool.mamba2_conv_tails_cache(
            layer.layer_id
        )
        conv_tails_layer = (
            conv_tails_per_layer[0] if conv_tails_per_layer else None
        )
        capture_conv_tails(
            mixed_qkv=mixed_qkv,
            seg_cu_seqlens=self._pic_phys_seg_cu_seqlens,
            persist_src_idx=self._pic_phys_persist_src,
            persist_dst_slot=self._pic_phys_persist_dst,
            conv_tails_layer=conv_tails_layer,
            persist_src_end_offset=self._pic_phys_persist_end_offset,
        )
        load_conv_history(
            conv_states=conv_states,
            conv_tails_layer=conv_tails_layer,
            seg_conv_indices=self._pic_phys_seg_conv_indices,
            prev_tail_slots=self._pic_phys_seg_prev_tail_slot,
            has_initial_state=self._pic_phys_seg_has_initial_states,
        )
        mixed_qkv_t = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=self._pic_phys_seg_has_initial_states,
            cache_indices=self._pic_phys_seg_conv_indices,
            query_start_loc=self._pic_phys_seg_cu_seqlens,
            seq_lens_cpu=self._pic_phys_seg_lengths,
        ).transpose(0, 1)[:seq_len]

        # ----- Split + reshape + gating -----
        query, key, value = torch.split(
            mixed_qkv_t, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1,
        )
        query = query.view(1, seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, seq_len, layer.num_v_heads, layer.head_v_dim)
        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

        _abs_pos = getattr(self, "_pic_abs_pos_flat", None)
        _diag_dump.dump_qkvgb(layer.layer_id, "pic", query[0], key[0], value[0], g[0], beta[0], abs_pos=_abs_pos)

        q_normed = _fla_l2norm(query.contiguous())
        k_normed = _fla_l2norm(key.contiguous())
        v_contig = value.contiguous()
        g_contig = g.contiguous()
        beta_contig = beta.contiguous()
        scale = layer.head_k_dim ** -0.5

        # cumsum + fwd_intra reset at compute-row boundaries (each row is a
        # contiguous span in mixed_qkv; per-row g_cumsum stays self-contained).
        compute_cu = self._pic_compute_cu_seqlens
        compute_indices = self._pic_compute_seg_indices
        g_cumsum = _fla_cumsum(g_contig, chunk_size=64, cu_seqlens=compute_cu)
        w, u, _A = _fla_fwd_intra(
            k=k_normed, v=v_contig, g=g_cumsum, beta=beta_contig,
            cu_seqlens=compute_cu,
        )

        temp_states = self._pic_recompute_temp_states
        temp_transitions = self._pic_recompute_temp_transitions
        kernel_h0_buf = self._pic_recompute_kernel_h0
        h_accum_buf = self._pic_h_accum_buf

        if C > 0:
            # ----- Pass1: full-seg tokens per row, init=0 -> S per row -----
            temp_states.zero_()
            _fla_fwd_h(
                k=k_normed, w=w, u=u, g=g_cumsum,
                initial_state=temp_states,
                initial_state_indices=compute_indices,
                cu_seqlens=compute_cu,
            )

            # ----- Pass2: per row, init=I, u=0 -> T per row -----
            temp_transitions.zero_()
            temp_transitions.diagonal(dim1=-2, dim2=-1).fill_(1)
            u_zero = self._pic_u_zero_buf
            # bf16 w (NOT w.float()) — triton 3.6.0 fp32 tl.dot is broken.
            _fla_fwd_h(
                k=k_normed, w=w, u=u_zero, g=g_cumsum,
                initial_state=temp_transitions,
                initial_state_indices=compute_indices,
                cu_seqlens=compute_cu,
            )

            # ----- Persist mid-miss/sys (S, T) full-seg state to pool -----
            # Future cache hits read this as the cold-prefill-equivalent full
            # seg state (h0=0 over all seg tokens).
            if self._pic_compute_persist_src.numel() > 0:
                ssm_states[self._pic_compute_persist_dst] = (
                    temp_states[self._pic_compute_persist_src]
                )
                transition_pool[self._pic_compute_persist_dst] = (
                    temp_transitions[self._pic_compute_persist_src].to(
                        transition_pool.dtype
                    )
                )

        # ----- Compose: per-req loop, sink -> interior per mid pic -----
        # For each mini-seg row, kernel_h0[row] = h_accum entering that row
        # (Pass3 then re-runs with this h0 → correct o under real prefix).
        kernel_h0_buf.zero_()
        h_accum_buf.zero_()
        batch_size = forward_batch.batch_size

        def _advance_batch(h, row):
            """Stage h_accum through a batch compute row (snapshot h0, then
            advance via Pass1 (S, T) for this row)."""
            kernel_h0_buf[row] = h.to(kernel_h0_buf.dtype)
            return (
                torch.bmm(h, temp_transitions[row].float())
                + temp_states[row].float()
            )

        def _advance_pool(h, slot):
            """Stage h_accum through pool state (no Pass3 row — interior not
            in batch at hit replay). transition_pool / ssm_states are fp32
            (mamba_ssm_dtype=float32), read directly."""
            return torch.bmm(h, transition_pool[slot]) + ssm_states[slot]

        for req_idx, plan in enumerate(self._pic_compose_plan):
            h_accum = h_accum_buf[req_idx]  # fp32 view, zeroed above
            for entry in plan["seg_order"]:
                kind = entry["kind"]
                slot = entry["mamba_slot"]
                sink_row = entry["sink_row"]
                interior_row = entry["interior_row"]
                use_pool = entry["use_pool_interior"]

                if kind == "sys":
                    # sys: 1 stage. Cold = batch (interior_row); hit = pool.
                    if use_pool:
                        h_accum = _advance_pool(h_accum, slot)
                    elif interior_row is not None:
                        h_accum = _advance_batch(h_accum, interior_row)
                elif kind == "query":
                    # query: 1 stage from batch (always miss).
                    if interior_row is not None:
                        h_accum = _advance_batch(h_accum, interior_row)
                else:
                    # mid pic (hit or miss): 3 stages.
                    # Stage 1: sink (batch). Skip if sink absent (x_sink=0).
                    if sink_row is not None:
                        h_accum = _advance_batch(h_accum, sink_row)
                    # Stage 2: interior (pool for hit, batch for miss).
                    if use_pool:
                        h_accum = _advance_pool(h_accum, slot)
                    elif interior_row is not None:
                        h_accum = _advance_batch(h_accum, interior_row)
            h_accum_buf[req_idx] = h_accum

        # ----- Final state -> ssm_states[req_dst] -----
        # Pass1 included query tokens, so compose's h_accum already advanced
        # past query. No post-Pass3 query_row extraction needed.
        ssm_states[self._pic_req_dst_indices_long] = h_accum_buf.to(ssm_states.dtype)

        # ----- Pass3: single fwd_h + fwd_o over all compute tokens with seeded h0 -----
        if C > 0:
            _h_p3, v_new_p3 = _fla_fwd_h(
                k=k_normed, w=w, u=u, g=g_cumsum,
                initial_state=kernel_h0_buf,
                initial_state_indices=compute_indices,
                cu_seqlens=compute_cu,
            )
            core_attn_out = _fla_fwd_o(
                q=q_normed, k=k_normed, v=v_new_p3, h=_h_p3, g=g_cumsum,
                scale=scale,
                cu_seqlens=compute_cu,
            )
        else:
            core_attn_out = torch.empty(
                (1, seq_len, layer.num_v_heads, layer.head_v_dim),
                dtype=q_normed.dtype, device=device,
            )

        _diag_dump.dump_o(layer.layer_id, "pic", core_attn_out, abs_pos=getattr(self, "_pic_abs_pos_flat", None))
        _diag_dump.dump_ssm_finals(layer.layer_id, h_accum_buf, batch_size)

        return core_attn_out
