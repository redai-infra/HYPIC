from typing import List, Optional, Tuple, Union

import torch
from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_intra import chunk_kda_fwd_intra
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.kda import chunk_gla_fwd_o_gk, chunk_kda
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
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
from sglang.srt.pic.conv_tails import (
    build_prev_tail_slots,
    capture_conv_tails,
    load_conv_history,
)
from sglang.srt.pic.policy import PICCompose
from sglang.srt.pic.state_composition import build_addition_prefix_states
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

# KDA always uses the triton causal_conv1d_fn (no CUDA override).
# Only causal_conv1d_update needs platform-specific overrides for decode.
if is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_update_npu

    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_update_cpu

    causal_conv1d_update = causal_conv1d_update_cpu

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class KDAKernelDispatcher:
    """Dispatches KDA kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonKDAKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("KDA CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
                CuteDSLKDAKernel,
            )

            self.decode_kernel = CuteDSLKDAKernel()
        else:
            raise ValueError(
                f"Unsupported KDA decode backend: {decode_backend}. "
                "KDA currently only supports 'triton'."
            )

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("KDA CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
                CuteDSLKDAKernel,
            )

            cutedsl_kernel = CuteDSLKDAKernel()
            if getattr(cutedsl_kernel, "supports_prefill", False):
                # SM100 chunk prefill pipeline.
                self.extend_kernel = cutedsl_kernel
            else:
                # CuTe DSL prefill kernels need SM100 (Blackwell); on older GPUs
                # fall back to the Triton chunk kernel.
                self.extend_kernel = triton_kernel
                rank0_log(
                    "KDA cutedsl prefill needs SM100; falling back to Triton extend."
                )
        else:
            raise ValueError(
                f"Unsupported KDA prefill backend: {prefill_backend}. "
                "KDA supports 'triton' or 'cutedsl' (cutedsl prefill needs SM100)."
            )

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"KDA kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__} "
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
        """Attempt packed decode. Returns output tensor or None if the decode
        kernel does not support packed decode."""
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
    ) -> torch.Tensor:
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


class KDAAttnBackend(MambaAttnBackendBase):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = KDAKernelDispatcher(decode_backend, prefill_backend)

    def forward_decode(
        self,
        layer: RadixLinearAttention,
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

        qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(-1, -2),
            layer.conv_weights,
            layer.bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        # Skip split + reshape by consuming the packed mixed_qkv directly in a
        # single fused Triton kernel (KDA per-K gate variant of GDN PR #20627).
        #
        # The packed kernel hard-assumes one token per sequence (T=1): it has no
        # query_start_loc / per-sequence loop. forward_decode is only entered in
        # decode mode (see HybridLinearAttnBackend.forward dispatch), where each
        # request contributes exactly one token, so #tokens == #requests. Multi-
        # token-per-seq speculative paths (target_verify / draft_extend) go
        # through forward_extend instead. Assert the invariant so a future
        # routing change fails loudly rather than silently corrupting state.
        if self.kernel_dispatcher.supports_packed_decode:
            assert qkv.shape[0] == cache_indices.shape[0], (
                "KDA packed decode requires one token per sequence (T=1): "
                f"got {qkv.shape[0]} tokens for {cache_indices.shape[0]} requests."
            )
            return self.kernel_dispatcher.packed_decode(
                mixed_qkv=qkv,
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

        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        return self.kernel_dispatcher.decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0].transpose(-1, -2)

        ssm_states = mamba_cache_params.temporal

        has_initial_state = forward_batch.extend_prefix_lens > 0

        splits = [layer.q_dim, layer.k_dim, layer.v_dim]
        q, k, v = mixed_qkv.transpose(0, 1).split(splits, dim=0)
        q_conv_weight, k_conv_weight, v_conv_weight = layer.conv_weights.split(
            splits, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_states.split(splits, dim=-2)
        if layer.bias is not None:
            q_bias, k_bias, v_bias = layer.bias.split(splits, dim=0)
        else:
            q_bias, k_bias, v_bias = None, None, None

        q = causal_conv1d_fn(
            q,
            q_conv_weight,
            q_bias,
            activation="silu",
            conv_states=q_conv_state,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)
        k = causal_conv1d_fn(
            k,
            k_conv_weight,
            k_bias,
            activation="silu",
            conv_states=k_conv_state,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)
        v = causal_conv1d_fn(
            v,
            v_conv_weight,
            v_bias,
            activation="silu",
            conv_states=v_conv_state,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        core_attn_out = self.kernel_dispatcher.extend(
            q=q,
            k=k,
            v=v,
            g=a,
            beta=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=getattr(layer, "lower_bound", None),
        )

        return core_attn_out

    # ------------------------------------------------------------------
    # PIC support
    # ------------------------------------------------------------------
    def init_pic_metadata(self, forward_batch: "ForwardBatch"):
        """Pre-compute PIC segment metadata for KDA layers."""
        device = forward_batch.input_ids.device
        pic_miss_segments = forward_batch.pic_miss_segments
        batch_size = forward_batch.batch_size
        req_cache_indices = self.forward_metadata.mamba_cache_indices
        is_addition = forward_batch.pic_policy.compose is PICCompose.ADDITION

        seg_lengths: List[int] = []
        seg_req_idx: List[int] = []

        for req_idx in range(batch_size):
            prev_start = -1
            for (start, end) in pic_miss_segments[req_idx]:
                assert start > prev_start, (
                    f"pic_miss_segments[{req_idx}] not start-monotonic: "
                    f"{pic_miss_segments[req_idx]}"
                )
                prev_start = start
                seg_lengths.append(end - start)
                seg_req_idx.append(req_idx)

        num_segments = len(seg_lengths)

        pic_hit_mamba_slots = forward_batch.pic_hit_mamba_slots
        pic_miss_mamba_slots = forward_batch.pic_miss_mamba_slots
        seg_offsets: list = [0]
        dst_indices_list: list = []
        persist_src_list: list = []
        persist_dst_list: list = []
        trans_persist_src_list: list = []
        trans_persist_dst_list: list = []
        seg_cursor = 0
        seg_conv_slot_list: list = []
        for req_idx in range(batch_size):
            miss_segs = pic_miss_segments[req_idx]
            req_seg_count = len(miss_segs)
            seg_offsets.append(seg_cursor + req_seg_count)

            miss_slots = pic_miss_mamba_slots[req_idx] if pic_miss_mamba_slots else {}
            req_dst_slot = req_cache_indices[req_idx].item()
            for local_i, (start, end) in enumerate(miss_segs):
                persist_slot_any = miss_slots.get((start, end))
                if persist_slot_any is not None:
                    trans_persist_src_list.append(seg_cursor + local_i)
                    trans_persist_dst_list.append(persist_slot_any)
                if local_i < req_seg_count - 1:
                    persist_slot = persist_slot_any
                    if persist_slot is not None:
                        persist_src_list.append(seg_cursor + local_i)
                        persist_dst_list.append(persist_slot)
                    seg_conv_slot_list.append(
                        persist_slot if persist_slot is not None else req_dst_slot
                    )
                else:
                    seg_conv_slot_list.append(req_dst_slot)
            seg_cursor += req_seg_count
            dst_indices_list.append(req_dst_slot)

        self._pic_fused_persist_src = torch.tensor(persist_src_list, dtype=torch.int32, device=device)
        self._pic_fused_persist_dst = torch.tensor(persist_dst_list, dtype=torch.int32, device=device)
        self._pic_trans_persist_src = torch.tensor(trans_persist_src_list, dtype=torch.long, device=device)
        self._pic_trans_persist_dst = torch.tensor(trans_persist_dst_list, dtype=torch.long, device=device)
        self._pic_req_last_miss_payload = (
            torch.tensor(seg_offsets[1:], dtype=torch.long, device=device) - 1
        )
        self._pic_seg_offsets = seg_offsets
        self._pic_has_multiple_misses = any(
            end - start > 1
            for start, end in zip(seg_offsets, seg_offsets[1:])
        )
        self._pic_req_dst_indices_long = torch.tensor(
            dst_indices_list, dtype=torch.long, device=device,
        )

        seg_lens_tensor = torch.tensor(seg_lengths, dtype=torch.int32, device=device)
        seg_cu_seqlens = torch.zeros(num_segments + 1, dtype=torch.int32, device=device)
        seg_cu_seqlens[1:] = torch.cumsum(seg_lens_tensor, dim=0)

        self._pic_seg_cu_seqlens = seg_cu_seqlens
        self._pic_seg_indices = torch.arange(num_segments, dtype=torch.int32, device=device)
        self._pic_seg_conv_indices = torch.tensor(
            seg_conv_slot_list, dtype=torch.int32, device=device,
        )
        self._pic_seg_lengths = seg_lengths

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

        # Allocate per-batch KDA workspaces
        layer0_cache = self.req_to_token_pool.mamba2_layer_cache(0)
        ssm0 = layer0_cache.temporal  # (N_slots, H, D, D)
        _, H, D_k, D_v = ssm0.shape
        state_shape = (num_segments, H, D_k, D_v)
        ssm_dtype = ssm0.dtype

        def _alloc(buf, shape, dtype):
            if (buf is None or buf.shape != shape
                    or buf.dtype != dtype or buf.device != device):
                return torch.empty(shape, dtype=dtype, device=device)
            return buf

        self._pic_temp_states_buf = _alloc(
            getattr(self, "_pic_temp_states_buf", None), state_shape, ssm_dtype,
        )
        if is_addition and self._pic_has_multiple_misses:
            self._pic_addition_h0_buf = _alloc(
                getattr(self, "_pic_addition_h0_buf", None),
                state_shape,
                ssm_dtype,
            )
            self._pic_addition_suffix_buf = _alloc(
                getattr(self, "_pic_addition_suffix_buf", None),
                (batch_size, H, D_k, D_v),
                ssm_dtype,
            )

    def forward_extend_pic_addition(
        self,
        layer: RadixLinearAttention,
        forward_batch: "ForwardBatch",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        """PIC addition for KDA linear-attn layers."""
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]
        device = mixed_qkv.device

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states_raw = mamba_cache_params.conv[0]  # (N+1, K-1, D)
        ssm_states = mamba_cache_params.temporal  # (N+1, H, D, D)

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

        # --- Conv1d (KDA: split q/k/v, 3 separate conv calls) ---
        conv_tails_per_layer = self.req_to_token_pool.mamba2_conv_tails_cache(
            layer.layer_id
        )
        conv_tails_layer = conv_tails_per_layer[0] if conv_tails_per_layer else None

        # Capture conv tails BEFORE conv1d mutates state
        capture_conv_tails(
            mixed_qkv=mixed_qkv,
            seg_cu_seqlens=self._pic_seg_cu_seqlens,
            persist_src_idx=self._pic_fused_persist_src,
            persist_dst_slot=self._pic_fused_persist_dst,
            conv_tails_layer=conv_tails_layer,
            layout="kda",
        )

        # Transpose conv_states to (N+1, D, K-1) for causal_conv1d_fn
        conv_states = conv_states_raw.transpose(-1, -2)

        # Load conv history from preceding segments
        load_conv_history(
            conv_states=conv_states,
            conv_tails_layer=conv_tails_layer,
            seg_conv_indices=self._pic_seg_conv_indices,
            prev_tail_slots=self._pic_prev_tail_slot,
            has_initial_state=self._pic_has_initial_states,
            layout="kda",
        )

        # Split mixed_qkv and conv weights for q/k/v
        splits = [layer.q_dim, layer.k_dim, layer.v_dim]
        mixed_qkv_t = mixed_qkv.transpose(0, 1)  # (D, seq) for conv1d
        q_raw, k_raw, v_raw = mixed_qkv_t.split(splits, dim=0)
        q_conv_weight, k_conv_weight, v_conv_weight = layer.conv_weights.split(
            splits, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_states.split(splits, dim=-2)
        if layer.bias is not None:
            q_bias, k_bias, v_bias = layer.bias.split(splits, dim=0)
        else:
            q_bias, k_bias, v_bias = None, None, None

        q = causal_conv1d_fn(
            q_raw, q_conv_weight, q_bias,
            activation="silu",
            conv_states=q_conv_state,
            has_initial_state=self._pic_has_initial_states,
            cache_indices=self._pic_seg_conv_indices,
            query_start_loc=self._pic_seg_cu_seqlens,
            seq_lens_cpu=self._pic_seg_lengths,
        ).transpose(0, 1)[:seq_len]
        k = causal_conv1d_fn(
            k_raw, k_conv_weight, k_bias,
            activation="silu",
            conv_states=k_conv_state,
            has_initial_state=self._pic_has_initial_states,
            cache_indices=self._pic_seg_conv_indices,
            query_start_loc=self._pic_seg_cu_seqlens,
            seq_lens_cpu=self._pic_seg_lengths,
        ).transpose(0, 1)[:seq_len]
        v = causal_conv1d_fn(
            v_raw, v_conv_weight, v_bias,
            activation="silu",
            conv_states=v_conv_state,
            has_initial_state=self._pic_has_initial_states,
            cache_indices=self._pic_seg_conv_indices,
            query_start_loc=self._pic_seg_cu_seqlens,
            seq_lens_cpu=self._pic_seg_lengths,
        ).transpose(0, 1)[:seq_len]

        # Reshape to (1, seq_len, H, D) for chunk_kda
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)

        # --- KDA kernel (g=a already pre-gated, beta=b already sigmoid'd) ---
        def _run_pic_segments(initial_states):
            return chunk_kda(
                q=q, k=k, v=v,
                g=a, beta=b,
                initial_state=initial_states,
                initial_state_indices=self._pic_seg_indices,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=self._pic_seg_cu_seqlens,
            )

        core_attn_out = _run_pic_segments(temp_states)

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
                request_suffix_states=self._pic_addition_suffix_buf,
            )
            core_attn_out = _run_pic_segments(addition_h0)
            final_states = addition_h0
        else:
            final_states = temp_states
        final_states = final_states[self._pic_req_last_miss_payload]
        if self._pic_has_multiple_misses:
            final_states = final_states + self._pic_addition_suffix_buf
        ssm_states[self._pic_req_dst_indices_long] = final_states

        return core_attn_out

    def forward_extend_pic_transition(
        self,
        layer: RadixLinearAttention,
        forward_batch: "ForwardBatch",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        """PIC transition mode for KDA linear-attn layers.

        Steps:
        1. Conv1d (same as addition)
        2. Inline KDA FLA steps to get w, u, kg, Aqk
        3. fwd_h with initial_state=0 → per-seg S
        4. fwd_h with initial_state=I, u=0 → per-seg T
        5. Persist S, T; compose h_accum = h_accum @ T + S
        6. Pass3: fwd_h + fwd_o with composed h0 → final attention output
        """
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]
        device = mixed_qkv.device

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states_raw = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        transition_pool = mamba_cache_params.transition
        assert transition_pool is not None, (
            "PIC transition mode requires transition buffer in MambaPool."
        )

        pic_hit_mamba_slots = forward_batch.pic_hit_mamba_slots
        pic_miss_mamba_slots = forward_batch.pic_miss_mamba_slots
        pic_miss_segments = forward_batch.pic_miss_segments
        pic_hit_segments = forward_batch.pic_hit_segments
        req_cache_indices = self.forward_metadata.mamba_cache_indices
        batch_size = forward_batch.batch_size

        # --- Conv1d (same as addition) ---
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
            layout="kda",
        )
        conv_states = conv_states_raw.transpose(-1, -2)
        load_conv_history(
            conv_states=conv_states,
            conv_tails_layer=conv_tails_layer,
            seg_conv_indices=self._pic_seg_conv_indices,
            prev_tail_slots=self._pic_prev_tail_slot,
            has_initial_state=self._pic_has_initial_states,
            layout="kda",
        )

        splits = [layer.q_dim, layer.k_dim, layer.v_dim]
        mixed_qkv_t = mixed_qkv.transpose(0, 1)
        q_raw, k_raw, v_raw = mixed_qkv_t.split(splits, dim=0)
        q_conv_weight, k_conv_weight, v_conv_weight = layer.conv_weights.split(
            splits, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_states.split(splits, dim=-2)
        if layer.bias is not None:
            q_bias, k_bias, v_bias = layer.bias.split(splits, dim=0)
        else:
            q_bias, k_bias, v_bias = None, None, None

        q = causal_conv1d_fn(
            q_raw, q_conv_weight, q_bias, activation="silu",
            conv_states=q_conv_state,
            has_initial_state=self._pic_has_initial_states,
            cache_indices=self._pic_seg_conv_indices,
            query_start_loc=self._pic_seg_cu_seqlens,
            seq_lens_cpu=self._pic_seg_lengths,
        ).transpose(0, 1)[:seq_len]
        k = causal_conv1d_fn(
            k_raw, k_conv_weight, k_bias, activation="silu",
            conv_states=k_conv_state,
            has_initial_state=self._pic_has_initial_states,
            cache_indices=self._pic_seg_conv_indices,
            query_start_loc=self._pic_seg_cu_seqlens,
            seq_lens_cpu=self._pic_seg_lengths,
        ).transpose(0, 1)[:seq_len]
        v = causal_conv1d_fn(
            v_raw, v_conv_weight, v_bias, activation="silu",
            conv_states=v_conv_state,
            has_initial_state=self._pic_has_initial_states,
            cache_indices=self._pic_seg_conv_indices,
            query_start_loc=self._pic_seg_cu_seqlens,
            seq_lens_cpu=self._pic_seg_lengths,
        ).transpose(0, 1)[:seq_len]

        # Reshape to (1, seq_len, H, D)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)

        # --- Inline KDA FLA steps ---
        # KDA uses l2norm on q/k before the kernel
        q_normed = l2norm_fwd(q.contiguous())
        k_normed = l2norm_fwd(k.contiguous())
        v_contig = v.contiguous()
        # a = forget_gate (already fused_kda_gate'd), b = beta (already sigmoid'd)
        g_contig = a.contiguous()
        beta_contig = b.contiguous()
        scale = layer.head_k_dim ** -0.5

        g_cumsum = chunk_local_cumsum(
            g_contig, chunk_size=64, cu_seqlens=self._pic_seg_cu_seqlens
        )
        w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
            q=q_normed, k=k_normed, v=v_contig,
            gk=g_cumsum, beta=beta_contig, scale=scale,
            cu_seqlens=self._pic_seg_cu_seqlens, chunk_size=64,
        )

        # --- Step 2: fwd_h with initial_state=0 → per-seg state S ---
        temp_states = self._pic_temp_states_buf
        temp_states.zero_()
        chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=g_cumsum,
            initial_state=temp_states,
            initial_state_indices=self._pic_seg_indices,
            cu_seqlens=self._pic_seg_cu_seqlens,
        )

        # --- Step 3: T computation (h0=I, u=0) ---
        _, H_dim, D_k, D_v = ssm_states.shape
        temp_transitions = getattr(self, "_pic_temp_transitions_buf", None)
        trans_shape = (num_segments, H_dim, D_v, D_k)
        if (temp_transitions is None or temp_transitions.shape != trans_shape
                or temp_transitions.device != device):
            temp_transitions = torch.empty(trans_shape, dtype=ssm_states.dtype, device=device)
            self._pic_temp_transitions_buf = temp_transitions
        temp_transitions.zero_()
        temp_transitions.diagonal(dim1=-2, dim2=-1).fill_(1)

        u_zero_shape = (1, seq_len, H_dim, D_v)
        u_zero = getattr(self, "_pic_u_zero_buf", None)
        if (u_zero is None or u_zero.shape != u_zero_shape
                or u_zero.device != device):
            u_zero = torch.zeros(u_zero_shape, dtype=v_contig.dtype, device=device)
            self._pic_u_zero_buf = u_zero

        chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u_zero, gk=g_cumsum,
            initial_state=temp_transitions,
            initial_state_indices=self._pic_seg_indices,
            cu_seqlens=self._pic_seg_cu_seqlens,
        )

        # --- Step 4: Persist S, T to pool ---
        if self._pic_trans_persist_src.numel() > 0:
            ssm_states[self._pic_trans_persist_dst] = temp_states[self._pic_trans_persist_src]
            transition_pool[self._pic_trans_persist_dst] = (
                temp_transitions[self._pic_trans_persist_src].to(transition_pool.dtype)
            )

        # --- Step 5: Compose h_accum = h_accum @ T + S ---
        kernel_h0_shape = (num_segments, H_dim, D_k, D_v)
        kernel_h0_buf = getattr(self, "_pic_kernel_h0_buf", None)
        if (kernel_h0_buf is None or kernel_h0_buf.shape != kernel_h0_shape
                or kernel_h0_buf.device != device):
            kernel_h0_buf = torch.empty(kernel_h0_shape, dtype=ssm_states.dtype, device=device)
            self._pic_kernel_h0_buf = kernel_h0_buf
        kernel_h0_buf.zero_()

        h_accum_shape = (batch_size, H_dim, D_k, D_v)
        h_accum_buf = getattr(self, "_pic_h_accum_buf", None)
        if (h_accum_buf is None or h_accum_buf.shape != h_accum_shape
                or h_accum_buf.device != device):
            h_accum_buf = torch.empty(h_accum_shape, dtype=torch.float32, device=device)
            self._pic_h_accum_buf = h_accum_buf

        seg_cursor = 0
        for req_idx in range(batch_size):
            hit_slots_dict = (
                pic_hit_mamba_slots[req_idx] if pic_hit_mamba_slots else {}
            )
            hit_segs = pic_hit_segments[req_idx] if pic_hit_segments else []
            miss_segs = pic_miss_segments[req_idx]
            miss_slots_dict = (
                pic_miss_mamba_slots[req_idx] if pic_miss_mamba_slots else {}
            )
            last_miss_payload = (
                seg_cursor + len(miss_segs) - 1 if miss_segs else -1
            )

            req_order = []
            for (s, e, h) in hit_segs:
                req_order.append((s, hit_slots_dict[h], None))
            for local_i, (s, e) in enumerate(miss_segs):
                req_order.append(
                    (s, miss_slots_dict[(s, e)], seg_cursor + local_i)
                )
            req_order.sort(key=lambda x: x[0])

            h_accum = torch.zeros(
                H_dim, D_k, D_v, dtype=torch.float32, device=device,
            )
            for _, slot, miss_payload in req_order:
                if miss_payload == last_miss_payload:
                    kernel_h0_buf[miss_payload] = h_accum.to(kernel_h0_buf.dtype)
                T = transition_pool[slot].float()
                S = ssm_states[slot].float()
                h_accum = torch.bmm(h_accum, T) + S
            h_accum_buf[req_idx] = h_accum
            seg_cursor += len(miss_segs)

        ssm_states[self._pic_req_dst_indices_long] = h_accum_buf.to(ssm_states.dtype)

        # --- Step 6: Pass3 — re-run with composed h0 ---
        _h2, v_new2 = chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=g_cumsum,
            initial_state=kernel_h0_buf,
            initial_state_indices=self._pic_seg_indices,
            cu_seqlens=self._pic_seg_cu_seqlens,
        )
        core_attn_out = chunk_gla_fwd_o_gk(
            q=q_normed, v=v_new2, g=g_cumsum, A=Aqk, h=_h2, o=v_contig,
            scale=scale,
            cu_seqlens=self._pic_seg_cu_seqlens, chunk_size=64,
        )
        return core_attn_out
