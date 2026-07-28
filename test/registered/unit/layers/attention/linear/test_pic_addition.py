import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.layers.attention.linear import gdn_backend, kda_backend
from sglang.srt.pic.policy import POLICIES
from sglang.srt.pic.state_composition import build_addition_prefix_states
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _identity_conv(x, *_args, **_kwargs):
    return x


class _FakeGDNKernelDispatcher:
    def __init__(self):
        self.call_count = 0

    def extend(
        self,
        *,
        q,
        v,
        ssm_states,
        cache_indices,
        query_start_loc,
        **_kwargs,
    ):
        self.call_count += 1
        output = torch.empty_like(v)
        for segment_idx, state_idx in enumerate(cache_indices.tolist()):
            start = query_start_loc[segment_idx].item()
            end = query_start_loc[segment_idx + 1].item()
            state = ssm_states[state_idx]
            for token_idx in range(start, end):
                state = 0.5 * state + v[:, token_idx]
                output[:, token_idx] = state
            ssm_states[state_idx] = state
        return output, None, None


def _fake_chunk_kda(
    *,
    v,
    initial_state,
    initial_state_indices,
    cu_seqlens,
    **_kwargs,
):
    output = torch.empty_like(v)
    for segment_idx, state_idx in enumerate(initial_state_indices.tolist()):
        start = cu_seqlens[segment_idx].item()
        end = cu_seqlens[segment_idx + 1].item()
        state = initial_state[state_idx]
        for token_idx in range(start, end):
            state = 0.5 * state + v[:, token_idx]
            output[:, token_idx] = state
        initial_state[state_idx] = state
    return output


class _FakeReqToTokenPool:
    def __init__(self):
        self.ssm_states = torch.zeros(5, 1, 1, 1)
        self.ssm_states[0] = 10
        self.ssm_states[1] = 20
        self.conv_states = torch.zeros(5, 2, 3)

    def mamba2_layer_cache(self, _layer_id):
        return SimpleNamespace(
            conv=[self.conv_states],
            temporal=self.ssm_states,
        )

    def mamba2_conv_tails_cache(self, _layer_id):
        return None


class TestPICAddition(CustomTestCase):
    def test_kda_allocates_prefix_workspace_only_for_addition(self):
        def make_backend():
            backend = object.__new__(kda_backend.KDAAttnBackend)
            backend.req_to_token_pool = _FakeReqToTokenPool()
            backend.forward_metadata = SimpleNamespace(
                mamba_cache_indices=torch.tensor([4], dtype=torch.int32)
            )
            return backend

        def make_batch(policy):
            return SimpleNamespace(
                input_ids=torch.tensor([1, 2]),
                batch_size=1,
                pic_policy=policy,
                pic_hit_segments=[[(0, 1, "hit")]],
                pic_hit_mamba_slots=[{"hit": 0}],
                pic_miss_segments=[[(1, 2), (2, 3)]],
                pic_miss_mamba_slots=[{(1, 2): 2, (2, 3): 3}],
            )

        transition_backend = make_backend()
        h0_sentinel = torch.empty(1)
        suffix_sentinel = torch.empty(1)
        transition_backend._pic_addition_h0_buf = h0_sentinel
        transition_backend._pic_addition_suffix_buf = suffix_sentinel
        transition_backend.init_pic_metadata(make_batch(POLICIES["transition"]))
        self.assertIs(transition_backend._pic_addition_h0_buf, h0_sentinel)
        self.assertIs(transition_backend._pic_addition_suffix_buf, suffix_sentinel)

        addition_backend = make_backend()
        addition_backend.init_pic_metadata(make_batch(POLICIES["addition"]))
        self.assertEqual(
            addition_backend._pic_addition_h0_buf.shape,
            (2, 1, 1, 1),
        )
        self.assertEqual(
            addition_backend._pic_addition_suffix_buf.shape,
            (1, 1, 1, 1),
        )

    def test_addition_prefix_states_handle_mixed_request_shapes(self):
        out = torch.empty(3, 1, 1, 1)
        suffix = torch.empty(2, 1, 1, 1)
        build_addition_prefix_states(
            segment_states=torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1, 1),
            state_pool=torch.tensor([10.0, 20.0, 30.0]).view(3, 1, 1, 1),
            hit_segments_per_request=[
                [(1, 2, "interleaved"), (3, 4, "suffix")],
                [(0, 1, "prefix")],
            ],
            hit_slots_per_request=[
                {"interleaved": 0, "suffix": 2},
                {"prefix": 1},
            ],
            miss_segments_per_request=[
                [(0, 1), (2, 3)],
                [(1, 2)],
            ],
            segment_offsets=[0, 2, 3],
            out=out,
            request_suffix_states=suffix,
        )

        torch.testing.assert_close(
            out.flatten(),
            torch.tensor([0.0, 11.0, 20.0]),
        )
        torch.testing.assert_close(
            suffix.flatten(),
            torch.tensor([30.0, 0.0]),
        )

    def _make_backend(self, backend_type):
        backend = object.__new__(backend_type)
        backend.req_to_token_pool = _FakeReqToTokenPool()
        backend.forward_metadata = SimpleNamespace(
            mamba_cache_indices=torch.tensor([4], dtype=torch.int32)
        )
        backend._pic_temp_states_buf = torch.empty(2, 1, 1, 1)
        backend._pic_seg_indices = torch.tensor([0, 1], dtype=torch.int32)
        backend._pic_seg_cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int32)
        backend._pic_seg_lengths = [1, 1]
        backend._pic_seg_conv_indices = torch.tensor([1, 3], dtype=torch.int32)
        backend._pic_has_initial_states = torch.tensor([False, False])
        backend._pic_prev_tail_slot = torch.tensor([-1, -1], dtype=torch.int32)
        backend._pic_fused_persist_src = torch.empty(0, dtype=torch.int32)
        backend._pic_fused_persist_dst = torch.empty(0, dtype=torch.int32)
        backend._pic_trans_persist_src = torch.tensor([0, 1], dtype=torch.long)
        backend._pic_trans_persist_dst = torch.tensor([2, 3], dtype=torch.long)
        backend._pic_req_dst_indices_long = torch.tensor([4], dtype=torch.long)
        backend._pic_req_last_miss_payload = torch.tensor([1], dtype=torch.long)
        backend._pic_seg_offsets = [0, 2]
        backend._pic_has_multiple_misses = True
        backend._pic_addition_h0_buf = torch.empty(2, 1, 1, 1)
        backend._pic_addition_suffix_buf = torch.empty(1, 1, 1, 1)
        return backend

    def _make_forward_batch(self):
        return SimpleNamespace(
            batch_size=1,
            pic_hit_mamba_slots=[{"hit": 0, "suffix": 1}],
            pic_hit_segments=[[(1, 2, "hit"), (3, 4, "suffix")]],
            pic_miss_mamba_slots=[{(0, 1): 2, (2, 3): 3}],
            pic_miss_segments=[[(0, 1), (2, 3)]],
        )

    def _assert_composed_result(self, output, states):
        torch.testing.assert_close(
            output.flatten(),
            torch.tensor([1.0, 7.5]),
        )
        torch.testing.assert_close(
            states.flatten(),
            torch.tensor([10.0, 20.0, 1.0, 2.0, 27.5]),
        )

    def test_gdn_addition_carries_history_across_multiple_misses(self):
        backend = self._make_backend(gdn_backend.GDNAttnBackend)
        backend.kernel_dispatcher = _FakeGDNKernelDispatcher()
        layer = SimpleNamespace(
            layer_id=0,
            q_dim=1,
            k_dim=1,
            v_dim=1,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=1,
            head_q_dim=1,
            head_k_dim=1,
            head_v_dim=1,
            conv_weights=torch.ones(3, 2),
            bias=None,
            activation="silu",
            A_log=torch.zeros(1),
            dt_bias=torch.zeros(1),
        )
        mixed_qkv = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])

        with (
            patch.object(gdn_backend, "causal_conv1d_fn", _identity_conv),
            patch.object(
                gdn_backend,
                "fused_gdn_gating",
                lambda *_args: (torch.zeros(1), torch.zeros(1)),
            ),
        ):
            output = backend.forward_extend_pic_addition(
                layer,
                self._make_forward_batch(),
                mixed_qkv,
                torch.zeros(1),
                torch.zeros(1),
            )

        self._assert_composed_result(
            output,
            backend.req_to_token_pool.ssm_states,
        )

    def test_gdn_single_miss_keeps_the_one_pass_path(self):
        backend = self._make_backend(gdn_backend.GDNAttnBackend)
        backend._pic_temp_states_buf = torch.empty(1, 1, 1, 1)
        backend._pic_seg_indices = torch.tensor([0], dtype=torch.int32)
        backend._pic_seg_cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)
        backend._pic_seg_lengths = [1]
        backend._pic_trans_persist_src = torch.tensor([0], dtype=torch.long)
        backend._pic_trans_persist_dst = torch.tensor([2], dtype=torch.long)
        backend._pic_req_last_miss_payload = torch.tensor([0], dtype=torch.long)
        backend._pic_seg_offsets = [0, 1]
        backend._pic_has_multiple_misses = False
        backend.kernel_dispatcher = _FakeGDNKernelDispatcher()
        layer = SimpleNamespace(
            layer_id=0,
            q_dim=1,
            k_dim=1,
            v_dim=1,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=1,
            head_q_dim=1,
            head_k_dim=1,
            head_v_dim=1,
            conv_weights=torch.ones(3, 2),
            bias=None,
            activation="silu",
            A_log=torch.zeros(1),
            dt_bias=torch.zeros(1),
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            pic_hit_mamba_slots=[{"hit": 0}],
            pic_hit_segments=[[(0, 1, "hit")]],
            pic_miss_mamba_slots=[{(1, 2): 2}],
            pic_miss_segments=[[(1, 2)]],
        )

        with (
            patch.object(gdn_backend, "causal_conv1d_fn", _identity_conv),
            patch.object(
                gdn_backend,
                "fused_gdn_gating",
                lambda *_args: (torch.zeros(1), torch.zeros(1)),
            ),
        ):
            output = backend.forward_extend_pic_addition(
                layer,
                forward_batch,
                torch.tensor([[0.0, 0.0, 1.0]]),
                torch.zeros(1),
                torch.zeros(1),
            )

        torch.testing.assert_close(output.flatten(), torch.tensor([6.0]))
        torch.testing.assert_close(
            backend.req_to_token_pool.ssm_states[4].flatten(),
            torch.tensor([6.0]),
        )
        self.assertEqual(backend.kernel_dispatcher.call_count, 1)

    def test_kda_addition_carries_history_across_multiple_misses(self):
        backend = self._make_backend(kda_backend.KDAAttnBackend)
        layer = SimpleNamespace(
            layer_id=0,
            q_dim=1,
            k_dim=1,
            v_dim=1,
            head_q_dim=1,
            head_k_dim=1,
            head_v_dim=1,
            conv_weights=torch.ones(3, 2),
            bias=None,
        )
        mixed_qkv = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])

        with (
            patch.object(kda_backend, "causal_conv1d_fn", _identity_conv),
            patch.object(kda_backend, "chunk_kda", _fake_chunk_kda),
        ):
            output = backend.forward_extend_pic_addition(
                layer,
                self._make_forward_batch(),
                mixed_qkv,
                torch.zeros(1),
                torch.zeros(1),
            )

        self._assert_composed_result(
            output,
            backend.req_to_token_pool.ssm_states,
        )


if __name__ == "__main__":
    unittest.main()
