"""Test that Req.init_next_round_input wires PIC fields from MatchResult."""

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeCache:
    page_size = 1

    def supports_mamba(self):
        return True

    def match_prefix(self, params):
        from sglang.srt.pic.picache import SegmentEntry

        entry = SegmentEntry(
            seg_hash=b"\x01" * 16,
            full_kv_slots=torch.tensor([100, 101, 102], dtype=torch.int64),
            mamba_state_slot=7,
            token_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
        )
        return MatchResult(
            device_indices=torch.tensor([100, 101, 102], dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
            pic_segment_entries=[entry, None],
        )


def _make_req():
    req = Req.__new__(Req)
    req.rid = "test-rid"
    req.fill_ids = []
    req.full_untruncated_fill_ids = []
    req.origin_input_ids = [1, 2, 3, 4, 5]
    req.output_ids = []
    req.positional_embed_overrides = None
    req.pic_segments = [(0, 3), (3, 5)]
    req.pic_hit_segments = []
    req.pic_miss_segments = []
    req.pic_segment_entries = {}
    req.pic_miss_token_positions = None
    req.return_logprob = False
    req.logprob_start_len = -1
    req.session = None
    req.extra_key = None
    req.is_retracted = False
    req.multimodal_inputs = None
    req.dllm_config = None
    # set_extend_input_len writes to extend_input_len
    req.extend_input_len = 0
    return req


def test_init_next_round_input_populates_pic_fields():
    req = _make_req()
    req.init_next_round_input(tree_cache=FakeCache())
    assert len(req.pic_segment_entries) == 1
    assert req.pic_hit_segments == [(0, 3, b"\x01" * 16)]
    assert req.pic_miss_segments == [(3, 5)]
    assert req.pic_miss_token_positions.tolist() == [3, 4]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
