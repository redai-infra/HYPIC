import torch

from sglang.srt.pic.picache import SegmentEntry
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_segment_entry_construction_defaults():
    e = SegmentEntry(
        seg_hash=b"\x00" * 16,
        full_kv_slots=torch.tensor([10, 11, 12], dtype=torch.int64),
        mamba_state_slot=7,
        token_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
    )
    assert e.lock_ref == 0
    assert e.last_access_time == 0.0
    assert e.creation_time == 0.0
    assert e.hit_count == 0
    assert e.priority == 0


def test_lrustrategy_works_on_segment_entry():
    from sglang.srt.mem_cache.evict_policy import LRUStrategy

    e = SegmentEntry(
        seg_hash=b"\x00" * 16,
        full_kv_slots=torch.empty(0, dtype=torch.int64),
        mamba_state_slot=0,
        token_ids=torch.empty(0, dtype=torch.int64),
        last_access_time=42.0,
    )
    strategy = LRUStrategy()
    assert strategy.get_priority(e) == 42.0  # duck-types as TreeNode


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
