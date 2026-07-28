import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_pic_segment_entries_default_none_backward_compat():
    r = MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=None,
        last_host_node=None,
        best_match_node=None,
    )
    assert r.pic_segment_entries is None


def test_pic_segment_entries_can_be_set():
    r = MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=None,
        last_host_node=None,
        best_match_node=None,
        pic_segment_entries=[None, "fake-entry-1", None],
    )
    assert r.pic_segment_entries[1] == "fake-entry-1"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
