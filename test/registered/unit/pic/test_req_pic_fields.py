"""Verify the 5 PIC fields are present on Req with correct defaults.

Test is intentionally minimal: we construct Req via the canonical minimal
signature and just check field defaults. Behavior is covered in PICache
tests (T10-T13) and integration (T23).
"""

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_req_has_pic_fields():
    r = Req(
        rid="test-rid",
        origin_input_text="hello",
        origin_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
    )
    assert r.pic_segments is None
    assert r.pic_hit_segments == []
    assert r.pic_miss_segments == []
    assert r.pic_segment_entries == {}
    assert r.pic_miss_token_positions is None


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
