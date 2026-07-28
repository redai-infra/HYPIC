import pytest

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_pic_segments_field_default_none():
    req = GenerateReqInput(text="hi")
    assert req.pic_segments is None


def test_pic_segments_field_accepts_list_of_pairs():
    req = GenerateReqInput(text="hi", pic_segments=[(0, 2), (2, 4)])
    assert req.pic_segments == [(0, 2), (2, 4)]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
