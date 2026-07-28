from sglang.srt.pic.scatter_xfer import partition_combine_first
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Req:
    """Minimal stand-in for sglang Req: only the attr the partition reads."""

    def __init__(self, name, is_pic_combine=False):
        self.name = name
        self.is_pic_combine = is_pic_combine

    def __repr__(self):
        return self.name


def test_combines_move_to_front_stable():
    s0 = _Req("seg0")
    c0 = _Req("combine0", is_pic_combine=True)
    s1 = _Req("seg1")
    c1 = _Req("combine1", is_pic_combine=True)
    s2 = _Req("seg2")
    q = [s0, c0, s1, c1, s2]

    partition_combine_first(q)

    # combines first, in their original relative order; then segments, in theirs.
    assert q == [c0, c1, s0, s1, s2]


def test_no_combines_is_noop():
    q = [_Req("seg0"), _Req("seg1"), _Req("seg2")]
    before = list(q)
    partition_combine_first(q)
    assert q == before


def test_all_combines_unchanged():
    q = [_Req("c0", True), _Req("c1", True)]
    before = list(q)
    partition_combine_first(q)
    assert q == before


def test_missing_attr_treated_as_segment():
    class _Bare:
        pass

    bare = _Bare()  # no is_pic_combine attr at all
    c = _Req("c", True)
    q = [bare, c]
    partition_combine_first(q)
    assert q == [c, bare]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
