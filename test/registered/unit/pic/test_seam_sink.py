from sglang.srt.pic import SEAM_SINK_DEFAULT, resolve_seam_sink_tokens
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_resolve_seam_sink_tokens_ratio_and_count():
    assert SEAM_SINK_DEFAULT == 8
    assert resolve_seam_sink_tokens(0, 100) == 0
    assert resolve_seam_sink_tokens(0.25, 100) == 25
    assert resolve_seam_sink_tokens(0.01, 10) == 1
    assert resolve_seam_sink_tokens(1, 100) == 100
    assert resolve_seam_sink_tokens(8, 100) == 8
    assert resolve_seam_sink_tokens(200, 100) == 100


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
