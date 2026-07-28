"""T14 sanity: PICache is importable alongside scheduler with no circular import."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_picache_importable_from_scheduler_path():
    from sglang.srt.managers import scheduler  # noqa: F401
    from sglang.srt.pic.picache import PICache  # noqa: F401


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
