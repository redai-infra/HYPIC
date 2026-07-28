import torch

from sglang.srt.pic.picache import PICache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _fake_picache():
    # 用最小 stub：只测 _entries / lock_ref 簿记，不碰真 pool
    c = PICache.__new__(PICache)
    c._entries = {}
    return c


def test_inject_new_then_dup():
    c = _fake_picache()
    h = b"\x01" * 16
    tok = torch.tensor([1, 2, 3], dtype=torch.int64)
    kv = torch.tensor([10, 11, 12], dtype=torch.int64)
    e1 = c.inject_received_segment(h, tok, kv, 5)
    assert e1.lock_ref == 1 and c._entries[h] is e1
    e2 = c.inject_received_segment(h, tok, kv, 7)  # dup hash
    assert e2 is e1 and e1.lock_ref == 2  # 不重插，引用+1


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
