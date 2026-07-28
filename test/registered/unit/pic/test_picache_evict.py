import time

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.pic.picache import PICache
from sglang.srt.pic.segmenter import segment_hash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _pc():
    class A:
        page_size = 1
        device = "cpu"
        freed = []

        def available_size(self):
            return 4096

        def alloc(self, n):
            return torch.arange(n, dtype=torch.int64)

        def free(self, idx):
            self.freed.append(idx)

    class M:
        freed = []

        def free(self, idx):
            self.freed.append(idx)

        def available_size(self):
            return 64

    class R:
        pass

    mamba_allocator = M()
    req_pool = R()
    req_pool.mamba_allocator = mamba_allocator
    return PICache(req_pool, A(), mamba_allocator, page_size=1, disable=False)


def test_evict_lru_skips_locked():
    pc = _pc()
    h1 = segment_hash([1, 2, 3])
    h2 = segment_hash([4, 5, 6])
    e1 = pc._insert_segment(
        h1, torch.tensor([1, 2, 3]), torch.tensor([10, 11, 12]), 100
    )
    time.sleep(0.001)
    pc._insert_segment(h2, torch.tensor([4, 5, 6]), torch.tensor([20, 21, 22]), 200)
    e1.lock_ref = 1
    r = pc.evict(EvictParams(num_tokens=3))
    assert r.num_tokens_evicted == 3
    assert h2 not in pc._entries
    assert h1 in pc._entries


def test_evict_picks_lru_among_unlocked():
    pc = _pc()
    h1 = segment_hash([1, 2, 3])
    h2 = segment_hash([4, 5, 6])
    pc._insert_segment(h1, torch.tensor([1, 2, 3]), torch.tensor([10, 11, 12]), 100)
    time.sleep(0.001)
    pc._insert_segment(h2, torch.tensor([4, 5, 6]), torch.tensor([20, 21, 22]), 200)
    r = pc.evict(EvictParams(num_tokens=3))
    assert r.num_tokens_evicted == 3
    assert h1 not in pc._entries
    assert h2 in pc._entries


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
