import pytest
import torch

from sglang.srt.pic.picache import PICache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeAllocator:
    page_size = 1

    def available_size(self):
        return 1024

    def alloc(self, n):
        return torch.arange(n, dtype=torch.int64)

    def free(self, idx):
        pass


class FakeReqPool:
    pass


class FakeMambaPool:
    def available_size(self):
        return 64

    def alloc_one(self):
        return 0

    def free(self, idx):
        pass


def test_picache_init_smoke():
    mamba_allocator = FakeMambaPool()
    req_pool = FakeReqPool()
    req_pool.mamba_allocator = mamba_allocator
    pc = PICache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=FakeAllocator(),
        mamba_pool=mamba_allocator,
        page_size=1,
        disable=False,
        pic_mode="addition",
    )
    assert pc.supports_mamba() is True
    assert pc.is_tree_cache() is True
    assert pc.is_chunk_cache() is False
    assert pc.page_size == 1
    assert pc.disable is False
    assert pc.evictable_size() == 0


def test_picache_rejects_unknown_mode():
    mamba_allocator = FakeMambaPool()
    req_pool = FakeReqPool()
    req_pool.mamba_allocator = mamba_allocator
    with pytest.raises((AssertionError, NotImplementedError)):
        PICache(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=FakeAllocator(),
            mamba_pool=mamba_allocator,
            page_size=1,
            disable=False,
            pic_mode="bogus",
        )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
