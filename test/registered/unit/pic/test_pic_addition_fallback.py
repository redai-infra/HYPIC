from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.pic.picache import PICache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class RecordingTokenAllocator:
    device = "cpu"

    def __init__(self):
        self.freed = []

    def free(self, slots):
        self.freed.extend(slots.tolist())


def test_full_recompute_releases_uncached_pic_slots():
    token_allocator = RecordingTokenAllocator()
    mamba_allocator = MambaSlotAllocator(size=3, device="cpu")
    cache = PICache(
        req_to_token_pool=SimpleNamespace(mamba_allocator=mamba_allocator),
        token_to_kv_pool_allocator=token_allocator,
        mamba_pool=SimpleNamespace(),
        page_size=1,
        disable=False,
    )
    slots = torch.arange(10, 15)
    req = SimpleNamespace(
        pic_full_recompute=True,
        pic_miss_segments=[(0, 5)],
        pic_miss_segment_slots={(0, 5): (slots, None)},
        pic_segment_entries={},
        pic_segments=[(0, 5)],
        req_pool_idx=None,
        mamba_pool_idx=None,
    )
    cache.add_inflight(5, 0)

    cache.cache_finished_req(req)

    assert token_allocator.freed == slots.tolist()
    assert cache._inflight_full_tokens == 0
    assert cache._inflight_mamba_slots == 0
