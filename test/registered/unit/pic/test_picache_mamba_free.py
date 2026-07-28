from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.pic.picache import PICache
from sglang.srt.pic.segmenter import segment_hash
from sglang.srt.server_args import (
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def set_pic_min_tokens():
    try:
        previous_server_args = get_global_server_args()
    except ValueError:
        previous_server_args = None

    def set_value(value):
        set_global_server_args_for_scheduler(
            SimpleNamespace(pic_segment_min_tokens=value)
        )

    yield set_value
    set_global_server_args_for_scheduler(previous_server_args)


class RecordingAllocator:
    page_size = 1
    device = "cpu"

    def __init__(self):
        self.freed = []

    def available_size(self):
        return 4096

    def free(self, slots):
        self.freed.append(slots.clone())


class RecordingReqPool:
    def __init__(self):
        self.mamba_allocator = RecordingAllocator()


def make_picache():
    req_pool = RecordingReqPool()
    kv_allocator = RecordingAllocator()
    cache = PICache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=kv_allocator,
        mamba_pool=SimpleNamespace(),
        page_size=1,
        disable=False,
    )
    return cache, kv_allocator, req_pool.mamba_allocator


def insert_entry(cache, token_ids, kv_slots, mamba_slot, last_access_time):
    token_ids = torch.tensor(token_ids, dtype=torch.int64)
    entry = cache._insert_segment(
        segment_hash(token_ids),
        token_ids,
        torch.tensor(kv_slots, dtype=torch.int64),
        mamba_slot,
    )
    entry.last_access_time = last_access_time
    return entry


def test_duplicate_cleanup_returns_all_mamba_slots_together(set_pic_min_tokens):
    set_pic_min_tokens(-1)
    cache, kv_allocator, mamba_allocator = make_picache()
    first = insert_entry(cache, [1, 2], [1, 2], 1, 1)
    second = insert_entry(cache, [3, 4], [3, 4], 2, 2)
    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3, 4, 5],
        pic_segments=[(0, 2), (2, 4), (4, 5)],
        pic_miss_segments=[(0, 2), (2, 4)],
        pic_miss_segment_slots={
            (0, 2): (torch.tensor([101, 102]), 11),
            (2, 4): (torch.tensor([103, 104]), 12),
        },
        pic_segment_entries={},
    )

    cache.cache_unfinished_req(req)

    assert [slots.tolist() for slots in kv_allocator.freed] == [
        [101, 102],
        [103, 104],
    ]
    assert [slots.tolist() for slots in mamba_allocator.freed] == [[11, 12]]
    assert req.pic_segment_entries == {
        first.seg_hash: first,
        second.seg_hash: second,
    }
    assert req.pic_freed_miss_segments == {(0, 2), (2, 4)}
    assert len(cache._entries) == 2


def test_finished_request_returns_last_and_skipped_mamba_slots_together(
    set_pic_min_tokens,
):
    set_pic_min_tokens(3)
    cache, kv_allocator, mamba_allocator = make_picache()
    req = SimpleNamespace(
        pic_segments=[(0, 2), (2, 4), (4, 5)],
        pic_miss_segment_slots={
            (0, 2): (torch.tensor([101, 102]), 11),
            (2, 4): (torch.tensor([103, 104]), 12),
            (4, 5): (torch.tensor([105]), 13),
        },
        pic_segment_entries={},
        req_pool_idx=None,
        mamba_pool_idx=None,
    )

    cache.cache_finished_req(req, is_insert=False)

    assert [slots.tolist() for slots in kv_allocator.freed] == [
        [105],
        [101, 102, 103, 104],
    ]
    assert [slots.tolist() for slots in mamba_allocator.freed] == [[13, 11, 12]]
    assert cache._entries == {}


def test_evict_returns_selected_mamba_slots_in_lru_order_together():
    cache, kv_allocator, mamba_allocator = make_picache()
    first = insert_entry(cache, [1], [101, 102], 11, 1)
    locked = insert_entry(cache, [2], [201], 21, 0)
    third = insert_entry(cache, [3], [301, 302, 303], 31, 2)
    retained = insert_entry(cache, [4], [401], 41, 3)
    locked.lock_ref = 1

    result = cache.evict(EvictParams(num_tokens=5, mamba_num=2))

    assert result.num_tokens_evicted == 5
    assert result.mamba_num_evicted == 2
    assert [slots.tolist() for slots in kv_allocator.freed] == [
        [101, 102],
        [301, 302, 303],
    ]
    assert [slots.tolist() for slots in mamba_allocator.freed] == [[11, 31]]
    assert list(cache._entries) == [locked.seg_hash, retained.seg_hash]
    assert first.seg_hash not in cache._entries
    assert third.seg_hash not in cache._entries


def test_evict_restores_real_mamba_allocator_capacity_and_order():
    req_pool = RecordingReqPool()
    req_pool.mamba_allocator = MambaSlotAllocator(4, "cpu")
    allocated = req_pool.mamba_allocator.alloc(4)
    cache = PICache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=RecordingAllocator(),
        mamba_pool=SimpleNamespace(),
        page_size=1,
        disable=False,
    )
    for offset, slot in enumerate(allocated.tolist()):
        insert_entry(
            cache,
            [offset],
            [100 + offset],
            slot,
            last_access_time=offset,
        )

    result = cache.evict(EvictParams(num_tokens=3, mamba_num=3))

    assert result.num_tokens_evicted == 3
    assert result.mamba_num_evicted == 3
    assert req_pool.mamba_allocator.available_size() == 3
    assert req_pool.mamba_allocator.free_slots.tolist() == [1, 2, 3]
    assert [entry.mamba_state_slot for entry in cache._entries.values()] == [4]
