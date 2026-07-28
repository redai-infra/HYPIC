"""Tests for sglang.srt.pic.pic_alloc.pic_alloc_for_extend."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import sglang.srt.pic.pic_alloc as pic_alloc
from sglang.srt.pic.pic_alloc import pic_alloc_for_extend
from sglang.srt.pic.picache import PICache
from sglang.srt.pic.segmenter import segment_hash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Args:
    pic_segment_min_tokens = -1


class _FakeAllocator:
    """Stands in for TokenToKVPoolAllocator."""

    page_size = 1
    device = "cpu"

    def __init__(self):
        self._cursor = 1000
        self.last_free = None

    def available_size(self):
        return 1 << 20

    def alloc(self, n):
        out = torch.arange(self._cursor, self._cursor + n, dtype=torch.int64)
        self._cursor += n
        return out

    def free(self, idx):
        self.last_free = idx


class _FakeMambaPool:
    def __init__(self):
        self._next = 50

    def available_size(self):
        return 64

    def alloc(self, n):
        out = torch.arange(self._next, self._next + n, dtype=torch.int64)
        self._next += n
        return out

    def free(self, idx):
        pass


class _FakeReqToTokenPool:
    """Stands in for HybridReqToTokenPool."""

    def __init__(self, size: int, max_context_len: int, mamba_pool):
        self.size = size
        self.max_context_len = max_context_len
        self.req_to_token = torch.zeros((size, max_context_len), dtype=torch.int64)
        self.free_slots = list(range(size))
        self.mamba_pool = mamba_pool
        self.mamba_allocator = mamba_pool

    def available_size(self):
        return len(self.free_slots)

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def alloc(self, reqs):
        need = sum(1 for r in reqs if r.req_pool_idx is None)
        if need > len(self.free_slots):
            return None
        chosen = self.free_slots[:need]
        self.free_slots = self.free_slots[need:]
        off = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = chosen[off]
                off += 1
        return [r.req_pool_idx for r in reqs]


def _make_req(req_pool_idx_initial=None):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx_initial,
        pic_hit_segments=[],
        pic_miss_segments=[],
        pic_segment_entries={},
        pic_miss_segment_slots={},
        is_chunked=0,
        kv_committed_len=0,
    )


def _make_picache(pic_mode="addition"):
    pic_alloc.get_global_server_args = lambda: _Args()
    mamba_allocator = _FakeMambaPool()
    return PICache(
        req_to_token_pool=SimpleNamespace(mamba_allocator=mamba_allocator),
        token_to_kv_pool_allocator=_FakeAllocator(),
        mamba_pool=mamba_allocator,
        page_size=1,
        disable=False,
        pic_mode=pic_mode,
    )


def test_pic_alloc_for_extend_hit_miss_last_segment():
    """One request with 3 segments: seg0 hit, seg1 miss (mid), seg2 miss (last)."""
    pic = _make_picache()

    # Seed PICache with a cached entry for seg0.
    seg0_ids = [10, 11, 12, 13]
    seg0_hash = segment_hash(seg0_ids)
    cached_slots = torch.tensor([7000, 7001, 7002, 7003], dtype=torch.int64)
    entry = pic._insert_segment(
        seg0_hash,
        torch.tensor(seg0_ids, dtype=torch.int64),
        cached_slots,
        mamba_state_slot=42,
    )

    # Build req with hit (0:4), miss (4:7), miss-last (7:10).
    req = _make_req()
    req.pic_hit_segments = [(0, 4, seg0_hash)]
    req.pic_miss_segments = [(4, 7), (7, 10)]
    req.pic_segment_entries = {seg0_hash: entry}

    mamba_pool = _FakeMambaPool()
    rttp = _FakeReqToTokenPool(size=8, max_context_len=64, mamba_pool=mamba_pool)

    batch = SimpleNamespace(
        reqs=[req],
        tree_cache=pic,
        req_to_token_pool=rttp,
        device="cpu",
    )

    out_cache_loc, rpi_device, rpi = pic_alloc_for_extend(batch)

    # Total miss tokens = 3 + 3 = 6.
    assert out_cache_loc.shape == (6,), out_cache_loc.shape
    assert rpi.tolist() == [0]
    assert rpi_device.tolist() == [0]
    assert req.req_pool_idx == 0

    # Hit range written with cached entry slots.
    assert rttp.req_to_token[0, 0:4].tolist() == cached_slots.tolist()
    # Miss range 4:7 written with first chunk of out_cache_loc.
    assert rttp.req_to_token[0, 4:7].tolist() == out_cache_loc[0:3].tolist()
    # Miss range 7:10 written with second chunk.
    assert rttp.req_to_token[0, 7:10].tolist() == out_cache_loc[3:6].tolist()

    # pic_miss_segment_slots populated for both miss segments.
    assert set(req.pic_miss_segment_slots.keys()) == {(4, 7), (7, 10)}
    slots0, mamba0 = req.pic_miss_segment_slots[(4, 7)]
    slots1, mamba1 = req.pic_miss_segment_slots[(7, 10)]
    assert slots0.tolist() == out_cache_loc[0:3].tolist()
    assert slots1.tolist() == out_cache_loc[3:6].tolist()
    assert isinstance(mamba0, int) and isinstance(mamba1, int)
    assert mamba0 != mamba1


def test_pic_alloc_for_extend_all_miss_no_hits():
    pic = _make_picache()
    req = _make_req()
    req.pic_hit_segments = []
    req.pic_miss_segments = [(0, 5)]

    mamba_pool = _FakeMambaPool()
    rttp = _FakeReqToTokenPool(size=4, max_context_len=32, mamba_pool=mamba_pool)

    batch = SimpleNamespace(
        reqs=[req],
        tree_cache=pic,
        req_to_token_pool=rttp,
        device="cpu",
    )

    out_cache_loc, _, _ = pic_alloc_for_extend(batch)
    assert out_cache_loc.shape == (5,)
    assert rttp.req_to_token[0, 0:5].tolist() == out_cache_loc.tolist()
    assert (0, 5) in req.pic_miss_segment_slots


def test_pic_alloc_transition_rope_single_segment_has_public_slots():
    pic = _make_picache(pic_mode="transition_rope")
    req = _make_req()
    req.pic_segments = [(0, 5)]
    req.pic_miss_segments = [(0, 5)]

    mamba_pool = _FakeMambaPool()
    rttp = _FakeReqToTokenPool(size=4, max_context_len=32, mamba_pool=mamba_pool)
    batch = SimpleNamespace(
        reqs=[req],
        tree_cache=pic,
        req_to_token_pool=rttp,
        device="cpu",
    )

    out_cache_loc, _, _ = pic_alloc_for_extend(batch)
    private, public, mamba = req.pic_miss_segment_slots[(0, 5)]
    assert out_cache_loc.tolist() == private.tolist()
    assert public is not None
    assert public.numel() == 5
    assert mamba is not None


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
