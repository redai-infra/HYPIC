import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.pic.picache import PICache
from sglang.srt.pic.segmenter import segment_hash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Req:
    def __init__(self, pic_segments, fill_ids):
        self.pic_segments = pic_segments
        self.fill_ids = fill_ids
        self.origin_input_ids = fill_ids


def _pc():
    class A:
        page_size = 1

        def available_size(self):
            return 4096

        def alloc(self, n):
            return torch.arange(n, dtype=torch.int64)

        def free(self, idx):
            pass

    class M:
        def available_size(self):
            return 64

        def free(self, idx):
            pass

    class R:
        pass

    mamba_allocator = M()
    req_pool = R()
    req_pool.mamba_allocator = mamba_allocator
    return PICache(req_pool, A(), mamba_allocator, page_size=1, disable=False)


def test_match_no_segments_returns_empty():
    pc = _pc()
    req = _Req(pic_segments=None, fill_ids=[1, 2, 3])
    r = pc.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3]), req=req))
    assert r.device_indices.numel() == 0
    assert r.pic_segment_entries is None or all(
        x is None for x in r.pic_segment_entries
    )


def test_match_some_hit_some_miss():
    pc = _pc()
    seg_ids = torch.tensor([10, 11, 12], dtype=torch.int64)
    h = segment_hash(seg_ids.tolist())
    entry = pc._insert_segment(
        h, seg_ids, torch.tensor([900, 901, 902], dtype=torch.int64), 7
    )

    req = _Req(
        pic_segments=[(0, 3), (3, 6), (6, 9)],
        fill_ids=[1, 2, 3, 10, 11, 12, 99, 98, 97],
    )
    r = pc.match_prefix(
        MatchPrefixParams(key=RadixKey(token_ids=req.fill_ids), req=req)
    )
    assert r.pic_segment_entries is not None
    assert r.pic_segment_entries[0] is None
    assert r.pic_segment_entries[1] is entry
    assert r.pic_segment_entries[2] is None
    assert r.device_indices.tolist() == [900, 901, 902]


def test_match_refreshes_last_access_time():
    pc = _pc()
    seg_ids = torch.tensor([10, 11, 12], dtype=torch.int64)
    h = segment_hash(seg_ids.tolist())
    entry = pc._insert_segment(
        h, seg_ids, torch.tensor([900, 901, 902], dtype=torch.int64), 7
    )
    old_ts = entry.last_access_time
    import time as _t

    _t.sleep(0.001)
    req = _Req(pic_segments=[(0, 3), (3, 6)], fill_ids=[1, 2, 3, 10, 11, 12])
    pc.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=req.fill_ids), req=req))
    assert entry.last_access_time == old_ts
    req2 = _Req(
        pic_segments=[(0, 3), (3, 6), (6, 7)], fill_ids=[1, 2, 3, 10, 11, 12, 5]
    )
    pc.match_prefix(MatchPrefixParams(key=RadixKey(token_ids=req2.fill_ids), req=req2))
    assert entry.last_access_time > old_ts


def test_inc_dec_lock_ref_on_req():
    pc = _pc()
    seg_ids = torch.tensor([10, 11, 12], dtype=torch.int64)
    h = segment_hash(seg_ids.tolist())
    entry = pc._insert_segment(
        h, seg_ids, torch.tensor([900, 901, 902], dtype=torch.int64), 7
    )

    class FakeReq:
        pass

    req = FakeReq()
    req.pic_segment_entries = {h: entry}

    assert entry.lock_ref == 0
    pc.inc_lock_ref(req)
    assert entry.lock_ref == 1
    pc.inc_lock_ref(req)
    assert entry.lock_ref == 2
    pc.dec_lock_ref(req)
    assert entry.lock_ref == 1
    pc.dec_lock_ref(req)
    assert entry.lock_ref == 0


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
