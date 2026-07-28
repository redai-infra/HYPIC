import torch

import sglang.srt.pic.picache as picache
from sglang.srt.pic.picache import PICache
from sglang.srt.pic.segmenter import segment_hash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Args:
    pic_segment_min_tokens = -1


def _pc():
    picache.get_global_server_args = lambda: _Args()

    class A:
        page_size = 1
        device = "cpu"

        def available_size(self):
            return 4096

        def alloc(self, n):
            return torch.arange(n, dtype=torch.int64)

        def free(self, idx):
            self.last_free = idx

    class M:
        def available_size(self):
            return 64

        def free(self, idx):
            self.last_free = idx

    class R:
        pass

    mamba_allocator = M()
    req_pool = R()
    req_pool.mamba_allocator = mamba_allocator
    return PICache(req_pool, A(), mamba_allocator, page_size=1, disable=False), A, M


def test_cache_unfinished_req_inserts_miss_segments():
    pc, _, _ = _pc()
    seg_ids = [1, 2, 3]
    h = segment_hash(seg_ids)

    class Req:
        pass

    req = Req()
    req.fill_ids = seg_ids + [4, 5]
    req.origin_input_ids = req.fill_ids
    req.pic_segments = [(0, 3), (3, 5)]
    req.pic_miss_segments = [(0, 3)]
    req.pic_segment_entries = {}
    req.pic_miss_segment_slots = {
        (0, 3): (torch.tensor([777, 778, 779], dtype=torch.int64), 99)
    }

    pc.cache_unfinished_req(req)
    assert h in pc._entries
    e = pc._entries[h]
    assert e.full_kv_slots.tolist() == [777, 778, 779]
    assert e.mamba_state_slot == 99
    assert h in req.pic_segment_entries


def test_cache_unfinished_req_inserts_single_segment():
    pc, _, _ = _pc()
    seg_ids = [1, 2, 3]
    h = segment_hash(seg_ids)

    class Req:
        pass

    req = Req()
    req.fill_ids = seg_ids
    req.origin_input_ids = req.fill_ids
    req.pic_segments = [(0, 3)]
    req.pic_miss_segments = [(0, 3)]
    req.pic_segment_entries = {}
    req.pic_miss_segment_slots = {
        (0, 3): (torch.tensor([777, 778, 779], dtype=torch.int64), 99)
    }

    pc.cache_unfinished_req(req)
    assert h in pc._entries
    assert pc._entries[h].full_kv_slots.tolist() == [777, 778, 779]


def test_cache_unfinished_req_idempotent_frees_dup():
    pc, A, M = _pc()
    seg_ids = [1, 2, 3]
    h = segment_hash(seg_ids)
    pre = pc._insert_segment(h, torch.tensor(seg_ids), torch.tensor([100, 101, 102]), 5)

    class Req:
        pass

    req = Req()
    req.fill_ids = seg_ids + [4]
    req.origin_input_ids = req.fill_ids
    req.pic_segments = [(0, 3), (3, 4)]
    req.pic_miss_segments = [(0, 3)]
    req.pic_segment_entries = {}
    dup_slots = torch.tensor([200, 201, 202], dtype=torch.int64)
    req.pic_miss_segment_slots = {(0, 3): (dup_slots, 6)}

    pc.cache_unfinished_req(req)
    assert pc._entries[h] is pre
    assert req.pic_segment_entries[h] is pre
    assert hasattr(pc.token_to_kv_pool_allocator, "last_free")


def test_cache_finished_req_dec_locks_all_entries():
    pc, _, _ = _pc()
    h = segment_hash([1, 2, 3])
    e = pc._insert_segment(h, torch.tensor([1, 2, 3]), torch.tensor([10, 11, 12]), 5)
    e.lock_ref = 1

    class Req:
        pass

    req = Req()
    req.pic_segment_entries = {h: e}
    pc.cache_finished_req(req)
    assert e.lock_ref == 0


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
