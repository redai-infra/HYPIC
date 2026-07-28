import types

import sglang.srt.pic.segmenter as seg
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Entry:
    def __init__(self):
        self.lock_ref = 0


def _make_scheduler():
    from sglang.srt.managers.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)  # bypass __init__
    s.waiting_queue = []
    s.disaggregation_mode = None
    s.enable_priority_scheduling = False  # v0.5.14 _add_request_to_queue gate
    s._entries = {}
    s.tree_cache = types.SimpleNamespace(_entries=s._entries)
    return s


def _make_req(token_ids):
    r = types.SimpleNamespace()
    r.pic_scatter_single_seg = True
    r.origin_input_ids = token_ids
    r.rid = "r1"
    r.priority = None  # v0.5.14 priority gate expects the attr
    return r


def test_hit_stashes_and_pins_not_enqueued():
    s = _make_scheduler()
    tok = [10, 11, 12]
    h = seg.segment_hash(tok)
    entry = _Entry()
    s.tree_cache._entries[h] = entry
    req = _make_req(tok)

    s._add_request_to_queue(req)

    assert req not in s.waiting_queue  # NOT enqueued for prefill
    assert getattr(s, "_pic_hit_retire", []) == [(req, h)]
    assert entry.lock_ref == 1  # hit-hold taken


def test_miss_enqueues_as_before():
    s = _make_scheduler()
    req = _make_req([99, 98])  # hash absent
    s._add_request_to_queue(req)
    assert req in s.waiting_queue
    assert getattr(s, "_pic_hit_retire", []) == []


def test_pic_drain_retires_hits(monkeypatch):
    import sglang.srt.pic.scatter_xfer as sx

    s = _make_scheduler()
    # _pic_drain also calls try_push_pending/drain_writes/drain_injects — stub them
    monkeypatch.setattr(sx, "try_push_pending", lambda s: None)
    monkeypatch.setattr(sx, "drain_writes", lambda s: None)
    monkeypatch.setattr(sx, "drain_injects", lambda s: None)
    retired = []
    monkeypatch.setattr(
        sx,
        "pic_scatter_retire",
        lambda sch, r, h, *, recomputed: retired.append((r, h, recomputed)),
    )
    req = _make_req([5, 6])
    h = seg.segment_hash([5, 6])
    s._pic_hit_retire = [(req, h)]

    s._pic_drain()

    assert retired == [(req, h, False)]
    assert s._pic_hit_retire == []


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
