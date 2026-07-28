"""Pending-hold eviction bug: a DEFERRED scatter push (combine dst handle not
yet arrived) must pin the cache entry across the _pic_push_pending window so it
can't be evicted before try_push_pending fires the WRITE. Net lock_ref per
segment stays 0: +1 at register, -1 at fire or timeout-drop."""

import types

import sglang.srt.pic.scatter_xfer as sx
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Entry:
    def __init__(self, lock_ref=0):
        self.lock_ref = lock_ref
        self.full_kv_slots = list(range(4))
        self.mamba_state_slot = 5


class _Sched:
    def __init__(self, entry, seg_hash):
        self.tree_cache = types.SimpleNamespace(_entries={seg_hash: entry})
        # combine on a DIFFERENT port than the meta below → REMOTE branch
        self.server_args = types.SimpleNamespace(port=30001, pic_scatter_timeout_s=30.0)
        self._pic_scatter_handles = {}  # empty → deferred


def _make_req():
    return types.SimpleNamespace(
        pic_scatter_meta={
            "combine_addr": "http://127.0.0.1:9999",  # port != 30001 → remote
            "scatter_room": 42,
            "seg_index": 0,
        },
        origin_input_ids=[1, 2, 3, 4],
    )


def test_deferred_push_takes_pending_hold(monkeypatch):
    """RED→GREEN #1: deferred push pins the entry (+1) and registers pending."""
    req = _make_req()
    seg_hash = sx.segment_hash(list(req.origin_input_ids))
    entry = _Entry(lock_ref=0)
    sched = _Sched(entry, seg_hash)

    sx.maybe_push_after_prefill(req, sched)

    assert entry.lock_ref == 1, "pending-hold not taken on deferred push"
    assert len(sched._pic_push_pending) == 1
    assert sched._pic_push_pending[0]["seg_hash"] == seg_hash


def test_fire_releases_pending_hold(monkeypatch):
    """#2: handle arrives → _start_write fires, pending-hold released, item removed."""
    req = _make_req()
    seg_hash = sx.segment_hash(list(req.origin_input_ids))
    entry = _Entry(lock_ref=0)
    sched = _Sched(entry, seg_hash)

    sx.maybe_push_after_prefill(req, sched)  # register + take hold
    assert entry.lock_ref == 1

    fired = {"n": 0}
    monkeypatch.setattr(
        sx,
        "_start_write",
        lambda s, h, e, room, seg, sh: fired.__setitem__("n", fired["n"] + 1),
    )
    sched._pic_scatter_handles = {(42, 0): object()}
    sx.try_push_pending(sched)

    assert fired["n"] == 1, "_start_write not invoked"
    assert entry.lock_ref == 0, "pending-hold not released on fire (net != 0)"
    assert sched._pic_push_pending == []
    assert (42, 0) not in sched._pic_scatter_handles


def test_fire_start_write_raises_does_not_leak_hold(monkeypatch):
    """#4: _start_write raises (PIC_SYNC_WRITE / transfer error) → pending-hold
    still released (net 0), not leaked into the pool → OOM path this fix guards."""
    req = _make_req()
    seg_hash = sx.segment_hash(list(req.origin_input_ids))
    entry = _Entry(lock_ref=0)
    sched = _Sched(entry, seg_hash)

    sx.maybe_push_after_prefill(req, sched)  # register + take hold
    assert entry.lock_ref == 1

    def _boom(*a, **k):
        raise RuntimeError("write failed")

    monkeypatch.setattr(sx, "_start_write", _boom)
    sched._pic_scatter_handles = {(42, 0): object()}
    sx.try_push_pending(sched)  # must swallow the exception, not leak the +1

    assert entry.lock_ref == 0, "pending-hold leaked when _start_write raised"
    assert sched._pic_push_pending == []


def test_timeout_drop_releases_pending_hold(monkeypatch):
    """#3: past deadline, handle absent → pending-hold released, item removed."""
    req = _make_req()
    seg_hash = sx.segment_hash(list(req.origin_input_ids))
    entry = _Entry(lock_ref=0)
    sched = _Sched(entry, seg_hash)

    sx.maybe_push_after_prefill(req, sched)
    assert entry.lock_ref == 1

    # force the deadline into the past; handle still absent
    sched._pic_push_pending[0]["deadline"] = 0.0
    sched._pic_scatter_handles = {}
    sx.try_push_pending(sched)

    assert entry.lock_ref == 0, "pending-hold not released on timeout-drop (net != 0)"
    assert sched._pic_push_pending == []


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
