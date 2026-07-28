import types

import sglang.srt.pic.scatter_xfer as sx
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Entry:
    def __init__(self):
        self.lock_ref = 1  # protect/hit hold present per contract


class _TC:
    def __init__(self, entry, seg_hash):
        self._entries = {seg_hash: entry}


class _Req:
    def __init__(self):
        self.rid = "r1"
        self.return_logprob = False
        self.finished_reason = None
        self.mamba_pool_idx = None
        self.origin_input_ids = [1, 2, 3]
        self.pic_scatter_meta = {
            "combine_addr": "http://127.0.0.1:1",
            "scatter_room": 7,
            "seg_index": 0,
        }


class _Sched:
    def __init__(self, entry, seg_hash):
        self.tree_cache = _TC(entry, seg_hash)
        self.server_args = types.SimpleNamespace(port=30001, pic_scatter_timeout_s=30.0)
        self.streamed = []
        self.freed_mamba = []
        self.req_to_token_pool = types.SimpleNamespace(
            free_mamba_cache=lambda r: self.freed_mamba.append(r),
            free=lambda r: None,
        )
        # v0.5.14 moved stream_output onto the SchedulerOutputStreamer.
        self.output_streamer = types.SimpleNamespace(
            stream_output=lambda reqs, logprob: self.streamed.append(reqs[0])
        )


def test_retire_hit_releases_hold_pushes_and_finishes(monkeypatch):
    seg_hash = sx.segment_hash([1, 2, 3])
    entry = _Entry()
    sched = _Sched(entry, seg_hash)
    req = _Req()
    called = {}
    monkeypatch.setattr(
        sx, "maybe_push_after_prefill", lambda r, s: called.setdefault("push", (r, s))
    )

    sx.pic_scatter_retire(sched, req, seg_hash, recomputed=False)

    assert called["push"] == (req, sched)  # push invoked
    assert entry.lock_ref == 0  # +1 released, net 0
    assert sched.freed_mamba == []  # hit path frees no mamba
    assert req.finished_reason is not None  # finished
    assert sched.streamed == [req]  # client notified


def test_retire_recomputed_frees_mamba(monkeypatch):
    seg_hash = sx.segment_hash([1, 2, 3])
    entry = _Entry()
    sched = _Sched(entry, seg_hash)
    req = _Req()
    req.mamba_pool_idx = 5
    monkeypatch.setattr(sx, "maybe_push_after_prefill", lambda r, s: None)

    sx.pic_scatter_retire(sched, req, seg_hash, recomputed=True)

    assert entry.lock_ref == 0  # +1 released
    assert sched.freed_mamba == [req]  # miss path frees mamba
    assert req.finished_reason is not None


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
