"""Unit test for PIC scatter cache-hit-rate accounting (design/plan 2026-07-17).

Runs on QS (`/usr/bin/python`); imports scatter_xfer which needs torch.
    /usr/bin/python -m sglang.srt.pic.test_scatter_hitrate
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeSched:
    def __init__(self, flags):
        self.__dict__["_pic_scatter_seg_recomputed"] = dict(flags)


class _FakeReq:
    def __init__(self, pic_segments, pic_combine):
        self.pic_segments = pic_segments
        self.pic_combine = pic_combine


def test_cached_tokens_counts_only_hits():
    from sglang.srt.pic.scatter_xfer import pic_combine_cached_tokens

    # 3 segments: seg0 hit (len 100), seg1 miss (len 50), seg2 flag-missing (len 30).
    # Only seg0 counts. seg2 missing flag → treated as miss (conservative).
    segs = [(0, 100), (100, 150), (150, 180)]
    combine = [
        {"room": "R", "seg_index": 0},
        {"room": "R", "seg_index": 1},
        {"room": "R", "seg_index": 2},
    ]
    flags = {("R", 0): False, ("R", 1): True}  # False=hit, True=recomputed(miss)
    sched = _FakeSched(flags)
    req = _FakeReq(segs, combine)

    got = pic_combine_cached_tokens(sched, req)
    assert got == 100, got
    # consumed flags popped; missing one never inserted.
    assert sched._pic_scatter_seg_recomputed == {}, sched._pic_scatter_seg_recomputed
    print("OK cached_tokens counts only hits ->", got)


def test_cached_tokens_empty():
    from sglang.srt.pic.scatter_xfer import pic_combine_cached_tokens

    assert pic_combine_cached_tokens(_FakeSched({}), _FakeReq([], [])) == 0
    print("OK empty combine -> 0")


def test_notif_parse_contract():
    # Mirrors the drain_injects ingest branch; guards the wire format
    # {room}|{seg}|mamba|r{0,1} that _fire_write emits.
    def ingest(messages):
        seen, flags = set(), {}
        for m in messages:
            parts = m.split(b"|")
            if len(parts) == 4 and parts[2] == b"mamba" and parts[3][:1] == b"r":
                flags[(int(parts[0]), int(parts[1]))] = parts[3] == b"r1"
                seen.add(b"|".join(parts[:3]))
            else:
                seen.add(m)
        return seen, flags

    seen, flags = ingest([b"12345|3|mamba|r1", b"12345|3|kv", b"12345|2|mamba|r0"])
    assert flags == {(12345, 3): True, (12345, 2): False}, flags
    assert b"12345|3|mamba" in seen and b"12345|3|kv" in seen, seen
    assert b"12345|2|mamba" in seen, seen
    # base mamba tag (not the |r suffixed one) is what want_mamba matches.
    assert b"12345|3|mamba|r1" not in seen, seen
    print("OK notif parse contract")


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
