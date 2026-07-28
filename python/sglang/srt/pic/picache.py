"""PICache: segment-granularity position-independent cache for hybrid models.

See qianyou/2026-05-28-pic-sglang-design.md §5.4.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.evict_policy import EvictionStrategy, LRUStrategy
from sglang.srt.pic.policy import POLICIES, PICCompose
from sglang.srt.pic.segmenter import segment_hash
from sglang.srt.server_args import get_global_server_args


@dataclass
class SegmentEntry:
    """One cached segment's footprint.

    Field naming intentionally mirrors `mem_cache/radix_cache.py:TreeNode` so
    that the existing `EvictionStrategy` classes (LRUStrategy/FIFOStrategy/...)
    duck-type on us without modification.
    """
    seg_hash: bytes
    full_kv_slots: torch.Tensor          # int64, len == segment length
    mamba_state_slot: int                # MambaPool slot id
    token_ids: torch.Tensor              # for hash-collision fallback compare
    lock_ref: int = 0
    last_access_time: float = 0.0
    creation_time: float = 0.0
    hit_count: int = 0
    priority: int = 0


class PICache(BasePrefixCache):
    """Segment-granularity position-independent cache.

    Match semantics: any subset of pic_segments may hit, in any order;
    last segment is never cached.
    """

    def __init__(
        self,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        mamba_pool,
        page_size: int,
        disable: bool,
        pic_mode: str = "addition",
        enable_metrics: bool = False,
        is_decode: bool = False,
    ):
        assert pic_mode in POLICIES, f"unknown pic_mode={pic_mode!r}"
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.mamba_pool = mamba_pool
        # v0.5.14 split slot alloc/free out of MambaPool into MambaSlotAllocator.
        # PICache manages mamba slots directly, so route through the allocator.
        self.mamba_allocator = req_to_token_pool.mamba_allocator
        self.page_size = page_size
        self.disable = disable
        self.pic_mode = pic_mode
        self.policy = POLICIES[pic_mode]
        # decode role receives the whole prompt KV via PD (never through PIC
        # entries), so its cache_finished_req must free [0, committed_len).
        self.is_decode = is_decode

        self._entries: Dict[bytes, SegmentEntry] = {}
        self.eviction_strategy: EvictionStrategy = LRUStrategy()
        self._inflight_full_tokens: int = 0
        self._inflight_mamba_slots: int = 0

        if enable_metrics:
            self.init_metrics_collector()

    def supports_mamba(self) -> bool:
        return True

    def is_chunk_cache(self) -> bool:
        return False

    def reset(self) -> None:
        # `Scheduler.flush_cache` calls this immediately before clearing the
        # backing KV allocator and HybridReqToTokenPool/MambaPool wholesale.
        # Freeing thousands of segment entries one by one would repeatedly
        # grow allocator free lists via torch.cat, which is slow and can OOM
        # when HAPIC keeps many segment states. Drop the index here and let
        # the pools' clear() methods rebuild their free lists in one shot.
        self._entries.clear()
        self._inflight_full_tokens = 0
        self._inflight_mamba_slots = 0

    def evictable_size(self) -> int:
        return sum(
            int(e.full_kv_slots.numel())
            for e in self._entries.values()
            if e.lock_ref == 0
        )

    def protected_size(self) -> int:
        return sum(
            int(e.full_kv_slots.numel())
            for e in self._entries.values()
            if e.lock_ref > 0
        )

    def total_size(self) -> int:
        return sum(int(e.full_kv_slots.numel()) for e in self._entries.values())

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def mamba_evictable_size(self) -> int:
        return sum(1 for e in self._entries.values() if e.lock_ref == 0)

    def mamba_protected_size(self) -> int:
        return sum(1 for e in self._entries.values() if e.lock_ref > 0) + self._inflight_mamba_slots

    def full_protected_size(self) -> int:
        return self.protected_size() + self._inflight_full_tokens

    def add_inflight(self, num_tokens: int, num_mamba_slots: int) -> None:
        self._inflight_full_tokens += num_tokens
        self._inflight_mamba_slots += num_mamba_slots

    def remove_inflight(self, num_tokens: int, num_mamba_slots: int) -> None:
        self._inflight_full_tokens = max(0, self._inflight_full_tokens - num_tokens)
        self._inflight_mamba_slots = max(0, self._inflight_mamba_slots - num_mamba_slots)

    def supports_swa(self) -> bool:
        return False

    def swa_evictable_size(self) -> int:
        return 0

    def swa_protected_size(self) -> int:
        return 0

    def sanity_check(self) -> None:
        pass

    def all_values_flatten(self):
        if not self._entries:
            return torch.empty(0, dtype=torch.int64)
        return torch.cat([e.full_kv_slots for e in self._entries.values()])

    def all_mamba_values_flatten(self):
        if not self._entries:
            return torch.empty(0, dtype=torch.int64)
        return torch.tensor(
            [e.mamba_state_slot for e in self._entries.values()], dtype=torch.int64
        )

    def session_held_full_tokens(self) -> int:
        return 0

    def session_held_swa_tokens(self) -> int:
        return 0

    def session_held_tokens(self) -> int:
        return 0

    def session_held_req_count(self) -> int:
        return 0

    def pretty_print(self) -> None:
        import logging

        logging.getLogger(__name__).info(
            "PICache: %d entries, %d tokens cached (%d protected)",
            len(self._entries),
            self.total_size(),
            self.protected_size(),
        )

    def _match_segment(self, seg_hash: bytes, token_ids: torch.Tensor) -> Optional[SegmentEntry]:
        entry = self._entries.get(seg_hash)
        if entry is None:
            return None
        if entry.token_ids.shape != token_ids.shape or not bool(
            torch.equal(entry.token_ids, token_ids.to(entry.token_ids.dtype))
        ):
            return None
        entry.last_access_time = time.monotonic()
        entry.hit_count += 1
        return entry

    def _insert_segment(
        self,
        seg_hash: bytes,
        token_ids: torch.Tensor,
        full_kv_slots: torch.Tensor,
        mamba_state_slot: int,
    ) -> SegmentEntry:
        """Idempotent: if seg_hash already present (and matches token_ids), return existing entry.
        Caller is responsible for freeing its own (full_kv_slots, mamba_state_slot) when this happens.
        """
        existing = self._match_segment(seg_hash, token_ids)
        if existing is not None:
            return existing
        now = time.monotonic()
        entry = SegmentEntry(
            seg_hash=seg_hash,
            # Clone to own storage. Mirrors RadixCache convention
            # (radix_cache.py:767 `new_node.value = value.clone()`):
            # slot-index tensors stored in long-lived cache structures must
            # not be views of the allocator's free_pages buffer, or they
            # pin a fresh ~16 MiB stale storage per entry after the next
            # free() rotates self.free_pages via torch.cat.
            full_kv_slots=full_kv_slots.clone(),
            mamba_state_slot=mamba_state_slot,
            token_ids=token_ids.detach().clone(),
            lock_ref=0,
            last_access_time=now,
            creation_time=now,
        )
        self._entries[seg_hash] = entry
        return entry

    def inject_received_segment(self, seg_hash, token_ids, full_kv_slots, mamba_state_slot):
        existing = self._entries.get(seg_hash)
        if existing is not None:
            existing.lock_ref += 1
            return existing
        entry = self._insert_segment(seg_hash, token_ids, full_kv_slots, mamba_state_slot)
        entry.lock_ref += 1
        return entry

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Caller-writes; does NOT touch lock_ref. Mirrors RadixCache.match_prefix.
        Lock acquired later in schedule_policy._req_inc_lock_ref.
        """
        req = params.req
        device = (self.token_to_kv_pool_allocator.device
                  if hasattr(self.token_to_kv_pool_allocator, "device") else "cpu")
        empty_indices = torch.empty((0,), dtype=torch.int64, device=device)
        empty = MatchResult(
            device_indices=empty_indices,
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
            pic_segment_entries=None,
        )
        if self.disable or req is None or getattr(req, "pic_segments", None) is None or len(req.pic_segments) == 0:
            return empty

        segments = req.pic_segments
        per_seg: List[Optional[SegmentEntry]] = [None] * len(segments)

        # transition_rope_recompute: re-forward each hit segment's sink seam.
        is_recompute = self.policy.recompute
        if is_recompute:
            from sglang.srt.pic import SEAM_SINK_DEFAULT, resolve_seam_sink_tokens
            seam_sink = SEAM_SINK_DEFAULT
            req.pic_hit_seam_positions = {}

        for i, (start, end) in enumerate(segments):
            if i == len(segments) - 1:
                continue
            min_tokens = getattr(get_global_server_args(), "pic_segment_min_tokens", -1)
            if min_tokens > 0 and end - start < min_tokens:
                continue
            seg_token_ids = torch.tensor(req.origin_input_ids[start:end], dtype=torch.int64)
            h_bytes = segment_hash(seg_token_ids)
            entry = self._match_segment(h_bytes, seg_token_ids)
            if entry is not None:
                per_seg[i] = entry

        hit_slot_tensors: List[torch.Tensor] = []
        for i, entry in enumerate(per_seg):
            if entry is None:
                continue
            start, end = segments[i]
            if is_recompute and i > 0:
                seg_len = end - start
                # Keep one interior token so hit+seam still composes through pool.
                x_sink_eff = resolve_seam_sink_tokens(seam_sink, seg_len - 1)
                if x_sink_eff > 0:
                    hit_slot_tensors.append(entry.full_kv_slots[x_sink_eff:])
                    req.pic_hit_seam_positions[(start, end)] = list(
                        range(start, start + x_sink_eff)
                    )
                    continue
            hit_slot_tensors.append(entry.full_kv_slots)

        if hit_slot_tensors:
            device_indices = torch.cat(hit_slot_tensors).to(device)
        else:
            device_indices = empty_indices

        num_hit = sum(1 for e in per_seg if e is not None)
        num_miss = sum(1 for e in per_seg if e is None) - 1  # last seg always None
        hit_tokens = sum(e.full_kv_slots.numel() for e in per_seg if e is not None)
        logger.info(
            "PICache.match_prefix: %d segments, %d hit (%d tokens), %d miss, cache_size=%d entries",
            len(segments), num_hit, hit_tokens, num_miss, len(self._entries),
        )

        return MatchResult(
            device_indices=device_indices,
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
            pic_segment_entries=per_seg,
        )

    def cache_unfinished_req(self, req, **kwargs) -> None:
        """Forward completed: register miss segments in the index.

        Pre-condition (set by pic_alloc_for_extend):
          addition/transition:   req.pic_miss_segment_slots[(s,e)] = (kv_slots, mamba)
          transition_rope:       req.pic_miss_segment_slots[(s,e)] = (private, public, mamba)
                                 (public is None only for the last/global segment)

        For transition/transition_rope: lock_ref handoff (entries get lock_ref++).
        For transition_rope: entry.full_kv_slots = public (already pos=0 by
        construction — the forward Phase A writes pos=0 K into public).
        """
        miss_segments: List = getattr(req, "pic_miss_segments", []) or []
        miss_slots: Dict = getattr(req, "pic_miss_segment_slots", {}) or {}
        if getattr(req, "pic_full_recompute", False):
            req.pic_cache_owned_miss_segments = set()
            req.pic_freed_miss_segments = set()
            return
        is_transition = self.policy.compose is PICCompose.TRANSITION
        is_rope = self.policy.rope

        inserted = 0
        inflight_tokens = 0
        inflight_mamba = 0
        req.pic_cache_owned_miss_segments = set()
        req.pic_freed_miss_segments = set()
        min_tokens = getattr(get_global_server_args(), "pic_segment_min_tokens", -1)
        for (start, end) in miss_segments:
            # ponytail: the last segment (Q) is normally not cached (recomputed by
            # decode). A scatter single-seg sub-request has exactly one segment that
            # IS the "last" one but MUST be cached + pushed to combine — don't skip.
            if (
                req.pic_segments
                and len(req.pic_segments) > 1
                and (start, end) == req.pic_segments[-1]
                and not getattr(req, "pic_scatter_single_seg", False)
            ):
                continue
            if min_tokens > 0 and end - start < min_tokens:
                continue
            # ponytail: on the PD-decode side the req carries pic_miss_segments
            # (from tokenization) but no locally-allocated slots — it received the
            # full KV via PD transfer and must not re-cache segments. Skip those.
            if (start, end) not in miss_slots:
                continue
            seg_ids = torch.tensor(req.origin_input_ids[start:end], dtype=torch.int64)
            seg_hash = segment_hash(seg_ids)
            slot_tuple = miss_slots[(start, end)]
            if is_rope:
                _private_slots, public_slots, mamba_slot = slot_tuple
                assert public_slots is not None, (
                    "local miss segment must have a public slot in rope PIC modes"
                )
                kv_slots = public_slots
            else:
                kv_slots, mamba_slot = slot_tuple
            assert mamba_slot is not None, (
                "cacheable miss segment must have a mamba slot"
            )
            existing = self._match_segment(seg_hash, seg_ids)
            if existing is not None:
                self.token_to_kv_pool_allocator.free(kv_slots)
                self.mamba_allocator.free(torch.tensor([mamba_slot], dtype=torch.int64, device=kv_slots.device))
                req.pic_segment_entries[seg_hash] = existing
                req.pic_freed_miss_segments.add((start, end))
            else:
                entry = self._insert_segment(seg_hash, seg_ids, kv_slots, mamba_slot)
                req.pic_segment_entries[seg_hash] = entry
                req.pic_cache_owned_miss_segments.add((start, end))
                inserted += 1

                # v2 transition fix: protect newly inserted entry until req finishes
                if is_transition:
                    entry.lock_ref += 1

            inflight_tokens += kv_slots.numel()
            inflight_mamba += 1

        # Transition mode: inflight -> lock_ref handoff (for the slots now owned
        # by entries — i.e. public-slot inflight for rope, kv-slot inflight for
        # plain transition). Private-slot inflight (rope only) is removed by
        # cache_finished_req when private slots are freed.
        if is_transition and (inflight_tokens > 0 or inflight_mamba > 0):
            self.remove_inflight(inflight_tokens, inflight_mamba)

        if inserted > 0:
            logger.info(
                "PICache.cache_unfinished_req: inserted %d new segments, total cache=%d entries",
                inserted, len(self._entries),
            )

    def cache_finished_req(self, req, is_insert: bool = True, **kwargs) -> None:
        miss_slots = getattr(req, "pic_miss_segment_slots", None)
        if miss_slots:
            is_transition = self.policy.compose is PICCompose.TRANSITION
            is_rope = self.policy.rope
            if getattr(req, "pic_full_recompute", False):
                assert not is_transition and not is_rope
                all_kv = torch.cat([slots for slots, _mamba in miss_slots.values()])
                self.token_to_kv_pool_allocator.free(all_kv)
                self.remove_inflight(all_kv.numel(), 0)
                miss_slots = {}
            if is_insert:
                if miss_slots:
                    self.cache_unfinished_req(req)
            if not is_transition:
                # addition: free inflight against the single miss kv_slots tuple.
                if is_rope:
                    inflight_tokens = sum(
                        priv.numel() + (0 if pub is None else pub.numel())
                        for priv, pub, _mamba in miss_slots.values()
                    )
                    inflight_mamba = sum(
                        1 for _priv, _pub, mamba in miss_slots.values()
                        if mamba is not None
                    )
                else:
                    inflight_tokens = sum(
                        slots.numel() for slots, _mamba in miss_slots.values()
                    )
                    inflight_mamba = sum(
                        1 for _slots, mamba in miss_slots.values()
                        if mamba is not None
                    )
                self.remove_inflight(inflight_tokens, inflight_mamba)
            # Free last segment's slots (never cached by design).
            pic_segments = getattr(req, "pic_segments", None)
            if miss_slots and pic_segments and len(pic_segments) > 1:
                last_seg = tuple(pic_segments[-1])
                if last_seg in miss_slots:
                    if is_rope:
                        # global: (private, None, mamba)
                        last_priv, last_pub, last_mamba = miss_slots[last_seg]
                        assert last_pub is None
                        self.token_to_kv_pool_allocator.free(last_priv)
                        last_dev = last_priv.device
                    else:
                        last_kv, last_mamba = miss_slots[last_seg]
                        self.token_to_kv_pool_allocator.free(last_kv)
                        last_dev = last_kv.device
                    if last_mamba is not None:
                        self.mamba_allocator.free(
                            torch.tensor([last_mamba], dtype=torch.int64,
                                         device=last_dev)
                        )
                    if is_transition and not is_rope:
                        self.remove_inflight(
                            last_kv.numel(),
                            1 if last_mamba is not None else 0,
                        )

            if miss_slots and not is_rope:
                min_tokens = getattr(
                    get_global_server_args(), "pic_segment_min_tokens", -1
                )
                cache_owned = getattr(req, "pic_cache_owned_miss_segments", set())
                already_freed = getattr(req, "pic_freed_miss_segments", set())
                pic_segments = getattr(req, "pic_segments", None)
                last_seg = tuple(pic_segments[-1]) if pic_segments else None
                skipped_slots = []
                skipped_mamba_slots = []
                skipped_tokens = 0
                for (start, end), (kv_slots, mamba_slot) in miss_slots.items():
                    seg = (start, end)
                    if seg == last_seg or seg in cache_owned or seg in already_freed:
                        continue
                    if min_tokens <= 0 or end - start >= min_tokens:
                        continue
                    skipped_slots.append(kv_slots)
                    skipped_tokens += int(kv_slots.numel())
                    if mamba_slot is not None:
                        skipped_mamba_slots.append(mamba_slot)
                if skipped_slots:
                    combined = torch.cat(skipped_slots)
                    self.token_to_kv_pool_allocator.free(combined)
                    if skipped_mamba_slots:
                        self.mamba_allocator.free(
                            torch.tensor(
                                skipped_mamba_slots,
                                dtype=torch.int64,
                                device=combined.device,
                            )
                        )
                    if is_transition:
                        self.remove_inflight(
                            skipped_tokens, len(skipped_mamba_slots)
                        )

            # transition_rope: free ALL miss private slots (local miss private was
            # never registered to an entry; global private was just freed above).
            # Also remove the matching inflight tokens.
            if is_rope:
                priv_to_free = []
                global_seg = (
                    tuple(pic_segments[-1])
                    if pic_segments and len(pic_segments) > 1
                    else None
                )
                for (start, end), (priv, _pub, _mamba) in miss_slots.items():
                    if (start, end) == global_seg:
                        continue  # already freed above
                    priv_to_free.append(priv)
                if priv_to_free:
                    combined = torch.cat(priv_to_free)
                    self.token_to_kv_pool_allocator.free(combined)
                    self.remove_inflight(combined.numel(), 0)
                # And free the global private inflight count (its slot was
                # freed above but inflight tokens still need decrementing).
                if global_seg is not None and global_seg in miss_slots:
                    gp = miss_slots[global_seg][0]
                    self.remove_inflight(gp.numel(), 1)  # +1 for global mamba

        # transition_rope[_recompute]: free private hit slots (rerotated copies, not cached).
        if self.policy.rope:
            hit_private = getattr(req, "pic_rope_hit_private_slots", None)
            if hit_private:
                all_private = []
                for (private_slots, _entry_slots) in hit_private.values():
                    all_private.append(private_slots)
                if all_private:
                    combined = torch.cat(all_private)
                    self.token_to_kv_pool_allocator.free(combined)
                    self.remove_inflight(combined.numel(), 0)

        # Free decode-generated KV slots: [prompt_end, kv_committed_len).
        # PIC's cache_unfinished_req only registers miss segments (prompt
        # ranges) into entries; tokens generated during decode are allocated
        # by alloc_for_decode -> alloc_token_slots but never inserted into
        # any cache entry, so nothing else will free them. Without this,
        # every PIC request leaks (kv_committed_len - prompt_end) slots.
        pic_segments = getattr(req, "pic_segments", None)
        if pic_segments and req.req_pool_idx is not None:
            prompt_end = pic_segments[-1][1]
            committed_len = getattr(req, "kv_committed_len", prompt_end)
            # decode-side prebuilt req: prompt KV [0, prompt_end) arrived via
            # PD, never registered into PIC entries, so nothing else frees it —
            # free the full req range. prefill/combine keep prompt_end (their
            # prompt KV is freed via miss_slots/entries above).
            free_start = 0 if self.is_decode else prompt_end
            if committed_len > free_start:
                decode_slots = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, free_start:committed_len
                ]
                self.token_to_kv_pool_allocator.free(decode_slots)

        # Free the per-request mamba_pool_idx allocated by
        # HybridReqToTokenPool.alloc for the decode-time recurrent state.
        # release_kv_cache skips this branch for caches with
        # supports_mamba()=True (PIC), so we must free it ourselves.
        if (
            getattr(req, "mamba_pool_idx", None) is not None
            and hasattr(self.req_to_token_pool, "free_mamba_cache")
        ):
            self.req_to_token_pool.free_mamba_cache(req)

        self.dec_lock_ref(req)

    def evict(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        need_tokens = params.num_tokens
        need_mamba = params.mamba_num
        if need_tokens <= 0 and need_mamba <= 0:
            return EvictResult()
        candidates = [e for e in self._entries.values() if e.lock_ref == 0]
        candidates.sort(key=lambda e: self.eviction_strategy.get_priority(e))
        evicted_tokens = 0
        evicted_mamba = 0
        for e in candidates:
            if evicted_tokens >= need_tokens and evicted_mamba >= need_mamba:
                break
            self.token_to_kv_pool_allocator.free(e.full_kv_slots)
            self.mamba_allocator.free(torch.tensor([e.mamba_state_slot], dtype=torch.int64,
                                             device=e.full_kv_slots.device))
            evicted_tokens += int(e.full_kv_slots.numel())
            evicted_mamba += 1
            del self._entries[e.seg_hash]
        self.update_eviction_metrics(evicted_tokens, start_time)
        return EvictResult(
            num_tokens_evicted=evicted_tokens,
            mamba_num_evicted=evicted_mamba,
        )

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        entries = getattr(node, "pic_segment_entries", None) or {}
        delta = 0
        for e in entries.values():
            e.lock_ref += 1
            delta += 1
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        entries = getattr(node, "pic_segment_entries", None) or {}
        delta = 0
        for e in entries.values():
            if e.lock_ref > 0:
                e.lock_ref -= 1
                delta -= 1
        return DecLockRefResult(delta=delta)
