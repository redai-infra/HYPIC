"""PIC variant of mem_cache/common.py:alloc_for_extend.

Allocates KV slots for miss segments, copies hit segment slots from cache,
writes req_to_token mapping, and tracks inflight accounting.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import alloc_req_slots, alloc_token_slots
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


def pic_alloc_for_extend(batch) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """PIC variant of alloc_for_extend.

    For each request:
      - Allocate a req_pool slot via the standard path.
      - For each hit segment, write the cached entry's full_kv_slots into
        req_to_token[req_idx, start:end].
      - For each miss segment, allocate a chunk of token slots and a mamba slot,
        record them on req.pic_miss_segment_slots[(start, end)], and write the
        token slots into req_to_token[req_idx, start:end].

    Returns (out_cache_loc, req_pool_indices_device, req_pool_indices).
    """
    tree_cache = batch.tree_cache
    policy = getattr(tree_cache, "policy", None)
    if policy is not None and policy.rope:
        return _pic_alloc_transition_rope(batch)
    return _pic_alloc_standard(batch)


def _pic_alloc_standard(batch) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Standard PIC alloc (addition/transition modes).

    Hit segments -> entry.full_kv_slots written to req_to_token.
    Miss segments -> freshly allocated slots.
    """
    tree_cache = batch.tree_cache
    device = batch.device

    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(device, non_blocking=True)

    total_miss = sum(
        sum(end - start for (start, end) in r.pic_miss_segments)
        for r in batch.reqs
    )
    out_cache_loc = alloc_token_slots(tree_cache, total_miss)

    mamba_pool = batch.req_to_token_pool.mamba_allocator
    min_tokens = getattr(get_global_server_args(), "pic_segment_min_tokens", -1)
    cursor = 0
    total_mamba_alloc = 0
    for i, req in enumerate(batch.reqs):
        req.pic_miss_segment_slots = {}
        req_idx = req_pool_indices[i]
        pic_segments = getattr(req, "pic_segments", None)
        last_seg = tuple(pic_segments[-1]) if pic_segments else None
        for hit in req.pic_hit_segments:
            start, end, seg_hash = hit
            entry = req.pic_segment_entries[seg_hash]
            batch.req_to_token_pool.write(
                (req_idx, slice(start, end)),
                entry.full_kv_slots.to(device),
            )
        for (start, end) in req.pic_miss_segments:
            n = end - start
            # .clone() so slot_chunk owns its storage. Mirrors RadixCache
            # (radix_cache.py:485 `.to(..., copy=True)`): slot tensors that
            # will be stashed on the request and later handed to PICache
            # must not be views into out_cache_loc / allocator.free_pages.
            slot_chunk = out_cache_loc[cursor:cursor + n].clone()
            cursor += n
            cacheable_local = not (
                (start, end) != last_seg
                and min_tokens > 0
                and n < min_tokens
            )
            mamba_slot = (
                _alloc_one_mamba(mamba_pool, tree_cache)
                if cacheable_local
                and not getattr(req, "pic_full_recompute", False)
                else None
            )
            req.pic_miss_segment_slots[(start, end)] = (slot_chunk, mamba_slot)
            if mamba_slot is not None:
                total_mamba_alloc += 1
            batch.req_to_token_pool.write(
                (req_idx, slice(start, end)),
                slot_chunk,
            )

    tree_cache.add_inflight(total_miss, total_mamba_alloc)
    # 3rd value is the CPU tensor mirror (v0.5.14 alloc_for_extend contract:
    # req_pool_indices_cpu), not the raw list — merge_batch torch.cat needs a tensor.
    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def _pic_alloc_transition_rope(batch) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """PIC alloc for transition_rope mode (per-segment isolated semantics).

    Slot policy:
      - LOCAL miss segment (non-last): allocate PRIVATE + PUBLIC.
          PRIVATE holds K @ real position (decode + global attention read it).
          PUBLIC holds K @ pos=0 (registered as entry.full_kv_slots for cross-req reuse).
      - GLOBAL miss segment (the last segment): allocate PRIVATE only.
          Last segment is never cached, so no public slot.
      - HIT segment: allocate PRIVATE only.
          Forward will rerotate entry.full_kv_slots (public, pos=0) into PRIVATE.

    req_to_token always points to PRIVATE slots (decode reads private,
    real-pos K, just like baseline path).

    out_cache_loc is the per-token private slot ordered to match input_ids.
    The model writes pre-rope-rotated q/k/v there; Phase A will overwrite
    those private K slots with the proper real-pos K (after Phase B rerotate).

    Records:
      req.pic_miss_segment_slots = {(start, end): (private, public_or_None, mamba)}
      req.pic_rope_hit_private_slots = {(start, end): (private, entry_public)}
    """
    tree_cache = batch.tree_cache
    device = batch.device

    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, tree_cache
    )
    req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_device = req_pool_indices_cpu.to(device, non_blocking=True)

    # Per-token counts. For miss: private (1) + (public if non-last) (1).
    # For hit: private (1).
    total_miss_priv = sum(
        sum(end - start for (start, end) in r.pic_miss_segments)
        for r in batch.reqs
    )
    total_local_miss_pub = 0
    for r in batch.reqs:
        miss = r.pic_miss_segments
        if not miss:
            continue
        # In multi-segment prompts, the last segment is the global/query segment
        # and is not cached. A single-segment warmup is cacheable.
        last_seg = (
            tuple(r.pic_segments[-1])
            if r.pic_segments and len(r.pic_segments) > 1
            else None
        )
        for (start, end) in miss:
            if (start, end) != last_seg:
                total_local_miss_pub += (end - start)
    total_hit = sum(
        sum(end - start for (start, end, _) in r.pic_hit_segments)
        for r in batch.reqs
    )
    total_alloc = total_miss_priv + total_local_miss_pub + total_hit
    all_slots = alloc_token_slots(tree_cache, total_alloc)

    # Pool layout: [miss_private | local_miss_public | hit_private]
    miss_priv_pool = all_slots[:total_miss_priv]
    pub_pool = all_slots[total_miss_priv:total_miss_priv + total_local_miss_pub]
    hit_priv_pool = all_slots[total_miss_priv + total_local_miss_pub:]
    # transition_rope_recompute: out_cache_loc must align with input_ids order
    # (real-pos sorted, includes both miss tokens and hit-seg seam tokens).
    # Build a per-token slot list later; default to miss_priv_pool for the
    # plain transition_rope path.
    _policy = getattr(tree_cache, "policy", None)
    is_recompute = _policy is not None and _policy.recompute
    if not is_recompute:
        out_cache_loc = miss_priv_pool

    mamba_pool = batch.req_to_token_pool.mamba_allocator
    miss_cursor = 0
    pub_cursor = 0
    hit_cursor = 0
    total_mamba_alloc = 0
    # For recompute mode: build per-req per-token slot map keyed by absolute pos.
    per_req_slot_at_pos: List[Dict[int, torch.Tensor]] = []
    for i, req in enumerate(batch.reqs):
        req.pic_miss_segment_slots = {}
        req.pic_rope_hit_private_slots = {}  # {(start, end): (private, entry_public)}
        req_idx = req_pool_indices[i]
        last_seg = (
            tuple(req.pic_segments[-1])
            if req.pic_segments and len(req.pic_segments) > 1
            else None
        )
        slot_at_pos: Dict[int, torch.Tensor] = {}

        # Hit segments: allocate PRIVATE; forward rerotates entry.public into it.
        for hit in req.pic_hit_segments:
            start, end, seg_hash = hit
            entry = req.pic_segment_entries[seg_hash]
            n = end - start
            private_slots = hit_priv_pool[hit_cursor:hit_cursor + n].clone()
            hit_cursor += n
            req.pic_rope_hit_private_slots[(start, end)] = (
                private_slots,
                entry.full_kv_slots.to(device),
            )
            batch.req_to_token_pool.write(
                (req_idx, slice(start, end)),
                private_slots,
            )
            # Recompute: hit-seg seam tokens re-enter forward; map abs pos to slot.
            if is_recompute:
                seam = getattr(req, "pic_hit_seam_positions", {}).get((start, end))
                if seam is not None:
                    for ap in seam:
                        ofs = int(ap) - start
                        slot_at_pos[ap] = private_slots[ofs:ofs + 1]

        # Miss segments: PRIVATE always; PUBLIC only for non-last.
        for (start, end) in req.pic_miss_segments:
            n = end - start
            # .clone() so the stashed slot tensor owns its storage (mirror of
            # _pic_alloc_standard rationale: don't pin allocator.free_pages).
            private_slots = miss_priv_pool[miss_cursor:miss_cursor + n].clone()
            miss_cursor += n
            if (start, end) != last_seg:
                public_slots = pub_pool[pub_cursor:pub_cursor + n].clone()
                pub_cursor += n
            else:
                public_slots = None
            mamba_slot = _alloc_one_mamba(mamba_pool, tree_cache)
            req.pic_miss_segment_slots[(start, end)] = (
                private_slots, public_slots, mamba_slot,
            )
            total_mamba_alloc += 1
            batch.req_to_token_pool.write(
                (req_idx, slice(start, end)),
                private_slots,
            )
            if is_recompute:
                for ofs in range(n):
                    slot_at_pos[start + ofs] = private_slots[ofs:ofs + 1]

        per_req_slot_at_pos.append(slot_at_pos)

    if is_recompute:
        # Build out_cache_loc by walking pic_miss_token_positions (real-pos
        # sorted) per req, looking up slot_at_pos.
        out_chunks: List[torch.Tensor] = []
        for i, req in enumerate(batch.reqs):
            pos = req.pic_miss_token_positions
            if pos is None:
                continue
            slot_at_pos = per_req_slot_at_pos[i]
            slots = [slot_at_pos[int(p)] for p in pos.tolist()]
            if slots:
                out_chunks.append(torch.cat(slots))
        out_cache_loc = (
            torch.cat(out_chunks)
            if out_chunks
            else torch.empty((0,), dtype=miss_priv_pool.dtype, device=miss_priv_pool.device)
        )

    tree_cache.add_inflight(total_alloc, total_mamba_alloc)
    logger.debug(
        "pic_alloc_transition_rope: priv=%d pub=%d hit_priv=%d (total=%d)",
        total_miss_priv, total_local_miss_pub, total_hit, total_alloc,
    )
    # 3rd value is the CPU tensor mirror (v0.5.14 alloc_for_extend contract:
    # req_pool_indices_cpu), not the raw list — merge_batch torch.cat needs a tensor.
    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


def _alloc_one_mamba(mamba_pool, tree_cache) -> int:
    """Return a single mamba slot id (int).

    MambaPool.alloc returns None when the pool is exhausted. Mirror the
    alloc_token_slots → evict_from_tree_cache pattern: if alloc fails,
    evict one mamba slot from PICache and retry. Raises RuntimeError if
    eviction makes no progress (all entries lock_ref>0).
    """
    res = mamba_pool.alloc(1)
    if res is None:
        evicted = tree_cache.evict(EvictParams(num_tokens=0, mamba_num=1))
        if evicted.mamba_num_evicted <= 0:
            raise RuntimeError(
                f"mamba_pool exhausted and PICache cannot evict "
                f"(available={mamba_pool.available_size()}, "
                f"cache_size={len(getattr(tree_cache, '_entries', {}))})"
            )
        res = mamba_pool.alloc(1)
        if res is None:
            raise RuntimeError("mamba_pool.alloc failed even after eviction")
    if isinstance(res, torch.Tensor):
        return int(res[0].item())
    if hasattr(res, "__getitem__"):
        return int(res[0])
    return int(res)
