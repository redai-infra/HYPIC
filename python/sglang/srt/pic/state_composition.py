from collections.abc import Mapping, Sequence

import torch


def build_addition_prefix_states(
    *,
    segment_states: torch.Tensor,
    state_pool: torch.Tensor,
    hit_segments_per_request: Sequence[Sequence[tuple]],
    hit_slots_per_request: Sequence[Mapping],
    miss_segments_per_request: Sequence[Sequence[tuple[int, int]]],
    segment_offsets: Sequence[int],
    out: torch.Tensor,
    request_suffix_states: torch.Tensor,
) -> torch.Tensor:
    """Build the additive prefix state entering each computed segment."""
    out.zero_()
    request_suffix_states.zero_()
    for req_idx, hit_slots in enumerate(hit_slots_per_request):
        start = segment_offsets[req_idx]
        end = segment_offsets[req_idx + 1]
        if start == end:
            continue

        steps = [
            (segment[0], "hit", segment[2])
            for segment in hit_segments_per_request[req_idx]
        ]
        steps.extend(
            (segment[0], "miss", miss_idx)
            for miss_idx, segment in enumerate(miss_segments_per_request[req_idx])
        )
        steps.sort(key=lambda step: step[0])

        accum = out.new_zeros(out.shape[1:])
        after_last_miss = False
        for _position, kind, payload in steps:
            if kind == "hit":
                hit_state = state_pool[hit_slots[payload]].to(out.dtype)
                if after_last_miss:
                    request_suffix_states[req_idx].add_(hit_state)
                else:
                    accum.add_(hit_state)
            else:
                segment_idx = start + payload
                out[segment_idx].copy_(accum)
                accum.add_(segment_states[segment_idx])
                after_last_miss = segment_idx == end - 1
    return out
