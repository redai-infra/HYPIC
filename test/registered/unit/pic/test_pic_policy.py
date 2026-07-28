from sglang.srt.pic.policy import POLICIES, PICCompose, PICPolicy, resolve_policy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_four_aliases_map_to_expected_attrs():
    assert POLICIES["addition"] == PICPolicy(
        PICCompose.ADDITION, rope=False, recompute=False
    )
    assert POLICIES["transition"] == PICPolicy(
        PICCompose.TRANSITION, rope=False, recompute=False
    )
    assert POLICIES["transition_rope"] == PICPolicy(
        PICCompose.TRANSITION, rope=True, recompute=False
    )
    assert POLICIES["transition_rope_recompute"] == PICPolicy(
        PICCompose.TRANSITION, rope=True, recompute=True
    )


def test_resolve_policy_known_and_unknown():
    assert resolve_policy("transition_rope") is POLICIES["transition_rope"]
    try:
        resolve_policy("bogus")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown mode should raise KeyError")


def test_derived_predicates_replace_string_checks():
    # addition check
    assert POLICIES["addition"].compose is PICCompose.ADDITION
    assert POLICIES["transition"].compose is not PICCompose.ADDITION
    # transition (non-addition) check
    for m in ("transition", "transition_rope", "transition_rope_recompute"):
        assert POLICIES[m].compose is PICCompose.TRANSITION
    # rope check (== old _ROPE_PIC_MODES membership)
    assert (
        POLICIES["transition_rope"].rope and POLICIES["transition_rope_recompute"].rope
    )
    assert not POLICIES["addition"].rope and not POLICIES["transition"].rope
    # mamba_idx = 2 if rope else 1
    assert (2 if POLICIES["transition_rope"].rope else 1) == 2
    assert (2 if POLICIES["transition"].rope else 1) == 1
    # is_recompute
    assert POLICIES["transition_rope_recompute"].recompute
    assert not POLICIES["transition_rope"].recompute


def test_policy_is_frozen_and_hashable():
    p = POLICIES["transition"]
    assert hash(p) == hash(
        PICPolicy(PICCompose.TRANSITION, rope=False, recompute=False)
    )
    try:
        p.rope = True
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("PICPolicy should be frozen")


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
