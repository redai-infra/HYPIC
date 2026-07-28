import logging

import pytest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_pic_flags_exist_with_defaults():
    args = ServerArgs(model_path="dummy")
    assert args.pic_enable is False
    assert args.pic_separator_str == "<<PIC_SEP>>"
    assert args.pic_mode == "addition"
    assert args.pic_segment_min_tokens == -1


def test_pic_enable_warns_for_unknown_model_family(caplog):
    args = ServerArgs(model_path="dummy")
    args.model_path = "meta-llama/Llama-3-8B"
    args.pic_enable = True
    args.chunked_prefill_size = -1
    with caplog.at_level(logging.WARNING):
        args.check_pic_constraints()
    assert "does not match known PIC-capable families" in caplog.text


def test_pic_enable_requires_chunked_prefill_disabled():
    with pytest.raises(AssertionError, match="chunked_prefill_size"):
        ServerArgs(
            model_path="Qwen/Qwen3.5-35B-A3B",
            pic_enable=True,
            chunked_prefill_size=2048,
        )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
