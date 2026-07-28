import dataclasses

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_FIELDS = {"pic_scatter_single_seg", "pic_scatter_meta", "pic_combine"}


def test_tokenized_req_has_scatter_fields():
    f = {x.name for x in dataclasses.fields(TokenizedGenerateReqInput)}
    assert _FIELDS <= f


def test_generate_req_has_scatter_fields():
    f = {x.name for x in dataclasses.fields(GenerateReqInput)}
    assert _FIELDS <= f


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
