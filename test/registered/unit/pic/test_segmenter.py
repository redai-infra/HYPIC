from sglang.srt.pic.segmenter import split_and_tokenize
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeTokenizer:
    """Whitespace tokenizer; each token = one int."""

    def encode(self, text, add_special_tokens=False):
        return [abs(hash(t)) % 1000 + 1 for t in text.split()]


def test_no_sep_yields_one_segment():
    tk = FakeTokenizer()
    ids, offs = split_and_tokenize("hello world", tk, separator="<<SEP>>")
    assert len(offs) == 1
    assert offs[0] == (0, len(ids))
    assert ids == tk.encode("hello world")


def test_sep_splits_and_offsets_are_concat_local():
    tk = FakeTokenizer()
    text = "sys here <<SEP>> doc one body <<SEP>> q stuff"
    ids, offs = split_and_tokenize(text, tk, separator="<<SEP>>")
    a = tk.encode("sys here")
    b = tk.encode("doc one body")
    c = tk.encode("q stuff")
    assert ids == a + b + c
    assert offs == [
        (0, len(a)),
        (len(a), len(a) + len(b)),
        (len(a) + len(b), len(a) + len(b) + len(c)),
    ]


def test_sep_string_never_in_ids():
    tk = FakeTokenizer()
    ids, offs = split_and_tokenize("a <<SEP>> b", tk, separator="<<SEP>>")
    sep_tok = tk.encode("<<SEP>>")
    assert all(t not in sep_tok for t in ids) or sep_tok == []


def test_empty_segments_skipped():
    tk = FakeTokenizer()
    text = "a <<SEP>>  <<SEP>> b"
    ids, offs = split_and_tokenize(text, tk, separator="<<SEP>>")
    assert all(end > start for start, end in offs)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
