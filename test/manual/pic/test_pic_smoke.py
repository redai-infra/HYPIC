"""End-to-end smoke test for PIC v1.

Requires H20 and a running sglang server.
Run via: bash qianyou/run_pic_smoke.sh

Currently blocked: T20's linear-attn lapic_addition path raises
NotImplementedError. See qianyou/2026-05-28-pic-sglang-plan.md task 20.
"""

import os
import time

import pytest
import requests

MODEL = "/root/qianyou/models/Qwen3.5-35B-A3B"
SEP = "<<PIC_SEP>>"
SYS = "You are a helpful assistant."
C1 = "Doc1: The animal in doc1 is dog. " * 200
C2 = "Doc2: The animal in doc2 is cat. " * 200
C3 = "Doc3: This is a distractor. " * 200
Q = " Based on the documents, what is the animal in doc2? Answer in one word:"
PROMPT = f"{SYS}{SEP}{C1}{SEP}{C2}{SEP}{C3}{SEP}{Q}"

BASELINE_URL = os.environ.get("PIC_BASELINE_URL", "http://localhost:30000")
PIC_URL = os.environ.get("PIC_PIC_URL", "http://localhost:30001")


def _generate(url: str, prompt: str, max_tokens: int = 8):
    t0 = time.perf_counter()
    r = requests.post(
        f"{url}/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens},
        },
        timeout=120,
    )
    r.raise_for_status()
    elapsed = time.perf_counter() - t0
    body = r.json()
    text = body.get("text", "")
    meta = body.get("meta_info", {})
    return text, elapsed, meta


def _warmup(url: str):
    for prompt in [
        f"{SYS}{SEP}{C1}{SEP}{Q}",
        f"{SYS}{SEP}{C3}{SEP}{Q}",
    ]:
        _generate(url, prompt, max_tokens=4)


@pytest.mark.skipif(
    os.environ.get("PIC_RUN_SMOKE") != "1",
    reason="set PIC_RUN_SMOKE=1 and have baseline+PIC servers running",
)
def test_pic_argmax_matches_baseline_and_latency_scales():
    _warmup(BASELINE_URL)
    _warmup(PIC_URL)
    baseline_text, baseline_lat, _ = _generate(BASELINE_URL, PROMPT)
    pic_text, pic_lat, _ = _generate(PIC_URL, PROMPT)
    assert (
        pic_text == baseline_text
    ), f"argmax diverged:\n  baseline={baseline_text!r}\n  pic={pic_text!r}"
    assert (
        pic_lat < baseline_lat * 0.6
    ), f"PIC latency not reduced enough: pic={pic_lat:.3f}s baseline={baseline_lat:.3f}s"
