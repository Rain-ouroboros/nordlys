import shutil
import time

import numpy as np
import pytest

from nordlys.encoder import Broadcast, Mp3Encoder


def test_broadcast_fanout_and_drop():
    b = Broadcast()
    i1, q1 = b.add()
    q1_size = q1.maxsize
    for k in range(q1_size + 10):
        b.publish(bytes([k % 256]))
    assert b.count == 1
    assert q1.qsize() == q1_size            # dropped oldest, never blocked
    first = q1.get_nowait()
    assert first == bytes([10])             # 10 oldest dropped
    b.remove(i1)
    assert b.count == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_encoder_produces_mp3():
    b = Broadcast()
    _, q = b.add()
    enc = Mp3Encoder(44100, b)
    t = np.arange(44100) / 44100
    block = np.stack([np.sin(2 * np.pi * 220 * t)] * 2).astype(np.float32) * 0.5
    data = b""
    for _ in range(4):
        enc.feed(block)
    deadline = time.time() + 10
    while len(data) < 8000 and time.time() < deadline:
        try:
            data += q.get(timeout=1.0)
        except Exception:
            pass
    enc.close()
    assert len(data) >= 8000
    assert b"\xff" in data[:4000]           # mp3 frame sync bytes present
