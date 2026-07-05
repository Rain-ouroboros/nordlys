import numpy as np
from nordlys.generator import MusicGenerator


def _gen():
    # small block for test speed: 0.25 s at 8 kHz
    return MusicGenerator(sr=8000, block=2000, seed=11, hour_fn=lambda: 12)


def test_blocks_valid_and_nonsilent():
    g = _gen()
    peak = 0.0
    for _ in range(60):
        b = g.next_block()
        assert b.shape == (2, 2000) and b.dtype == np.float32
        assert np.isfinite(b).all() and np.abs(b).max() <= 1.0
        peak = max(peak, float(np.abs(b).max()))
    assert peak > 0.02


def test_deterministic_given_seed():
    a, b = _gen(), _gen()
    for _ in range(10):
        assert np.array_equal(a.next_block(), b.next_block())


def test_mood_and_skip_change_status():
    g = _gen()
    g.next_block()
    g.set_mood(brightness=0.9)
    assert g.status()["mood"]["target"]["brightness"] == 0.9
    before = g.status()["scene"]
    g.skip()
    for _ in range(200):
        g.next_block()
    assert g.status()["scene"] != before or "→" in g.status()["scene"]


def test_status_shape():
    g = _gen()
    g.next_block()
    st = g.status()
    assert {"mood", "scene", "key", "chord", "blocks"} <= set(st)
