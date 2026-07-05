import numpy as np
from nordlys.mixer import Ducker

SR = 8000


def test_ducks_under_voice_and_releases():
    d = Ducker(SR, attack=0.05, release=0.2)
    music = np.ones((2, SR), np.float32) * 0.5
    voice = np.zeros((2, SR), np.float32)
    voice[:, : SR // 2] = 0.5
    out = d.process(music, voice)
    assert out[0, SR // 4] < 1.0  # not plain sum (0.5+0.5)
    # after release, second all-silent-voice block returns music toward full level
    out2 = d.process(music, np.zeros_like(voice))
    assert out2[0, -1] > 0.45


def test_no_voice_passthrough():
    d = Ducker(SR)
    music = (np.random.default_rng(0).standard_normal((2, SR)) * 0.1).astype(np.float32)
    out = d.process(music, np.zeros_like(music))
    assert np.allclose(out, music, atol=1e-4)


def test_attenuation_depth():
    d = Ducker(SR, depth_db=-10, attack=0.01)
    music = np.ones((2, SR), np.float32) * 0.5
    voice = np.ones((2, SR), np.float32) * 0.3
    out = d.process(music, voice)
    g = (out[0, -1] - 0.3) / 0.5  # residual music gain at block end
    assert 0.25 < g < 0.45        # ~ -10 dB ≈ 0.316
