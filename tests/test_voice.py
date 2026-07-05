import shutil
import wave

import numpy as np
import pytest

from nordlys.voice import VoiceBus, load_wav, TTSBackend


def _write_wav(path, sr=8000, n=4000, ch=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        t = np.arange(n) / sr
        x = (np.sin(2 * np.pi * 300 * t) * 20000).astype("<i2")
        frames = np.repeat(x, ch).tobytes()
        w.writeframes(frames)


def test_load_wav_resamples_and_normalizes(tmp_path):
    p = tmp_path / "a.wav"
    _write_wav(p, sr=8000, n=8000)
    x = load_wav(str(p), 16000)
    assert x.shape[0] == 2 and abs(x.shape[1] - 16000) < 20
    assert 0.65 < np.abs(x).max() <= 0.72


def test_pull_plays_then_silence(tmp_path):
    p = tmp_path / "a.wav"
    _write_wav(p, sr=8000, n=4000)
    bus = VoiceBus(8000, str(tmp_path / "spool"), tts_cmd=None)
    bus.enqueue_wav(str(p))
    got = bus.pull(3000)
    assert got.shape == (2, 3000) and np.abs(got).max() > 0.1 and bus.speaking
    bus.pull(3000)  # rest of message + start of gap
    for _ in range(5):
        tailing = bus.pull(3000)
    assert np.abs(tailing).max() == 0.0 and not bus.speaking


def test_spool_pickup(tmp_path):
    spool = tmp_path / "spool"
    spool.mkdir()
    _write_wav(spool / "msg.wav", sr=8000, n=2000)
    bus = VoiceBus(8000, str(spool), tts_cmd=None)
    bus.poll_spool()
    assert bus.queued == 1
    assert not list(spool.glob("*.wav"))  # moved to done/


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="no espeak-ng")
def test_tts_backend_espeak(tmp_path):
    b = TTSBackend("espeak-ng -v ru -s 140 -w {out} {text}")
    path = b.synth("привет")
    x = load_wav(path, 8000)
    assert np.abs(x).max() > 0.1
