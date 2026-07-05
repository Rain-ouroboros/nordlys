import numpy as np
from nordlys.theory import MODES, Key, ChordWalk, scale_midi, midi_hz, drift_key


def test_scale_midi_within_mode():
    key = Key(root=2, mode="lydian")
    for m in scale_midi(key):
        assert (m - key.root) % 12 in MODES["lydian"]


def test_midi_hz_a440():
    assert abs(midi_hz(69) - 440.0) < 1e-6


def test_chord_walk_stays_in_scale():
    rng = np.random.default_rng(7)
    key = Key(root=0, mode="dorian")
    walk = ChordWalk(key, rng)
    for _ in range(50):
        chord = walk.next_chord()
        assert len(chord) == 3
        for m in chord:
            assert (m - key.root) % 12 in MODES["dorian"]


def test_chord_walk_voice_leading_bounded():
    rng = np.random.default_rng(7)
    walk = ChordWalk(Key(0, "lydian"), rng)
    prev = walk.next_chord()
    for _ in range(30):
        cur = walk.next_chord()
        assert abs(cur[0] - prev[0]) <= 12
        prev = cur


def test_drift_key_neighbor_fifths():
    rng = np.random.default_rng(3)
    key = Key(root=4, mode="lydian")
    for _ in range(20):
        nk = drift_key(key, rng)
        assert nk.root in {(key.root + 5) % 12, (key.root + 7) % 12}
        assert nk.mode in MODES
