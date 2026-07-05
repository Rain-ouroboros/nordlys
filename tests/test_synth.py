import numpy as np
from nordlys.mood import Mood
from nordlys.synth import Voice, PadLayer, DroneLayer, SequenceLayer, TextureLayer

SR = 44100
MOOD = Mood()


def _check(block, n):
    assert block.shape == (2, n) and block.dtype == np.float32
    assert np.isfinite(block).all()
    assert np.abs(block).max() <= 1.0


def test_voice_shape_phase_continuity():
    v = Voice(220.0, SR)
    a = v.render(4096, 0.5)
    b = v.render(4096, 0.5)
    _check(a, 4096)
    _check(b, 4096)
    # no click at the seam: derivative bounded by max slope of summed partials
    seam = abs(float(b[0, 0]) - float(a[0, -1]))
    step = np.abs(np.diff(a[0])).max()
    assert seam <= step * 3


def test_voice_release_dies():
    v = Voice(220.0, SR, attack=0.01, release=0.05)
    v.render(4096, 0.5)
    v.release_now()
    for _ in range(20):
        v.render(4096, 0.5)
    assert v.dead


def test_pluck_decays_and_dies():
    v = Voice(440.0, SR, pluck=0.1)
    first = v.render(4096, 0.5)
    assert np.abs(first).max() > 0.01
    for _ in range(40):
        last = v.render(4096, 0.5)
    assert v.dead and np.abs(last).max() < 1e-3


def test_layers_render_and_sound():
    rng = np.random.default_rng(5)
    n = SR  # 1s
    pad, drone = PadLayer(SR, rng), DroneLayer(SR, rng)
    seq, tex = SequenceLayer(SR, rng), TextureLayer(SR, rng)
    pad.set_chord([60, 64, 67])
    drone.set_root(48)
    seq.set_scale([72, 74, 76, 79, 81])
    dense = Mood(density=1.0, energy=1.0)
    got_sound = {"pad": False, "drone": False, "seq": False, "tex": False}
    for _ in range(8):
        for name, layer in (("pad", pad), ("drone", drone), ("seq", seq), ("tex", tex)):
            blk = layer.render(n, dense)
            _check(blk, n)
            if np.abs(blk).max() > 1e-4:
                got_sound[name] = True
    assert all(got_sound.values()), got_sound


def test_pad_chord_change_no_explosion():
    rng = np.random.default_rng(5)
    pad = PadLayer(SR, rng)
    pad.set_chord([60, 64, 67])
    pad.render(SR, MOOD)
    pad.set_chord([62, 65, 69])
    blk = pad.render(SR, MOOD)
    assert np.abs(blk).max() <= 1.0
