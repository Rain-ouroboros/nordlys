import numpy as np
from nordlys.scene import SCENES, SceneScheduler, daynight_bias


def test_deterministic_with_seed():
    a = SceneScheduler(np.random.default_rng(9))
    b = SceneScheduler(np.random.default_rng(9))
    for _ in range(500):
        a.step(8.0)
        b.step(8.0)
        assert a.name == b.name and a.gains() == b.gains()


def test_transitions_happen_and_fade_blends():
    s = SceneScheduler(np.random.default_rng(3), min_dur=20, max_dur=30, fade=10)
    seen = {s.name.split("→")[0]}
    for _ in range(200):
        s.step(5.0)
        g = s.gains()
        assert set(g) == {"pad", "drone", "seq", "texture"}
        assert all(0.0 <= v <= 1.2 for v in g.values())
        seen.add(s.name.split("→")[0])
    assert s.transitions >= 3 and len(seen) >= 2


def test_skip_forces_transition():
    s = SceneScheduler(np.random.default_rng(4), min_dur=1000, max_dur=2000, fade=5)
    before = s.transitions
    s.skip()
    for _ in range(5):
        s.step(2.0)
    assert s.transitions == before + 1


def test_daynight_bias_dark_at_night():
    night, noon = daynight_bias(3), daynight_bias(13)
    assert night["brightness"] < noon["brightness"]
