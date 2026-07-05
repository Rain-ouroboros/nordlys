from nordlys.mood import Mood, MoodState


def test_step_converges_no_overshoot():
    st = MoodState(Mood(energy=0.0), tau=10.0)
    st.set_targets(energy=1.0)
    prev = 0.0
    for _ in range(100):
        cur = st.step(5.0).energy
        assert prev <= cur <= 1.0
        prev = cur
    assert cur > 0.99


def test_set_targets_clamps_and_ignores_unknown():
    st = MoodState()
    st.set_targets(warmth=7.0, bogus=1.0)
    assert st.target.warmth == 1.0
    assert not hasattr(st.target, "bogus")


def test_step_is_slow():
    st = MoodState(Mood(density=0.0), tau=45.0)
    st.set_targets(density=1.0)
    st.step(8.0)
    assert st.current.density < 0.25
