import numpy as np
from nordlys.reverb import Reverb, make_ir


def test_ir_shape_decay():
    rng = np.random.default_rng(1)
    ir = make_ir(8000, 1.0, rng)
    assert ir.shape[0] == 2 and ir.dtype == np.float32
    head = np.abs(ir[:, : ir.shape[1] // 4]).mean()
    tail = np.abs(ir[:, -ir.shape[1] // 4:]).mean()
    assert tail < head * 0.3


def test_impulse_produces_tail_across_blocks():
    rng = np.random.default_rng(1)
    block = 2048
    rv = Reverb(make_ir(8000, 0.5, rng), block)
    x = np.zeros((2, block), np.float32)
    x[:, 0] = 1.0
    first = rv.process(x)
    second = rv.process(np.zeros((2, block), np.float32))
    assert np.abs(first).max() > 0
    assert np.abs(second).max() > 0  # tail carried into next block
    assert np.isfinite(first).all() and np.isfinite(second).all()


def test_energy_bounded():
    rng = np.random.default_rng(2)
    block = 2048
    rv = Reverb(make_ir(8000, 0.5, rng), block)
    x = rng.standard_normal((2, block)).astype(np.float32) * 0.1
    for _ in range(10):
        y = rv.process(x)
    assert np.abs(y).max() < 1.0
