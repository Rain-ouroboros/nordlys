import numpy as np


def make_ir(sr: int, seconds: float, rng) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n) / sr
    env = np.exp(-3.0 * t / seconds)
    ir = rng.standard_normal((2, n)) * env
    onset = int(0.02 * sr)
    ir[:, :onset] *= np.linspace(0.0, 1.0, onset)
    ir *= 0.9 / np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    return ir.astype(np.float32)


class Reverb:
    def __init__(self, ir: np.ndarray, block: int):
        self.block = block
        m = ir.shape[1]
        size = 1
        while size < block + m - 1:
            size *= 2
        self.size = size
        self.H = np.fft.rfft(ir, size, axis=1)
        self.overlap = np.zeros((2, size - block))

    def process(self, x: np.ndarray) -> np.ndarray:
        X = np.fft.rfft(x, self.size, axis=1)
        y = np.fft.irfft(X * self.H, self.size, axis=1)
        y[:, : self.overlap.shape[1]] += self.overlap
        self.overlap = y[:, self.block:].copy()
        return y[:, : self.block].astype(np.float32)
