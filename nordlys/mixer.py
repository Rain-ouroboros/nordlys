import numpy as np


class Ducker:
    """Sidechain: music dips depth_db while voice is active; frame-based envelope."""

    def __init__(self, sr, depth_db=-10.0, attack=0.5, release=1.5, thresh=0.01, hop=512):
        self.sr, self.hop, self.thresh = sr, hop, thresh
        self.floor = 10.0 ** (depth_db / 20.0)
        self.a_att = float(np.exp(-hop / (attack * sr)))
        self.a_rel = float(np.exp(-hop / (release * sr)))
        self.env = 0.0  # 0 = no ducking, 1 = fully ducked

    def process(self, music: np.ndarray, voice: np.ndarray) -> np.ndarray:
        n = music.shape[1]
        pad = (-n) % self.hop
        v = np.abs(voice).max(axis=0)
        if pad:
            v = np.pad(v, (0, pad))
        frames = v.reshape(-1, self.hop).max(axis=1)
        env = np.empty(len(frames))
        e = self.env
        for i, active in enumerate(frames > self.thresh):
            a = self.a_att if active else self.a_rel
            e = a * e + (1.0 - a) * (1.0 if active else 0.0)
            env[i] = e
        self.env = float(e)
        gain = 1.0 - (1.0 - self.floor) * np.repeat(env, self.hop)[:n]
        return (music * gain.astype(np.float32) + voice).astype(np.float32)
