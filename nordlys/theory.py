from dataclasses import dataclass

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MODES = {
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "major_pent": (0, 2, 4, 7, 9),
    "minor_pent": (0, 3, 5, 7, 10),
}


@dataclass(frozen=True)
class Key:
    root: int
    mode: str


def midi_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12)


def scale_midi(key: Key, lo: int = 36, hi: int = 84) -> list[int]:
    steps = MODES[key.mode]
    return [m for m in range(lo, hi + 1) if (m - key.root) % 12 in steps]


class ChordWalk:
    """Random walk over diatonic degrees; triads stacked in mode thirds."""

    def __init__(self, key: Key, rng):
        self.key = key
        self.rng = rng
        self.degree = 0

    def next_chord(self) -> list[int]:
        steps = MODES[self.key.mode]
        n = len(steps)
        self.degree = int((self.degree + self.rng.choice([-2, -1, 1, 2])) % n)
        base = 48 + self.key.root
        notes = []
        for k in (0, 2, 4):
            j = self.degree + k
            notes.append(base + steps[j % n] + 12 * (j // n))
        return notes


def drift_key(key: Key, rng) -> Key:
    root = int((key.root + rng.choice([5, 7])) % 12)
    mode = key.mode if rng.random() < 0.7 else str(rng.choice(list(MODES)))
    return Key(root, mode)
