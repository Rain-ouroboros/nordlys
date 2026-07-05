"""Standalone Silero TTS for nordlys' TTS_CMD.

Usage: python -m nordlys.tts_silero OUT_WAV "text ..."
Env:
  NORDLYS_SILERO_MODEL_PATH  explicit path to a silero .pt (wins)
  NORDLYS_SILERO_SPEAKER     default "xenia"
  NORDLYS_SILERO_RATE        default 24000

Must run under a python that has torch (+ optionally the `silero`
package for model auto-discovery). Kept import-light at module top so
nordlys' own venv (numpy-only) can import the package without torch.
"""
import glob
import inspect
import os
import sys
import wave
from array import array

THREAD_CAPS = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "TORCH_NUM_THREADS": "2",
}


def _resolve_model() -> str:
    explicit = os.environ.get("NORDLYS_SILERO_MODEL_PATH", "").strip()
    if explicit and os.path.exists(explicit):
        return explicit
    try:
        import silero
        base = os.path.join(os.path.dirname(silero.__file__), "model")
        found = sorted(glob.glob(os.path.join(base, "*.pt")))
        if found:
            return found[-1]
    except ImportError:
        pass
    raise SystemExit("no silero model: set NORDLYS_SILERO_MODEL_PATH")


def main() -> None:
    for k, v in THREAD_CAPS.items():
        os.environ.setdefault(k, v)
    out_wav, text = sys.argv[1], " ".join(sys.argv[2:]).strip()
    if not text:
        raise SystemExit("empty text")
    speaker = os.environ.get("NORDLYS_SILERO_SPEAKER", "xenia")
    rate = int(os.environ.get("NORDLYS_SILERO_RATE", "24000"))

    import torch
    torch.set_num_threads(2)
    model = torch.package.PackageImporter(_resolve_model()).load_pickle(
        "tts_models", "model")
    kwargs = {"text": text, "speaker": speaker, "sample_rate": rate}
    sig = inspect.signature(model.apply_tts)
    for key in ("put_accent", "put_yo", "put_stress_homo", "put_yo_homo"):
        if key in sig.parameters:
            kwargs[key] = True
    audio = model.apply_tts(**kwargs)
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().tolist()
    pcm = array("h", (max(-32767, min(32767, int(s * 32767))) for s in audio))
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
