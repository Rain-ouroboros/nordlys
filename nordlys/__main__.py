import argparse
import threading

from .config import PORT, TTS_CMD
from .server import Station, make_server


def main():
    ap = argparse.ArgumentParser("nordlys")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-tts", action="store_true")
    args = ap.parse_args()

    station = Station(seed=args.seed, tts_cmd=None if args.no_tts else TTS_CMD)
    threading.Thread(target=station.run_pipeline, daemon=True).start()
    srv = make_server(station, args.port)
    print(f"nordlys on air: http://0.0.0.0:{args.port}/  (stream: /stream.mp3)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        station.stop.set()
        station.encoder.close()


if __name__ == "__main__":
    main()
