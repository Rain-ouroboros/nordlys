import json
import threading
import urllib.error
import urllib.request

import pytest

from nordlys.server import Station, make_server


@pytest.fixture()
def station(tmp_path):
    st = Station(sr=8000, block=2000, seed=1, spool_dir=str(tmp_path / "spool"),
                 tts_cmd=None)
    srv = make_server(st, port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield st, srv.server_address[1]
    srv.shutdown()
    st.encoder.close()


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.status, r.read()


def _post(port, path, obj):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(obj).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_status(station):
    st, port = station
    code, body = _get(port, "/status")
    data = json.loads(body)
    assert code == 200
    assert {"mood", "scene", "listeners", "uptime_sec", "on_air", "speaking"} <= set(data)


def test_mood_roundtrip(station):
    st, port = station
    code, resp = _post(port, "/mood", {"energy": 0.9, "junk": 5})
    assert code == 200
    assert st.gen.status()["mood"]["target"]["energy"] == 0.9


def test_skip(station):
    st, port = station
    code, _ = _post(port, "/skip", {})
    assert code == 200


def test_say_without_tts_is_503(station):
    st, port = station
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(port, "/say", {"text": "привет"})
    assert e.value.code == 503


def test_index_served(station):
    st, port = station
    code, body = _get(port, "/")
    assert code == 200 and b"<html" in body.lower()


def test_404(station):
    st, port = station
    with pytest.raises(urllib.error.HTTPError):
        _get(port, "/nope")


def test_index_is_nordlys_ui(station):
    st, port = station
    _, body = _get(port, "/")
    html = body.decode()
    assert "canvas" in html and "#2e3440" in html.lower()


def test_query_string_stripped(station):
    st, port = station
    # UI requests /stream.mp3?<cache-buster>; must not 404
    r = urllib.request.urlopen(f"http://127.0.0.1:{port}/stream.mp3?123", timeout=5)
    assert r.status == 200
    assert r.headers["Content-Type"] == "audio/mpeg"
    r.close()
    code, _ = _get(port, "/status?x=1")
    assert code == 200


def test_pipeline_feeds_paced_chunks(tmp_path, monkeypatch):
    import threading
    import time as _time
    import nordlys.server as srv

    monkeypatch.setattr(srv, "LEAD_SEC", 1000.0)  # no pacing sleeps in test
    st = Station(sr=8000, block=24000, seed=2, spool_dir=str(tmp_path / "sp"),
                 tts_cmd=None)
    st.encoder.close()
    feeds = []
    st.encoder = type("FakeEnc", (), {"feed": lambda self, b: feeds.append(b.shape[1]),
                                      "close": lambda self: None})()
    t = threading.Thread(target=st.run_pipeline, daemon=True)
    t.start()
    deadline = _time.time() + 10
    while len(feeds) < 6 and _time.time() < deadline:
        _time.sleep(0.05)
    st.stop.set()
    t.join(timeout=5)
    assert len(feeds) >= 6
    # 3s blocks at 1s chunks: every feed is exactly 1s of audio
    assert set(feeds[:6]) == {8000}
