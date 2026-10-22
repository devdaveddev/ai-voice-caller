"""End-to-end websocket test with fake Twilio, STT, TTS and mock LLM. No network, no real calls."""
import base64
import json
import os
import time

import numpy as np
import pytest

os.environ["LLM_PROVIDER"] = "mock"
os.environ["VALIDATE_TWILIO_SIGNATURE"] = "false"
os.environ["TWILIO_AUTH_TOKEN"] = "x"
os.environ["SILENCE_TIMEOUT_S"] = "1"
os.environ["ENDPOINT_SILENCE_MS"] = "300"

from fastapi.testclient import TestClient

from app import audio, main, storage


class FakeSTT:
    async def transcribe(self, pcm):
        return "kal pay kar dunga"


class FakeTTS:
    async def synthesize(self, text, cache=False):
        return audio.pcm16_to_mulaw((2000 * np.sin(np.arange(4000) * 0.2)).astype(np.int16))  # 0.5 s


hung = []


async def fake_hangup(sid):
    hung.append(sid)


def frame(loud):
    pcm = (4000 * np.sin(np.arange(160) * 0.3)).astype(np.int16) if loud else (30 * np.random.randn(160)).astype(np.int16)
    return json.dumps({"event": "media", "streamSid": "MZ1", "media": {"track": "inbound", "payload": base64.b64encode(audio.pcm16_to_mulaw(pcm)).decode()}})


def recv_until(ws, pred, limit=200):
    seen = []
    for _ in range(limit):
        m = json.loads(ws.receive_text())
        seen.append(m)
        if pred(m):
            return seen
    raise AssertionError(f"not found in {[x['event'] for x in seen]}")


def test_full_call(monkeypatch):
    monkeypatch.setattr(main, "stt", FakeSTT())
    monkeypatch.setattr(main, "tts", FakeTTS())
    monkeypatch.setattr(main.dialer, "hangup", fake_hangup)
    sid = "CAtest123"
    client = TestClient(main.app)
    with client.websocket_connect("/media-stream") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(json.dumps({"event": "start", "streamSid": "MZ1", "start": {
            "streamSid": "MZ1", "callSid": sid,
            "customParameters": {"bot": "loan", "to": "+910000000000", "token": main._stream_token(sid)}}}))
        # greeting spoken first
        msgs = recv_until(ws, lambda m: m["event"] == "mark")
        assert any(m["event"] == "media" for m in msgs)
        ws.send_text(json.dumps({"event": "mark", "streamSid": "MZ1", "mark": {"name": "m1"}}))

        # user speaks -> endpointing -> reply streams back
        for _ in range(25): ws.send_text(frame(True))
        for _ in range(30): ws.send_text(frame(False))
        msgs = recv_until(ws, lambda m: m["event"] == "mark")
        assert any(m["event"] == "media" for m in msgs)

        # barge-in: user talks while bot audio is still "playing" (mark not acked)
        for _ in range(25): ws.send_text(frame(True))
        recv_until(ws, lambda m: m["event"] == "clear")

        # finish the utterance, then answer up to END_CALL
        for _ in range(30): ws.send_text(frame(False))
        for _ in range(3):
            for _ in range(25): ws.send_text(frame(True))
            for _ in range(30): ws.send_text(frame(False))
            time.sleep(0.3)
            for _ in range(10):
                ws.send_text(json.dumps({"event": "mark", "streamSid": "MZ1", "mark": {"name": "x"}}))
            if hung: break
        time.sleep(1.5)
        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ1", "stop": {"callSid": sid}}))
    time.sleep(0.5)
    assert hung == [sid]
    data = json.loads((storage.RECORDINGS_DIR / f"{storage.base_for(sid)}.json").read_text(encoding="utf-8"))
    assert data["end_reason"] == "agent_end_call"
    assert any(t["role"] == "user" for t in data["transcript"])
    assert not any("END_CALL" in t["text"] for t in data["transcript"])


def test_bad_token_rejected():
    client = TestClient(main.app)
    with pytest.raises(Exception):
        with client.websocket_connect("/media-stream") as ws:
            ws.send_text(json.dumps({"event": "start", "streamSid": "x", "start": {"streamSid": "x", "callSid": "CA1", "customParameters": {"token": "nope"}}}))
            ws.receive_text()


def test_twiml():
    os.environ["PUBLIC_BASE_URL"] = "https://abc.ngrok-free.app"
    r = TestClient(main.app).post("/twiml?bot=insurance", data={"CallSid": "CA1", "To": "+91"})
    assert 'wss://abc.ngrok-free.app/media-stream' in r.text and 'name="bot" value="insurance"' in r.text
