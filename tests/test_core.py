import asyncio
import json
import os
import time
from datetime import datetime

import numpy as np

os.environ["LLM_PROVIDER"] = "mock"
os.environ["VALIDATE_TWILIO_SIGNATURE"] = "false"
os.environ["TWILIO_AUTH_TOKEN"] = "x"

from app import audio, dialer
from app.config import BotConfig, fill


def test_mulaw_roundtrip():
    t = np.arange(8000) / 8000
    pcm = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    back = audio.mulaw_to_pcm16(audio.pcm16_to_mulaw(pcm))
    assert np.abs(back.astype(int) - pcm).max() < 600  # mulaw quantisation error
    assert audio.pcm16_to_mulaw(np.array([0], dtype=np.int16)) == b"\xff"


def test_wav_and_resample():
    pcm = (3000 * np.random.randn(2400)).astype(np.int16)
    out, rate = audio.wav_to_pcm16(audio.pcm16_to_wav(pcm, 24000))
    assert rate == 24000 and len(out) == 2400
    assert len(audio.wav_to_mulaw8k(audio.pcm16_to_wav(pcm, 24000))) == 800


def test_vad_start_end():
    vad = audio.EnergyVAD()
    quiet = (50 * np.random.randn(160)).astype(np.int16)
    loud = (4000 * np.sin(np.arange(160) * 0.3)).astype(np.int16)
    ev = [vad.process(quiet) for _ in range(20)]
    assert not any(ev)
    ev = [vad.process(loud) for _ in range(30)]
    assert ev.count("start") == 1
    ev = [vad.process(quiet) for _ in range(40)]
    assert ev.count("end") == 1
    assert len(vad.pop_utterance()) > 160 * 30


def test_placeholders_and_prompts():
    assert fill("a {{x}} {{y}}", {"x": 1}) == "a 1 {{y}}"
    for name in ("loan", "insurance"):
        b = BotConfig(name)
        assert "{{" not in b.system_prompt and "{{" not in b.greeting, name
        assert "END_CALL" in b.system_prompt


def test_conversation_splits_and_end_call():
    from app.llm import MockLLM
    from app.session import Conversation
    c = Conversation(BotConfig("loan"), MockLLM())

    async def go(text):
        return [s async for s in c.reply(text)]
    s1 = asyncio.run(go("hello"))
    assert s1 == ["जी ठीक है।", "आप ईएमआई कब तक भर पाएँगे?"] and not c.end_call
    asyncio.run(go("kal"))
    s3 = asyncio.run(go("5 ko"))
    assert c.end_call and all("END" not in s for s in s3)
    assert c.history[-1]["role"] == "assistant"


def test_compliance_window():
    ist = dialer.IST
    assert dialer.compliance_check(False, "+91", "loan", datetime(2026, 1, 1, 10, 0, tzinfo=ist))[0]
    assert not dialer.compliance_check(False, "+91", "loan", datetime(2026, 1, 1, 19, 0, tzinfo=ist))[0]
    assert not dialer.compliance_check(False, "+91", "loan", datetime(2026, 1, 1, 7, 59, tzinfo=ist))[0]
    ok, d = dialer.compliance_check(True, "+91", "loan", datetime(2026, 1, 1, 23, 0, tzinfo=ist))
    assert ok and d == "FORCED_OUTSIDE_WINDOW"
