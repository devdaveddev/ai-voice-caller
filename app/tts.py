"""Text-to-speech wrapper (Sarvam Bulbul REST). Returns mulaw 8 kHz bytes ready for Twilio.

We request WAV (documented) and convert ourselves, so we don't depend on how Sarvam
containers a raw `mulaw` response.
"""
import base64
import logging
import re

import httpx

from .audio import wav_to_mulaw8k
from .config import env, env_float

log = logging.getLogger("tts")


class TTSError(Exception):
    pass


def clean_for_speech(text: str) -> str:
    text = re.sub(r"\[?END_CALL\]?", "", text)
    text = re.sub(r"[*_`#>~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class SarvamTTS:
    URL = "https://api.sarvam.ai/text-to-speech"

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=15)
        self._cache: dict[str, bytes] = {}

    async def synthesize(self, text: str, cache: bool = False) -> bytes:
        text = clean_for_speech(text)
        if not text:
            return b""
        if cache and text in self._cache:
            return self._cache[text]
        payload = {
            "text": text[:2500],
            "language_code": env("SARVAM_TTS_LANGUAGE", "hi-IN"),
            "speaker": env("SARVAM_TTS_SPEAKER", "shubh"),
            "model": env("SARVAM_TTS_MODEL", "bulbul:v3"),
            "pace": env_float("SARVAM_TTS_PACE", 1.0),
            "speech_sample_rate": 8000,
            "output_audio_codec": "wav",
        }
        last = None
        for attempt in range(2):
            try:
                r = await self.client.post(
                    self.URL,
                    headers={"api-subscription-key": env("SARVAM_API_KEY")},
                    json=payload,
                )
                if r.status_code != 200:
                    raise TTSError(f"HTTP {r.status_code}: {r.text[:200]}")
                audio = wav_to_mulaw8k(base64.b64decode(r.json()["audios"][0]))
                if cache:
                    self._cache[text] = audio
                return audio
            except (httpx.HTTPError, TTSError, KeyError, ValueError) as e:
                last = e
                log.warning("TTS attempt %d failed: %r", attempt + 1, e)
        raise TTSError(str(last))
