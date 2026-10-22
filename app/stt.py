"""Speech-to-text wrapper (Sarvam REST). Swap the class to change provider.

Sarvam also has a streaming WebSocket (wss://api.sarvam.ai/speech-to-text/ws, 8 kHz supported),
but we already do our own endpointing, so one REST call per utterance is simple and robust.
"""
import logging

import httpx
import numpy as np

from .audio import pcm16_to_wav, resample
from .config import env

log = logging.getLogger("stt")


class STTError(Exception):
    pass


class SarvamSTT:
    URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=15)
        self.model = env("SARVAM_STT_MODEL", "saaras:v4")
        self.language = env("SARVAM_STT_LANGUAGE", "hi-IN")

    async def transcribe(self, pcm8k: np.ndarray) -> str:
        wav = pcm16_to_wav(resample(pcm8k, 8000, 16000), 16000)  # Sarvam works best at 16 kHz
        try:
            r = await self.client.post(
                self.URL,
                headers={"api-subscription-key": env("SARVAM_API_KEY")},
                files={"file": ("utt.wav", wav, "audio/wav")},
                data={"model": self.model, "language_code": self.language},
            )
        except httpx.HTTPError as e:
            raise STTError(f"network: {e!r}") from e
        if r.status_code != 200:
            raise STTError(f"HTTP {r.status_code}: {r.text[:200]}")
        return (r.json().get("transcript") or "").strip()
