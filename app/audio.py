"""Audio helpers: mulaw <-> PCM16, resampling, WAV parsing, and an energy VAD.

Pure numpy (audioop was removed in Python 3.13).
"""
import io
import struct
import wave
from collections import deque

import numpy as np

_BIAS = 0x84
_CLIP = 32635


def _build_ulaw_decode_table() -> np.ndarray:
    t = np.zeros(256, dtype=np.int16)
    for i in range(256):
        u = ~i & 0xFF
        sign, exp, man = u & 0x80, (u >> 4) & 7, u & 0x0F
        sample = (((man << 3) + _BIAS) << exp) - _BIAS
        t[i] = -sample if sign else sample
    return t


_ULAW_DECODE = _build_ulaw_decode_table()


def mulaw_to_pcm16(data: bytes) -> np.ndarray:
    return _ULAW_DECODE[np.frombuffer(data, dtype=np.uint8)]


def pcm16_to_mulaw(pcm: np.ndarray) -> bytes:
    x = pcm.astype(np.int32)
    sign = np.where(x < 0, 0x80, 0)
    x = np.minimum(np.abs(x), _CLIP) + _BIAS
    exp = np.floor(np.log2(x)).astype(np.int32) - 7
    exp = np.clip(exp, 0, 7)
    man = (x >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | man) & 0xFF).astype(np.uint8).tobytes()


def resample(pcm: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear-interpolation resample. Good enough for telephony speech."""
    if src == dst or len(pcm) == 0:
        return pcm
    n = int(round(len(pcm) * dst / src))
    xs = np.linspace(0, len(pcm) - 1, n)
    return np.interp(xs, np.arange(len(pcm)), pcm.astype(np.float32)).astype(np.int16)


def pcm16_to_wav(pcm: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


def wav_to_pcm16(data: bytes) -> tuple[np.ndarray, int]:
    """Parse WAV bytes -> (mono int16, sample_rate). Tolerates streaming headers with bogus sizes."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        # Header with 0/0xFFFFFFFF data size: find "data" chunk manually.
        if data[:4] != b"RIFF":
            raise
        idx = data.find(b"data")
        ch, rate, _, _, width_bits = struct.unpack("<HIIHH", data[22:36])
        width = width_bits // 8
        raw = data[idx + 8:]
    if width != 2:
        raise ValueError(f"unsupported WAV sample width {width}")
    pcm = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return pcm, rate


def wav_to_mulaw8k(data: bytes) -> bytes:
    pcm, rate = wav_to_pcm16(data)
    return pcm16_to_mulaw(resample(pcm, rate, 8000))


class EnergyVAD:
    """Frame-based endpointing for 8 kHz PCM (20 ms frames).

    Feed frames with `process(pcm)`; returns events "start" / "end" / None.
    Threshold adapts to the noise floor so phone-line hiss doesn't trigger it.
    """

    def __init__(self, frame_ms: int = 20, start_ms: int = 160, end_silence_ms: int = 700,
                 min_rms: float = 350.0, noise_factor: float = 3.0, preroll_ms: int = 300):
        self.frame_ms = frame_ms
        self.start_frames = max(1, start_ms // frame_ms)
        self.end_frames = max(1, end_silence_ms // frame_ms)
        self.min_rms = min_rms
        self.noise_factor = noise_factor
        self.noise_floor = 100.0
        self.in_speech = False
        self._voiced_run = 0
        self._silent_run = 0
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, preroll_ms // frame_ms))
        self._utt: list[np.ndarray] = []

    @property
    def threshold(self) -> float:
        return max(self.min_rms, self.noise_floor * self.noise_factor)

    def process(self, pcm: np.ndarray) -> str | None:
        rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) if len(pcm) else 0.0
        voiced = rms > self.threshold
        if not voiced and not self.in_speech:
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms  # adapt only on silence

        if not self.in_speech:
            self._preroll.append(pcm)
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run >= self.start_frames:
                self.in_speech = True
                self._silent_run = 0
                self._utt = list(self._preroll)
                self._preroll.clear()
                return "start"
            return None

        self._utt.append(pcm)
        self._silent_run = 0 if voiced else self._silent_run + 1
        if self._silent_run >= self.end_frames:
            self.in_speech = False
            self._voiced_run = 0
            return "end"
        return None

    def pop_utterance(self) -> np.ndarray:
        """Audio of the last utterance, with trailing silence trimmed to ~200 ms."""
        trim = max(0, self._silent_run - 10)
        frames = self._utt[: len(self._utt) - trim] if trim else self._utt
        self._utt = []
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)

    def reset(self) -> None:
        self.in_speech = False
        self._voiced_run = self._silent_run = 0
        self._preroll.clear()
        self._utt = []
