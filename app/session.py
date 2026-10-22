"""Per-call state and the realtime loop: audio -> VAD -> STT -> LLM -> sentence TTS -> audio."""
import asyncio
import base64
import json
import logging
import re
import time
from datetime import datetime
from typing import Awaitable, Callable

import numpy as np

from . import storage
from .audio import EnergyVAD, mulaw_to_pcm16
from .config import BotConfig, env_float
from .stt import STTError
from .tts import TTSError

log = logging.getLogger("session")

END_RE = re.compile(r"\[?END_?CALL\]?")
SENT_RE = re.compile(r"[.!?।]+\s+")
FRAME = 160  # samples per 20 ms at 8 kHz


class Conversation:
    """History + streaming reply split into sentences. Shared by phone calls and the terminal tester."""

    def __init__(self, bot: BotConfig, llm):
        self.bot, self.llm = bot, llm
        self.history: list[dict] = []
        self.end_call = False

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    async def reply(self, user_text: str, timing: dict | None = None):
        """Async generator of speakable sentences. Sets self.end_call if the model emitted END_CALL."""
        t0 = time.monotonic()
        self.history.append({"role": "user", "content": user_text})
        buf, started = "", False

        def record(sentence: str) -> None:  # history only holds what was actually handed to TTS
            if self.history[-1]["role"] != "assistant":
                self.history.append({"role": "assistant", "content": sentence})
            else:
                self.history[-1]["content"] += " " + sentence

        async for delta in self.llm.stream(self.bot.system_prompt, self.history):
            if timing is not None and "llm_first_token" not in timing:
                timing["llm_first_token"] = time.monotonic() - t0
            buf += delta
            if END_RE.search(buf):
                self.end_call = True
            while True:
                cut = next((m.end() for m in SENT_RE.finditer(buf) if len(buf[: m.end()].strip()) >= 10), None)
                if cut is None:
                    break
                sentence, buf = END_RE.sub("", buf[:cut]).strip(), buf[cut:]
                if sentence:
                    if timing is not None and not started:
                        timing["llm_first_sentence"] = time.monotonic() - t0
                    started = True
                    record(sentence)
                    yield sentence
        rest = END_RE.sub("", buf).strip()
        if END_RE.search(buf):
            self.end_call = True
        if rest:
            if timing is not None and not started:
                timing["llm_first_sentence"] = time.monotonic() - t0
            record(rest)
            yield rest


class CallSession:
    def __init__(self, ws, bot: BotConfig, stt, tts, llm, hangup: Callable[[str], Awaitable[None]],
                 call_sid: str = "", to: str = ""):
        self.ws, self.bot, self.stt, self.tts, self.llm, self._hangup = ws, bot, stt, tts, llm, hangup
        self.convo = Conversation(bot, llm)
        self.vad = EnergyVAD(end_silence_ms=int(env_float("ENDPOINT_SILENCE_MS", 700)))
        self.silence_timeout = env_float("SILENCE_TIMEOUT_S", 5)
        self.stream_sid = ""
        self.call_sid, self.to = call_sid, to
        self.started_at = datetime.now()
        self.transcript: list[dict] = []
        self.turn_metrics: list[dict] = []
        self.end_reason = "unknown"

        self.pending_marks = 0
        self._mark_n = 0
        self.turn_task: asyncio.Task | None = None
        self.turn_stage = ""          # "", "stt", "llm", "speaking"
        self.user_speaking = False
        self.last_activity = time.monotonic()
        self.checkins = 0
        self.carry_audio = np.zeros(0, dtype=np.int16)
        self._resid = np.zeros(0, dtype=np.int16)
        self._cur_audio = np.zeros(0, dtype=np.int16)
        self._hung_up = False
        self._bg: list[asyncio.Task] = []
        self.turn_no = 0

    # ---------- plumbing ----------
    @property
    def bot_busy(self) -> bool:
        return (self.turn_task is not None and not self.turn_task.done()) or self.pending_marks > 0

    async def _send(self, obj: dict) -> None:
        await self.ws.send_text(json.dumps(obj))

    def _log_line(self, role: str, text: str) -> None:
        self.transcript.append({"t": datetime.now().isoformat(timespec="seconds"), "role": role, "text": text})
        log.info("[%s] %s: %s", self.call_sid[-6:], role.upper(), text)

    async def _send_audio(self, mulaw: bytes) -> None:
        for i in range(0, len(mulaw), 4000):
            await self._send({"event": "media", "streamSid": self.stream_sid,
                              "media": {"payload": base64.b64encode(mulaw[i:i + 4000]).decode()}})
        self._mark_n += 1
        self.pending_marks += 1
        await self._send({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": f"m{self._mark_n}"}})

    def _start_turn(self, coro) -> None:
        self.turn_task = asyncio.create_task(coro)
        self.turn_task.add_done_callback(self._turn_done)

    def _turn_done(self, t: asyncio.Task) -> None:
        if not t.cancelled() and t.exception():
            log.error("turn crashed: %r", t.exception())
        if t is self.turn_task:
            self.turn_stage = ""

    # ---------- main loop ----------
    async def run(self, initial: list[dict] | None = None) -> None:
        queued = list(initial or [])
        try:
            self._bg.append(asyncio.create_task(self._silence_watcher()))
            while True:
                msg = queued.pop(0) if queued else json.loads(await self.ws.receive_text())
                ev = msg.get("event")
                if ev == "start":
                    self.stream_sid = msg["start"]["streamSid"]
                    self.call_sid = self.call_sid or msg["start"].get("callSid", "")
                    storage.register_call(self.call_sid, self.bot.name)
                    log.info("stream started call=%s bot=%s", self.call_sid, self.bot.name)
                    self.last_activity = time.monotonic()
                    self._start_turn(self._say_fixed(self.bot.greeting))
                elif ev == "media":
                    if msg["media"].get("track", "inbound") == "inbound":
                        self._on_audio(base64.b64decode(msg["media"]["payload"]))
                elif ev == "mark":
                    self._on_mark()
                elif ev == "stop":
                    self.end_reason = self.end_reason if self.end_reason != "unknown" else "remote_hangup"
                    break
        except Exception as e:  # websocket disconnect etc.
            if type(e).__name__ != "WebSocketDisconnect":
                log.exception("session loop error")
            if self.end_reason == "unknown":
                self.end_reason = "disconnect"
        finally:
            await self._finalize()

    # ---------- inbound audio / VAD ----------
    def _on_audio(self, mulaw: bytes) -> None:
        pcm = np.concatenate([self._resid, mulaw_to_pcm16(mulaw)])
        n = len(pcm) // FRAME * FRAME
        self._resid = pcm[n:]
        for i in range(0, n, FRAME):
            ev = self.vad.process(pcm[i:i + FRAME])
            if ev == "start":
                self._on_user_start()
            elif ev == "end":
                self._on_user_end()

    def _on_user_start(self) -> None:
        self.user_speaking = True
        self.last_activity = time.monotonic()
        self.checkins = 0
        if self.bot_busy:
            asyncio.create_task(self._barge_in())

    def _on_user_end(self) -> None:
        self.user_speaking = False
        self.last_activity = time.monotonic()
        audio = np.concatenate([self.carry_audio, self.vad.pop_utterance()])
        self.carry_audio = np.zeros(0, dtype=np.int16)
        if len(audio) < 8000 * 0.3:  # <300 ms: noise/cough
            return
        self._cur_audio = audio
        self.turn_no += 1
        self._start_turn(self._run_turn(audio, time.monotonic(), self.turn_no))

    async def _barge_in(self) -> None:
        stage = self.turn_stage
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
            if stage == "stt":  # their previous words weren't transcribed yet: keep the audio
                self.carry_audio = self._cur_audio
        self.convo.end_call = False
        if self.pending_marks > 0 or stage in ("speaking", "llm"):
            await self._send({"event": "clear", "streamSid": self.stream_sid})
            log.info("[%s] BARGE-IN (stage=%s) -> clear", self.call_sid[-6:], stage)
        self.pending_marks = 0

    def _on_mark(self) -> None:
        self.pending_marks = max(0, self.pending_marks - 1)
        if self.pending_marks == 0:
            self.last_activity = time.monotonic()

    # ---------- bot turns ----------
    async def _say_fixed(self, text: str, record: bool = True) -> None:
        self.turn_stage = "speaking"
        if record:
            self._log_line("bot", text)
            self.convo.add_assistant(text)
        try:
            await self._send_audio(await self.tts.synthesize(text, cache=True))
        except TTSError as e:
            log.error("TTS failed for fixed line: %r", e)

    async def _run_turn(self, audio: np.ndarray, t_end: float, n: int) -> None:
        m = {"turn": n}
        self.turn_stage = "stt"
        try:
            text = await self.stt.transcribe(audio)
        except STTError as e:
            log.error("STT failed: %r", e)
            await self._say_fixed(self.bot.fallbacks.get("stt_error", "Sorry, please repeat."), record=False)
            return
        m["stt"] = time.monotonic() - t_end
        if not text:
            return
        self._log_line("user", text)
        self.turn_stage = "llm"

        q: asyncio.Queue = asyncio.Queue()
        player = asyncio.create_task(self._play(q, m, t_end))
        tts_tasks: list[asyncio.Task] = []
        try:
            try:
                async for sentence in self.convo.reply(text, m):
                    self._log_line("bot", sentence)
                    t = asyncio.create_task(self.tts.synthesize(sentence))  # prefetch while earlier audio plays
                    tts_tasks.append(t)
                    await q.put(t)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("LLM failed: %r", e)
                if not tts_tasks:
                    await q.put(asyncio.create_task(self.tts.synthesize(self.bot.fallbacks.get("llm_error", "Sorry."), )))
            await q.put(None)
            await player
        except asyncio.CancelledError:
            player.cancel()
            for t in tts_tasks:
                t.cancel()
            raise
        m["total_to_first_audio"] = m.get("first_audio_at", 0) and m["first_audio_at"] - t_end
        self.turn_metrics.append(m)
        log.info("[%s] TURN %d latency: stt=%.0fms llm_first_token=%.0fms first_sentence=%.0fms "
                 "tts_first_audio=%.0fms | speech_end->first_audio=%.0fms", self.call_sid[-6:], n,
                 m.get("stt", 0) * 1000, m.get("llm_first_token", 0) * 1000, m.get("llm_first_sentence", 0) * 1000,
                 m.get("tts", 0) * 1000, (m.get("total_to_first_audio") or 0) * 1000)
        if self.convo.end_call:
            self.end_reason = "agent_end_call"
            await self._end_after_playback()

    async def _play(self, q: asyncio.Queue, m: dict, t_end: float) -> None:
        first = True
        while (task := await q.get()) is not None:
            tw = time.monotonic()
            try:
                audio = await task
            except TTSError as e:
                log.error("TTS failed: %r", e)
                continue
            if first:
                self.turn_stage = "speaking"
                m["tts"] = time.monotonic() - tw
                m["first_audio_at"] = time.monotonic()
                first = False
            if audio:
                await self._send_audio(audio)

    async def _end_after_playback(self) -> None:
        deadline = time.monotonic() + 20
        while self.pending_marks > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.4)
        await self._do_hangup()

    async def _do_hangup(self) -> None:
        if self._hung_up:
            return
        self._hung_up = True
        log.info("[%s] hanging up (%s)", self.call_sid[-6:], self.end_reason)
        try:
            await self._hangup(self.call_sid)
        except Exception as e:
            log.error("hangup failed: %r", e)

    # ---------- silence ----------
    async def _silence_watcher(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            if not self.stream_sid or self.user_speaking or self.bot_busy:
                continue
            if time.monotonic() - self.last_activity < self.silence_timeout:
                continue
            self.checkins += 1
            self.last_activity = time.monotonic()
            if self.checkins > 3:
                self.end_reason = "silence_timeout"
                self._start_turn(self._goodbye())
                return
            self._start_turn(self._say_fixed(self.bot.checkin))

    async def _goodbye(self) -> None:
        await self._say_fixed(self.bot.fallbacks.get("goodbye", "Goodbye."))
        await self._end_after_playback()

    # ---------- wrap-up ----------
    async def _finalize(self) -> None:
        for t in self._bg + ([self.turn_task] if self.turn_task else []):
            if t and not t.done() and t is not asyncio.current_task():
                t.cancel()
        ended = datetime.now()
        text = "\n".join(f"[{x['t']}] {x['role'].upper()}: {x['text']}" for x in self.transcript)
        outcome: dict
        if not any(x["role"] == "user" for x in self.transcript):
            outcome = {"committed": False, "committed_date": None, "committed_amount": None,
                       "refused": False, "reason": None, "summary": "Customer never spoke."}
        else:
            try:
                outcome = await asyncio.wait_for(self.llm.extract_outcome(text), 25)
            except Exception as e:
                log.error("outcome extraction failed: %r", e)
                outcome = {"error": repr(e)}
        lat = [x for x in self.turn_metrics if x.get("total_to_first_audio")]
        record = {
            "call_sid": self.call_sid, "bot": self.bot.name, "to": self.to,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": ended.isoformat(timespec="seconds"),
            "duration_s": round((ended - self.started_at).total_seconds(), 1),
            "end_reason": self.end_reason, "outcome": outcome,
            "avg_speech_end_to_first_audio_ms": round(1000 * sum(x["total_to_first_audio"] for x in lat) / len(lat))
            if lat else None,
            "transcript": self.transcript,
        }
        try:
            storage.write_call_files(self.call_sid or "nosid", text, record)
            log.info("[%s] saved transcript+outcome: %s", self.call_sid[-6:], outcome)
        except Exception:
            log.exception("failed to save call files")
