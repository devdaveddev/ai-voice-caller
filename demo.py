"""Demo caller without a phone line: you type as the customer, the bot answers via the LLM
set in .env (anthropic by default; ollama/mock optional) and speaks with Sarvam TTS (text-only if TTS fails). Each bot sentence is saved to demo_out/ as a WAV (8 kHz,
exactly what would be sent to Twilio) and played if a player is available.

  python demo.py --bot loan
"""
import argparse
import asyncio
import os
import shutil
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

from app.audio import mulaw_to_pcm16
from app.config import BotConfig
from app.llm import make_llm
from app.session import Conversation
from app.tts import SarvamTTS, TTSError

OUT = Path("demo_out")


def save_wav(path: Path, mulaw: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(mulaw_to_pcm16(mulaw).astype(np.int16).tobytes())


def play(path: Path) -> None:
    for player in (["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(player[0]):
            subprocess.run([*player, str(path)], check=False)
            return


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="loan", choices=["loan", "insurance"])
    ap.add_argument("--no-play", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    bot, tts, n = BotConfig(args.bot), SarvamTTS(), 0
    convo = Conversation(bot, make_llm())

    async def say(text: str) -> None:
        nonlocal n
        n += 1
        t0 = time.monotonic()
        try:
            audio = await tts.synthesize(text)
        except TTSError as e:
            print(f"   [TTS unavailable, continuing text-only: {str(e)[:80]}]")
            return
        p = OUT / f"{args.bot}_{n:02d}.wav"
        save_wav(p, audio)
        print(f"   [TTS {1000 * (time.monotonic() - t0):.0f}ms -> {p}]")
        if not args.no_play:
            await asyncio.get_running_loop().run_in_executor(None, play, p)

    print(f"BOT: {bot.greeting}")
    convo.add_assistant(bot.greeting)
    await say(bot.greeting)
    loop = asyncio.get_running_loop()
    while not convo.end_call:
        try:
            user = (await loop.run_in_executor(None, input, "YOU: ")).strip()
        except EOFError:
            break
        if user == "/quit":
            break
        async for sentence in convo.reply(user or "(customer is silent)"):
            print(f"BOT: {sentence}")
            await say(sentence)
    print("--- call ended ---" if convo.end_call else "--- quit ---")


if __name__ == "__main__":
    asyncio.run(main())
